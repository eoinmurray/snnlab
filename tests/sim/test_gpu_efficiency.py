"""Execution efficiency contracts, with actual Inductor checks on CUDA."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from snnlab import lang
from snnlab.sim import (
    DatasetEncoder,
    DatasetSnapshotBinding,
    DenseArrayBinding,
    GraphExecutor,
    MeasurementWindow,
    RecordingSpec,
    SignalRecording,
    TrainingMetrics,
    plan_graph,
    save_training_checkpoint,
)
from snnlab.sim.cli import execution_spec_from_args
from snnlab.sim.inputs.provider import DatasetProvider, iter_batches
from snnlab.sim.requests import handlers as execution_functions
from snnlab.sim.requests.handlers import simulate
from snnlab.sim.requests.spec import ExecutionSpec
from snnlab.sim.training import trainer as training_execution
from snnlab.sim.training.trainer import train
from tests.sim._execution_builders import direct_train_bundle

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA unavailable"
        ),
    ),
]


def circuit(constructor):
    net = lang.Network("efficiency", dt=0.5 * lang.ms)
    source = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cells = net.population(
        "cells",
        size=3,
        neuron=constructor(refractory=1 * lang.ms, tau_mem=20 * lang.ms),
    )
    current = cells.neuron.kind.startswith("cuba_")
    synapse = (lang.ExponentialCurrent if current else lang.AMPA)(tau=3 * lang.ms)
    drive = net.connect(
        source,
        cells.excitatory,
        name="drive",
        synapse=synapse,
        weight=lang.Constant(1.0 if current else 0.1),
    )
    net.connect(
        cells.spikes,
        cells.excitatory,
        name="recurrence",
        connection="recurrent",
        delay=0.5 * lang.ms,
        synapse=synapse,
        weight=lang.Constant(0.1),
        allow_autapses=False,
    )
    scores = lang.FinalVoltage(source=cells.spikes, classes=2, name="scores")
    net.output("voltage", cells.voltage)
    net.output("spikes", cells.spikes)
    net.output("class_scores", scores)
    if "alif" in cells.neuron.kind or "adex" in cells.neuron.kind:
        net.output("adaptation", cells.state("adaptation"))
    return lang.compile(net).graph, drive.weight.id


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "constructor",
    [
        lang.CUBA_LIF,
        lang.COBA_LIF,
        lang.CUBA_ALIF,
        lang.COBA_ALIF,
        lang.CUBA_ADEX,
        lang.COBA_ADEX,
    ],
)
def test_compiled_outputs_gradients_and_continuation(constructor, device, monkeypatch):
    if device == "cpu":
        original = torch.compile
        monkeypatch.setattr(
            torch,
            "compile",
            lambda fn, **kw: original(
                fn, backend="eager", fullgraph=True, dynamic=False
            ),
        )
    graph, parameter = circuit(constructor)
    values = torch.ones(12, 2, 2, device=device)
    results = []
    for mode in (None, "default"):
        model = GraphExecutor(
            plan_graph(graph),
            seed=7,
            trainable_parameters=(parameter,),
            compile_mode=mode,
        ).to(device)
        full = model({"events": values}, diagnostics=False)
        full.outputs["class_scores"].sum().backward()
        gradient = model.parameter_map()[parameter].grad.clone()
        with torch.no_grad():
            first = model({"events": values[:5]}, diagnostics=False)
            tail = model(
                {"events": values[5:]},
                diagnostics=False,
                runtime_state=first.runtime_state,
            )
        torch.testing.assert_close(
            torch.cat((first.outputs["voltage"], tail.outputs["voltage"])),
            full.outputs["voltage"],
            atol=2e-5,
            rtol=2e-5,
        )
        assert (
            model.parameter_map()["recurrence.weight"].diagonal().count_nonzero() == 0
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        optimizer.step()
        model.enforce_constraints()
        results.append(
            (
                full,
                gradient,
                {
                    name: value.detach().clone()
                    for name, value in model.parameter_map().items()
                },
            )
        )
    for name in results[0][0].outputs:
        torch.testing.assert_close(
            results[0][0].outputs[name],
            results[1][0].outputs[name],
            atol=2e-5,
            rtol=2e-5,
        )
    torch.testing.assert_close(results[0][1], results[1][1], atol=2e-5, rtol=2e-5)
    for name in results[0][2]:
        torch.testing.assert_close(
            results[0][2][name], results[1][2][name], atol=2e-5, rtol=2e-5
        )


def dataset(tmp_path, graph):
    path = tmp_path / "dataset.npz"
    np.savez(
        path,
        features=np.asarray(
            [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.5, 0.5]],
            dtype=np.float32,
        ),
        labels=np.asarray([0, 1, 0, 1, 0]),
    )
    return DatasetSnapshotBinding(
        path=path,
        input_id="events",
        target_id="label",
        dataset_id="efficiency",
        split="train",
        encoder=DatasetEncoder(
            "rate_poisson", duration_ms=0.3, max_rate_hz=1000, seed=17
        ),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_prefetch_preserves_spikes_metadata_rng_and_partial_batch(tmp_path, device):
    bundle = direct_train_bundle()
    provider = DatasetProvider(
        bundle.graph, dataset(tmp_path, bundle.graph), device=device, execution_seed=5
    )
    requests = [
        (torch.arange(start, min(start + 2, 5)), "train", 2, None)
        for start in range(0, 5, 2)
    ]
    state = torch.get_rng_state().clone()
    reference = list(iter_batches(provider, requests, prefetch=0))
    actual = list(iter_batches(provider, requests, prefetch=2))
    assert torch.equal(state, torch.get_rng_state())
    assert len(actual) == 3
    for (_, tensors, meta), (_, expected, expected_meta) in zip(actual, reference):
        assert meta == expected_meta
        torch.testing.assert_close(
            tensors["events"], expected["events"], atol=0, rtol=0
        )


@pytest.mark.parametrize("device", DEVICES)
def test_prefetched_training_and_resume_are_exact(tmp_path, device):
    bundle = direct_train_bundle()
    spec = ExecutionSpec(
        kind="train",
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(dataset(tmp_path, bundle.graph),),
        device=device,
        diagnostics=False,
        epochs=2,
        batch_size=2,
        shuffle=True,
        seed=5,
    )
    reference = train(spec)
    actual = train(replace(spec, dataset_prefetch=2))
    assert reference.metrics["updates"] == actual.metrics["updates"]
    for name in reference.parameters:
        torch.testing.assert_close(
            reference.parameters[name], actual.parameters[name], atol=0, rtol=0
        )
    partial = train(replace(spec, dataset_prefetch=2, updates=2))
    save_training_checkpoint(tmp_path / "checkpoint", partial.training_checkpoint)
    resumed = train(
        replace(spec, checkpoint=tmp_path / "checkpoint", dataset_prefetch=2)
    )
    for name in reference.parameters:
        torch.testing.assert_close(
            reference.parameters[name], resumed.parameters[name], atol=0, rtol=0
        )


def test_discarded_candidates_do_not_snapshot(tmp_path, monkeypatch):
    bundle = direct_train_bundle()
    recipe = copy.deepcopy(bundle.training)
    for group in recipe["parameter_groups"]:
        group["lr"] = 0
    calls = []
    original = training_execution.capture_training_rng_state

    def capture(device):
        calls.append(device)
        return original(device)

    monkeypatch.setattr(training_execution, "capture_training_rng_state", capture)
    result = train(
        ExecutionSpec(
            kind="train",
            graph=bundle.graph,
            training=recipe,
            input_bindings=(DenseArrayBinding("events", torch.ones(3, 2, 2)),),
            targets={"label": torch.tensor([0, 1])},
            device="cpu",
            diagnostics=False,
            updates=4,
            epochs=4,
            metrics=TrainingMetrics(update_history=True),
        )
    )
    assert len(calls) == 2  # First selected candidate and final checkpoint.
    assert result.selected_checkpoint.completed_updates == 1
    assert all(isinstance(row["loss"], float) for row in result.metrics["updates"])
    assert all(
        isinstance(value, float)
        for row in result.metrics["updates"]
        for value in row["components"].values()
    )
    assert result.gradients


@pytest.mark.parametrize("device", DEVICES)
def test_recording_blocks_preserve_events_cells_and_bounds(device):
    graph, _ = circuit(lang.CUBA_ALIF)
    inputs = {"events": torch.ones(11, 2, 2, device=device)}
    signals = (
        SignalRecording("cells.spikes", cells=(2, 0), kind="spike_events"),
        SignalRecording("cells.voltage", cells=(1,)),
    )
    reference = GraphExecutor(plan_graph(graph)).to(device)(
        inputs,
        diagnostics=False,
        recording=RecordingSpec(signals, window=MeasurementWindow(1, 10)),
    )
    blocks = []
    actual = GraphExecutor(plan_graph(graph)).to(device)(
        inputs,
        diagnostics=False,
        recording=RecordingSpec(
            signals,
            window=MeasurementWindow(1, 10),
            sink=blocks.append,
            sink_id="efficiency/v1",
            block_steps=4,
        ),
        batch_offset=7,
    )
    for row in signals:
        selected = [b for b in blocks if b.signal == row.signal]
        assert [(b.start_step, b.end_step) for b in selected] == [
            (1, 5),
            (5, 9),
            (9, 10),
        ]
        payload = torch.cat([b.values for b in selected])
        expected = reference.recorded_signals[row.signal].clone()
        if row.kind == "spike_events":
            expected[:, 1] += 7
        torch.testing.assert_close(payload, expected, atol=0, rtol=0)
    for name in reference.outputs:
        torch.testing.assert_close(
            actual.outputs[name], reference.outputs[name], atol=0, rtol=0
        )


def test_timing_modes_use_explicit_boundaries(monkeypatch):
    graph, _ = circuit(lang.CUBA_LIF)
    calls = []
    monkeypatch.setattr(execution_functions, "_synchronize_device", calls.append)
    spec = ExecutionSpec(
        kind="infer",
        graph=graph,
        input_bindings=(DenseArrayBinding("events", torch.ones(3, 1, 2)),),
        device="cpu",
        diagnostics=False,
    )
    result = simulate(spec)
    assert calls == ["cpu", "cpu"]
    assert result.metrics["simulate_s"] >= 0
    calls.clear()
    result = simulate(replace(spec, timing_mode="enqueue"))
    assert not calls
    assert "simulate_s" not in result.metrics
    assert result.metrics["simulate_enqueue_s"] >= 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"compile_mode": "bad"},
        {"dataset_prefetch": -1},
        {"dataset_prefetch": True},
        {"timing_mode": "bad"},
    ],
)
def test_invalid_efficiency_options(kwargs):
    with pytest.raises(ValueError):
        ExecutionSpec(kind="build", **kwargs)


def test_cuda_graph_mode_is_rejected():
    with pytest.raises(ValueError, match="compile_mode"):
        ExecutionSpec(kind="build", compile_mode="reduce-overhead")
    graph, _ = circuit(lang.CUBA_ALIF)
    with pytest.raises(ValueError, match="compile_mode"):
        GraphExecutor(plan_graph(graph), compile_mode="reduce-overhead")


def test_cli_efficiency_options_map_to_request():
    from snnlab.sim.cli import parse_args

    spec = execution_spec_from_args(
        parse_args(
            [
                "sim",
                "--compile-mode",
                "default",
                "--dataset-prefetch",
                "2",
                "--timing-mode",
                "enqueue",
            ]
        )
    )
    assert spec.compile_mode == "default"
    assert spec.dataset_prefetch == 2
    assert spec.timing_mode == "enqueue"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("cells", [None, (2, 0)])
def test_cuda_recording_stages_cell_indices_before_stepping(cells):
    from snnlab.sim.runtime.recording import Recorder

    values = torch.arange(11 * 2 * 3, device="cuda", dtype=torch.float32).reshape(
        11, 2, 3
    )
    recorder = Recorder(
        RecordingSpec((SignalRecording("cells.voltage", cells=cells),), block_steps=4),
        axes={"cells.voltage": ("time", "batch", 3)},
        start=0,
        steps=11,
        dt_ms=1,
        device=values.device,
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        for step in range(11):
            recorder.observe({"cells.voltage": values[step]}, step)
        recorded = recorder.finish(2)
    counts = {row.key: row.count for row in profile.key_averages()}
    # Python-list indexing previously copied a fresh CPU index tensor to CUDA
    # on every observation. Chunk copies use copy_ directly, without Tensor.to.
    assert counts.get("aten::_to_copy", 0) == 0
    expected = values.cpu() if cells is None else values.cpu()[:, :, list(cells)]
    torch.testing.assert_close(recorded["cells.voltage"], expected, atol=0, rtol=0)
