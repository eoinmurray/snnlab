"""Default fast projection semantics and optional public result snapshots."""

from dataclasses import replace

import pytest
import torch

from snnlab import extensions as E
from snnlab import lang
from snnlab.sim import (
    DenseArrayBinding,
    GraphExecutor,
    RecordingSpec,
    SignalRecording,
    load_runtime_state,
    plan_graph,
    retention_identity,
    runtime_state_signature,
    save_runtime_state,
    save_training_checkpoint,
)
from snnlab.sim.requests.handlers import simulate
from snnlab.sim.requests.spec import ExecutionSpec
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


def circuit(*, current=False, tau=0.001, recurrence=True):
    net = lang.Network("zero_decay", dt=1 * lang.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cells = net.population(
        "cells",
        size=2,
        neuron=lang.CUBA_ALIF(refractory=1 * lang.ms)
        if current
        else lang.LeakyIntegrator(tau=3 * lang.ms),
    )
    synapse = (
        lang.ExponentialCurrent(tau=tau * lang.ms)
        if current
        else lang.LeakyIntegrator(tau=3 * lang.ms)
    )
    projection = net.connect(
        events, cells.excitatory, name="drive", synapse=synapse, weight=lang.Constant(1)
    )
    if recurrence:
        net.connect(
            cells.spikes,
            cells.inhibitory,
            name="feedback",
            connection="recurrent",
            delay=2 * lang.ms,
            synapse=synapse,
            weight=lang.Constant(0.1),
            allow_autapses=False,
        )
    net.output("voltage", cells.voltage)
    net.output("spikes", cells.spikes)
    net.output("drive_state", projection.current if current else projection.conductance)
    return lang.compile(net).graph


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("current,tau", [(False, 3), (True, 0.001), (True, 3)])
def test_fast_strict_finite_trajectories_gradients_updates_and_continuation(
    device, current, tau, tmp_path
):
    graph = circuit(current=current, tau=tau)
    plan = plan_graph(graph)
    trainable = tuple(p.parameter for p in plan.projections)
    values = torch.ones(12, 3, 2, device=device)
    values[4:7] = 0
    results = []
    for mode in ("strict", "fast"):
        model = GraphExecutor(
            plan, seed=7, trainable_parameters=trainable, projection_mode=mode
        ).to(device)
        full = model({"events": values}, diagnostics=False)
        full.outputs["voltage"].sum().backward()
        gradients = {
            name: p.grad.clone()
            for name, p in model.parameter_map().items()
            if p.grad is not None
        }
        first = model({"events": values[:5]}, diagnostics=False)
        save_runtime_state(tmp_path / mode, first.runtime_state)
        resumed = model(
            {"events": values[5:]},
            diagnostics=False,
            runtime_state=load_runtime_state(tmp_path / mode, device=device),
        )
        for name in full.outputs:
            torch.testing.assert_close(
                torch.cat([first.outputs[name], resumed.outputs[name]]),
                full.outputs[name],
                atol=0,
                rtol=0,
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.step()
        model.enforce_constraints()
        results.append(
            (
                full,
                gradients,
                {name: p.detach().clone() for name, p in model.parameter_map().items()},
            )
        )
    for name in results[0][0].outputs:
        torch.testing.assert_close(
            results[0][0].outputs[name], results[1][0].outputs[name], atol=0, rtol=0
        )
    for kind in (1, 2):
        for name in results[0][kind]:
            torch.testing.assert_close(
                results[0][kind][name], results[1][kind][name], atol=0, rtol=0
            )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_previous_projection_is_cleared_only_in_fast_mode(bad):
    graph = circuit(recurrence=False)
    values = torch.ones(3, 1, 2)
    values[0] = bad
    for mode in ("strict", "fast"):
        result = GraphExecutor(plan_graph(graph), projection_mode=mode)(
            {"events": values}, diagnostics=False
        )
        drive = result.outputs["drive_state"]
        assert not torch.isfinite(drive[0]).all()
        assert bool(torch.isfinite(drive[1:]).all()) == (mode == "fast")


def test_fast_skips_zero_edges_even_with_nonfinite_backward():
    # The removed zero edge no longer transmits NaN from infinity * zero backward.
    graph = circuit(recurrence=False)
    for mode in ("strict", "fast"):
        values = torch.ones(3, 1, 2, requires_grad=True)
        model = GraphExecutor(plan_graph(graph), projection_mode=mode)
        result = model({"events": values}, diagnostics=False)
        (result.outputs["drive_state"][-1].sum() * float("inf")).backward()
        if mode == "strict":
            assert torch.isnan(values.grad[0]).all()
        else:
            assert torch.equal(values.grad[0], torch.zeros_like(values.grad[0]))


@pytest.mark.parametrize(
    "flags",
    [
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ],
)
def test_independent_snapshot_controls_preserve_results_and_recordings(flags):
    graph = circuit(current=True)
    model = GraphExecutor(plan_graph(graph), trainable_parameters=("drive.weight",))
    inputs = {"events": torch.ones(8, 2, 2)}
    recording = RecordingSpec((SignalRecording("cells.voltage"),), block_steps=3)
    reference = model(inputs, diagnostics=False, recording=recording)
    actual = model(
        inputs,
        diagnostics=False,
        recording=recording,
        retain_parameters=flags[0],
        retain_final_state=flags[1],
        retain_runtime_state=flags[2],
    )
    assert bool(actual.parameters) == flags[0]
    assert bool(actual.final_state) == flags[1]
    assert (actual.runtime_state is not None) == flags[2]
    for name in actual.outputs:
        torch.testing.assert_close(
            actual.outputs[name], reference.outputs[name], atol=0, rtol=0
        )
    torch.testing.assert_close(
        actual.recorded_signals["cells.voltage"],
        reference.recorded_signals["cells.voltage"],
        atol=0,
        rtol=0,
    )
    assert actual.metrics["result_snapshots"] == dict(
        zip(("parameters", "final_state", "runtime_state"), flags)
    )
    actual.outputs["voltage"].sum().backward()
    assert torch.isfinite(model.parameter_map()["drive.weight"].grad).all()


def test_runtime_modes_and_cache_retention_cannot_silently_change(tmp_path):
    graph = circuit()
    plan = plan_graph(graph)
    values = {"events": torch.ones(4, 1, 2)}
    strict = GraphExecutor(plan, projection_mode="strict")(values, diagnostics=False)
    fast = GraphExecutor(plan)(values, diagnostics=False)
    assert strict.runtime_state.signature == runtime_state_signature(
        plan, projection_mode="strict"
    )
    assert fast.runtime_state.signature != strict.runtime_state.signature
    with pytest.raises(ValueError, match="incompatible"):
        GraphExecutor(plan)(values, runtime_state=strict.runtime_state)
    base = dict(
        graph=graph, start_step=0, steps_count=4, batch_size=1, diagnostics=False
    )
    assert retention_identity(**base) != retention_identity(
        **base, projection_mode="strict"
    )
    assert retention_identity(**base) != retention_identity(
        **base, retain_runtime_state=False
    )
    assert "projection_mode" not in retention_identity(**base, projection_mode="strict")


def test_training_protocol_preserves_legacy_strict_and_rejects_mode_switch(tmp_path):
    bundle = direct_train_bundle()
    spec = ExecutionSpec(
        kind="train",
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(DenseArrayBinding("events", torch.ones(3, 2, 2)),),
        targets={"label": torch.tensor([0, 1])},
        device="cpu",
        diagnostics=False,
        updates=2,
        projection_mode="strict",
    )
    result = train(spec)
    assert "projection_mode" not in result.training_checkpoint.execution_protocol
    save_training_checkpoint(tmp_path / "checkpoint", result.training_checkpoint)
    continued = train(replace(spec, checkpoint=tmp_path / "checkpoint"))
    assert continued.training_checkpoint.completed_updates == 4
    with pytest.raises(ValueError, match="execution protocol"):
        train(replace(spec, checkpoint=tmp_path / "checkpoint", projection_mode="fast"))
    fast = train(replace(spec, projection_mode="fast"))
    assert fast.training_checkpoint.execution_protocol["projection_mode"] == "fast"


def test_request_defaults_and_snapshot_controls_reach_inference():
    graph = circuit()
    spec = ExecutionSpec(
        kind="infer",
        graph=graph,
        input_bindings=(DenseArrayBinding("events", torch.ones(4, 1, 2)),),
        device="cpu",
        diagnostics=False,
        retain_parameters=False,
        retain_final_state=False,
        retain_runtime_state=False,
    )
    result = simulate(spec)
    assert result.metrics["projection_mode"] == "fast"
    assert (
        not result.parameters
        and not result.final_state
        and result.runtime_state is None
    )
    assert result.outputs


@pytest.mark.parametrize("mode", ["fast", "strict"])
def test_custom_synapses_always_execute_callbacks(mode):
    name = f"zero-decay.callback-{mode}/v1"
    calls = []

    def initialize(ctx):
        return {"value": torch.zeros(ctx.shape, device=ctx.device, dtype=ctx.dtype)}

    def step(ctx):
        calls.append(1)
        return {"value": ctx.state["value"] * 0.5 + ctx.drive}

    E.register_synapse(name, step, initialize=initialize, output_unit="nA")
    net = lang.Network("custom_synapse", dt=1 * lang.ms)
    source = net.input(
        "events", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    cells = net.population("cells", size=1, neuron=lang.CUBA_LIF())
    p = net.connect(
        source,
        cells.excitatory,
        name="drive",
        synapse=lang.CustomSynapse(name),
        weight=lang.Constant(1),
    )
    net.output("current", p.current)
    result = GraphExecutor(plan_graph(lang.compile(net).graph), projection_mode=mode)(
        {"events": torch.ones(4, 1, 1)}, diagnostics=False
    )
    assert len(calls) == 4
    torch.testing.assert_close(
        result.outputs["current"][:, 0, 0], torch.tensor([1.0, 1.5, 1.75, 1.875])
    )


def test_cli_modes_and_no_snapshot_flags():
    from snnlab.sim.cli import execution_spec_from_args, parse_args

    spec = execution_spec_from_args(
        parse_args(
            [
                "sim",
                "--projection-mode",
                "strict",
                "--no-retain-parameters",
                "--no-retain-final-state",
                "--no-retain-runtime-state",
            ]
        )
    )
    assert spec.projection_mode == "strict"
    assert (
        not spec.retain_parameters
        and not spec.retain_final_state
        and not spec.retain_runtime_state
    )


def test_no_snapshots_reduces_clone_work_without_changing_gradients():
    graph = circuit(current=True)
    values = {"events": torch.ones(6, 2, 2)}
    model = GraphExecutor(plan_graph(graph), trainable_parameters=("drive.weight",))
    counts = []
    gradients = []
    for keep in (True, False):
        model.zero_grad(set_to_none=True)
        with torch.profiler.profile() as profile:
            result = model(
                values,
                diagnostics=False,
                retain_parameters=keep,
                retain_final_state=keep,
                retain_runtime_state=keep,
            )
        counts.append(
            sum(row.count for row in profile.key_averages() if row.key == "aten::clone")
        )
        result.outputs["voltage"].sum().backward()
        gradients.append(model.parameter_map()["drive.weight"].grad.clone())
    assert counts[1] < counts[0]
    torch.testing.assert_close(gradients[0], gradients[1], atol=0, rtol=0)


def test_dataset_snapshot_policy_and_authenticated_cache(tmp_path):
    import numpy as np

    from snnlab.sim import (
        DatasetEncoder,
        DatasetSnapshotBinding,
        validate_inference_artifacts,
        write_inference_artifacts,
    )

    graph = circuit(current=True)
    path = tmp_path / "inputs.npz"
    np.savez(
        path,
        features=np.ones((4, 5, 2), dtype=np.uint8),
        labels=np.zeros(5, dtype=np.int64),
    )
    binding = DatasetSnapshotBinding(
        path=path,
        input_id="events",
        dataset_id="zero-decay-policy",
        split="test",
        encoder=DatasetEncoder("prebinned_spikes"),
    )
    spec = ExecutionSpec(
        kind="infer",
        graph=graph,
        input_bindings=(binding,),
        batch_size=2,
        device="cpu",
        diagnostics=False,
        retain_parameters=False,
        retain_final_state=False,
        retain_runtime_state=False,
    )
    result = simulate(spec)
    assert (
        not result.parameters
        and not result.final_state
        and result.runtime_state is None
    )
    assert result.outputs["voltage"].shape == (4, 5, 2)
    identity = retention_identity(
        graph=graph,
        start_step=0,
        steps_count=4,
        batch_size=5,
        diagnostics=False,
        retain_parameters=False,
        retain_final_state=False,
        retain_runtime_state=False,
    )
    assert result.metrics["inference_retention"] == identity
    write_inference_artifacts(tmp_path / "cache", result, graph=graph, seed=spec.seed)
    validate_inference_artifacts(tmp_path / "cache", expected_retention=identity)
    with pytest.raises(ValueError, match="retention"):
        validate_inference_artifacts(
            tmp_path / "cache",
            expected_retention=retention_identity(
                graph=graph,
                start_step=0,
                steps_count=4,
                batch_size=5,
                diagnostics=False,
            ),
        )


def test_cli_cannot_save_omitted_runtime_snapshot():
    from snnlab.sim.cli import execution_spec_from_args, parse_args

    with pytest.raises(ValueError, match="requires"):
        execution_spec_from_args(
            parse_args(
                ["sim", "--no-retain-runtime-state", "--save-runtime-state", "state"]
            )
        )
