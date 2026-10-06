"""Authenticated emitted-spike replacement, delay causality and continuation."""

from dataclasses import replace

import pytest
import torch

from snnlab import lang as snn
from snnlab.sim.execution import (
    AddPoissonSpikes,
    DenseArrayBinding,
    DenseSpikeReplay,
    DropSpikes,
    ExecutionSpec,
    GraphExecutor,
    ReplaySpikes,
    SparseSpikeReplay,
    execution_spec_from_args,
    intervention_identity,
    load_spike_replay,
    plan_graph,
    save_spike_replay,
    simulate,
    validate_inference_artifacts,
    write_inference_artifacts,
)
from snnlab.sim.tool import parse_args
from tests._circuits import author_ping
from tests.sim._execution_builders import expose_graph_diagnostics


def graph_fixture(neuron=None, bundle_path=None):
    net = snn.Network("replay", dt=1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    source = net.population(
        "source",
        size=2,
        neuron=neuron or snn.LeakyIntegrator(tau=2 * snn.ms, soft_reset_threshold=1),
        spiking=True,
    )
    immediate = net.population(
        "immediate", size=2, neuron=snn.LeakyIntegrator(tau=3 * snn.ms), spiking=False
    )
    delayed = net.population(
        "delayed", size=2, neuron=snn.LeakyIntegrator(tau=3 * snn.ms), spiking=False
    )
    net.connect(
        events,
        source.excitatory,
        name="drive",
        synapse=snn.ExponentialCurrent(tau=2 * snn.ms)
        if neuron is not None
        else snn.LeakyIntegrator(tau=2 * snn.ms),
        weight=snn.Constant(10),
    )
    zero = net.connect(
        source.spikes,
        immediate.excitatory,
        name="zero",
        synapse=snn.LeakyIntegrator(tau=2 * snn.ms),
        weight=snn.Constant(2),
    )
    lag = net.connect(
        source.spikes,
        delayed.excitatory,
        name="lag",
        synapse=snn.LeakyIntegrator(tau=2 * snn.ms),
        weight=snn.Constant(2),
        delay=2 * snn.ms,
    )
    net.expose(
        source.spikes,
        source.voltage,
        immediate.voltage,
        delayed.voltage,
        zero.conductance,
        lag.conductance,
        name="traces",
    )
    net.output("emitted", source.spikes)
    net.output("immediate_out", immediate.voltage)
    net.output("delayed_out", delayed.voltage)
    bundle = snn.compile(net, target=None)
    if bundle_path is not None:
        bundle.write(bundle_path)
    graph = expose_graph_diagnostics(bundle.graph)
    if neuron is not None:
        graph["observables"] = [
            row for row in graph["observables"] if row["signal"] != "drive.conductance"
        ]
    return graph


def stream():
    value = torch.zeros(6, 2, 2)
    value[0, 0, 0] = 1
    value[2, 1, 1] = 1
    value[3, 0, 1] = 1
    value[5, 1, 0] = 1
    return value


def dense(value=None, start_step=0):
    return DenseSpikeReplay.from_tensor(
        stream() if value is None else value, start_step=start_step, dt_ms=1
    )


def sparse(value=None):
    value = stream() if value is None else value
    coords = torch.nonzero(value)
    return SparseSpikeReplay.from_events(
        steps=coords[:, 0],
        batches=coords[:, 1],
        cells=coords[:, 2],
        steps_count=value.shape[0],
        batch_size=value.shape[1],
        cells_count=value.shape[2],
        dt_ms=1,
    )


def run(graph=None, replay=None, inputs=None, interventions=None):
    return simulate(
        ExecutionSpec(
            kind="infer",
            graph=graph or graph_fixture(),
            input_bindings=(
                DenseArrayBinding(
                    "events", torch.zeros(6, 2, 2) if inputs is None else inputs
                ),
            ),
            seed=17,
            device="cpu",
            interventions=interventions
            if interventions is not None
            else (ReplaySpikes("source", replay or dense()),),
        )
    )


def test_replacement_reaches_zero_delay_delayed_edges_and_recordings():
    result = run()
    expected = stream()
    torch.testing.assert_close(result.outputs["emitted"], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        result.diagnostics["source.spikes"], expected, rtol=0, atol=0
    )
    assert torch.count_nonzero(result.outputs["immediate_out"][0]) > 0
    assert torch.count_nonzero(result.outputs["delayed_out"][:2]) == 0
    assert torch.count_nonzero(result.outputs["delayed_out"][2]) > 0
    identity = result.metrics["inference_interventions"]
    assert identity["window"]["start_step"] == 0
    assert identity["resolved"][0]["replay"]["sha256"] == dense().sha256
    assert result.runtime_state.completed_steps == 6


@pytest.mark.parametrize("replay_factory", [dense, sparse])
def test_dense_sparse_and_chunked_execution_agree(replay_factory):
    graph = graph_fixture()
    replay = ReplaySpikes("source", replay_factory())
    inputs = {"events": torch.zeros(6, 2, 2)}
    model = GraphExecutor(plan_graph(graph), seed=17)
    full = model(inputs, interventions=(replay,))
    first = model({"events": inputs["events"][:3]}, interventions=(replay,))
    second = model(
        {"events": inputs["events"][3:]},
        runtime_state=first.runtime_state,
        interventions=(replay,),
    )
    for name in full.outputs:
        torch.testing.assert_close(
            full.outputs[name],
            torch.cat((first.outputs[name], second.outputs[name])),
            rtol=0,
            atol=0,
        )
    for name in full.runtime_state.voltages:
        torch.testing.assert_close(
            full.runtime_state.voltages[name],
            second.runtime_state.voltages[name],
            rtol=0,
            atol=0,
        )
    for name in full.runtime_state.population_histories:
        torch.testing.assert_close(
            full.runtime_state.population_histories[name],
            second.runtime_state.population_histories[name],
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        full.outputs["immediate_out"],
        run(replay=dense()).outputs["immediate_out"],
        rtol=0,
        atol=0,
    )


def test_absolute_sparse_coordinates_and_nonzero_dense_coverage():
    model = GraphExecutor(plan_graph(graph_fixture()), seed=17)
    first = model({"events": torch.zeros(3, 2, 2)})
    payload = dense(stream()[3:], start_step=3)
    second = model(
        {"events": torch.zeros(3, 2, 2)},
        runtime_state=first.runtime_state,
        interventions=(ReplaySpikes("source", payload),),
    )
    torch.testing.assert_close(second.outputs["emitted"], stream()[3:], rtol=0, atol=0)
    events = torch.nonzero(stream()[3:])
    sparse_payload = SparseSpikeReplay.from_events(
        steps=events[:, 0] + 3,
        batches=events[:, 1],
        cells=events[:, 2],
        start_step=3,
        steps_count=3,
        batch_size=2,
        cells_count=2,
        dt_ms=1,
    )
    sparse_second = model(
        {"events": torch.zeros(3, 2, 2)},
        runtime_state=first.runtime_state,
        interventions=(ReplaySpikes("source", sparse_payload),),
    )
    torch.testing.assert_close(
        second.outputs["emitted"], sparse_second.outputs["emitted"], rtol=0, atol=0
    )


def test_zero_replay_replaces_natural_spikes_but_preserves_local_neuron_state():
    inputs = torch.ones(6, 2, 2)
    baseline = run(inputs=inputs, interventions=())
    assert torch.count_nonzero(baseline.outputs["emitted"]) > 0
    zero = run(inputs=inputs, replay=dense(torch.zeros_like(inputs)))
    assert torch.count_nonzero(zero.outputs["emitted"]) == 0
    assert torch.count_nonzero(zero.outputs["immediate_out"]) == 0
    torch.testing.assert_close(
        zero.diagnostics["source.voltage"],
        baseline.diagnostics["source.voltage"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        zero.runtime_state.refractory["source"],
        baseline.runtime_state.refractory["source"],
        rtol=0,
        atol=0,
    )
    identity = run(inputs=inputs, replay=dense(baseline.outputs["emitted"]))
    for name in baseline.outputs:
        torch.testing.assert_close(
            identity.outputs[name], baseline.outputs[name], rtol=0, atol=0
        )
    for name in baseline.parameters:
        torch.testing.assert_close(
            identity.parameters[name], baseline.parameters[name], rtol=0, atol=0
        )


def test_intervention_composition_is_ordered_and_replay_is_not_addition():
    replay = ReplaySpikes("source", dense())
    drop = DropSpikes("source", 1)
    add = AddPoissonSpikes("source", 1000)
    assert (
        torch.count_nonzero(run(interventions=(replay, drop)).outputs["emitted"]) == 0
    )
    torch.testing.assert_close(
        run(interventions=(drop, replay)).outputs["emitted"], stream(), rtol=0, atol=0
    )
    assert torch.all(run(interventions=(replay, add)).outputs["emitted"] == 1)
    torch.testing.assert_close(
        run(interventions=(add, replay)).outputs["emitted"], stream(), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "change, message",
    [
        ({"sha256": "sha256:" + "0" * 64}, "digest mismatch"),
        ({"value": torch.ones(6, 2, 2) * 0.5}, "binary"),
        ({"value": torch.zeros(6, 2)}, "binary"),
        ({"start_step": -1}, "integer"),
        ({"dt_ms": 0}, "positive"),
    ],
)
def test_invalid_dense_payloads_fail(change, message):
    with pytest.raises(ValueError, match=message):
        run(replay=replace(dense(), **change))


@pytest.mark.parametrize(
    "payload,message",
    [
        (lambda: dense(torch.zeros(5, 2, 2)), "cover"),
        (lambda: dense(torch.zeros(6, 1, 2)), "batch/cell"),
        (lambda: dense(torch.zeros(6, 2, 3)), "batch/cell"),
        (lambda: DenseSpikeReplay.from_tensor(stream(), dt_ms=2), "timestep"),
    ],
)
def test_authenticated_payload_must_cover_exact_target(payload, message):
    with pytest.raises(ValueError, match=message):
        run(replay=payload())


@pytest.mark.parametrize(
    "change,message",
    [
        ({"steps": torch.tensor([0, 2, 3, 6])}, "out of range"),
        ({"batches": torch.tensor([0, 1, 0, 2])}, "out of range"),
        ({"cells": torch.tensor([0, 1, 1, 2])}, "out of range"),
        ({"steps": torch.tensor([0.0, 2.0, 3.0, 5.0])}, "integer"),
        (
            {
                "steps": torch.tensor([0, 0, 0, 0]),
                "batches": torch.zeros(4, dtype=torch.long),
                "cells": torch.zeros(4, dtype=torch.long),
            },
            "duplicate",
        ),
        ({"steps": torch.tensor([5, 2, 3, 0])}, "ordered"),
        ({"steps": torch.tensor([0, 2, 3])}, "lengths"),
    ],
)
def test_sparse_coordinates_fail_closed(change, message):
    with pytest.raises(ValueError, match=message):
        run(replay=replace(sparse(), **change))


def test_nonspiking_unknown_duplicate_targets_and_old_option_fail():
    for target, message in [
        ("missing", "unknown population"),
        ("immediate", "does not emit spikes"),
    ]:
        with pytest.raises(ValueError, match=message):
            run(interventions=(ReplaySpikes(target, dense()),))
    with pytest.raises(ValueError, match="repeats"):
        run(
            interventions=(
                ReplaySpikes("source", dense()),
                ReplaySpikes("source", dense()),
            )
        )
    with pytest.raises(TypeError, match="objects"):
        run(interventions=({"kind": "replay_spikes"},))
    with pytest.raises(ValueError, match="ExecutionSpec.interventions"):
        simulate(
            ExecutionSpec(
                kind="infer",
                graph=graph_fixture(),
                options={"inference_interventions": []},
            )
        )


def test_cuba_lif_spikes_can_be_replayed():
    result = run(graph=graph_fixture(snn.CUBA_LIF()))
    torch.testing.assert_close(result.outputs["emitted"], stream(), rtol=0, atol=0)


def test_mutation_after_digest_is_rejected_and_factory_owns_snapshot():
    original = stream()
    replay = DenseSpikeReplay.from_tensor(original, dt_ms=1)
    original.zero_()
    torch.testing.assert_close(
        run(replay=replay).outputs["emitted"], stream(), rtol=0, atol=0
    )
    replay.value.zero_()
    with pytest.raises(ValueError, match="digest mismatch"):
        run(replay=replay)


def test_replay_files_cli_pin_and_identical_content_relocation(tmp_path):
    for payload in [dense(), sparse()]:
        path = tmp_path / "replay.npz"
        file_digest = save_spike_replay(path, payload)
        relocated = tmp_path / "copy.npz"
        relocated.write_bytes(path.read_bytes())
        loaded = load_spike_replay(relocated, sha256=file_digest)
        torch.testing.assert_close(
            run(replay=loaded).outputs["emitted"], stream(), rtol=0, atol=0
        )
        spec = execution_spec_from_args(
            parse_args(
                [
                    "sim",
                    "--executor",
                    "graph",
                    "--intervention",
                    f"replay:source={path}@{file_digest}",
                ]
            )
        )
        assert isinstance(spec.interventions[0], ReplaySpikes)
        assert "intervention" not in spec.options
        save_spike_replay(path, dense(torch.zeros_like(stream())))
        with pytest.raises(ValueError, match="file digest"):
            load_spike_replay(path, sha256=file_digest)


def test_cache_rejects_different_replay_and_order_and_preserves_integrity(tmp_path):
    graph = graph_fixture()
    interventions = (ReplaySpikes("source", dense()), DropSpikes("source", 0.5))
    result = run(graph=graph, interventions=interventions)
    write_inference_artifacts(tmp_path, result, graph=graph, seed=17)
    expected = intervention_identity(
        interventions, graph=graph, seed=17, steps_count=6, batch_size=2
    )
    validate_inference_artifacts(
        tmp_path, graph=graph, seed=17, expected_interventions=expected
    )
    changes = [
        (
            ReplaySpikes("source", dense(torch.zeros_like(stream()))),
            DropSpikes("source", 0.5),
        ),
        tuple(reversed(interventions)),
    ]
    for changed in changes:
        identity = intervention_identity(
            changed, graph=graph, seed=17, steps_count=6, batch_size=2
        )
        with pytest.raises(ValueError, match="expected identity"):
            validate_inference_artifacts(tmp_path, expected_interventions=identity)
    with pytest.raises(ValueError, match="expected identity"):
        validate_inference_artifacts(tmp_path, expected_interventions=None)
    path = tmp_path / "outputs.npz"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="digest"):
        validate_inference_artifacts(tmp_path, expected_interventions=expected)


def test_diagnostics_do_not_change_outputs_or_rng():
    spec = ExecutionSpec(
        kind="infer",
        graph=graph_fixture(),
        input_bindings=(DenseArrayBinding("events", torch.zeros(6, 2, 2)),),
        seed=17,
        device="cpu",
        interventions=(
            ReplaySpikes("source", dense()),
            AddPoissonSpikes("source", 500),
        ),
    )
    first = simulate(spec)
    state = torch.get_rng_state().clone()
    second = simulate(replace(spec, diagnostics=False))
    assert torch.equal(state, torch.get_rng_state())
    assert not second.diagnostics
    for name in first.outputs:
        torch.testing.assert_close(
            first.outputs[name], second.outputs[name], rtol=0, atol=0
        )


def test_direct_forward_artifacts_retain_authenticated_replay_identity(tmp_path):
    graph = graph_fixture()
    interventions = (ReplaySpikes("source", dense()),)
    model = GraphExecutor(plan_graph(graph), seed=17)
    result = model({"events": torch.zeros(6, 2, 2)}, interventions=interventions)
    expected = intervention_identity(
        interventions, graph=graph, seed=17, steps_count=6, batch_size=2
    )
    write_inference_artifacts(tmp_path, result, graph=graph, seed=17)
    validate_inference_artifacts(tmp_path, expected_interventions=expected)


def test_replay_cli_runs_typed_request_and_writes_authenticated_artifacts(tmp_path):
    import numpy as np

    from snnlab.sim.bundle import load_graph_bundle
    from snnlab.sim.tool import main

    bundle_path = tmp_path / "network.bundle"
    graph_fixture(bundle_path=bundle_path)
    _, graph = load_graph_bundle(bundle_path)
    input_path = tmp_path / "inputs.npy"
    np.save(input_path, np.zeros((6, 2, 2), dtype=np.float32))
    replay_path = tmp_path / "replay.npz"
    file_digest = save_spike_replay(replay_path, dense())
    out = tmp_path / "out"
    main(
        [
            "sim",
            "--executor",
            "graph",
            "--bundle",
            str(bundle_path),
            "--input-file",
            str(input_path),
            "--out-dir",
            str(out),
            "--seed",
            "17",
            "--device",
            "cpu",
            "--intervention",
            f"replay:source={replay_path}@{file_digest}",
        ]
    )
    expected = intervention_identity(
        (ReplaySpikes("source", dense()),),
        graph=graph,
        seed=17,
        steps_count=6,
        batch_size=2,
    )
    validate_inference_artifacts(
        out, graph=graph, seed=17, expected_interventions=expected
    )
    with np.load(out / "outputs.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["emitted"], stream().numpy())


def test_inhibitory_replay_drives_conductance_on_the_declared_delay():
    net = snn.Network("replay_delay", dt=0.1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    author_ping(net, name="cell", n_e=4, n_i=1, source=events)
    graph = expose_graph_diagnostics(snn.compile(net).graph)
    model = GraphExecutor(plan_graph(graph), seed=7)
    with torch.no_grad():
        for value in model.parameter_map().values():
            value.zero_()
        model.parameter_map()["cell_I_to_E.weight"].fill_(5)
    replacement = torch.zeros(40, 2, 1)
    replacement[::3, 0] = 1
    replacement[1::4, 1] = 1
    result = model(
        {"events": torch.zeros(40, 2, 2)},
        interventions=(
            ReplaySpikes(
                "cell_I", DenseSpikeReplay.from_tensor(replacement, dt_ms=0.1)
            ),
        ),
    )
    torch.testing.assert_close(
        result.diagnostics["cell_I.spikes"], replacement, rtol=0, atol=0
    )
    decay = torch.exp(torch.tensor(-0.1 / 9.0))
    state = torch.zeros(2, 4)
    reference = []
    for step in range(40):
        kick = replacement[step - 1].expand(2, 4) * 5 if step else torch.zeros(2, 4)
        state = state * decay + kick
        reference.append(state)
    torch.testing.assert_close(
        result.diagnostics["cell_I_to_E.conductance"],
        torch.stack(reference),
        rtol=0,
        atol=0,
    )
