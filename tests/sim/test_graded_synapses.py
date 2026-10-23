"""Continuous population source timing, physical units and persistence."""

import copy
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from snnlab import extensions as ext
from snnlab import lang
from snnlab.sim import GraphExecutor, load_runtime_state, plan_graph, save_runtime_state

KEYS = (
    "a_to_b_gate",
    "b_to_a_gate",
    "a_to_b_conductance",
    "b_to_a_conductance",
    "a_voltage",
    "b_voltage",
)


@pytest.fixture
def example(monkeypatch):
    monkeypatch.setattr(ext, "_REGISTRY", {})
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "examples/graded-synapses")
    )
    module = importlib.import_module("graded_synapses")
    module.register()
    return module


def executor(example, **kwargs):
    bundle = lang.compile(example.network(**kwargs))
    return GraphExecutor(plan_graph(bundle.graph), compile_mode=None).double()


@pytest.mark.parametrize("delay_steps", [0, 1, 3])
@pytest.mark.parametrize("state_source", [False, True])
def test_reference_batch_and_saved_continuation(
    example, tmp_path, delay_steps, state_source
):
    model = executor(
        example, delay_steps=delay_steps, state_source=state_source, weight=0.75
    )
    clock = torch.zeros(80, 3, 1, dtype=torch.float64)
    full = model({"clock": clock})
    expected = example.reference(
        80, delay_steps=delay_steps, state_source=state_source, weight=0.75
    )
    for column, key in enumerate(KEYS):
        np.testing.assert_allclose(
            full.outputs[key].detach().numpy(),
            np.broadcast_to(expected[:, column, None, None], (80, 3, 1)),
            rtol=1e-12,
            atol=1e-12,
        )
    # First step independently detects zero initialization or newly advanced sources.
    gate = -60.0 + 0.1 / 2.0 * (0.75 * -40.0 + 60.0)
    assert full.outputs["a_to_b_gate"][0, 0, 0].item() == pytest.approx(gate, abs=1e-14)
    first = model({"clock": clock[:2]})
    saved = load_runtime_state(
        save_runtime_state(tmp_path / "state", first.runtime_state)
    )
    second = model({"clock": clock[2:]}, runtime_state=saved)
    for key in KEYS:
        torch.testing.assert_close(
            torch.cat((first.outputs[key], second.outputs[key])),
            full.outputs[key],
            rtol=0,
            atol=0,
        )
    for key, value in full.runtime_state.custom_state.items():
        torch.testing.assert_close(
            second.runtime_state.custom_state[key], value, rtol=0, atol=0
        )
    single = model({"clock": clock[:, :1]})
    torch.testing.assert_close(
        single.outputs["a_voltage"], full.outputs["a_voltage"][:, :1], rtol=0, atol=0
    )


def test_population_order_does_not_change_schedule(example):
    bundle = lang.compile(example.network())
    graph = copy.deepcopy(bundle.graph)
    graph["populations"].reverse()
    # Swap names so the planner's lexicographic traversal also reverses.
    for collection in ("populations", "projections", "outputs"):
        for row in graph[collection]:
            for field in ("id", "source", "target", "signal"):
                if field in row:
                    value = row[field]
                    row[field] = {"a": "b", "b": "a"}.get(value[0], value[0]) + value[
                        1:
                    ]
    left = GraphExecutor(plan_graph(bundle.graph), compile_mode=None).double()
    right = GraphExecutor(plan_graph(graph), compile_mode=None).double()
    values = {"clock": torch.zeros(10, 1, 1, dtype=torch.float64)}
    a, b = left(values), right(values)
    torch.testing.assert_close(
        a.outputs["a_voltage"], b.outputs["b_voltage"], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ("input", "input unit"),
        ("weight", "parameter.*unit"),
        ("target", "incompatible"),
        ("source", "unsupported"),
        ("legacy", "spikes port"),
    ],
)
def test_invalid_contracts_rejected_in_compile_and_plan(example, change, match):
    net = example.network()
    if change == "input":
        ext.register_synapse(
            "test.wrong/v1", example.graded_step, input_unit="nA", output_unit="uS"
        )
        net.projections[0]["synapse"] = lang.CustomSynapse("test.wrong/v1").json()
    elif change == "weight":
        net.parameters[0]["unit"] = "uS"
    elif change == "target":
        net.populations[1]["neuron"] = lang.CUBA_LIF().json()
    elif change == "source":
        net.projections[0]["source"] = "a.pre_reset_voltage"
    else:
        net.projections[0]["synapse"] = lang.GABA().json()
    with pytest.raises(ValueError, match=match):
        lang.compile(net)
    from snnlab.lang.compilation.serialization import graph_dict

    with pytest.raises(ValueError, match=match):
        plan_graph(graph_dict(net))


def test_portable_bundle_restores_input_unit(example, tmp_path):
    bundle = lang.compile(example.network())
    path = bundle.write(tmp_path / "bundle")
    ext._REGISTRY.clear()
    restored = lang.load_bundle(path)
    plan = plan_graph(restored.graph)
    assert ext.get("synapse", "example.graded_synapse/v1").input_unit == "mV"
    assert all(row["unit"] == "1" for row in restored.graph["parameters"])
    assert all(edge.continuous and edge.delay_steps == 0 for edge in plan.projections)


def test_gradient_crosses_continuous_source(example):
    model = executor(example)
    for weight in model.weights.values():
        weight.requires_grad_(True)
    result = model({"clock": torch.zeros(8, 1, 1, dtype=torch.float64)})
    result.outputs["a_voltage"].sum().backward()
    assert all(
        torch.isfinite(w.grad).all() and w.grad.abs().sum() > 0
        for w in model.weights.values()
    )


def test_gate_conductance_and_voltage_recording(example):
    from snnlab import sim

    model = executor(example)
    values = {"clock": torch.zeros(12, 2, 1, dtype=torch.float64)}
    full = model(values)
    signals = ("a_to_b.gate", "a_to_b.conductance", "b.voltage")
    recorded = model(
        values,
        diagnostics=False,
        retain_outputs=False,
        recording=sim.RecordingSpec(tuple(sim.SignalRecording(s) for s in signals)),
    )
    for signal, output, unit in zip(
        signals, ("a_to_b_gate", "a_to_b_conductance", "b_voltage"), ("mV", "uS", "mV")
    ):
        torch.testing.assert_close(
            recorded.recorded_signals[signal], full.outputs[output], rtol=0, atol=0
        )
        assert recorded.numpy().recording.info[signal].unit == unit


@pytest.mark.parametrize(
    "delay,match",
    [(-0.1, "non-negative"), (float("inf"), "finite"), (0.15, "integer number")],
)
def test_invalid_delays(example, delay, match):
    net = example.network()
    net.projections[0]["delay"] = {"value": delay, "unit": "ms"}
    with pytest.raises(ValueError, match=match):
        lang.compile(net)


def test_default_feedforward_continuous_cycle_is_valid(example):
    net = example.network()
    for edge in net.projections:
        edge["connection"] = "feedforward"
    graph = lang.compile(net).graph
    model = GraphExecutor(plan_graph(graph), compile_mode=None).double()
    result = model({"clock": torch.zeros(10, 1, 1, dtype=torch.float64)})
    np.testing.assert_allclose(
        result.outputs["a_voltage"][:, 0, 0].detach().numpy(),
        example.reference(10)[:, 4],
        atol=1e-12,
        rtol=1e-12,
    )
