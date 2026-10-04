"""Focused acceptance tests for the typed seam and graph executor."""

from __future__ import annotations

import copy

import pytest
import torch

from snnlab import lang as snn
from snnlab.sim import config
from snnlab.sim import models as M
from snnlab.sim.execution import (
    DelayBuffer,
    DenseArrayBinding,
    ExecutionSpec,
    GraphExecutor,
    build,
    plan_graph,
    simulate,
)
from tests.sim._bundle_builders import ping_classifier
from tests.sim._execution_builders import expose_graph_diagnostics


def _coupled_graph(*, direction="reciprocal", delay_ms=0.1):
    net = snn.Network("coupled_ping_gate", dt=0.1 * snn.ms)
    drive_a = net.input(
        "drive_a", shape=("time", "batch", 3), signal_type="spikes", unit="spike"
    )
    drive_b = net.input(
        "drive_b", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    a = snn.components.ping(net, name="a", n_e=4, n_i=1, source=drive_a)
    b = snn.components.ping(net, name="b", n_e=6, n_i=2, source=drive_b)
    if direction in {"unidirectional", "reciprocal"}:
        net.connect(
            a.I.spikes,
            b.E.inhibitory,
            name="a_I_to_b_E",
            synapse=snn.GABA(tau=9 * snn.ms),
            weight=snn.Constant(4.0),
            constraint=snn.NonNegative(),
            connection="feedback",
            delay=delay_ms * snn.ms,
        )
    if direction == "reciprocal":
        net.connect(
            b.I.spikes,
            a.E.inhibitory,
            name="b_I_to_a_E",
            synapse=snn.GABA(tau=9 * snn.ms),
            weight=snn.Constant(4.0),
            constraint=snn.NonNegative(),
            connection="feedback",
            delay=delay_ms * snn.ms,
        )
    net.expose(a.E.spikes, a.I.spikes, b.E.spikes, b.I.spikes, name="coupled")
    return expose_graph_diagnostics(snn.compile(net, target=None).graph)


def test_delay_lowering_is_exact_in_steps_and_feedback_is_causal():
    one = plan_graph(_coupled_graph(delay_ms=0.1))
    delayed = plan_graph(_coupled_graph(delay_ms=0.3))
    assert next(p for p in one.projections if p.id == "a_I_to_b_E").delay_steps == 1
    assert next(p for p in delayed.projections if p.id == "a_I_to_b_E").delay_steps == 3

    pulse = torch.tensor([[1.0]])
    zeros = torch.zeros_like(pulse)
    buffer = DelayBuffer(3, pulse)
    received = []
    for emitted in (pulse, zeros, zeros, zeros, zeros):
        received.append(float(buffer.read().item()))
        buffer.push(emitted)
    assert received == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_zero_delay_population_readout_uses_current_spikes_and_can_resume():
    net = snn.Network("explicit_readout", dt=0.1 * snn.ms)
    inputs = net.input(
        "inputs", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    cells = net.population("E", size=2, neuron=snn.COBA_LIF(tau_mem=20 * snn.ms))
    net.connect(
        inputs,
        cells.excitatory,
        name="drive",
        synapse=snn.AMPA(tau=2 * snn.ms),
        weight=snn.Constant(100.0),
    )
    readout = net.population(
        "readout", size=1, neuron=snn.LeakyIntegrator(tau=2 * snn.ms), spiking=False
    )
    net.connect(
        cells.spikes,
        readout.excitatory,
        name="readout_projection",
        synapse=snn.LeakyIntegrator(tau=2 * snn.ms),
        weight=snn.Constant(1.0),
    )
    net.output("voltage", readout.voltage)
    net.expose(cells.spikes, name="spikes")
    graph = snn.compile(net, target=None).graph
    assert (
        next(
            p for p in plan_graph(graph).projections if p.id == "readout_projection"
        ).delay_steps
        == 0
    )

    def run(steps, state=None):
        return simulate(
            ExecutionSpec(
                kind="simulate",
                graph=graph,
                device="cpu",
                input_bindings=(DenseArrayBinding("inputs", torch.ones(steps, 1, 1)),),
                runtime_state=state,
            )
        )

    full = run(20)
    first = run(8)
    second = run(12, first.runtime_state)
    voltage = full.outputs["voltage"]
    spikes = full.diagnostics["spikes"]
    first_spike = int(torch.nonzero(spikes.sum(dim=(1, 2)))[0])
    assert voltage[first_spike].item() > 0
    assert torch.count_nonzero(voltage[:first_spike]) == 0
    torch.testing.assert_close(
        voltage, torch.cat((first.outputs["voltage"], second.outputs["voltage"]))
    )


def test_disabled_projection_keeps_initialization_position_but_carries_no_drive():
    enabled_graph = _coupled_graph(direction="uncoupled")
    disabled_graph = copy.deepcopy(enabled_graph)
    disabled_graph["projections"][0]["enabled"] = False
    enabled = GraphExecutor(plan_graph(enabled_graph), seed=29)
    disabled = GraphExecutor(plan_graph(disabled_graph), seed=29)
    assert disabled.plan.projections[0].enabled is False
    for name, value in enabled.parameter_map().items():
        torch.testing.assert_close(
            value, disabled.parameter_map()[name], rtol=0, atol=0
        )
    result = disabled(
        {
            "drive_a": torch.ones(2, 1, 3),
            "drive_b": torch.zeros(2, 1, 2),
        }
    )
    projection_id = disabled.plan.projections[0].id
    assert torch.count_nonzero(result.diagnostics[f"{projection_id}.conductance"]) == 0


def test_explicit_initializers_constraints_units_and_realized_statistics():
    graph = _coupled_graph(direction="uncoupled")
    rows = {row["id"]: row for row in graph["parameters"]}
    first_id = graph["projections"][0]["parameters"][0]
    rows[first_id]["initializer"] = {
        "kind": "lower_clamped_normal",
        "mean": 0.5,
        "std": 0.1,
        "initial_zero_fraction": 0.5,
        "zeroing": "exact_k",
    }
    model = GraphExecutor(plan_graph(graph), seed=37)
    metadata = model.initialization_metadata[first_id]
    assert metadata["unit"] == "uS"
    assert metadata["constraint"] == {"kind": "non_negative"}
    assert metadata["scaling"] == "fan_in_normalized"
    assert metadata["statistics"]["count"] == 4
    assert metadata["statistics"]["zero_fraction"] == pytest.approx(0.5)
    assert (
        metadata
        == GraphExecutor(plan_graph(graph), seed=37).initialization_metadata[first_id]
    )


def test_graph_planning_rejects_non_microsiemens_projection_weights():
    graph = _coupled_graph(direction="uncoupled")
    parameter_id = graph["projections"][0]["parameters"][0]
    parameter = next(row for row in graph["parameters"] if row["id"] == parameter_id)
    parameter["unit"] = "nS"

    with pytest.raises(ValueError, match="requires unit uS, got nS"):
        plan_graph(graph)


def test_signed_normal_and_uniform_initializers_have_distinct_semantics():
    graph = _coupled_graph(direction="uncoupled")
    rows = {row["id"]: row for row in graph["parameters"]}
    ids = [projection["parameters"][0] for projection in graph["projections"][:2]]
    rows[ids[0]]["initializer"] = {"kind": "signed_normal", "mean": -1.0, "std": 0.0}
    rows[ids[0]]["constraint"] = None
    rows[ids[1]]["initializer"] = {"kind": "uniform", "low": 2.0, "high": 2.0}
    model = GraphExecutor(plan_graph(graph), seed=1)
    assert torch.all(model.parameter_map()[ids[0]] < 0)
    assert torch.all(model.parameter_map()[ids[1]] > 0)


def test_input_delay_pulse_arrives_on_exact_timestep_and_handles_boundary():
    graph = _coupled_graph(direction="uncoupled")
    projection = next(p for p in graph["projections"] if p["id"] == "a_input")
    projection["delay"] = {"value": 0.3, "unit": "ms"}
    parameter = next(p for p in graph["parameters"] if p["id"] == "a_input.weight")
    parameter["initializer"] = {"kind": "constant", "value": 1.0}
    inputs = {"drive_a": torch.zeros(5, 1, 3), "drive_b": torch.zeros(5, 1, 2)}
    inputs["drive_a"][0, 0, 0] = 1.0
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            input_bindings=tuple(
                DenseArrayBinding(name, value) for name, value in (inputs).items()
            ),
        )
    )
    conductance = result.diagnostics["a_input.conductance"][:, 0]
    assert torch.count_nonzero(conductance[:3]) == 0
    assert torch.count_nonzero(conductance[3]) > 0


def test_non_integral_delay_is_rejected_before_execution():
    graph = _coupled_graph(delay_ms=0.15)
    try:
        plan_graph(graph)
    except ValueError as exc:
        assert "integer number" in str(exc)
    else:
        raise AssertionError("fractional-step delay must fail planning")


def test_single_ping_seeded_parameters_and_forward_match_legacy_exactly():
    bundle = ping_classifier()
    graph = bundle.graph
    torch.manual_seed(17)
    M.N_IN = 784
    M.N_OUT = 10
    config.set_sim_dt(0.1, 1.2)
    M.T_steps = 12
    legacy = config.build_net(
        "ping",
        w_in=(0.2, 0.03),
        w_in_initial_zero_fraction=0.0,
        w_ei=(0.5, 0.05),
        w_ie=(1.0, 0.1),
        ei_strength=0.5,
        ei_ratio=2.0,
        recurrent_initial_zero_fraction=0.0,
        hidden_sizes=[256],
        readout_mode="mem-mean",
    )
    graph_model = build(
        ExecutionSpec(kind="build", executor="graph", graph=graph, seed=17)
    ).model
    assert isinstance(graph_model, GraphExecutor)
    mapping = {
        "sensory_ping_input.weight": legacy.W_ff[0],
        "classifier_projection.weight": legacy.W_ff[1],
        "sensory_ping_E_to_I.weight": legacy.W_ei["1"],
        "sensory_ping_I_to_E.weight": legacy.W_ie["1"],
    }
    for name, expected in mapping.items():
        torch.testing.assert_close(
            graph_model.parameter_map()[name], expected, rtol=0, atol=0
        )

    spikes = torch.zeros(12, 2, 784)
    spikes[0::2, :, :48] = 1.0
    legacy.recording = True
    legacy_logits = legacy(input_spikes=spikes)
    native = graph_model({"image": spikes})
    torch.testing.assert_close(
        native.outputs["class_logits"], legacy_logits, rtol=0, atol=2e-7
    )
    torch.testing.assert_close(
        native.diagnostics["cell_0"], legacy.spike_record["hid"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        native.diagnostics["cell_1"], legacy.spike_record["inh"], rtol=0, atol=0
    )
