"""Built-in MQIF equations, hybrid reset, interchange and runtime contracts."""

import copy
import math

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from snnlab import lang
from snnlab.sim import GraphExecutor, load_runtime_state, plan_graph, save_runtime_state
from snnlab.sim.graph import extensions as X

MODELS = (lang.CUBA_MQIF, lang.COBA_MQIF)


def network(model, **settings):
    net = lang.Network("mqif", dt=0.25 * lang.ms)
    events = net.input(
        "events", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    settings.setdefault("spike_mv", -10.0)
    cells = net.population("cells", size=2, neuron=model(**settings))
    current = cells.neuron.kind.startswith("cuba_")
    projection = net.connect(
        events,
        cells.excitatory,
        name="drive",
        synapse=(lang.ExponentialCurrent if current else lang.AMPA)(tau=3 * lang.ms),
        weight=lang.Constant(3.0 if current else 0.1),
    )
    net.output("voltage", cells.voltage)
    net.output("spikes", cells.spikes)
    net.output("slow_voltage", cells.state("slow_voltage"))
    return net, cells, projection


def state(v=-65.0, slow=-65.0, ref=0):
    return {
        "voltage": torch.tensor([[v]], dtype=torch.float64),
        "slow_voltage": torch.tensor([[slow]], dtype=torch.float64),
        "refractory": torch.tensor([[ref]], dtype=torch.long),
    }


def step(model, previous, excitation=0.0, inhibition=0.0, dt=0.1):
    return X.mqif_neuron(
        previous,
        torch.full_like(previous["voltage"], excitation),
        torch.full_like(previous["voltage"], inhibition),
        dt_ms=dt,
        config=model.json(),
        spike_function=lambda x: (x >= 0).to(x.dtype),
    )


def test_alias():
    assert lang.MQIF().json() == lang.CUBA_MQIF().json()


@pytest.mark.parametrize("model", MODELS)
def test_bundle_gradients_and_portable_continuation(model, tmp_path):
    net, cells, projection = network(model, refractory=0.5 * lang.ms)
    assert cells.state("slow_voltage").unit == "mV"
    bundle = lang.compile(net, target="tools/snnsim")
    assert not bundle.manifest.get("extensions")
    restored = lang.load_bundle(bundle.write(tmp_path / "bundle"))
    assert restored.graph == bundle.graph
    executor = GraphExecutor(
        plan_graph(restored.graph),
        trainable_parameters=(projection.weight.id,),
        compile_mode=None,
    )
    inputs = torch.ones(100, 2, 1)
    inputs[:, 1] = 0
    full = executor({"events": inputs})
    assert full.outputs["spikes"][:, 0].sum() > 0
    assert full.outputs["spikes"][:, 1].sum() == 0
    assert not torch.equal(
        full.outputs["slow_voltage"][:, 0], full.outputs["slow_voltage"][:, 1]
    )
    (full.outputs["spikes"].sum() + full.outputs["slow_voltage"].sum()).backward()
    grad = executor.parameter_map()[projection.weight.id].grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    first = executor({"events": inputs[:37]})
    saved = save_runtime_state(tmp_path / "state", first.runtime_state)
    tail = executor({"events": inputs[37:]}, runtime_state=load_runtime_state(saved))
    for key in full.outputs:
        torch.testing.assert_close(
            torch.cat((first.outputs[key], tail.outputs[key])),
            full.outputs[key],
            rtol=0,
            atol=0,
        )
    damaged = copy.deepcopy(first.runtime_state)
    damaged.custom_state.pop("neuron/cells/slow_voltage")
    with pytest.raises(ValueError, match="missing neuron/cells/slow_voltage"):
        executor({"events": inputs[37:]}, runtime_state=damaged)


@pytest.mark.parametrize("model", MODELS)
def test_converges_to_independent_continuous_equations(model):
    conductance = model == lang.COBA_MQIF
    excitation, inhibition = (0.002, 0.001) if conductance else (0.15, 0.04)

    def rhs(t, y):
        v, slow = y
        drive = (
            excitation * (0 - v) + inhibition * (-80 - v)
            if conductance
            else excitation - inhibition
        )
        return [
            0.001 * (v + 40) ** 2 - 0.002 * (slow + 45) ** 2 + drive,
            (v - slow) / 100,
        ]

    reference = solve_ivp(rhs, (0, 10), [-60, -62], rtol=1e-11, atol=1e-12).y[:, -1]
    errors = []
    for dt in (0.2, 0.1, 0.05):
        previous = state(-60, -62)
        for _ in range(round(10 / dt)):
            previous, spikes = step(model(), previous, excitation, inhibition, dt)
            assert spikes.item() == 0
        actual = np.array([previous["voltage"].item(), previous["slow_voltage"].item()])
        errors.append(np.linalg.norm(actual - reference))
    assert errors[1] < 0.6 * errors[0]
    assert errors[2] < 0.6 * errors[1]
    assert errors[2] < 0.003


@pytest.mark.parametrize("model", MODELS)
def test_reset_and_refractory_slow_relaxation(model):
    neuron = model(
        spike_mv=-50, reset_mv=-65, slow_reset_mv=-55, refractory=0.2 * lang.ms
    )
    updated, spikes = step(neuron, state(-50, -60), excitation=100)
    assert spikes.item() == 1
    assert updated["voltage"].item() == -65
    assert updated["slow_voltage"].item() == -55
    assert updated["refractory"].item() == 2
    for remaining in (1, 0):
        previous_slow = updated["slow_voltage"].item()
        updated, spikes = step(neuron, updated, excitation=100)
        assert spikes.item() == 0 and updated["voltage"].item() == -65
        assert updated["refractory"].item() == remaining
        assert updated["slow_voltage"].item() == pytest.approx(
            -65 + (previous_slow + 65) * math.exp(-0.001)
        )
    _, spikes = step(neuron, updated, excitation=1000)
    assert spikes.item() == 1


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "settings",
    [
        {"tau_slow": 0 * lang.ms},
        {"tau_slow": 10 * lang.mV},
        {"tau_slow": float("nan") * lang.ms},
        {"capacitance_nf": False},
        {"fast_coefficient_na_per_mv2": 0},
        {"slow_coefficient_na_per_mv2": -1},
        {"slow_center_mv": float("inf")},
        {"initial_slow_voltage_mv": float("nan")},
        {"reset_mv": 30},
        {"refractory": -1 * lang.ms},
        {"voltage_grad_dampen": 0},
    ],
)
def test_invalid_parameters_rejected_by_authoring_and_runtime(model, settings):
    with pytest.raises(ValueError):
        lang.compile(network(model, **settings)[0])
    graph = lang.compile(network(model)[0]).graph
    graph["populations"][0]["neuron"] = model(**settings).json()
    with pytest.raises(ValueError):
        plan_graph(graph)


@pytest.mark.parametrize("model", MODELS)
def test_wrong_synaptic_units_rejected(model):
    net, *_ = network(model)
    wrong = lang.AMPA if model == lang.CUBA_MQIF else lang.ExponentialCurrent
    net.projections[0]["synapse"] = wrong().json()
    with pytest.raises(ValueError, match="incompatible"):
        lang.compile(net)


@pytest.mark.parametrize("model", MODELS)
def test_compiled_blocks_match_eager_outputs_gradients_and_continuation(
    model, monkeypatch
):
    original = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kw: original(fn, backend="eager", fullgraph=True, dynamic=False),
    )
    net, _, projection = network(model)
    graph = lang.compile(net).graph
    results = []
    for size in (1, 4):
        executor = GraphExecutor(
            plan_graph(graph),
            trainable_parameters=(projection.weight.id,),
            compile_mode=None if size == 1 else "default",
            compile_block_steps=size,
        )
        values = torch.ones(41, 2, 1, requires_grad=True)
        full = executor({"events": values}, diagnostics=False)
        (full.outputs["voltage"].sum() + full.outputs["slow_voltage"].sum()).backward()
        results.append(
            (full, executor.parameter_map()[projection.weight.id].grad, values.grad)
        )
        first = executor({"events": values[:19].detach()}, diagnostics=False)
        tail = executor(
            {"events": values[19:].detach()},
            runtime_state=first.runtime_state,
            diagnostics=False,
        )
        for key in full.outputs:
            torch.testing.assert_close(
                torch.cat((first.outputs[key], tail.outputs[key])), full.outputs[key]
            )
    assert results[1][0].metrics["execution_path"]["selected"] == "compiled_blocks"
    for key in results[0][0].outputs:
        torch.testing.assert_close(
            results[0][0].outputs[key], results[1][0].outputs[key]
        )
    for eager, compiled in zip(results[0][1:], results[1][1:], strict=True):
        torch.testing.assert_close(eager, compiled)


@pytest.mark.parametrize("model", MODELS)
def test_inhibitory_drive_and_spike_boundary(model):
    baseline, _ = step(model(), state())
    inhibited, spikes = step(model(), state(), inhibition=0.1)
    assert inhibited["voltage"].item() < baseline["voltage"].item()
    assert spikes.item() == 0
    # At both parabola centers the intrinsic current is exactly zero.
    neuron = model(fast_center_mv=-40, slow_center_mv=-45, spike_mv=-40)
    updated, spikes = step(neuron, state(-40, -45))
    assert spikes.item() == 1
    assert updated["voltage"].item() == -65


@pytest.mark.parametrize("model", MODELS)
def test_initial_states_and_recorded_units(model, tmp_path):
    import json

    from snnlab.sim import (
        DenseArrayBinding,
        NPZRecordingSink,
        RecordingSpec,
        SignalRecording,
    )
    from snnlab.sim.requests.handlers import simulate
    from snnlab.sim.requests.spec import ExecutionSpec

    graph = lang.compile(
        network(model, initial_voltage_mv=-60, initial_slow_voltage_mv=-62)[0]
    ).graph
    expected, _ = step(
        model(initial_voltage_mv=-60, initial_slow_voltage_mv=-62),
        state(-60, -62),
        dt=0.25,
    )
    actual = GraphExecutor(plan_graph(graph), compile_mode=None)(
        {"events": torch.zeros(1, 1, 1)}
    )
    torch.testing.assert_close(
        actual.outputs["voltage"][0], expected["voltage"].float().expand(1, 2)
    )
    torch.testing.assert_close(
        actual.outputs["slow_voltage"][0], expected["slow_voltage"].float().expand(1, 2)
    )
    root = tmp_path / "recording"
    simulate(
        ExecutionSpec(
            kind="simulate",
            graph=graph,
            input_bindings=(DenseArrayBinding("events", torch.zeros(8, 1, 1)),),
            recording=RecordingSpec(
                signals=(SignalRecording("cells.slow_voltage"),),
                sink=NPZRecordingSink(root),
                sink_id=NPZRecordingSink.identity,
            ),
        )
    )
    rows = [
        json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()
    ]
    slow_rows = [
        row for row in rows if row["signal"] in {"slow_voltage", "cells.slow_voltage"}
    ]
    assert slow_rows and all(row["unit"] == "mV" for row in slow_rows)
