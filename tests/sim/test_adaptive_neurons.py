"""Adaptive neuron contracts, independent dynamics and portable continuation."""

import copy
import json
import math

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from snnlab import lang
from snnlab.sim import extensions as X
from snnlab.sim.execution import (
    DenseArrayBinding,
    ExecutionSpec,
    GraphExecutor,
    NPZRecordingSink,
    RecordingSpec,
    SignalRecording,
    load_runtime_state,
    plan_graph,
    save_runtime_state,
    simulate,
)

MODELS = (lang.CUBA_ALIF, lang.COBA_ALIF, lang.CUBA_ADEX, lang.COBA_ADEX)


def network(model, **settings):
    net = lang.Network("adaptive", dt=0.25 * lang.ms)
    events = net.input(
        "events", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
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
    net.output("adaptation", cells.state("adaptation"))
    return net, cells, projection


def state(voltage=-65.0, adaptation=0.0, refractory=0):
    return {
        "voltage": torch.tensor([[voltage]], dtype=torch.float64),
        "refractory": torch.tensor([[refractory]], dtype=torch.long),
        "adaptation": torch.tensor([[adaptation]], dtype=torch.float64),
    }


def step(model, previous, drive=0.0, inhibition=0.0, dt=0.1):
    return X.adaptive_neuron(
        previous,
        torch.full_like(previous["voltage"], drive),
        torch.full_like(previous["voltage"], inhibition),
        dt_ms=dt,
        config=model.json(),
        spike_function=lambda x: (x >= 0).to(x.dtype),
    )


@pytest.mark.parametrize("model", MODELS)
def test_bundle_gradients_and_saved_continuation(model, tmp_path):
    net, cells, projection = network(model, refractory_steps=2)
    bundle = lang.compile(net, target="tools/snnsim")
    restored = lang.load_bundle(bundle.write(tmp_path / "bundle"))
    assert restored.graph == bundle.graph
    assert cells.state("adaptation").unit == (
        "mV" if cells.neuron.kind.endswith("alif") else "nA"
    )
    assert not bundle.manifest.get("extensions")
    executor = GraphExecutor(
        plan_graph(restored.graph), trainable_parameters=(projection.weight.id,)
    )
    values = torch.ones(80, 2, 1)
    values[:, 1] = 0
    full = executor({"events": values})
    assert full.outputs["spikes"][:, 0].sum() > 0
    assert full.outputs["adaptation"][:, 0].max() > 0
    assert full.outputs["spikes"][:, 1].sum() == 0
    loss = full.outputs["spikes"].sum() + full.outputs["adaptation"].sum()
    loss.backward()
    gradient = executor.parameter_map()[projection.weight.id].grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    first = executor({"events": values[:31]})
    saved = save_runtime_state(tmp_path / "state", first.runtime_state)
    resumed = executor({"events": values[31:]}, runtime_state=load_runtime_state(saved))
    for name in full.outputs:
        torch.testing.assert_close(
            torch.cat((first.outputs[name], resumed.outputs[name])),
            full.outputs[name],
            rtol=0,
            atol=0,
        )
    for name, tensor in full.runtime_state.custom_state.items():
        assert torch.equal(tensor, resumed.runtime_state.custom_state[name])
    damaged = copy.deepcopy(first.runtime_state)
    damaged.custom_state.pop("neuron/cells/adaptation")
    with pytest.raises(ValueError, match="missing neuron/cells/adaptation"):
        executor({"events": values[31:]}, runtime_state=damaged)


def test_aliases():
    assert lang.ALIF().json() == lang.CUBA_ALIF().json()
    assert lang.ADEX().json() == lang.CUBA_ADEX().json()


@pytest.mark.parametrize("model", MODELS)
def test_rejects_incompatible_synaptic_units(model):
    net, *_ = network(model)
    wrong = lang.AMPA if model().kind.startswith("cuba_") else lang.ExponentialCurrent
    net.projections[0]["synapse"] = wrong(tau=3 * lang.ms).json()
    with pytest.raises(ValueError, match="incompatible"):
        lang.compile(net)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "settings",
    [
        {"tau_mem": 0 * lang.ms},
        {"tau_adaptation": 1 * lang.mV},
        {"tau_adaptation": float("nan") * lang.ms},
        {"capacitance_nf": False},
        {"refractory_steps": -1},
        {"initial_voltage_mv": float("inf")},
    ],
)
def test_invalid_common_settings_fail_compilation_and_planning(model, settings):
    net, *_ = network(model, **settings)
    with pytest.raises(ValueError):
        lang.compile(net)
    graph = lang.compile(network(model)[0]).graph
    graph["populations"][0]["neuron"] = model(**settings).json()
    with pytest.raises(ValueError):
        plan_graph(graph)


@pytest.mark.parametrize(
    "model,settings",
    [
        (lang.ALIF, {"adaptation_increment_mv": -1}),
        (lang.ALIF, {"initial_adaptation_mv": -1}),
        (lang.ADEX, {"delta_t_mv": 0}),
        (lang.ADEX, {"spike_mv": -51}),
        (lang.ADEX, {"a_us": -0.01}),
        (lang.ADEX, {"b_na": -0.1}),
        (lang.COBA_ADEX, {"excitatory_reversal_mv": float("nan")}),
    ],
)
def test_invalid_model_settings(model, settings):
    with pytest.raises(ValueError):
        lang.compile(network(model, **settings)[0])


def test_alif_adapts_threshold_then_decays_during_refractory():
    model = lang.ALIF(
        threshold_mv=-64.5,
        adaptation_increment_mv=2,
        tau_adaptation=10 * lang.ms,
        refractory_steps=2,
    )
    first, spikes = step(model, state(), drive=1, dt=1)
    assert spikes.item() == 1 and first["adaptation"].item() == 2
    next_state, spikes = step(model, first, drive=100, dt=1)
    assert spikes.item() == 0 and next_state["voltage"].item() == -65
    assert next_state["adaptation"].item() == pytest.approx(2 * math.exp(-0.1))
    adapted, spikes = step(model, state(adaptation=2), drive=1, dt=1)
    assert spikes.item() == 0
    assert adapted["voltage"].item() > -64.5


def test_alif_with_zero_adaptation_matches_current_lif_exactly():
    config = lang.ALIF(adaptation_increment_mv=0).json()
    previous = state()
    for drive in (0.0, 1.0, 100.0, 0.0):
        values = torch.full_like(previous["voltage"], drive)

        def spike(x):
            return (x >= 0).to(x.dtype)

        expected = X.current_lif(
            previous, values, values * 0, dt_ms=0.1, config=config, spike_function=spike
        )
        actual, spikes = step(lang.ALIF(adaptation_increment_mv=0), previous, drive)
        assert torch.equal(actual["voltage"], expected[0])
        assert torch.equal(spikes, expected[1])
        assert torch.equal(actual["refractory"], expected[2])
        previous = actual


@pytest.mark.parametrize("model", MODELS)
def test_inhibition_hyperpolarizes(model):
    actual, spikes = step(model(), state(), inhibition=0.1)
    baseline, _ = step(model(), state())
    assert actual["voltage"] < baseline["voltage"]
    assert spikes.item() == 0


def test_adex_spike_cutoff_and_spike_triggered_adaptation():
    model = lang.ADEX(a_us=0, b_na=0.3, refractory_steps=1)
    updated, spikes = step(model, state(voltage=-49), dt=0.1)
    assert spikes.item() == 0  # Crossing V_T alone is not a spike.
    updated, spikes = step(model, state(voltage=-31), drive=100, dt=0.1)
    assert spikes.item() == 1
    assert updated["adaptation"].item() == pytest.approx(0.3)
    refractory, spikes = step(model, updated, drive=100, dt=0.1)
    assert spikes.item() == 0
    assert refractory["adaptation"].item() < 0.3


@pytest.mark.parametrize("model", (lang.CUBA_ADEX, lang.COBA_ADEX))
def test_adex_converges_to_independent_continuous_equations(model):
    conductance = model().kind.startswith("coba_")
    drive = 0.002 if conductance else 0.15

    def rhs(t, y):
        v, w = y
        synaptic = drive * (0 - v) if conductance else drive
        return [
            (-0.05 * (v + 65) + 0.1 * np.exp((v + 50) / 2) - w + synaptic),
            (0.002 * (v + 65) - w) / 200,
        ]

    reference = solve_ivp(rhs, (0, 10), [-60, 0.05], rtol=1e-11, atol=1e-12).y[:, -1]
    errors = []
    for dt in (0.2, 0.1, 0.05):
        previous = state(voltage=-60, adaptation=0.05)
        for _ in range(round(10 / dt)):
            previous, spikes = step(model(), previous, drive, dt=dt)
            assert spikes.item() == 0
        actual = [previous["voltage"].item(), previous["adaptation"].item()]
        errors.append(np.linalg.norm(np.asarray(actual) - reference))
    assert errors[1] < errors[0] * 0.6
    assert errors[2] < errors[1] * 0.6
    assert errors[2] < 0.002


@pytest.mark.parametrize("model", (lang.CUBA_ADEX, lang.COBA_ADEX))
def test_extreme_exponential_does_not_overflow(model):
    updated, spikes = step(model(delta_t_mv=0.0001), state(voltage=-40))
    assert spikes.item() == 1
    assert all(torch.isfinite(tensor).all() for tensor in updated.values())


@pytest.mark.parametrize("model,unit", [(lang.ALIF, "mV"), (lang.ADEX, "nA")])
def test_recording_retains_adaptation_units(model, unit, tmp_path):
    graph = lang.compile(network(model)[0]).graph
    root = tmp_path / "recording"
    simulate(
        ExecutionSpec(
            kind="simulate",
            graph=graph,
            input_bindings=(DenseArrayBinding("events", torch.ones(8, 1, 1)),),
            recording=RecordingSpec(
                signals=(SignalRecording("cells.adaptation"),),
                sink=NPZRecordingSink(root),
                sink_id=NPZRecordingSink.identity,
            ),
        )
    )
    rows = [
        json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()
    ]
    adaptation_rows = [
        row for row in rows if row["signal"] in {"adaptation", "cells.adaptation"}
    ]
    assert adaptation_rows and all(row["unit"] == unit for row in adaptation_rows)
