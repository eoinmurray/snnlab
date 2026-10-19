"""Scheduled projection gains preserve physical dynamics and continuation."""

import copy
import json
import math
import os
import subprocess
import sys

import pytest
import torch

from snnlab import lang
from snnlab._shared.gains import sample
from snnlab.sim import GraphExecutor, load_runtime_state, plan_graph, save_runtime_state
from tests.sim import _gain_callbacks as callbacks


def network(modulation=None, *, current=True, delay=None, enabled=True, dt=1.0):
    net = lang.Network("scheduled", dt=dt * lang.ms)
    events = net.input(
        "events", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    cells = net.population(
        "cells",
        size=1,
        neuron=lang.CUBA_LIF(threshold_mv=1000)
        if current
        else lang.COBA_LIF(tau_mem=20 * lang.ms),
    )
    projection = net.connect(
        events,
        cells.excitatory,
        name="input_projection",
        synapse=lang.ExponentialCurrent(tau=2 * lang.ms)
        if current
        else lang.AMPA(tau=2 * lang.ms),
        weight=lang.Constant(2),
        initialization_scaling="direct",
        modulation=modulation,
        delay=delay,
        enabled=enabled,
    )
    net.output("drive", projection.current if current else projection.conductance)
    return net, projection


def executor(net, projection=None):
    return GraphExecutor(
        plan_graph(lang.compile(net).graph),
        trainable_parameters=() if projection is None else (projection.weight.id,),
    )


def samples(spec, steps=5, *, start=0, dt=1):
    return sample(spec.json(), start_step=start, steps=steps, dt_ms=dt)


@pytest.mark.parametrize(
    "schedule,expected",
    [
        (lang.ConstantGain(), [1, 1, 1, 1, 1]),
        (lang.StepGain(), [0, 0, 0, 0, 0]),
        (lang.StepGain(at=2.2 * lang.ms, after=3), [1, 1, 1, 3, 3]),
        (lang.RampGain(start=1 * lang.ms, end=3 * lang.ms), [0, 0, 0.5, 1, 1]),
        (
            lang.RampGain(start=1 * lang.ms, end=3 * lang.ms, before=2, after=0),
            [2, 2, 1, 0, 0],
        ),
        (lang.SineGain(frequency=250 * lang.Hz), [1, 1.5, 1, 0.5, 1]),
        (
            lang.PiecewiseGain(times=[1.2 * lang.ms, 3 * lang.ms], values=[0, 2, 1]),
            [0, 0, 2, 1, 1],
        ),
        (
            lang.CustomGain(callbacks.configured_step, config={"at_ms": 2, "after": 3}),
            [1, 1, 3, 3, 3],
        ),
    ],
)
def test_analytic_schedules(schedule, expected):
    assert samples(schedule) == pytest.approx(expected)
    assert samples(schedule, 2, start=3) == pytest.approx(expected[3:])


def test_defaults_units_phase_and_roundoff_boundaries():
    assert samples(lang.RampGain(), 2, dt=50) == (0, 0.5)
    assert samples(lang.SineGain(), 2, dt=6.25) == pytest.approx((1, 1.5))
    assert samples(
        lang.SineGain(frequency=0 * lang.Hz, depth=1, phase=math.pi / 2)
    ) == pytest.approx([2] * 5)
    for spec in [
        lang.StepGain(at=0.9 * lang.ms),
        lang.PiecewiseGain(times=[0.9 * lang.ms], values=[1, 0]),
    ]:
        assert samples(spec, 5, dt=0.3) == (1, 1, 1, 0, 0)


@pytest.mark.parametrize("current", [True, False])
@pytest.mark.parametrize(
    "schedule",
    [
        lang.StepGain(at=2 * lang.ms),
        lang.RampGain(end=4 * lang.ms),
        lang.SineGain(frequency=250 * lang.Hz),
        lang.CustomGain(callbacks.configured_step),
    ],
)
def test_drive_matches_independent_synaptic_recurrence(current, schedule):
    net, projection = network(schedule, current=current)
    model = executor(net, projection)
    source = torch.ones(5, 2, 1)
    result = model({"events": source})
    previous, expected = 0.0, []
    for gain in samples(schedule):
        previous = previous * math.exp(-0.5) + 2 * gain
        expected.append(previous)
    assert result.outputs["drive"][:, 0, 0].tolist() == pytest.approx(expected)
    torch.testing.assert_close(
        result.outputs["drive"][:, 0], result.outputs["drive"][:, 1]
    )
    assert result.parameters[projection.weight.id].item() == 2
    result.outputs["drive"].sum().backward()
    expected_gradient = 2 * sum(
        sum(samples(schedule)[j] * math.exp(-0.5 * (i - j)) for j in range(i + 1))
        for i in range(5)
    )
    assert model.parameter_map()[projection.weight.id].grad.item() == pytest.approx(
        expected_gradient
    )


def test_delayed_input_uses_arrival_time_and_existing_current_decays():
    net, _ = network(lang.StepGain(at=2 * lang.ms), delay=1 * lang.ms)
    result = executor(net)({"events": torch.ones(4, 1, 1)})
    assert result.outputs["drive"].flatten().tolist() == pytest.approx(
        [0, 2, 2 * math.exp(-0.5), 2 * math.exp(-1)]
    )


@pytest.mark.parametrize(
    "schedule",
    [
        lang.SineGain(),
        lang.StepGain(at=2.2 * lang.ms),
        lang.RampGain(end=8 * lang.ms),
        lang.CustomGain(callbacks.configured_step),
    ],
)
def test_portable_split_run_and_schedule_compatibility(tmp_path, schedule):
    net, _ = network(schedule, delay=1 * lang.ms)
    model = executor(net)
    source = torch.ones(8, 1, 1)
    full = model({"events": source})
    first = model({"events": source[:3]})
    path = save_runtime_state(tmp_path / "state", first.runtime_state)
    second = executor(net)(
        {"events": source[3:]}, runtime_state=load_runtime_state(path)
    )
    torch.testing.assert_close(
        torch.cat([first.outputs["drive"], second.outputs["drive"]]),
        full.outputs["drive"],
        rtol=0,
        atol=0,
    )
    other, _ = network(lang.ConstantGain(3), delay=1 * lang.ms)
    with pytest.raises(ValueError, match="compatib|signature|modulation"):
        executor(other)({"events": source[3:]}, runtime_state=first.runtime_state)


def test_shared_baseline_can_have_different_projection_gains():
    net, projection = network(lang.ConstantGain(2))
    second = net.connect(
        net._signal("events.value"),
        "cells.inhibitory",
        name="second",
        synapse=lang.ExponentialCurrent(tau=2 * lang.ms),
        weight=projection.weight,
        modulation=lang.ConstantGain(0.5),
    )
    net.output("second_drive", second.current)
    model = executor(net, projection)
    result = model({"events": torch.ones(3, 1, 1)})
    torch.testing.assert_close(
        result.outputs["drive"], 4 * result.outputs["second_drive"]
    )
    assert len(model.parameter_map()) == 1


def test_gain_preserves_autapse_mask_and_optimizer_constraints():
    net = lang.Network("masked", dt=1 * lang.ms)
    cells = net.population("cells", size=2, neuron=lang.CUBA_LIF())
    projection = net.connect(
        cells.spikes,
        cells.excitatory,
        name="feedback",
        synapse=lang.ExponentialCurrent(),
        weight=lang.Constant(1),
        constraint=lang.NonNegative(),
        connection="recurrent",
        modulation=lang.ConstantGain(3),
    )
    net.output("voltage", cells.voltage)
    # Give execution a time/batch reference even without external drive.
    net.input("clock", shape=("time", "batch", 1), signal_type="spikes", unit="spike")
    model = executor(net, projection)
    weight = model.parameter_map()[projection.weight.id]
    with torch.no_grad():
        weight.fill_(2)
    result = model({"clock": torch.ones(3, 1, 1)})
    assert result.parameters[projection.weight.id].diagonal().count_nonzero() == 0
    result.outputs["voltage"].sum().backward()
    assert weight.grad.diagonal().count_nonzero() == 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.step()
    model.enforce_constraints()
    assert weight.diagonal().count_nonzero() == 0


def test_plain_callable_is_wrapped_and_config_is_isolated():
    net, projection = network(callbacks.configured_step)
    assert projection.modulation.kind == "function_gain"
    assert executor(net)({"events": torch.ones(3, 1, 1)}).outputs["drive"][2].item() < 4
    config = {"nested": {"calls": 0}}
    schedule = lang.CustomGain(callbacks.mutating_config, config=config)
    config["nested"]["calls"] = 50
    assert samples(schedule) == (1,) * 5
    assert schedule.json()["config"]["nested"]["calls"] == 0


def test_custom_bundle_roundtrip_in_fresh_process(tmp_path):
    net, _ = network(
        lang.CustomGain(
            "tests.sim._gain_callbacks:configured_step",
            config={"at_ms": 2, "after": 0.5},
        )
    )
    bundle = lang.compile(net, target="tools/snnsim")
    root = bundle.write(tmp_path / "network.bundle")
    saved = json.loads((root / "graph.json").read_text())
    assert saved["projections"][0]["modulation"]["config"] == {"at_ms": 2, "after": 0.5}
    assert any(
        "modulation:function_gain" in row["features"]
        for row in bundle.manifest["required_capabilities"]["elements"]
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from snnlab import lang; from snnlab.sim import GraphExecutor, plan_graph; import torch, sys; b=lang.load_bundle(sys.argv[1]); r=GraphExecutor(plan_graph(b.graph))({'events':torch.ones(3,1,1)}); assert r.outputs['drive'][2].item() > 1",
            str(root),
        ],
        check=True,
        env={**os.environ},
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda: lang.ConstantGain(-1),
        lambda: lang.ConstantGain(True),
        lambda: lang.StepGain(at=-1 * lang.ms),
        lambda: lang.StepGain(at=1 * lang.Hz),
        lambda: lang.RampGain(start=2 * lang.ms, end=1 * lang.ms),
        lambda: lang.SineGain(depth=2),
        lambda: lang.SineGain(frequency=float("nan") * lang.Hz),
        lambda: lang.PiecewiseGain(times=[2 * lang.ms, 1 * lang.ms], values=[1, 2, 3]),
        lambda: lang.PiecewiseGain(times=[1 * lang.ms], values=[1]),
        lambda: lang.CustomGain(lambda t, c: 1),
        lambda: lang.CustomGain("missing_timing_module:gain"),
        lambda: lang.CustomGain("math:pi"),
        lambda: lang.CustomGain(callbacks.configured_step, config={"x": float("nan")}),
        lambda: lang.CustomGain(callbacks.configured_step, config={1: "bad"}),
    ],
)
def test_invalid_authoring_fails(make):
    with pytest.raises((ValueError, TypeError)):
        make()


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, [1], "1"])
def test_invalid_callback_result_fails_before_simulation(value):
    # Nonfinite results are produced from a valid, finite config.
    schedule = lang.CustomGain(callbacks.invalid_gain, config={"value": 1})
    net, _ = network(schedule)
    from unittest.mock import patch

    with patch("snnlab._shared.gains.import_function", return_value=lambda t, c: value):
        with pytest.raises(ValueError, match="gain at t=0"):
            executor(net)({"events": torch.ones(2, 1, 1)})


def test_disabled_callbacks_are_not_evaluated_and_exceptions_propagate():
    net, _ = network(lang.CustomGain(callbacks.broken), enabled=False)
    assert (
        executor(net)({"events": torch.ones(2, 1, 1)}).outputs["drive"].count_nonzero()
        == 0
    )
    net, _ = network(lang.CustomGain(callbacks.broken))
    with pytest.raises(RuntimeError, match="timing callback failed"):
        executor(net)({"events": torch.ones(2, 1, 1)})


@pytest.mark.parametrize(
    "spec",
    [
        None,
        {"kind": "unknown"},
        {"kind": "constant_gain", "value": -1},
        {"kind": "constant_gain", "value": 1, "typo": 2},
        {"kind": "function_gain", "function": "missing_module:gain", "config": {}},
    ],
)
def test_invalid_raw_graph_rejected_by_compiler_and_planner(spec):
    net, _ = network()
    graph = copy.deepcopy(lang.compile(net).graph)
    graph["projections"][0]["modulation"] = spec
    assert any(row.code == "E118" for row in lang.validate_graph(graph).diagnostics)
    with pytest.raises((ValueError, TypeError)):
        plan_graph(graph)


def test_omission_preserves_graph_shape_and_unit_gain_execution():
    net, _ = network()
    graph = lang.compile(net).graph
    assert "modulation" not in graph["projections"][0]
    scheduled, _ = network(lang.ConstantGain())
    inputs = {"events": torch.ones(5, 1, 1)}
    plain, unit = executor(net)(inputs), executor(scheduled)(inputs)
    torch.testing.assert_close(
        plain.outputs["drive"], unit.outputs["drive"], rtol=0, atol=0
    )


def test_native_training_matches_direct_pytorch_gradient_and_update():
    from snnlab.sim import DenseArrayBinding
    from snnlab.sim.requests.spec import ExecutionSpec
    from snnlab.sim.training.trainer import train

    net = lang.Network("trained_schedule", dt=1 * lang.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cells = net.population(
        "cells", size=2, neuron=lang.LeakyIntegrator(tau=20 * lang.ms), spiking=False
    )
    projection = net.connect(
        events,
        cells.excitatory,
        name="projection",
        synapse=lang.AMPA(tau=2 * lang.ms),
        weight=lang.Constant(1),
        initialization_scaling="direct",
        modulation=lang.StepGain(at=1 * lang.ms),
    )
    scores = lang.reduce(cells.voltage, operation="mean", over="time", name="scores")
    net.output("prediction", scores)
    bundle = lang.compile(
        net,
        training=lang.TrainSpec(
            objectives=[lang.CrossEntropy(prediction=scores, target="label")],
            parameter_groups=[
                lang.ParameterGroup([projection.weight.id], name="weights", lr=0.1)
            ],
            optimizer=lang.AdamW(weight_decay=0),
            presentation_duration=3 * lang.ms,
        ),
    )
    inputs = torch.eye(2).unsqueeze(0).expand(3, -1, -1).clone()
    labels = torch.tensor([0, 1])
    result = train(
        ExecutionSpec(
            kind="train",
            graph=bundle.graph,
            training=bundle.training,
            device="cpu",
            input_bindings=(DenseArrayBinding("events", inputs),),
            targets={"label": labels},
        )
    )
    weight = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = torch.optim.AdamW([weight], lr=0.1, weight_decay=0)
    state = torch.zeros(2, 2)
    voltage = torch.zeros(2, 2)
    traces = []
    for index in range(3):
        state = state * math.exp(-0.5) + (inputs[index] @ weight) * (
            1 if index == 0 else 0
        )
        voltage = math.exp(-0.05) * voltage + (1 - math.exp(-0.05)) * state
        traces.append(voltage)
    torch.nn.functional.cross_entropy(torch.stack(traces).mean(0), labels).backward()
    expected_gradient = weight.grad.clone()
    optimizer.step()
    torch.testing.assert_close(
        result.gradients[projection.weight.id], expected_gradient
    )
    torch.testing.assert_close(result.parameters[projection.weight.id], weight)


def test_custom_synapse_receives_modulated_drive(monkeypatch):
    from snnlab import extensions as ext
    from tests.sim import _extension_callbacks as synapse_callbacks

    monkeypatch.setattr(ext, "_REGISTRY", {})
    ext.register_synapse(
        "tests.scheduled_synapse/v1",
        synapse_callbacks.synapse_step,
        initialize=synapse_callbacks.synapse_init,
        output_unit="nA",
        state_units={"trace": "nA"},
    )
    net, _ = network(lang.StepGain(at=2 * lang.ms))
    net.projections[0]["synapse"] = lang.CustomSynapse(
        "tests.scheduled_synapse/v1"
    ).json()
    result = executor(net)({"events": torch.ones(4, 1, 1)})
    assert result.outputs["drive"].flatten().tolist() == pytest.approx(
        [2, 3, 1.5, 0.75]
    )


@pytest.mark.accelerator
@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_scheduled_execution_and_gradients_on_accelerator(device):
    available = (
        torch.cuda.is_available()
        if device == "cuda"
        else torch.backends.mps.is_available()
    )
    if not available:
        pytest.skip(f"{device} unavailable")
    net, projection = network(lang.SineGain())
    model = executor(net, projection).to(device)
    result = model({"events": torch.ones(5, 1, 1, device=device)})
    result.outputs["drive"].sum().backward()
    reference = executor(net, projection)
    expected = reference({"events": torch.ones(5, 1, 1)})
    expected.outputs["drive"].sum().backward()
    torch.testing.assert_close(result.outputs["drive"].cpu(), expected.outputs["drive"])
    torch.testing.assert_close(
        model.parameter_map()[projection.weight.id].grad.cpu(),
        reference.parameter_map()[projection.weight.id].grad,
    )
