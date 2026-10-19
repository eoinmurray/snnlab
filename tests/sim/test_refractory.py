"""Physical refractory durations, common scheduling and archived graph timing."""

import copy
import json

import pytest
import torch

from snnlab import lang
from snnlab._shared.timing import refractory_steps
from snnlab.lang.compilation.serialization import digest
from snnlab.sim import (
    GraphExecutor,
    PoissonInputBinding,
    load_runtime_state,
    plan_graph,
    save_runtime_state,
)
from snnlab.sim.artifacts.bundle import load_graph_bundle
from snnlab.sim.requests.handlers import simulate
from snnlab.sim.requests.spec import ExecutionSpec

MODELS = (
    lang.CUBA_LIF,
    lang.COBA_LIF,
    lang.CUBA_ALIF,
    lang.COBA_ALIF,
    lang.CUBA_ADEX,
    lang.COBA_ADEX,
)


def graph_for(model=lang.COBA_LIF, dt=0.1, duration=1.2):
    net = lang.Network("refractory", dt=dt * lang.ms)
    events = net.input(
        "events", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    settings = {"tau_mem": 20 * lang.ms, "refractory": duration * lang.ms}
    if "ALIF" in model.__name__:
        settings["adaptation_increment_mv"] = 0
    if "ADEX" in model.__name__:
        settings.update(a_us=0, b_na=0)
    neuron = model(**settings)
    cells = net.population("cells", size=1, neuron=neuron)
    current = neuron.kind.startswith("cuba_")
    net.connect(
        events,
        cells.excitatory,
        name="drive",
        synapse=(lang.ExponentialCurrent if current else lang.AMPA)(tau=5 * lang.ms),
        weight=lang.Constant(1000 if current else 100),
        initialization_scaling="direct",
    )
    net.output("spikes", cells.spikes)
    net.output("voltage", cells.voltage)
    return lang.compile(net)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "dt,count", [(0.05, 24), (0.1, 12), (0.2, 6), (0.3, 4), (0.6, 2)]
)
def test_all_models_hold_exact_physical_duration(model, dt, count):
    bundle = graph_for(model, dt)
    neuron = bundle.graph["populations"][0]["neuron"]
    assert neuron["refractory"] == {"value": 1.2, "unit": "ms"}
    assert "refractory_steps" not in neuron
    assert "refractory_policy" not in neuron
    plan = plan_graph(bundle.graph)
    timing = plan.refractory["cells"]
    assert timing["refractory_steps"] == timing["held_steps"] == count
    assert timing["requested_refractory_ms"] == 1.2
    assert timing["realized_refractory_ms"] == pytest.approx(1.2)
    assert timing["held_duration_ms"] == pytest.approx(1.2)
    assert timing["timing"] == "following_steps"
    steps = 3 * (count + 1)
    result = GraphExecutor(plan)({"events": torch.ones(steps, 1, 1)})
    indices = torch.where(result.outputs["spikes"].flatten() > 0)[0].tolist()
    assert indices == [0, count + 1, 2 * (count + 1)]
    assert torch.all(result.outputs["voltage"][1 : count + 1] == -65)
    assert result.metrics["refractory"]["cells"] == timing


@pytest.mark.parametrize("model", MODELS)
def test_zero_duration_allows_spiking_on_next_step(model):
    plan = plan_graph(graph_for(model, duration=0).graph)
    result = GraphExecutor(plan)({"events": torch.ones(4, 1, 1)})
    assert result.outputs["spikes"].flatten().tolist() == [1, 1, 1, 1]
    assert plan.refractory["cells"]["refractory_steps"] == 0


@pytest.mark.parametrize("model", MODELS)
def test_nondivisible_duration_fails_compilation_and_planning(model):
    graph = graph_for(model).graph
    graph["timebase"]["dt"]["value"] = 0.25
    with pytest.raises(ValueError, match="cells.*compatible timestep"):
        plan_graph(graph)
    assert lang.validate_graph(graph).errors
    with pytest.raises(ValueError, match="compatible timestep"):
        graph_for(model, dt=0.25)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("setting", ["refractory_steps", "refractory_policy"])
def test_constructors_reject_old_configuration(model, setting):
    with pytest.raises(TypeError, match=setting):
        model(**{setting: 2 if setting.endswith("steps") else "nearest"})


@pytest.mark.parametrize(
    "quantity",
    [
        -1,
        float("nan"),
        float("inf"),
        True,
        "1.2",
    ],
)
def test_invalid_serialized_durations_fail_both_paths(quantity):
    graph = graph_for().graph
    graph["populations"][0]["neuron"]["refractory"]["value"] = quantity
    with pytest.raises(ValueError, match="non-negative finite ms"):
        plan_graph(graph)
    assert lang.validate_graph(graph).errors


@pytest.mark.parametrize("value", [0.6, 0.6 * lang.mV])
def test_refractory_requires_ms_quantity(value):
    net = lang.Network("invalid")
    net.population("cells", size=1, neuron=lang.CUBA_LIF(refractory=value))
    with pytest.raises(ValueError, match="ms quantity"):
        lang.compile(net)


@pytest.mark.parametrize("legacy", [False, True])
def test_conflicting_representations_are_rejected(legacy):
    graph = graph_for().graph
    if legacy:
        graph.pop("refractory_timing")
    graph["populations"][0]["neuron"]["refractory_steps"] = 12
    with pytest.raises(ValueError):
        plan_graph(graph)
    assert lang.validate_graph(graph).errors
    with pytest.raises(ValueError):
        lang.migrate_refractory(graph)


@pytest.mark.parametrize("model", MODELS)
def test_legacy_bundle_keeps_identity_and_saved_continuation(model, tmp_path):
    bundle = graph_for(model)
    bundle.graph.pop("refractory_timing")
    neuron = bundle.graph["populations"][0]["neuron"]
    neuron.pop("refractory")
    neuron["refractory_steps"] = 12
    bundle.manifest["graph_digest"] = digest(bundle.graph)
    bundle.manifest["files"][0]["digest"] = digest(bundle.graph)
    path = bundle.write(tmp_path / "old")
    original = (path / "graph.json").read_bytes()
    restored = lang.load_bundle(path)
    manifest, graph = load_graph_bundle(path)
    assert manifest["graph_digest"] == digest(graph) == digest(restored.graph)
    plan = plan_graph(graph)
    expected_gap = 12 if model == lang.COBA_LIF else 13
    executor = GraphExecutor(plan)
    values = torch.ones(40, 1, 1)
    full = executor({"events": values})
    assert torch.where(full.outputs["spikes"].flatten() > 0)[0].tolist() == list(
        range(0, 40, expected_gap)
    )
    first = executor({"events": values[:5]})
    saved = save_runtime_state(tmp_path / "state", first.runtime_state)
    resumed = executor({"events": values[5:]}, runtime_state=load_runtime_state(saved))
    for name in full.outputs:
        assert torch.equal(
            torch.cat((first.outputs[name], resumed.outputs[name])), full.outputs[name]
        )
    assert (path / "graph.json").read_bytes() == original
    migrated = lang.migrate_refractory(graph)
    assert digest(migrated) != manifest["graph_digest"]
    assert graph == restored.graph
    assert "refractory_steps" not in migrated["populations"][0]["neuron"]
    assert migrated["populations"][0]["neuron"]["refractory"]["value"] == pytest.approx(
        1.2
    )
    migrated_result = GraphExecutor(plan_graph(migrated))({"events": values})
    for name in full.outputs:
        assert torch.equal(migrated_result.outputs[name], full.outputs[name])
    with pytest.raises(ValueError, match="incompatible with graph plan"):
        GraphExecutor(plan_graph(migrated))(
            {"events": values[5:]}, runtime_state=first.runtime_state
        )


def test_legacy_implicit_rounding_is_confined_to_archived_graphs():
    graph = graph_for(dt=0.25, duration=1).graph
    graph.pop("refractory_timing")
    neuron = graph["populations"][0]["neuron"]
    neuron.pop("refractory")
    graph["timebase"]["dt"]["value"] = 0.7
    plan = plan_graph(graph)
    timing = plan.refractory["cells"]
    assert timing["requested_refractory_ms"] == 3
    assert timing["realized_refractory_ms"] == pytest.approx(2.8)
    assert timing["held_duration_ms"] == pytest.approx(2.1)
    migrated = lang.migrate_refractory(graph)
    assert migrated["populations"][0]["neuron"]["refractory"]["value"] == pytest.approx(
        2.8
    )
    assert plan_graph(migrated).refractory["cells"]["refractory_steps"] == 4


@pytest.mark.parametrize("legacy", [False, True])
def test_timestep_override_preserves_source_physical_duration(legacy):
    graph = graph_for().graph
    if legacy:
        graph.pop("refractory_timing")
        neuron = graph["populations"][0]["neuron"]
        neuron.pop("refractory")
        neuron["refractory_steps"] = 12
    original = copy.deepcopy(graph)
    spec = ExecutionSpec(
        kind="simulate",
        graph=graph,
        device="cpu",
        input_bindings=(PoissonInputBinding("events", 60, (1000,), seed=17),),
        options={"inference_overrides": {"timestep_ms": 0.2}},
    )
    result = simulate(spec)
    timing = result.metrics["refractory"]["cells"]
    assert timing["refractory_steps"] == 6
    assert timing["requested_refractory_ms"] == pytest.approx(1.2)
    assert timing["realized_refractory_ms"] == pytest.approx(1.2)
    assert graph == original
    spec.options["inference_overrides"]["timestep_ms"] = 0.25
    with pytest.raises(ValueError, match="compatible timestep"):
        simulate(spec)


def test_default_durations_and_idempotent_migration():
    graph = graph_for().graph
    assert lang.migrate_refractory(graph) == graph
    for tau, duration in ((20, 3), (5, 1.5)):
        net = lang.Network("defaults", dt=0.1 * lang.ms)
        net.population("cells", size=1, neuron=lang.COBA_LIF(tau_mem=tau * lang.ms))
        graph = lang.compile(net).graph
        assert (
            plan_graph(graph).refractory["cells"]["requested_refractory_ms"] == duration
        )
        assert lang.migrate_refractory(graph) == graph
    assert lang.CUBA_LIF().json()["refractory"] == {"value": 0, "unit": "ms"}
    assert refractory_steps(1.2, 0.1) == 12


def test_serialized_contract_marker_is_validated():
    graph = graph_for().graph
    graph["refractory_timing"] = "nearest"
    assert lang.validate_graph(graph).errors
    with pytest.raises(ValueError, match="refractory_timing"):
        plan_graph(graph)
    assert json.loads(json.dumps(graph))["refractory_timing"] == "nearest"


@pytest.mark.parametrize("count", [0, 1, 2])
def test_legacy_zero_and_one_counts_keep_historical_release(count):
    graph = graph_for().graph
    graph.pop("refractory_timing")
    neuron = graph["populations"][0]["neuron"]
    neuron.pop("refractory")
    neuron["refractory_steps"] = count
    migrated = lang.migrate_refractory(graph)
    values = {"events": torch.ones(6, 1, 1)}
    old = GraphExecutor(plan_graph(graph))(values)
    new = GraphExecutor(plan_graph(migrated))(values)
    assert torch.where(old.outputs["spikes"].flatten() > 0)[0].tolist() == list(
        range(0, 6, max(count, 1))
    )
    assert torch.equal(old.outputs["spikes"], new.outputs["spikes"])


def test_serialized_policy_cannot_enable_rounding():
    graph = graph_for().graph
    graph["populations"][0]["neuron"]["refractory_policy"] = "nearest"
    assert lang.validate_graph(graph).errors
    with pytest.raises(ValueError, match="refractory_policy"):
        plan_graph(graph)


def test_tiny_positive_duration_is_not_zero():
    with pytest.raises(ValueError, match="compatible timestep"):
        refractory_steps(1e-300, 1e300)
