"""Physical current dynamics and complete registered extension workflows."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from snnlab import extensions as ext
from snnlab import lang
from snnlab.lang import training as recipe
from snnlab.sim.execution import (
    DatasetEncoder,
    DatasetSnapshotBinding,
    DenseArrayBinding,
    ExecutionSpec,
    GraphExecutor,
    load_runtime_state,
    plan_graph,
    save_runtime_state,
    simulate,
    train,
)
from tests.sim import _extension_callbacks as callbacks


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(ext, "_REGISTRY", {})


def current_network(**values):
    net = lang.Network("current", dt=1 * lang.ms)
    inputs = net.input(
        "inputs", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    cells = net.population("cells", size=1, neuron=lang.CUBA_LIF(**values))
    projection = net.connect(
        inputs,
        cells.excitatory,
        name="input",
        synapse=lang.ExponentialCurrent(tau=3 * lang.ms),
        weight=lang.Constant(1),
    )
    net.output("voltage", cells.voltage)
    net.output("spikes", cells.spikes)
    net.expose(projection.current, name="current")
    return net, inputs, cells, projection


def run(bundle, values, state=None):
    return simulate(
        ExecutionSpec(
            kind="simulate",
            graph=bundle.graph,
            input_bindings=(DenseArrayBinding("inputs", values),),
            device="cpu",
            runtime_state=state,
        )
    )


def test_current_matches_independent_analytic_recurrence():
    net, _, _, projection = current_network(threshold_mv=1000, initial_voltage_mv=-60)
    bundle = lang.compile(net, target="tools/snnsim")
    assert not any(
        d.code.startswith("C") or d.severity == "error" for d in bundle.diagnostics
    )
    assert bundle.graph["parameters"][0]["unit"] == "nA"
    assert projection.current.unit == "nA"
    values = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0]).reshape(-1, 1, 1)
    result = run(bundle, values)
    voltage, current = -60.0, 0.0
    expected = []
    currents = []
    beta = math.exp(-1 / 20)
    for event in values.flatten().tolist():
        current = current * math.exp(-1 / 3) + event
        voltage = -65 + (voltage + 65) * beta + current * 20 * (1 - beta)
        expected.append(voltage)
        currents.append(current)
    np.testing.assert_allclose(
        result.numpy(batch=0).outputs["voltage"].ravel(), expected, rtol=1e-6
    )
    np.testing.assert_allclose(
        result.numpy(batch=0).diagnostics["current"].ravel(), currents, rtol=1e-6
    )


def test_current_inhibition_and_refractory_reset():
    net, inputs, cells, _ = current_network(
        threshold_mv=-64.5, reset_mv=-67, refractory_steps=2
    )
    bundle = lang.compile(net)
    result = run(bundle, torch.ones(8, 1, 1))
    spikes = result.outputs["spikes"].flatten()
    assert spikes[0] == 1
    assert spikes[1:3].sum() == 0
    assert torch.all(result.outputs["voltage"][1:3] == -67)
    net.projections[0]["target"] = cells.inhibitory
    net.projections[0]["polarity"] = "inhibitory"
    inhibited = run(lang.compile(net), torch.ones(8, 1, 1))
    assert inhibited.outputs["spikes"].sum() == 0
    assert inhibited.outputs["voltage"][-1] < -65
    assert lang.LIF().kind == "cuba_lif"


def test_current_continuation_and_saved_state(tmp_path):
    net, *_ = current_network(threshold_mv=1000)
    bundle = lang.compile(net)
    values = torch.ones(12, 1, 1)
    full = run(bundle, values)
    first = run(bundle, values[:5])
    state = load_runtime_state(
        save_runtime_state(tmp_path / "state", first.runtime_state)
    )
    second = run(bundle, values[5:], state)
    torch.testing.assert_close(
        torch.cat([first.outputs["voltage"], second.outputs["voltage"]]),
        full.outputs["voltage"],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "model",
    [lang.COBA_LIF(tau_mem=20 * lang.ms), lang.LeakyIntegrator(tau=20 * lang.ms)],
)
def test_reject_current_conductance_family_mismatch(model):
    net, _, _, _ = current_network()
    net.populations[0]["neuron"] = model.json()
    with pytest.raises(ValueError, match="incompatible"):
        lang.compile(net)
    with pytest.raises(ValueError, match="incompatible"):
        plan_graph(net.__dict__ | {"timebase": {"dt": {"value": 1}}})


@pytest.mark.parametrize(
    "values",
    [
        {"tau_mem": 0 * lang.ms},
        {"capacitance_nf": 0},
        {"refractory_steps": -1},
        {"threshold_mv": float("nan")},
        {"reset_mv": -40},
    ],
)
def test_invalid_current_neuron_parameters(values):
    net, *_ = current_network(**values)
    with pytest.raises(ValueError):
        lang.compile(net)


def register_dynamics():
    ext.register_neuron(
        "test.adaptive/v1",
        "tests.sim._extension_callbacks:neuron_step",
        initialize="tests.sim._extension_callbacks:neuron_init",
        validate="tests.sim._extension_callbacks:validate_neuron",
        state_units={"adaptation": "nA"},
    )
    ext.register_synapse(
        "test.trace/v1",
        callbacks.synapse_step,
        initialize=callbacks.synapse_init,
        state_units={"trace": "nA"},
    )


def custom_network():
    register_dynamics()
    net = lang.Network("custom", dt=1 * lang.ms)
    inputs = net.input(
        "inputs", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    cells = net.population(
        "cells", size=1, neuron=lang.CustomNeuron("test.adaptive/v1", threshold=0.9)
    )
    projection = net.connect(
        inputs,
        cells.excitatory,
        name="input",
        synapse=lang.CustomSynapse("test.trace/v1"),
        weight=lang.Constant(0.2),
    )
    net.output("spikes", cells.spikes)
    net.output("voltage", cells.voltage)
    net.expose(cells.state("adaptation"), name="adaptation")
    net.expose(projection.state("trace"), name="trace")
    net.expose(projection.current, name="current")
    return net, inputs, cells, projection


def test_custom_dynamics_bundle_state_and_gradient(tmp_path):
    net, _, _, projection = custom_network()
    bundle = lang.compile(net, target="tools/snnsim")
    assert {x["category"] for x in bundle.manifest["extensions"]} == {
        "neuron",
        "synapse",
    }
    restored = lang.load_bundle(bundle.write(tmp_path / "bundle"))
    model = GraphExecutor(
        plan_graph(restored.graph), trainable_parameters=(projection.weight.id,)
    )
    values = torch.ones(14, 2, 1)
    full = model({"inputs": values})
    full.outputs["spikes"].sum().backward()
    assert model.parameter_map()[projection.weight.id].grad.abs().sum() > 0
    assert not full.diagnostics["adaptation"].requires_grad
    first = model({"inputs": values[:6]})
    path = save_runtime_state(tmp_path / "state", first.runtime_state)
    assert json.loads((path / "manifest.json").read_text())["schema_version"] == 2
    second = model({"inputs": values[6:]}, runtime_state=load_runtime_state(path))
    for key in full.outputs:
        torch.testing.assert_close(
            torch.cat([first.outputs[key], second.outputs[key]]),
            full.outputs[key],
            rtol=0,
            atol=0,
        )
    for key in full.diagnostics:
        torch.testing.assert_close(
            torch.cat([first.diagnostics[key], second.diagnostics[key]]),
            full.diagnostics[key],
            rtol=0,
            atol=0,
        )
    assert full.runtime_state.custom_state


def test_custom_initializer_constraint_operation_surrogate_and_regression_training(
    tmp_path,
):
    ext.register_initializer(
        "test.normal/v1",
        callbacks.normal,
    )
    ext.register_constraint(
        "test.bounds/v1",
        callbacks.bounds,
    )
    ext.register_operation(
        "test.affine/v1",
        callbacks.affine,
    )
    ext.register_objective(
        "test.mse/v1",
        callbacks.mse,
    )
    ext.register_regularizer("test.energy/v1", callbacks.energy)
    ext.register_optimizer("test.sgd/v1", callbacks.sgd)
    ext.register_surrogate("test.triangle/v1", callbacks.triangle)
    net = lang.Network("regression", dt=1 * lang.ms)
    inputs = net.input(
        "inputs", shape=("time", "batch", 1), signal_type="continuous", unit="1"
    )
    gain = net.parameter(
        "gain",
        shape=(1,),
        initializer=lang.CustomInitializer("test.normal/v1", value=0.1),
        constraint=lang.CustomConstraint("test.bounds/v1", limit=0.5),
    )
    scores = lang.ops.custom(
        "test.affine/v1",
        inputs,
        name="scores",
        shape=("batch", 1),
        unit="1",
        parameters=(gain,),
    )
    net.output("prediction", scores)
    training = lang.TrainSpec(
        objectives=(
            recipe.CustomObjective("test.mse/v1", prediction=scores, target="value"),
        ),
        parameter_groups=(recipe.ParameterGroup((gain,), name="all", lr=0.2),),
        optimizer=recipe.CustomOptimizer("test.sgd/v1", momentum=0.5),
        regularizers=(
            recipe.CustomRegularizer(
                "test.energy/v1", signals=(scores,), strength=0.01
            ),
        ),
        surrogate=recipe.CustomSurrogate("test.triangle/v1"),
    )
    bundle = lang.compile(net, training=training)
    path = bundle.write(tmp_path / "bundle")
    ext._REGISTRY.clear()
    args = dict(
        kind="train",
        bundle=path,
        input_bindings=(DenseArrayBinding("inputs", torch.ones(3, 4, 1)),),
        targets={"value": torch.ones(4, 1)},
        device="cpu",
        epochs=2,
        batch_size=2,
    )
    first = train(
        ExecutionSpec(**args, updates=2, save_final_checkpoint=tmp_path / "checkpoint")
    )
    resumed = train(ExecutionSpec(**args, checkpoint=tmp_path / "checkpoint"))
    full = train(ExecutionSpec(**args))
    torch.testing.assert_close(
        resumed.parameters["gain"], full.parameters["gain"], rtol=0, atol=0
    )
    assert first.parameters["gain"].item() > 0.1
    assert full.parameters["gain"].item() <= 0.5
    assert "train_accuracy" not in full.metrics["epochs"][0]
    assert (
        full.metrics["epochs"][-1]["train_loss"]
        < full.metrics["epochs"][0]["train_loss"]
    )
    script = """
import sys
import torch
from snnlab import extensions
from snnlab.sim.execution import DenseArrayBinding, ExecutionSpec, train
assert not extensions.definitions()
result = train(ExecutionSpec(kind="train", bundle=sys.argv[1],
    input_bindings=(DenseArrayBinding("inputs", torch.ones(3, 4, 1)),),
    targets={"value": torch.ones(4, 1)}, device="cpu", epochs=2, batch_size=2))
print(result.parameters["gain"].item())
"""
    root = Path(__file__).resolve().parents[2]
    environment = dict(
        os.environ, PYTHONPATH=os.pathsep.join((str(root), str(root / "src")))
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert float(completed.stdout.strip()) == full.parameters["gain"].item()


def test_custom_surrogate_derivative_reaches_current_weights():
    ext.register_surrogate(
        "test.constant/v1",
        lambda value, config: torch.full_like(value, config["derivative"]),
    )
    net, _, _, projection = current_network()
    model = GraphExecutor(
        plan_graph(lang.compile(net).graph),
        trainable_parameters=(projection.weight.id,),
        surrogate=recipe.CustomSurrogate("test.constant/v1", derivative=2).json(),
    )
    result = model({"inputs": torch.ones(1, 1, 1)})
    result.outputs["spikes"].sum().backward()
    expected = 2 * 20 * (1 - math.exp(-1 / 20))
    assert model.parameter_map()[projection.weight.id].grad.item() == pytest.approx(
        expected
    )


def test_missing_duplicate_and_invalid_registration():
    with pytest.raises(ValueError, match="unregistered"):
        lang.CustomNeuron("test.missing/v1")
    with pytest.raises(ValueError, match="versioned"):
        ext.register_initializer("bad", lambda *a: None)
    ext.register_initializer("test.same/v1", lambda *a: None)
    with pytest.raises(ValueError, match="already registered"):
        ext.register_initializer("test.same/v1", lambda *a: None)


def test_bad_custom_shape_and_state_change_fail():
    ext.register_initializer("test.bad/v1", lambda shape, config, **kw: torch.zeros(9))
    net, *_ = current_network()
    net.parameters[0]["initializer"] = lang.CustomInitializer("test.bad/v1").json()
    with pytest.raises(ValueError, match="shape"):
        run(lang.compile(net), torch.ones(3, 1, 1))
    net, *_ = custom_network()
    definition = ext.get("neuron", "test.adaptive/v1")
    # A separate version has an incompatible step state contract.
    ext.register_neuron(
        "test.badstate/v1",
        lambda ctx: (
            {"voltage": ctx.state["voltage"]},
            torch.zeros_like(ctx.state["voltage"]),
        ),
        initialize=definition.initialize,
    )
    net.populations[0]["neuron"] = lang.CustomNeuron("test.badstate/v1").json()
    net.observables = []
    with pytest.raises(ValueError, match="state keys"):
        run(lang.compile(net), torch.ones(3, 1, 1))


def test_custom_dataset_encoder(tmp_path):
    ext.register_encoder(
        "test.encode/v1",
        lambda arrays, selected, **kw: torch.ones(5, len(selected), kw["channels"]),
    )
    path = tmp_path / "data.npz"
    np.savez(path, features=np.ones((3, 1)), labels=np.zeros(3, dtype=np.int64))
    net, *_ = current_network()
    binding = DatasetSnapshotBinding(
        path,
        "inputs",
        "test",
        "train",
        DatasetEncoder("custom", definition="test.encode/v1"),
    )
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            graph=lang.compile(net).graph,
            input_bindings=(binding,),
            device="cpu",
        )
    )
    assert result.outputs["voltage"].shape == (5, 3, 1)
    assert (
        result.metrics["execution_protocol"]["dataset_binding"]["encoder"]["definition"]
        == "test.encode/v1"
    )


def test_custom_constraint_applied_after_fanin_normalization():
    ext.register_constraint("test.floor/v1", lambda value, config: value.clamp(min=1.0))
    net = lang.Network("floor")
    inputs = net.input(
        "inputs", shape=("time", "batch", 4), signal_type="spikes", unit="spike"
    )
    cells = net.population("cells", size=1, neuron=lang.CUBA_LIF())
    projection = net.connect(
        inputs,
        cells.excitatory,
        name="input",
        synapse=lang.ExponentialCurrent(),
        weight=lang.Constant(1.0),
        constraint=lang.CustomConstraint("test.floor/v1"),
    )
    model = GraphExecutor(plan_graph(lang.compile(net).graph))
    assert torch.all(model.parameter_map()[projection.weight.id] == 1.0)
    with torch.no_grad():
        model.parameter_map()[projection.weight.id].fill_(0)
    model.enforce_constraints()
    assert torch.all(model.parameter_map()[projection.weight.id] == 1.0)


@pytest.mark.parametrize("loader", ["lang", "sim", "cli"])
def test_saved_extensions_load_and_simulate_in_fresh_process(tmp_path, loader):
    net, *_ = custom_network()
    bundle = lang.compile(net)
    path = bundle.write(tmp_path / "bundle")
    expected = run(bundle, torch.ones(14, 2, 1))
    assert not any(
        file.suffix in {".py", ".pkl", ".pickle"} for file in path.rglob("*")
    )
    assert bundle.graph["extensions"] == bundle.manifest["extensions"]
    environment = dict(os.environ)
    root = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = os.pathsep.join((str(root), str(root / "src")))
    script = """
import sys
import numpy as np
import torch
from snnlab import extensions, lang
from snnlab.sim.execution import DenseArrayBinding, ExecutionSpec, simulate
assert not extensions.definitions()
if sys.argv[3] == "lang":
    bundle = lang.load_bundle(sys.argv[1])
result = simulate(ExecutionSpec(kind="simulate", bundle=sys.argv[1],
    input_bindings=(DenseArrayBinding("inputs", torch.ones(14, 2, 1)),), device="cpu"))
np.savez(sys.argv[2], **result.numpy().outputs)
"""
    output = tmp_path / "result.npz"
    command = [sys.executable, "-c", script, str(path), str(output), loader]
    if loader == "cli":
        values_path = tmp_path / "inputs.npy"
        np.save(values_path, np.ones((14, 2, 1), dtype=np.float32))
        output = tmp_path / "run" / "outputs.npz"
        command = [
            sys.executable,
            "-m",
            "snnlab.sim",
            "sim",
            "--bundle",
            str(path),
            "--input-file",
            str(values_path),
            "--out-dir",
            str(output.parent),
            "--device",
            "cpu",
        ]
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    with np.load(output) as actual:
        for name, value in expected.numpy().outputs.items():
            np.testing.assert_array_equal(actual[name], value)
    ext._REGISTRY.clear()
    # Standalone graph JSON also resolves its dependencies without manual imports.
    model = GraphExecutor(plan_graph(json.loads((path / "graph.json").read_text())))
    actual = model({"inputs": torch.ones(14, 2, 1)})
    torch.testing.assert_close(
        actual.outputs["spikes"], expected.outputs["spikes"], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "reference",
    [
        "missing_snnlab_test_module:step",
        "tests.sim._extension_callbacks:missing_step",
        "tests.sim._extension_callbacks:not_callable",
    ],
)
def test_missing_or_invalid_saved_callback_fails_before_simulation(reference):
    net, *_ = custom_network()
    graph = lang.compile(net).graph
    graph["extensions"][0]["implementation"]["function"] = reference
    ext._REGISTRY.clear()
    with pytest.raises((ValueError, TypeError), match="callback"):
        simulate(
            ExecutionSpec(
                kind="simulate",
                graph=graph,
                input_bindings=(DenseArrayBinding("inputs", torch.ones(2, 1, 1)),),
                device="cpu",
            )
        )


def test_saved_package_version_checked_even_with_cached_definition(monkeypatch):
    monkeypatch.setattr(ext.metadata, "version", lambda package: "1.2.3")
    ext.register_initializer(
        "test.package/v1", callbacks.normal, package="example-neurons"
    )
    net, *_ = current_network()
    net.parameters[0]["initializer"] = lang.CustomInitializer(
        "test.package/v1", value=1.0
    ).json()
    graph = lang.compile(net).graph
    assert graph["extensions"][0]["package_version"] == "1.2.3"
    monkeypatch.setattr(ext.metadata, "version", lambda package: "2.0.0")
    with pytest.raises(ValueError, match="requires version 1.2.3, installed 2.0.0"):
        plan_graph(graph)


def test_missing_extension_package_has_clear_error():
    with pytest.raises(ValueError, match="package.*not installed"):
        ext.register_initializer(
            "test.package/v1", callbacks.normal, package="snnlab-missing-test-package"
        )


def test_nonimportable_callbacks_cannot_be_saved(tmp_path):
    ext.register_initializer(
        "test.local/v1", lambda shape, config, **kw: torch.ones(shape, **kw)
    )
    net, *_ = current_network()
    net.parameters[0]["initializer"] = lang.CustomInitializer("test.local/v1").json()
    bundle = lang.compile(net)
    run(bundle, torch.ones(2, 1, 1))  # Ephemeral callbacks remain usable in memory.
    with pytest.raises(ValueError, match="cannot save.*importable module functions"):
        bundle.write(tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_existing_definition_cannot_override_saved_implementation():
    net, *_ = custom_network()
    graph = lang.compile(net).graph
    ext._REGISTRY.clear()
    ext.register_neuron(
        "test.adaptive/v1", callbacks.synapse_step, initialize=callbacks.neuron_init
    )
    with pytest.raises(ValueError, match="conflicts with saved definition"):
        plan_graph(graph)


def test_import_references_are_verified_before_loading(tmp_path):
    net, *_ = custom_network()
    bundle = lang.compile(net)
    path = bundle.write(tmp_path / "bundle")
    graph = json.loads((path / "graph.json").read_text())
    graph["extensions"][0]["implementation"]["function"] = (
        "missing_snnlab_test_module:step"
    )
    (path / "graph.json").write_text(json.dumps(graph))
    ext._REGISTRY.clear()
    from snnlab.sim.bundle import load_graph_bundle

    for loader in (lang.load_bundle, load_graph_bundle):
        with pytest.raises(ValueError, match="digest"):
            loader(path)
    assert not ext.definitions()


def test_older_name_only_bundle_requires_registration(tmp_path):
    net, *_ = custom_network()
    bundle = lang.compile(net)
    bundle.graph.pop("extensions")
    from snnlab.lang.compiler import canonical_json, digest

    path = bundle.write(tmp_path / "bundle")
    bundle.manifest["graph_digest"] = digest(bundle.graph)
    bundle.manifest["files"][0]["digest"] = digest(bundle.graph)
    (path / "manifest.json").write_bytes(canonical_json(bundle.manifest))
    ext._REGISTRY.clear()
    with pytest.raises(ValueError, match="registration module"):
        lang.load_bundle(path)


def test_stateless_optimizer_resume(tmp_path):
    ext.register_optimizer("test.stateless/v1", callbacks.sgd)
    net = lang.Network("stateless")
    inputs = net.input(
        "inputs", shape=("time", "batch", 2), signal_type="continuous", unit="1"
    )
    scores = lang.ops.linear(inputs, size=2, name="readout")
    scores = lang.ops.reduce(scores, operation="mean", over="time", name="average")
    net.output("prediction", scores)
    training = lang.TrainSpec(
        objectives=(recipe.CrossEntropy(prediction=scores, target="class"),),
        parameter_groups=(
            recipe.ParameterGroup(("readout.weight",), name="readout", lr=0.1),
        ),
        optimizer=recipe.CustomOptimizer("test.stateless/v1"),
    )
    bundle = lang.compile(net, training=training)
    args = dict(
        kind="train",
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(
            DenseArrayBinding("inputs", torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])),
        ),
        targets={"class": torch.tensor([0, 1])},
        device="cpu",
    )
    first = train(
        ExecutionSpec(**args, updates=1, save_final_checkpoint=tmp_path / "checkpoint")
    )
    resumed = train(
        ExecutionSpec(**args, updates=1, checkpoint=tmp_path / "checkpoint")
    )
    full = train(ExecutionSpec(**args, updates=2))
    assert first.optimizer_state == {"readout.weight": {}}
    torch.testing.assert_close(
        resumed.parameters["readout.weight"],
        full.parameters["readout.weight"],
        rtol=0,
        atol=0,
    )
