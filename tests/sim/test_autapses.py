"""Declared self-connectivity survives initialization, execution and training."""

import copy
import io

import pytest
import torch

from snnlab import extensions as ext
from snnlab import lang
from snnlab.lang import training
from snnlab.lang.compiler import graph_dict, validate_graph
from snnlab.sim.execution import (
    DenseArrayBinding,
    ExecutionSpec,
    GraphExecutor,
    load_runtime_state,
    load_training_checkpoint,
    plan_graph,
    save_runtime_state,
    simulate,
    train,
)


def recurrent_network(
    *,
    size=4,
    polarity="excitatory",
    initializer=None,
    allowed=False,
    scaling="fan_in_normalized",
    connection="recurrent",
):
    net = lang.Network("autapses", dt=0.1 * lang.ms)
    events = net.input(
        "events", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    cells = net.population(
        "cells", size=size, neuron=lang.COBA_LIF(tau_mem=20 * lang.ms)
    )
    net.connect(
        events,
        cells.excitatory,
        name="drive",
        synapse=lang.AMPA(),
        weight=lang.Constant(10),
    )
    projection = net.connect(
        cells.spikes,
        getattr(cells, polarity),
        name="recurrence",
        synapse=lang.AMPA() if polarity == "excitatory" else lang.GABA(),
        weight=initializer or lang.Constant(1),
        constraint=lang.NonNegative(),
        connection=connection,
        delay=0.1 * lang.ms,
        allow_autapses=allowed,
        initialization_scaling=scaling,
    )
    net.output("feedback", projection.conductance)
    return net, cells, projection


def executor(net, projection, *, seed=0):
    graph = lang.compile(net, target=None).graph
    return GraphExecutor(
        plan_graph(graph), seed=seed, trainable_parameters=(projection.weight.id,)
    )


@pytest.mark.parametrize("polarity", ["excitatory", "inhibitory"])
@pytest.mark.parametrize("zeroing", ["dense", "bernoulli", "exact_k"])
@pytest.mark.parametrize("allowed", [False, True])
def test_positive_initializers_obey_policy_and_exact_partner_count(
    polarity, zeroing, allowed
):
    initializer = lang.LowerClampedNormal(
        1,
        0,
        initial_zero_fraction=0 if zeroing == "dense" else 0.5,
        zeroing="bernoulli" if zeroing == "dense" else zeroing,
    )
    net, _, projection = recurrent_network(
        size=8, polarity=polarity, initializer=initializer, allowed=allowed
    )
    model = executor(net, projection)
    weight = model.parameter_map()[projection.weight.id]
    if not allowed:
        assert torch.count_nonzero(weight.diagonal()) == 0
    elif zeroing == "dense":
        assert torch.all(weight.diagonal() > 0)
    else:
        assert torch.count_nonzero(weight.diagonal()) > 0
    eligible = 8 if allowed else 7
    if zeroing == "dense":
        assert torch.all((weight != 0).sum(0) == eligible)
        assert torch.all(weight[weight != 0] == 1 / eligible)
    elif zeroing == "exact_k":
        # Both round(7 * 0.5) and round(8 * 0.5) select four partners.
        assert torch.all((weight != 0).sum(0) == 4)
        assert torch.all(weight[weight != 0] == 0.25)
        assert torch.all(weight.sum(0) == 1)
    metadata = model.initialization_metadata[projection.weight.id]["autapses"][
        projection.id
    ]
    assert metadata["requested"] is allowed
    assert metadata["eligible_fan_in"] == eligible


@pytest.mark.parametrize("connection", ["feedforward", "recurrent", "feedback"])
def test_population_identity_controls_exclusion_independently_of_label(connection):
    net, _, projection = recurrent_network(connection=connection)
    model = executor(net, projection)
    assert (
        torch.count_nonzero(model.parameter_map()[projection.weight.id].diagonal()) == 0
    )


@pytest.mark.parametrize("size", [1, 2, 3])
@pytest.mark.parametrize("zeroing", ["dense", "bernoulli", "exact_k"])
@pytest.mark.parametrize("scaling", ["direct", "fan_in_normalized"])
def test_small_populations_have_no_impossible_partners(size, zeroing, scaling):
    initializer = lang.LowerClampedNormal(
        1,
        0,
        initial_zero_fraction=0 if zeroing == "dense" else 0.99,
        zeroing="bernoulli" if zeroing == "dense" else zeroing,
    )
    net, _, projection = recurrent_network(
        size=size, initializer=initializer, scaling=scaling
    )
    weight = executor(net, projection).parameter_map()[projection.weight.id]
    assert torch.isfinite(weight).all()
    assert torch.count_nonzero(weight.diagonal()) == 0
    if size == 1:
        assert torch.count_nonzero(weight) == 0
    elif zeroing == "exact_k":
        assert torch.all((weight != 0).sum(0) == 1)
        assert torch.all(weight.sum(0) == (size - 1 if scaling == "direct" else 1))


def test_equal_sized_different_populations_keep_matching_indices():
    net, cells, _ = recurrent_network()
    other = net.population("other", size=4, neuron=lang.COBA_LIF())
    projection = net.connect(
        cells.spikes,
        other.excitatory,
        name="cross",
        synapse=lang.AMPA(),
        weight=lang.Constant(1),
    )
    model = executor(net, projection)
    weight = model.parameter_map()[projection.weight.id]
    assert torch.all(weight == 0.25)
    assert model.autapse_metadata[projection.id]["effective"] == "not_applicable"


def test_single_neuron_can_opt_in_to_self_feedback():
    net, _, projection = recurrent_network(size=1, allowed=True)
    model = executor(net, projection)
    assert torch.equal(model.parameter_map()[projection.weight.id], torch.ones(1, 1))
    assert model.autapse_metadata[projection.id]["eligible_fan_in"] == 1


@pytest.mark.parametrize("invalid", [None, 0, 1, "false"])
def test_policy_requires_a_boolean_in_authoring_and_saved_graphs(invalid):
    with pytest.raises(TypeError, match="allow_autapses must be boolean"):
        recurrent_network(allowed=invalid)
    net, _, _ = recurrent_network()
    graph = lang.compile(net, target=None).graph
    graph["projections"][-1]["allow_autapses"] = invalid
    with pytest.raises(ValueError, match="allow_autapses must be boolean"):
        validate_graph(graph).raise_for_errors()
    with pytest.raises(ValueError, match="allow_autapses must be boolean"):
        plan_graph(graph)


@pytest.mark.parametrize("zeroing", ["dense", "bernoulli", "exact_k"])
def test_historical_graphs_preserve_draws_and_initialization_metadata(zeroing):
    initializer = lang.LowerClampedNormal(
        1,
        0,
        initial_zero_fraction=0 if zeroing == "dense" else 0.5,
        zeroing="bernoulli" if zeroing == "dense" else zeroing,
    )
    net, _, projection = recurrent_network(
        size=8, initializer=initializer, allowed=True
    )
    graph = lang.compile(net, target=None).graph
    explicit = GraphExecutor(plan_graph(graph), seed=17)
    historical = copy.deepcopy(graph)
    for row in historical["projections"]:
        row.pop("allow_autapses")
    before = copy.deepcopy(historical)
    legacy = GraphExecutor(plan_graph(historical), seed=17)
    assert historical == before
    for name, weight in explicit.parameter_map().items():
        assert torch.equal(weight, legacy.parameter_map()[name])
        expected = copy.deepcopy(explicit.initialization_metadata[name])
        expected.pop("autapses")
        assert legacy.initialization_metadata[name] == expected
    assert legacy.autapse_metadata[projection.id]["origin"] == "legacy_default"
    assert legacy.autapse_metadata[projection.id]["effective"] == "allowed"
    old_state_dict = {
        key: value
        for key, value in legacy.state_dict().items()
        if key != "_extra_state"
    }
    legacy.load_state_dict(old_state_dict)
    excluded_graph = copy.deepcopy(graph)
    excluded_graph["projections"][-1]["allow_autapses"] = False
    excluded = GraphExecutor(plan_graph(excluded_graph))
    with pytest.raises(RuntimeError, match="checkpoint has no autapse policy"):
        excluded.load_state_dict(old_state_dict)


def test_training_keeps_only_structural_zeros_and_handles_optimizer_momentum():
    net, _, projection = recurrent_network(
        initializer=lang.LowerClampedNormal(
            1, 0, initial_zero_fraction=0.8, zeroing="exact_k"
        )
    )
    model = executor(net, projection)
    weight = model.parameter_map()[projection.weight.id]
    initial = weight.detach().clone()
    optimizer = torch.optim.SGD([weight], lr=0.1, momentum=0.9)
    (-weight.sum()).backward()
    assert torch.all(weight.grad.diagonal() == 0)
    optimizer.step()
    # Simulate stale optimizer state capable of moving a forbidden entry.
    optimizer.state[weight]["momentum_buffer"].fill_diagonal_(1)
    optimizer.zero_grad()
    (-weight.sum()).backward()
    optimizer.step()
    model.enforce_constraints()
    assert torch.all(weight.diagonal() == 0)
    off_diagonal = ~torch.eye(4, dtype=torch.bool)
    assert torch.all(weight[(initial == 0) & off_diagonal] > 0)


def test_forward_masks_manual_diagonal_edits_and_continuation(tmp_path):
    net, _, projection = recurrent_network()
    model = executor(net, projection)
    values = {"events": torch.ones(10, 1, 1)}
    baseline = model(values)
    weight = model.parameter_map()[projection.weight.id]
    with torch.no_grad():
        weight.fill_diagonal_(100)
    result = model(values)
    assert torch.equal(result.outputs["feedback"], baseline.outputs["feedback"])
    assert result.outputs["feedback"].sum() > 0
    result.outputs["feedback"].sum().backward()
    assert torch.all(weight.grad.diagonal() == 0)
    assert torch.count_nonzero(weight.grad) > 0
    first = model({"events": values["events"][:4]})
    saved = save_runtime_state(tmp_path / "runtime", first.runtime_state)
    continued = load_runtime_state(saved)
    second = model({"events": values["events"][4:]}, runtime_state=continued)
    assert torch.equal(
        torch.cat((first.outputs["feedback"], second.outputs["feedback"])),
        result.outputs["feedback"],
    )
    assert result.metrics["autapses"][projection.id]["effective"] == "excluded"
    permitted_net, _, permitted = recurrent_network(allowed=True)
    with pytest.raises(ValueError, match="runtime state is incompatible"):
        executor(permitted_net, permitted)(
            {"events": values["events"][4:]}, runtime_state=continued
        )


@pytest.mark.parametrize("assign", [False, True])
def test_bundle_and_native_checkpoints_preserve_and_validate_policy(tmp_path, assign):
    net, _, projection = recurrent_network()
    bundle = lang.compile(net, target=None)
    loaded = lang.load_bundle(bundle.write(tmp_path / "network.bundle"))
    assert loaded.graph == bundle.graph
    assert loaded.graph["projections"][-1]["allow_autapses"] is False
    model = executor(net, projection)
    with torch.no_grad():
        model.parameter_map()[projection.weight.id].fill_diagonal_(100)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    state_dict = torch.load(stream, weights_only=True)
    assert torch.all(state_dict["weights.recurrence__weight"].diagonal() == 0)
    state_dict["weights.recurrence__weight"].fill_diagonal_(100)
    restored = GraphExecutor(
        plan_graph(loaded.graph), trainable_parameters=(projection.weight.id,)
    )
    restored.load_state_dict(state_dict, assign=assign)
    assert torch.all(restored.parameter_map()[projection.weight.id].diagonal() == 0)
    restored.parameter_map()[projection.weight.id].sum().backward()
    assert torch.all(
        restored.parameter_map()[projection.weight.id].grad.diagonal() == 0
    )
    permitted_net, _, permitted = recurrent_network(allowed=True)
    with pytest.raises(ValueError, match="checkpoint autapse policy"):
        executor(permitted_net, permitted).load_state_dict(state_dict)
    checkpoint = tmp_path / "model.pt"
    torch.save(restored.state_dict(), checkpoint)
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            graph=bundle.graph,
            checkpoint=checkpoint,
            input_bindings=(DenseArrayBinding("events", torch.zeros(10, 1, 1)),),
        )
    )
    assert torch.all(result.parameters[projection.weight.id].diagonal() == 0)


@pytest.mark.parametrize("shared_use", ["allowed", "cross_population", "operation"])
def test_incompatible_shared_structural_masks_fail_compilation_and_planning(shared_use):
    net, cells, projection = recurrent_network()
    if shared_use == "operation":
        net.operation(
            "linear",
            cells.spikes,
            name="linear",
            shape=("time", "batch", 4),
            unit="uS",
            parameters=(projection.weight,),
            size=4,
        )
    else:
        target = (
            cells
            if shared_use == "allowed"
            else net.population("other", size=4, neuron=lang.COBA_LIF())
        )
        net.connect(
            cells.spikes,
            target.excitatory,
            name="shared",
            synapse=lang.AMPA(),
            weight=projection.weight,
            connection="recurrent",
            allow_autapses=shared_use == "allowed",
        )
    with pytest.raises(ValueError, match="incompatible autapse masks"):
        lang.compile(net, target=None)
    with pytest.raises(ValueError, match="incompatible autapse masks"):
        plan_graph(graph_dict(net))


def test_compatible_shared_weights_initialize_once():
    net, cells, projection = recurrent_network()
    net.connect(
        cells.spikes,
        cells.inhibitory,
        name="shared",
        synapse=lang.GABA(),
        weight=projection.weight,
        connection="recurrent",
    )
    model = executor(net, projection)
    assert torch.all(model.parameter_map()[projection.weight.id].diagonal() == 0)
    assert set(model.initialization_metadata[projection.weight.id]["autapses"]) == {
        "recurrence",
        "shared",
    }


def test_builtin_training_and_checkpoint_resume_preserve_exclusion(tmp_path):
    net, cells, projection = recurrent_network()
    prediction = lang.readouts.MeanVoltage(
        source=cells.spikes, classes=2, name="scores", weight=lang.Uniform(0.1, 1.0)
    )
    net.output("class_scores", prediction)
    recipe = lang.TrainSpec(
        objectives=[training.CrossEntropy(prediction=prediction, target="label")],
        parameter_groups=[
            training.ParameterGroup(
                [row["id"] for row in net.parameters], name="all", lr=0.01
            )
        ],
        optimizer=training.AdamW(weight_decay=0),
        presentation_duration=10 * lang.ms,
    )
    bundle = lang.compile(net, training=recipe, target=None)
    common = dict(
        kind="train",
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(DenseArrayBinding("events", torch.ones(100, 1, 1)),),
        targets={"label": torch.tensor([0])},
        seed=7,
    )
    full = train(ExecutionSpec(**common, updates=4))
    root = tmp_path / "checkpoint"
    train(ExecutionSpec(**common, updates=2, save_final_checkpoint=root))
    checkpoint = load_training_checkpoint(root)
    assert (
        checkpoint.initialization[projection.weight.id]["autapses"][projection.id][
            "effective"
        ]
        == "excluded"
    )
    assert torch.all(checkpoint.parameters[projection.weight.id].diagonal() == 0)
    resumed = train(ExecutionSpec(**common, updates=2, checkpoint=root))
    for name in full.parameters:
        assert torch.equal(full.parameters[name], resumed.parameters[name])
    assert torch.all(full.parameters[projection.weight.id].diagonal() == 0)
    assert torch.all(full.gradients[projection.weight.id].diagonal() == 0)
    assert torch.count_nonzero(full.gradients[projection.weight.id]) > 0
    assert full.metrics["updates"][-2:] == resumed.metrics["updates"]
    altered = copy.deepcopy(bundle.graph)
    next(row for row in altered["projections"] if row["id"] == projection.id)[
        "allow_autapses"
    ] = True
    with pytest.raises(
        ValueError, match="checkpoint.*(graph digest|initializer metadata)"
    ):
        train(ExecutionSpec(**{**common, "graph": altered}, updates=1, checkpoint=root))


def fill_weights(value, config):
    return torch.ones_like(value)


def test_structural_exclusion_takes_precedence_over_custom_constraints(monkeypatch):
    monkeypatch.setattr(ext, "_REGISTRY", {})
    ext.register_constraint("test.fill/v1", function=fill_weights)
    net, _, projection = recurrent_network()
    net.parameters[-1]["constraint"] = lang.CustomConstraint("test.fill/v1").json()
    model = executor(net, projection)
    model.enforce_constraints()
    weight = model.parameter_map()[projection.weight.id]
    assert torch.all(weight.diagonal() == 0)
    assert torch.all(weight[~torch.eye(4, dtype=torch.bool)] == 1)
