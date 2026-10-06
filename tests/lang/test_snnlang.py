from __future__ import annotations

import json
import shutil

import pytest

from snnlab import lang as snn
from snnlab.lang import training
from snnlab.lang.compiler import canonical_json, graph_dict, validate_graph
from tests._circuits import author_ping


def small_network():
    net = snn.Network("small")
    x = net.input("x", shape=("time", "batch", 8), signal_type="spikes", unit="spike")
    cell = author_ping(net, name="cell", n_e=12, n_i=3, source=x)
    return net, cell


def test_graph_shaped_authoring_and_component_expansion():
    net, cell = small_network()
    assert {p["id"] for p in net.populations} == {"cell_E", "cell_I"}
    assert {p["connection"] for p in net.projections} == {"feedforward", "recurrent"}
    assert cell.E.spikes.shape == ("time", "batch", 12)
    assert net.groups["cell"].members


def test_disabled_projection_remains_structural_and_explicit():
    net, cell = small_network()
    loop = net.connect(
        cell.E.spikes,
        cell.I.excitatory,
        name="disabled_extra_loop",
        synapse=snn.AMPA(tau=5 * snn.ms),
        weight=snn.Normal(0.2, 0.01),
        connection="recurrent",
        enabled=False,
    )
    graph = snn.compile(net, target=None).graph
    row = next(p for p in graph["projections"] if p["id"] == loop.id)
    assert row["enabled"] is False
    parameter = next(p for p in graph["parameters"] if p["id"] == loop.weight.id)
    assert parameter["shape"] == [3, 12]


def test_initializer_vocabulary_serializes_without_opaque_options():
    assert snn.SignedNormal(-0.1, 0.2).json() == {
        "kind": "signed_normal",
        "mean": -0.1,
        "std": 0.2,
    }
    assert snn.Uniform(0.1, 0.3).json() == {"kind": "uniform", "low": 0.1, "high": 0.3}
    assert snn.Zeros().json() == {"kind": "zeros"}
    assert snn.LowerClampedNormal(
        0.2, 0.03, initial_zero_fraction=0.75, zeroing="exact_k"
    ).json() == {
        "kind": "lower_clamped_normal",
        "mean": 0.2,
        "std": 0.03,
        "initial_zero_fraction": 0.75,
        "zeroing": "exact_k",
    }


def test_names_are_unique():
    net = snn.Network("bad")
    net.input("x", shape=(1,), signal_type="continuous")
    with pytest.raises(ValueError, match="duplicate"):
        net.input("x", shape=(1,), signal_type="continuous")


def test_feedback_requires_delay():
    net, cell = small_network()
    net.connect(
        cell.E.spikes,
        cell.I.excitatory,
        name="feedback",
        synapse=snn.AMPA(),
        connection="feedback",
    )
    with pytest.raises(ValueError, match="feedback"):
        snn.compile(net)


@pytest.mark.parametrize("kind", ["mean", "final", "count", "rate", "cumulative"])
def test_readouts_expand_to_serialisable_ops(kind):
    net, cell = small_network()
    if kind == "mean":
        value = snn.readouts.MeanVoltage(source=cell.E.spikes, classes=4, name="r")
    elif kind == "final":
        value = snn.readouts.FinalVoltage(source=cell.E.spikes, classes=4, name="r")
    elif kind == "count":
        value = snn.readouts.SpikeCount(source=cell.E.spikes, classes=4, name="r")
    elif kind == "rate":
        mask = net.input("valid", shape=("time", "batch"), signal_type="mask")
        value = snn.readouts.SpikeRate(
            source=cell.E.spikes, classes=4, name="r", mask=mask
        )
    else:
        value = snn.readouts.CumulativePotential(
            source=cell.E.spikes, classes=4, name="r"
        )
    net.output("scores", value)
    bundle = snn.compile(net)
    assert bundle.graph["operations"]
    if kind in {"mean", "final", "count", "rate"}:
        assert value.shape == ("batch", 4)
    else:
        assert value.shape == ("time", "batch", 4)
    canonical_json(bundle.graph)


def test_spike_rate_rejects_ambiguous_duration():
    net, cell = small_network()
    with pytest.raises(ValueError, match="duration"):
        snn.readouts.SpikeRate(source=cell.E.spikes, classes=4, name="rate")


def test_validation_rejects_invalid_rate_mask_shape():
    net, cell = small_network()
    mask = net.input("valid", shape=("batch", 3), signal_type="mask")
    rate = snn.readouts.SpikeRate(
        source=cell.E.spikes, classes=4, name="rate", mask=mask
    )
    net.output("rates", rate)
    with pytest.raises(ValueError, match="valid-duration mask"):
        snn.compile(net)


def test_validation_rejects_non_positive_rate_duration():
    net, cell = small_network()
    rate = snn.readouts.SpikeRate(
        source=cell.E.spikes, classes=4, name="rate", duration=0
    )
    net.output("rates", rate)
    with pytest.raises(ValueError, match="positive seconds"):
        snn.compile(net)


def test_validation_rejects_linear_readout_parameter_shape_drift():
    net, cell = small_network()
    scores = snn.readouts.SpikeCount(source=cell.E.spikes, classes=4, name="scores")
    net.output("class_scores", scores)
    graph = graph_dict(net)
    parameter = next(
        p for p in graph["parameters"] if p["id"] == "scores_projection.weight"
    )
    parameter["shape"] = [3, 12]
    result = validate_graph(graph)
    assert any(d.code == "E209" for d in result.errors)
    with pytest.raises(ValueError, match="linear parameter shape"):
        snn.compile(net)


def test_projection_weights_use_microsiemens_and_reject_unit_drift():
    net, _ = small_network()
    graph = graph_dict(net)
    projection_parameter_ids = {
        parameter
        for projection in graph["projections"]
        for parameter in projection["parameters"]
    }
    rows = {row["id"]: row for row in graph["parameters"]}
    assert {rows[parameter]["unit"] for parameter in projection_parameter_ids} == {"uS"}

    rows[next(iter(projection_parameter_ids))]["unit"] = "nS"
    result = validate_graph(graph)
    assert any(d.code == "E113" for d in result.errors)


def test_validation_rejects_operation_unit_drift():
    net, cell = small_network()
    bad = net.operation(
        "reduce_sum",
        cell.E.spikes,
        name="bad_units",
        shape=("batch", 12),
        unit="mV",
    )
    net.output("bad_result", bad)
    with pytest.raises(ValueError, match="incompatible"):
        snn.compile(net)


def test_training_selects_and_freezes_parameters():
    net, cell = small_network()
    scores = snn.readouts.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    ids = [p["id"] for p in net.parameters]
    selected = "scores_projection.weight"
    spec = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            training.ParameterGroup([selected], name="selected", lr=1e-3),
            training.ParameterGroup(
                [pid for pid in ids if pid != selected],
                name="frozen",
                lr=0,
                frozen=True,
            ),
        ],
        optimizer=training.AdamW(),
        stop_gradients=[training.StopGradient.at(cell.E.spikes)],
        surrogate=training.FastSigmoid(slope=1.25),
    )
    bundle = snn.compile(net, training=spec)
    assert bundle.training is not None
    assert bundle.training["graph_digest"] == bundle.manifest["graph_digest"]
    assert bundle.training["parameter_groups"][1]["frozen"]
    assert bundle.training["resolved_parameters"] == {
        "trainable": [selected],
        "frozen": sorted(pid for pid in ids if pid != selected),
        "learning_rates": {selected: 1e-3},
    }
    assert bundle.training["surrogate"] == {"kind": "fast_sigmoid", "slope": 1.25}
    assert bundle.training["resolved_gradients"] == {
        "surrogate": {"kind": "fast_sigmoid", "slope": 1.25},
        "voltage_gradient_dampening": {"cell_E": 80.0, "cell_I": 80.0},
    }


def test_gradient_vocabulary_rejects_invalid_surrogate_and_dampening():
    net, cell = small_network()
    scores = snn.readouts.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    ids = [p["id"] for p in net.parameters]
    spec = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[training.ParameterGroup(ids, name="all", lr=1e-3)],
        optimizer=training.AdamW(),
        surrogate=training.FastSigmoid(slope=0),
    )
    with pytest.raises(ValueError, match="surrogate slope must be positive"):
        snn.compile(net, training=spec)
    net.populations[0]["neuron"]["voltage_grad_dampen"] = 0
    with pytest.raises(ValueError, match="voltage_grad_dampen must be a positive"):
        snn.compile(net)


def test_spike_budget_and_presentation_duration_compile_exact_physical_contract():
    net, cell = small_network()
    scores = snn.readouts.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    ids = [p["id"] for p in net.parameters]
    spec = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[training.ParameterGroup(ids, name="all", lr=1e-3)],
        optimizer=training.AdamW(),
        regularizers=[
            training.SpikeBudgetPenalty(
                signals=(cell.E.spikes, cell.I.spikes), ceiling_hz=10.0, strength=0.01
            )
        ],
        presentation_duration=200 * snn.ms,
    )
    recipe = snn.compile(net, training=spec).training
    assert recipe["presentation_duration"] == {"value": 200.0, "unit": "ms"}
    assert recipe["regularizers"] == [
        {
            "kind": "spike_budget",
            "signals": ["cell_E.spikes", "cell_I.spikes"],
            "strength": 0.01,
            "config": {
                "ceiling": {"value": 10.0, "unit": "Hz"},
                "penalty": "squared_hinge",
                "aggregation": "mean_presentations_then_layers_of_population_mean_rate",
            },
        }
    ]


def test_loss_and_duration_vocabulary_fail_closed():
    net, cell = small_network()
    scores = snn.readouts.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    ids = [p["id"] for p in net.parameters]
    spec = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[training.ParameterGroup(ids, name="all", lr=1e-3)],
        optimizer=training.AdamW(),
        regularizers=[
            training.SpikeBudgetPenalty(signals=(), ceiling_hz=-1, strength=-1)
        ],
        presentation_duration=0.15 * snn.ms,
    )
    with pytest.raises(ValueError) as exc:
        snn.compile(net, training=spec)
    message = str(exc.value)
    assert "regularizer requires at least one signal" in message
    assert "ceiling must be non-negative Hz" in message
    assert "strength must be non-negative and finite" in message
    assert "integer number of graph timesteps" in message


def test_objective_rejects_stop_gradient_route_with_no_downstream_trainable_parameter():
    net, cell = small_network()
    scores = snn.readouts.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    ids = [p["id"] for p in net.parameters]
    input_weight = "cell_input.weight"
    spec = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            training.ParameterGroup([input_weight], name="input", lr=1e-3),
            training.ParameterGroup(
                [pid for pid in ids if pid != input_weight],
                name="frozen",
                lr=0,
                frozen=True,
            ),
        ],
        optimizer=training.AdamW(),
        stop_gradients=[training.StopGradient.at(cell.E.spikes)],
    )
    with pytest.raises(ValueError, match="objective has no differentiable route"):
        snn.compile(net, training=spec)


def test_trainable_zero_initialized_recurrent_variant_has_named_gradient_route():
    net, cell = small_network()
    recurrent = next(
        projection
        for projection in net.projections
        if projection["id"] == "cell_E_to_I"
    )
    recurrent_parameter = recurrent["parameters"][0]
    parameter = next(p for p in net.parameters if p["id"] == recurrent_parameter)
    parameter["initializer"] = snn.Zeros().json()
    scores = snn.readouts.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    ids = [p["id"] for p in net.parameters]
    spec = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            training.ParameterGroup([recurrent_parameter], name="recurrent", lr=4e-4),
            training.ParameterGroup(
                [pid for pid in ids if pid != recurrent_parameter],
                name="frozen",
                lr=0,
                frozen=True,
            ),
        ],
        optimizer=training.AdamW(),
        surrogate=training.FastSigmoid(slope=1),
    )
    recipe = snn.compile(net, training=spec).training
    assert recipe["resolved_parameters"]["trainable"] == [recurrent_parameter]
    assert parameter["initializer"] == {"kind": "zeros"}


def test_disabled_trainable_projection_is_not_a_differentiable_route():
    net, cell = small_network()
    disabled = net.connect(
        cell.E.spikes,
        cell.I.excitatory,
        name="disabled_trainable",
        synapse=snn.AMPA(tau=2 * snn.ms),
        weight=snn.Zeros(),
        connection="recurrent",
        enabled=False,
    )
    scores = snn.readouts.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    ids = [p["id"] for p in net.parameters]
    pid = disabled.weight.id
    spec = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            training.ParameterGroup([pid], name="disabled", lr=1e-3),
            training.ParameterGroup(
                [item for item in ids if item != pid], name="frozen", lr=0, frozen=True
            ),
        ],
        optimizer=training.AdamW(),
    )
    with pytest.raises(ValueError, match="objective has no differentiable route"):
        snn.compile(net, training=spec)


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        (
            lambda ids: [training.ParameterGroup(ids[:1], name="partial", lr=1e-3)],
            "omitted",
        ),
        (
            lambda ids: [
                training.ParameterGroup(ids, name="first", lr=1e-3),
                training.ParameterGroup(ids[:1], name="second", lr=1e-3),
            ],
            "already selected",
        ),
        (
            lambda ids: [
                training.ParameterGroup(ids, name="frozen", lr=1e-3, frozen=True)
            ],
            "frozen parameter group learning rate must be zero",
        ),
        (
            lambda ids: [training.ParameterGroup(ids, name="trainable", lr=0)],
            "trainable parameter group learning rate must be positive",
        ),
    ],
)
def test_training_parameter_groups_fail_closed(groups, message):
    net, cell = small_network()
    scores = snn.readouts.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    ids = [p["id"] for p in net.parameters]
    spec = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=groups(ids),
        optimizer=training.AdamW(),
    )
    with pytest.raises(ValueError, match=message):
        snn.compile(net, training=spec)


def test_training_rejects_unknown_parameter():
    net, cell = small_network()
    scores = snn.readouts.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    spec = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[training.ParameterGroup(["ghost"], name="bad", lr=1e-3)],
        optimizer=training.AdamW(),
    )
    with pytest.raises(ValueError, match="unknown training parameter"):
        snn.compile(net, training=spec)


def test_compile_is_deterministic():
    first, cell = small_network()
    first.output("spikes", cell.E.spikes)
    second, cell2 = small_network()
    second.output("spikes", cell2.E.spikes)
    assert canonical_json(snn.compile(first).graph) == canonical_json(
        snn.compile(second).graph
    )
    assert snn.compile(first).manifest == snn.compile(second).manifest


def test_bundle_round_trip_and_tamper_detection(tmp_path):
    net, cell = small_network()
    net.output("spikes", cell.E.spikes)
    root = snn.compile(net).write(tmp_path / "bundle")
    loaded = snn.load_bundle(root)
    assert loaded.graph == snn.compile(net).graph
    graph = json.loads((root / "graph.json").read_text())
    graph["name"] = "tampered"
    (root / "graph.json").write_text(json.dumps(graph))
    with pytest.raises(ValueError, match="digest"):
        snn.load_bundle(root)


def test_assets_keep_logical_names_and_copy_physical_bytes(tmp_path):
    net, cell = small_network()
    net.output("spikes", cell.E.spikes)
    net.asset("connectivity", media_type="application/octet-stream")
    source = tmp_path / "matrix.bin"
    source.write_bytes(b"weights")
    root = snn.compile(net, assets={"connectivity": source}).write(tmp_path / "bundle")
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["assets"][0]["id"] == "connectivity"
    assert (root / manifest["assets"][0]["path"]).read_bytes() == b"weights"
    assert str(tmp_path) not in (root / "graph.json").read_text()
    copied = snn.load_bundle(root).write(tmp_path / "copied")
    assert (copied / manifest["assets"][0]["path"]).read_bytes() == b"weights"


def test_disconnected_population_is_warning_not_backend_failure():
    net = snn.Network("warning")
    net.population("lonely", size=2, neuron=snn.LIF())
    result = validate_graph(graph_dict(net))
    assert not result.errors
    assert any(d.code == "W101" for d in result.warnings)


def test_backend_capability_is_separate_from_validity():
    net, cell = small_network()
    custom = net.operation(
        "future_op",
        cell.E.spikes,
        name="future",
        shape=cell.E.spikes.shape,
        unit="spike",
    )
    net.output("future_result", custom)
    bundle = snn.compile(net, target="tools/snnsim")
    assert any(d.code == "C101" for d in bundle.diagnostics)


def test_manifest_archives_versioned_element_capabilities():
    net, cell = small_network()
    net.expose(cell.E.spikes, name="e_spikes")
    bundle = snn.compile(net)
    required = bundle.manifest["required_capabilities"]
    assert required["schema"] == "snnlang.capabilities/v1"
    by_element = {row["element"]: row["features"] for row in required["elements"]}
    assert "neuron:coba_lif" in by_element["cell_E"]
    assert "connection:recurrent" in by_element["cell_I_to_E"]
    assert "recording:spikes" in by_element["e_spikes"]


def test_visualisation_is_deterministic_and_has_stable_ids(tmp_path):
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' is required for snnlang visualisation")
    net, cell = small_network()
    net.output("spikes", cell.E.spikes)
    bundle = snn.compile(net)
    a = bundle.visualise(tmp_path / "a.svg", view="circuit")
    b = bundle.visualise(tmp_path / "b.svg", view="circuit")
    assert a.read_bytes() == b.read_bytes()
    assert "n_cell" in a.read_text()
    assert 'class="node node component"' in a.read_text()
    png_a = bundle.visualise(tmp_path / "a.png", view="circuit", scale=2)
    png_b = bundle.visualise(tmp_path / "b.png", view="circuit", scale=2)
    assert png_a.read_bytes() == png_b.read_bytes()


def test_bundle_lowers_to_snnviz_diagram_before_rendering():
    net, cell = small_network()
    net.output("spikes", cell.E.spikes)

    visual = snn.diagram(snn.compile(net), view="circuit")

    assert visual.metadata == {"view": "circuit", "source": "snnlang"}
    assert {node.id for node in visual.nodes} >= {"x", "cell", "out:spikes"}


def test_expanded_diagram_groups_component_populations_in_target_columns():
    net, cell = small_network()
    net.output("spikes", cell.E.spikes)

    visual = snn.diagram(snn.compile(net), view="expanded")
    group = next(group for group in visual.groups if group.id == "cell")
    titles = {node.id: node.title for node in visual.nodes}

    assert group.members == ("cell_E", "cell_I")
    assert group.same_rank
    assert (titles["cell_E"], titles["cell_I"]) == ("E", "I")


def test_all_visual_views_render(tmp_path):
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' is required for snnlang visualisation")
    net, cell = small_network()
    scores = snn.readouts.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    spec = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            training.ParameterGroup(
                [p["id"] for p in net.parameters], name="all", lr=1e-3
            )
        ],
        optimizer=training.AdamW(),
    )
    bundle = snn.compile(net, training=spec)
    for view in ("circuit", "training", "expanded"):
        assert (
            bundle.visualise(tmp_path / f"{view}.svg", view=view).stat().st_size > 500
        )


def test_expanded_visualisation_retains_recurrent_self_projection(tmp_path):
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' is required for snnlang visualisation")
    net = snn.Network("self_loop")
    e = net.population("E", size=8, neuron=snn.COBA_LIF())
    net.connect(
        e.spikes,
        e.excitatory,
        name="E_to_E",
        synapse=snn.AMPA(),
        weight=snn.Constant(0.1),
        connection="recurrent",
    )

    output = snn.compile(net).visualise(tmp_path / "expanded.svg", view="expanded")

    assert "n_E_to_E" in output.read_text()


def test_circuit_visualisation_can_expand_selected_groups(tmp_path):
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' is required for snnlang visualisation")
    net, cell = small_network()
    net.output("spikes", cell.E.spikes)
    bundle = snn.compile(net)

    output = bundle.visualise(
        tmp_path / "selected.svg",
        view="circuit",
        expand_groups=("cell",),
    )
    svg = output.read_text()

    assert "cluster_n_cell" in svg
    assert "n_cell_E" in svg
    assert "n_cell_I" in svg
    assert "n_cell_E_to_I" in svg


def test_circuit_visualisation_rejects_unknown_expanded_group(tmp_path):
    net, cell = small_network()
    net.output("spikes", cell.E.spikes)
    bundle = snn.compile(net)

    with pytest.raises(ValueError, match="cannot expand unknown groups: missing"):
        bundle.visualise(
            tmp_path / "selected.svg",
            view="circuit",
            expand_groups=("missing",),
        )


def test_spike_count_parameters_can_define_its_training_group():
    net = snn.Network("readout_parameters")
    inputs = net.input(
        "inputs", shape=("time", "batch", 4), signal_type="spikes", unit="spike"
    )
    scores = snn.readouts.SpikeCount(source=inputs, classes=2, name="scores")
    net.output("class_scores", scores)
    recipe = snn.TrainSpec(
        objectives=(training.CrossEntropy(prediction=scores, target="class"),),
        parameter_groups=(
            training.ParameterGroup(scores.parameters, name="readout", lr=0.01),
        ),
        optimizer=training.AdamW(),
    )
    bundle = snn.compile(net, training=recipe, target="tools/snnsim")
    assert bundle.training["resolved_parameters"]["trainable"] == [
        "scores_projection.weight"
    ]
