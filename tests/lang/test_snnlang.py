from __future__ import annotations

import copy
import json
import math
import shutil

import pytest

from snnlab import lang as snn
from snnlab.lang import validate_graph
from snnlab.lang.compilation.serialization import canonical_json, graph_dict
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
        value = snn.MeanVoltage(source=cell.E.spikes, classes=4, name="r")
    elif kind == "final":
        value = snn.FinalVoltage(source=cell.E.spikes, classes=4, name="r")
    elif kind == "count":
        value = snn.SpikeCount(source=cell.E.spikes, classes=4, name="r")
    elif kind == "rate":
        mask = net.input("valid", shape=("time", "batch"), signal_type="mask")
        value = snn.SpikeRate(source=cell.E.spikes, classes=4, name="r", mask=mask)
    else:
        value = snn.CumulativePotential(source=cell.E.spikes, classes=4, name="r")
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
        snn.SpikeRate(source=cell.E.spikes, classes=4, name="rate")


def test_validation_rejects_invalid_rate_mask_shape():
    net, cell = small_network()
    mask = net.input("valid", shape=("batch", 3), signal_type="mask")
    rate = snn.SpikeRate(source=cell.E.spikes, classes=4, name="rate", mask=mask)
    net.output("rates", rate)
    with pytest.raises(ValueError, match="valid-duration mask"):
        snn.compile(net)


def test_validation_rejects_non_positive_rate_duration():
    net, cell = small_network()
    rate = snn.SpikeRate(source=cell.E.spikes, classes=4, name="rate", duration=0)
    net.output("rates", rate)
    with pytest.raises(ValueError, match="positive seconds"):
        snn.compile(net)


def test_validation_rejects_linear_readout_parameter_shape_drift():
    net, cell = small_network()
    scores = snn.SpikeCount(source=cell.E.spikes, classes=4, name="scores")
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
    scores = snn.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    ids = [p["id"] for p in net.parameters]
    selected = "scores_projection.weight"
    spec = snn.TrainSpec(
        objectives=[snn.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            snn.ParameterGroup([selected], name="selected", lr=1e-3),
            snn.ParameterGroup(
                [pid for pid in ids if pid != selected],
                name="frozen",
                lr=0,
                frozen=True,
            ),
        ],
        optimizer=snn.AdamW(),
        stop_gradients=[snn.StopGradient.at(cell.E.spikes)],
        surrogate=snn.FastSigmoid(slope=1.25),
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
    scores = snn.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    ids = [p["id"] for p in net.parameters]
    spec = snn.TrainSpec(
        objectives=[snn.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[snn.ParameterGroup(ids, name="all", lr=1e-3)],
        optimizer=snn.AdamW(),
        surrogate=snn.FastSigmoid(slope=0),
    )
    with pytest.raises(ValueError, match="surrogate slope must be positive"):
        snn.compile(net, training=spec)
    net.populations[0]["neuron"]["voltage_grad_dampen"] = 0
    with pytest.raises(ValueError, match="voltage_grad_dampen must be a positive"):
        snn.compile(net)


def test_spike_budget_and_presentation_duration_compile_exact_physical_contract():
    net, cell = small_network()
    scores = snn.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    ids = [p["id"] for p in net.parameters]
    spec = snn.TrainSpec(
        objectives=[snn.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[snn.ParameterGroup(ids, name="all", lr=1e-3)],
        optimizer=snn.AdamW(),
        regularizers=[
            snn.SpikeBudgetPenalty(
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
    scores = snn.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    ids = [p["id"] for p in net.parameters]
    spec = snn.TrainSpec(
        objectives=[snn.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[snn.ParameterGroup(ids, name="all", lr=1e-3)],
        optimizer=snn.AdamW(),
        regularizers=[snn.SpikeBudgetPenalty(signals=(), ceiling_hz=-1, strength=-1)],
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
    scores = snn.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    ids = [p["id"] for p in net.parameters]
    input_weight = "cell_input.weight"
    spec = snn.TrainSpec(
        objectives=[snn.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            snn.ParameterGroup([input_weight], name="input", lr=1e-3),
            snn.ParameterGroup(
                [pid for pid in ids if pid != input_weight],
                name="frozen",
                lr=0,
                frozen=True,
            ),
        ],
        optimizer=snn.AdamW(),
        stop_gradients=[snn.StopGradient.at(cell.E.spikes)],
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
    scores = snn.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    ids = [p["id"] for p in net.parameters]
    spec = snn.TrainSpec(
        objectives=[snn.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            snn.ParameterGroup([recurrent_parameter], name="recurrent", lr=4e-4),
            snn.ParameterGroup(
                [pid for pid in ids if pid != recurrent_parameter],
                name="frozen",
                lr=0,
                frozen=True,
            ),
        ],
        optimizer=snn.AdamW(),
        surrogate=snn.FastSigmoid(slope=1),
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
    scores = snn.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    ids = [p["id"] for p in net.parameters]
    pid = disabled.weight.id
    spec = snn.TrainSpec(
        objectives=[snn.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            snn.ParameterGroup([pid], name="disabled", lr=1e-3),
            snn.ParameterGroup(
                [item for item in ids if item != pid], name="frozen", lr=0, frozen=True
            ),
        ],
        optimizer=snn.AdamW(),
    )
    with pytest.raises(ValueError, match="objective has no differentiable route"):
        snn.compile(net, training=spec)


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        (
            lambda ids: [snn.ParameterGroup(ids[:1], name="partial", lr=1e-3)],
            "omitted",
        ),
        (
            lambda ids: [
                snn.ParameterGroup(ids, name="first", lr=1e-3),
                snn.ParameterGroup(ids[:1], name="second", lr=1e-3),
            ],
            "already selected",
        ),
        (
            lambda ids: [snn.ParameterGroup(ids, name="frozen", lr=1e-3, frozen=True)],
            "frozen parameter group learning rate must be zero",
        ),
        (
            lambda ids: [snn.ParameterGroup(ids, name="trainable", lr=0)],
            "trainable parameter group learning rate must be positive",
        ),
    ],
)
def test_training_parameter_groups_fail_closed(groups, message):
    net, cell = small_network()
    scores = snn.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    ids = [p["id"] for p in net.parameters]
    spec = snn.TrainSpec(
        objectives=[snn.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=groups(ids),
        optimizer=snn.AdamW(),
    )
    with pytest.raises(ValueError, match=message):
        snn.compile(net, training=spec)


def test_training_rejects_unknown_parameter():
    net, cell = small_network()
    scores = snn.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    spec = snn.TrainSpec(
        objectives=[snn.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[snn.ParameterGroup(["ghost"], name="bad", lr=1e-3)],
        optimizer=snn.AdamW(),
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
    assert 'class="node node population spiking"' in a.read_text()
    assert "cluster_n_cell" in a.read_text()
    png_a = bundle.visualise(tmp_path / "a.png", view="circuit", scale=2)
    png_b = bundle.visualise(tmp_path / "b.png", view="circuit", scale=2)
    assert png_a.read_bytes() == png_b.read_bytes()


def test_bundle_lowers_to_snnviz_diagram_before_rendering():
    net, cell = small_network()
    net.output("spikes", cell.E.spikes)

    visual = snn.diagram(snn.compile(net), view="circuit")

    assert visual.metadata == {"view": "circuit", "source": "snnlang"}
    assert {node.id for node in visual.nodes} >= {"x", "cell_E", "cell_I", "out:spikes"}
    assert all(node.kind != "component" for node in visual.nodes)


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
    scores = snn.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    spec = snn.TrainSpec(
        objectives=[snn.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            snn.ParameterGroup([p["id"] for p in net.parameters], name="all", lr=1e-3)
        ],
        optimizer=snn.AdamW(),
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


@pytest.mark.parametrize("view", ["circuit", "training", "expanded"])
def test_every_visual_view_shows_group_contents(tmp_path, view):
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' is required for snnlang visualisation")
    net, cell = small_network()
    net.output("spikes", cell.E.spikes)
    bundle = snn.compile(net)

    output = bundle.visualise(
        tmp_path / "selected.svg",
        view=view,
    )
    svg = output.read_text()

    assert "cluster_n_cell" in svg
    assert "n_cell_E" in svg
    assert "n_cell_I" in svg
    assert "n_cell_E_to_I" in svg


def test_visualisation_no_longer_accepts_selective_expansion(tmp_path):
    net, cell = small_network()
    net.output("spikes", cell.E.spikes)
    bundle = snn.compile(net)

    with pytest.raises(TypeError, match="expand_groups"):
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
    scores = snn.SpikeCount(source=inputs, classes=2, name="scores")
    net.output("class_scores", scores)
    recipe = snn.TrainSpec(
        objectives=(snn.CrossEntropy(prediction=scores, target="class"),),
        parameter_groups=(
            snn.ParameterGroup(scores.parameters, name="readout", lr=0.01),
        ),
        optimizer=snn.AdamW(),
    )
    bundle = snn.compile(net, training=recipe, target="tools/snnsim")
    assert bundle.training["resolved_parameters"]["trainable"] == [
        "scores_projection.weight"
    ]


# stdp_simple acceptance tests (T22, T25, T34). The rest are in
# tests/sim/test_plasticity_stdp_simple.py.


def _stdp_simple(**overrides):
    values = {
        "a_plus": 0.01,
        "a_minus": 0.0105,
        "tau_plus": 16.8 * snn.ms,
        "tau_minus": 33.7 * snn.ms,
        "w_max": 1.0,
        **overrides,
    }
    return snn.STDPSimple(**values)


def _stdp_network(
    plasticity=None,
    *,
    port="excitatory",
    source_signal="spikes",
    spiking_target=True,
    **connect,
):
    net = snn.Network("stdp_simple")
    if source_signal == "spikes":
        source = net.input(
            "pre", shape=("time", "batch", 3), signal_type="spikes", unit="spike"
        )
    else:
        source = net.input(
            "pre", shape=("time", "batch", 3), signal_type="current", unit="nA"
        )
    if spiking_target:
        post = net.population("post", size=2, neuron=snn.CUBA_LIF())
        synapse = snn.ExponentialCurrent(tau=2 * snn.ms)
        signal = post.spikes
    else:
        post = net.population(
            "post", size=2, neuron=snn.LeakyIntegrator(tau=2 * snn.ms), spiking=False
        )
        synapse = snn.AMPA(tau=2 * snn.ms)
        signal = post.voltage
    if plasticity is not None:
        connect["plasticity"] = plasticity
    net.connect(
        source,
        getattr(post, port),
        name="syn",
        synapse=synapse,
        weight=snn.Constant(0.5),
        initialization_scaling="direct",
        **connect,
    )
    net.output("post_signal", signal)
    return net


_STDP_SIMPLE_RULE = object()  # default: the networks of T25 carry stdp_simple


def _stdp_shared_weight_network(plasticity=_STDP_SIMPLE_RULE):
    if plasticity is _STDP_SIMPLE_RULE:
        plasticity = _stdp_simple()
    net = snn.Network("stdp_shared")
    pre = net.input(
        "pre", shape=("time", "batch", 3), signal_type="spikes", unit="spike"
    )
    post = net.population("post", size=2, neuron=snn.CUBA_LIF())
    shared = net.parameter(
        "shared", shape=(2, 3), initializer=snn.Constant(0.5), unit="nA"
    )
    rule = {} if plasticity is None else {"plasticity": plasticity}
    for name, extra in (("a", rule), ("b", {})):
        net.connect(
            pre,
            post.excitatory,
            name=name,
            synapse=snn.ExponentialCurrent(tau=2 * snn.ms),
            weight=shared,
            **extra,
        )
    net.output("post_spikes", post.spikes)
    return net


def _stdp_conv2d_network(plasticity=_STDP_SIMPLE_RULE):
    if plasticity is _STDP_SIMPLE_RULE:
        plasticity = _stdp_simple()
    rule = {} if plasticity is None else {"plasticity": plasticity}
    net = snn.Network("stdp_conv")
    source = net.input(
        "pre",
        shape=("time", "batch", 40),
        spatial_shape=(2, 4, 5),
        signal_type="spikes",
        unit="spike",
    )
    cells = net.population("post", spatial_shape=(3, 2, 3), neuron=snn.CUBA_LIF())
    net.connect(
        source,
        cells.excitatory,
        name="syn",
        connectivity=snn.Conv2d(kernel_size=(2, 3), stride=(2, 1), padding=(0, 0)),
        synapse=snn.ExponentialCurrent(tau=2 * snn.ms),
        weight=snn.Constant(0.1),
        initialization_scaling="direct",
        **rule,
    )
    net.output("post_spikes", cells.spikes)
    return net


def test_t22_stdp_simple_spec_states_its_rule_and_is_opt_in():
    spec = _stdp_simple().json()
    assert spec["kind"] == "stdp_simple"
    assert (spec["update"], spec["bounds"]) == ("additive", "hard")
    assert (spec["pairing"], spec["w_min"], spec["w_max"]) == ("all_to_all", 0.0, 1.0)
    plastic = snn.compile(_stdp_network(_stdp_simple()), target=None).graph
    static = snn.compile(_stdp_network(), target=None).graph
    row = next(r for r in plastic["projections"] if r["id"] == "syn")
    assert row["plasticity"] == spec
    static_row = next(r for r in static["projections"] if r["id"] == "syn")
    assert "plasticity" not in static_row


_STDP_REJECTED = {
    "inhibitory_receptor": lambda: _stdp_network(_stdp_simple(), port="inhibitory"),
    "modulatory_receptor": lambda: _stdp_network(_stdp_simple(), port="modulatory"),
    "weight_modulation": lambda: _stdp_network(
        _stdp_simple(), modulation=snn.ConstantGain(0.5)
    ),
    "shared_weight": _stdp_shared_weight_network,
    "conv2d_kernel": _stdp_conv2d_network,
    "non_spiking_source": lambda: _stdp_network(
        _stdp_simple(), source_signal="current"
    ),
    "non_spiking_target": lambda: _stdp_network(_stdp_simple(), spiking_target=False),
}


@pytest.mark.parametrize("case", sorted(_STDP_REJECTED))
def test_t25_stdp_simple_rejects_unsupported_projections(case):
    # stdp_simple is a per-projection rule for projections onto an excitatory
    # receptor, from a spiking source to a spiking target, with its own weights.
    with pytest.raises(ValueError, match="stdp_simple|plastic"):
        snn.compile(_STDP_REJECTED[case](), target=None)


def _stdp_graph_with(mutation):
    graph = copy.deepcopy(snn.compile(_stdp_network(_stdp_simple()), target=None).graph)
    row = next(r for r in graph["projections"] if r["id"] == "syn")
    mutation(graph, row)
    return graph


def _move_to_inhibitory(graph, row):
    # A hand-edited bundle: the same projection, now onto the inhibitory receptor.
    row["target"] = "post.inhibitory"
    row["polarity"] = "inhibitory"


_STDP_CORRUPTED = {
    "inhibitory_receptor": _move_to_inhibitory,
    "non_finite_a_plus": lambda graph, row: row["plasticity"].update(a_plus=math.inf),
    "nan_w_max": lambda graph, row: row["plasticity"].update(w_max=math.nan),
    "unknown_kind": lambda graph, row: row["plasticity"].update(kind="stdp_magic"),
    "unknown_field": lambda graph, row: row["plasticity"].update(mu_plus=0.5),
    "multiplicative": lambda graph, row: row["plasticity"].update(
        update="multiplicative"
    ),
}


@pytest.mark.parametrize("case", sorted(_STDP_CORRUPTED))
def test_t34_graph_validation_rejects_unsupported_or_corrupted_plasticity(case):
    # Bundles loaded from disk pass through validate_graph, so a graph edited
    # outside Python is held to the same rules as Network.connect.
    graph = _stdp_graph_with(_STDP_CORRUPTED[case])
    with pytest.raises(ValueError, match="stdp_simple|plasticity"):
        snn.validate_graph(graph).raise_for_errors()


# stdp_nonlinear acceptance tests (N01-N04). The rest are in
# tests/sim/test_plasticity_stdp_nonlinear.py. Skipped until lang provides
# STDPNonlinear.

requires_stdp_nonlinear = pytest.mark.skipif(
    not hasattr(snn, "STDPNonlinear"), reason="STDPNonlinear is not implemented yet"
)

_NONLINEAR_DEFAULTS = {
    "a_plus": 0.01,
    "a_minus": 0.0105,
    "mu_plus": 0.3,
    "mu_minus": 0.7,
    "tau_plus": 16.8 * snn.ms,
    "tau_minus": 33.7 * snn.ms,
    "w_max": 1.0,
}


def _stdp_nonlinear(**overrides):
    return snn.STDPNonlinear(**{**_NONLINEAR_DEFAULTS, **overrides})  # PROVISIONAL


@requires_stdp_nonlinear
def test_n01_spec_states_nonlinear_soft_bounds_and_defaults():
    spec = _stdp_nonlinear().json()
    assert spec == {
        "kind": "stdp_nonlinear",
        "update": "nonlinear",
        "bounds": "soft",
        "pairing": "all_to_all",
        "a_plus": 0.01,
        "a_minus": 0.0105,
        "mu_plus": 0.3,
        "mu_minus": 0.7,
        "tau_plus": {"value": 16.8, "unit": "ms"},
        "tau_minus": {"value": 33.7, "unit": "ms"},
        "w_min": 0.0,
        "w_max": 1.0,
        "freeze_zero_weights": False,
    }
    chosen = _stdp_nonlinear(w_min=0.2, pairing="nearest", freeze_zero_weights=True)
    assert (chosen.json()["w_min"], chosen.json()["pairing"]) == (0.2, "nearest")
    assert chosen.json()["freeze_zero_weights"] is True
    # Opt-in: only a projection with the rule carries it.
    plastic = snn.compile(_stdp_network(_stdp_nonlinear()), target=None).graph
    static = snn.compile(_stdp_network(), target=None).graph
    row = next(r for r in plastic["projections"] if r["id"] == "syn")
    assert row["plasticity"] == spec
    static_row = next(r for r in static["projections"] if r["id"] == "syn")
    assert "plasticity" not in static_row
    # Keyword-only, and every physical parameter is required.
    with pytest.raises(TypeError):
        snn.STDPNonlinear(*_NONLINEAR_DEFAULTS.values())
    for name in _NONLINEAR_DEFAULTS:
        missing = {k: v for k, v in _NONLINEAR_DEFAULTS.items() if k != name}
        with pytest.raises(TypeError):
            snn.STDPNonlinear(**missing)
    with pytest.raises(TypeError):
        _stdp_nonlinear(update="additive")  # the update is not a parameter


@requires_stdp_nonlinear
@pytest.mark.parametrize(
    "override",
    [
        {"a_plus": -0.01},
        {"a_minus": -0.01},
        {"a_plus": math.nan},
        {"mu_plus": -0.1},
        {"mu_minus": -0.1},
        {"mu_plus": math.nan},
        {"mu_minus": math.inf},
        {"mu_plus": True},
        {"tau_plus": 0.0 * snn.ms},
        {"tau_minus": -1.0 * snn.ms},
        {"tau_plus": 16.8},
        {"w_min": -0.1},
        {"w_min": 1.0},
        {"w_min": 0.6, "w_max": 0.5},
        {"w_max": math.inf},
        {"pairing": "presynaptic_centred"},
        {"freeze_zero_weights": "yes"},
    ],
    ids=[
        "a_plus<0",
        "a_minus<0",
        "a_plus=nan",
        "mu_plus<0",
        "mu_minus<0",
        "mu_plus=nan",
        "mu_minus=inf",
        "mu_plus=bool",
        "tau_plus=0",
        "tau_minus<0",
        "tau_plus_not_a_time",
        "w_min<0",
        "w_min=w_max",
        "w_min>w_max",
        "w_max=inf",
        "unknown_pairing",
        "freeze_not_bool",
    ],
)
def test_n02_invalid_parameters_raise(override):
    with pytest.raises(ValueError, match="stdp_nonlinear"):
        _stdp_nonlinear(**override)


_NONLINEAR_CORRUPTED = {
    "unknown_field": lambda spec: spec.update(alpha=1.05),
    "additive_update": lambda spec: spec.update(update="additive"),
    "hard_bounds": lambda spec: spec.update(bounds="hard"),
    "negative_mu_minus": lambda spec: spec.update(mu_minus=-0.5),
    "non_finite_a_plus": lambda spec: spec.update(a_plus=math.inf),
    "tau_without_unit": lambda spec: spec.update(tau_plus=16.8),
}


@requires_stdp_nonlinear
def test_n03_spec_round_trips_and_bad_specs_give_e120(tmp_path):
    from snnlab._shared.plasticity import stdp_nonlinear_from_json  # PROVISIONAL

    rule = _stdp_nonlinear(w_min=0.1, pairing="nearest", freeze_zero_weights=True)
    bundle = snn.compile(_stdp_network(rule), target=None)
    loaded = snn.load_bundle(bundle.write(tmp_path / "bundle"))
    row = next(r for r in loaded.graph["projections"] if r["id"] == "syn")
    assert row["plasticity"] == rule.json()
    values = stdp_nonlinear_from_json(row["plasticity"])
    assert values["mu_plus"] == 0.3 and values["mu_minus"] == 0.7
    assert (values["tau_plus_ms"], values["tau_minus_ms"]) == (16.8, 33.7)
    assert (values["w_min"], values["w_max"], values["pairing"]) == (
        0.1,
        1.0,
        "nearest",
    )
    assert values["freeze_zero_weights"] is True
    errors = [
        d for d in snn.validate_graph(loaded.graph).diagnostics if d.severity == "error"
    ]
    assert errors == []
    for name, mutation in _NONLINEAR_CORRUPTED.items():
        graph = copy.deepcopy(bundle.graph)
        mutation(
            next(r for r in graph["projections"] if r["id"] == "syn")["plasticity"]
        )
        codes = [d.code for d in snn.validate_graph(graph).diagnostics]
        assert "E120" in codes, name


_NONLINEAR_REJECTED = {
    "inhibitory_receptor": lambda rule: _stdp_network(rule, port="inhibitory"),
    "modulatory_receptor": lambda rule: _stdp_network(rule, port="modulatory"),
    "weight_modulation": lambda rule: _stdp_network(
        rule, modulation=snn.ConstantGain(0.5)
    ),
    "shared_weight": _stdp_shared_weight_network,
    "conv2d_kernel": _stdp_conv2d_network,
    "non_spiking_source": lambda rule: _stdp_network(rule, source_signal="current"),
    "non_spiking_target": lambda rule: _stdp_network(rule, spiking_target=False),
}


@requires_stdp_nonlinear
@pytest.mark.parametrize("case", sorted(_NONLINEAR_REJECTED))
def test_n04_unsupported_projections_give_e121(case):
    # Each network compiles without the rule, so the rule alone is refused.
    snn.compile(_NONLINEAR_REJECTED[case](None), target=None)
    with pytest.raises(ValueError, match="E121.*stdp_nonlinear"):
        snn.compile(_NONLINEAR_REJECTED[case](_stdp_nonlinear()), target=None)
