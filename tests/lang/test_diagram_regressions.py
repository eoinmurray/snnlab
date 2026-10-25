"""Semantic regressions exposed by real-world diagram review cases."""

import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest

from snnlab import lang, viz

EXAMPLE = Path(__file__).parents[2] / "examples" / "diagrams"
spec = importlib.util.spec_from_file_location("diagram_example", EXAMPLE / "example.py")
diagram_cases = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diagram_cases
spec.loader.exec_module(diagram_cases)
all_box_types = diagram_cases.all_box_types
case_diagram = diagram_cases.case_diagram


def snapshot(name):
    path = EXAMPLE / "inputs" / f"{name}.json"
    payload = json.loads(path.read_text())
    return lang.Bundle(payload["graph"], payload["training"], payload["manifest"], [])


@pytest.mark.parametrize("view", ["circuit", "training", "expanded"])
def test_sustained_cue_case_preserves_inputs_groups_and_pathways(view):
    bundle = snapshot("exp136_sustained_cue_ping")
    visual = lang.diagram(bundle, view=view)
    assert {node.id for node in visual.nodes} == {
        row["id"] for row in bundle.graph["inputs"] + bundle.graph["populations"]
    }
    assert {group.id: set(group.members) for group in visual.groups} == {
        group["id"]: set(group["members"]) & {node.id for node in visual.nodes}
        for group in bundle.graph["groups"]
    }
    assert {edge.id for edge in visual.edges} == {
        row["id"] for row in bundle.graph["projections"]
    }
    assert len(visual.nodes) == 13
    assert len(visual.groups) == 5
    assert len(visual.edges) == 17
    pathways = {(edge.source, edge.target) for edge in visual.edges}
    for sender in ("S1", "S2"):
        assert (f"cue_{sender}_E", f"{sender}_E") in pathways
        assert (f"{sender}_E", "R_E") in pathways
        assert (f"{sender}_E", "R_I") in pathways
    assert not any(
        source.startswith("R_") and target.startswith("S")
        for source, target in pathways
    )
    assert not any(
        source.startswith("cue_") and target.startswith("R_")
        for source, target in pathways
    )


@pytest.mark.parametrize("view", ["circuit", "training", "expanded"])
def test_all_views_preserve_every_group_member_and_projection(view):
    bundle = snapshot("exp133_coupled")
    visual = lang.diagram(bundle, view=view)
    assert {
        row["id"] for row in bundle.graph["populations"] + bundle.graph["operations"]
    } <= {node.id for node in visual.nodes}
    assert all(node.kind != "component" for node in visual.nodes)
    assert {row["id"] for row in bundle.graph["projections"]} == {
        edge.id for edge in visual.edges if edge.id
    }


def test_all_box_types_example_is_complete_and_explicit():

    visual = case_diagram(all_box_types(), "training")
    assert {node.kind for node in visual.nodes} == {
        "input",
        "population",
        "operation",
        "output",
        "training",
        "objective",
        "neutral",
    }
    assert {node.badge for node in visual.nodes if node.kind == "training"} == {
        "trainable",
        "frozen",
    }
    assert {node.badge for node in visual.nodes if node.kind == "population"} == {
        "spiking",
        "analogue",
    }
    assert any(group.members == ("circuit_E", "circuit_I") for group in visual.groups)


def test_box_information_is_concise_and_semantically_ordered():

    visual = case_diagram(all_box_types(), "training")
    nodes = {node.id: node for node in visual.nodes}
    assert nodes["inputs"].detail.splitlines() == ["Spikes", "time × batch × 16"]
    assert nodes["circuit_E"].detail.splitlines() == [
        "16 neurons",
        "COBA-LIF",
        "Membrane τ: 20 ms",
    ]
    assert nodes["readout"].detail.splitlines() == ["4 units", "Leaky integrator"]
    assert nodes["mean_scores"].detail.splitlines() == [
        "Mean over time",
        "batch × 4",
        "Unit: mV",
    ]
    assert nodes["train:learned"].detail.splitlines() == [
        "Trainable",
        "Learning rate: 0.001",
        "320 parameters",
    ]
    assert nodes["train:fixed_recurrence"].detail.splitlines() == [
        "Frozen",
        "128 parameters",
    ]
    assert nodes["objective:0"].detail == "Target: label"
    assert all(len(node.detail.splitlines()) <= 3 for node in visual.nodes)


def test_box_type_name_and_information_have_clear_visual_hierarchy(tmp_path):
    if any(shutil.which(program) is None for program in ("dot", "neato")):
        pytest.skip("Graphviz is required")

    visual = case_diagram(all_box_types(), "training")
    path = viz.render_diagram(visual, tmp_path / "typography.svg")
    ns = {"svg": "http://www.w3.org/2000/svg"}
    groups = ET.parse(path).getroot().findall(".//svg:g", ns)
    cards = [group for group in groups if "node" in group.get("class", "").split()]
    assert len(cards) == len(visual.nodes)
    type_colours = {}
    for card in cards:
        texts = card.findall("svg:text", ns)
        header, name, information = texts[:3]
        assert len(texts) == (4 if header.text == "POPULATION" else 3)
        assert all(text.get("text-anchor") == "start" for text in texts)
        assert len({text.get("x") for text in texts}) == 1
        box = card.find("svg:polygon", ns)
        type_colours.setdefault(header.text, box.get("fill"))
        assert type_colours[header.text] == box.get("fill")
        assert box.get("stroke") != box.get("fill")
        assert header.text == header.text.upper()
        assert float(header.get("font-size")) == 10
        assert float(name.get("font-size")) == 14
        assert name.get("font-weight") == "bold"
        assert float(information.get("font-size")) == 11
        assert information.get("font-weight") != "bold"
        assert (
            float(header.get("y")) < float(name.get("y")) < float(information.get("y"))
        )
    assert len(set(type_colours.values())) == len(type_colours)


def test_collapsed_component_box_is_rejected():
    visual = viz.Diagram(
        "removed", (viz.DiagramNode("group", "Group", "", "", kind="component"),), ()
    )
    with pytest.raises(ValueError, match="unknown diagram node kind: component"):
        viz.diagram_to_dot(visual)


@pytest.mark.parametrize(
    "case_name",
    [
        "all_box_types",
        "exp124_ping",
        "exp133_coupled",
        "exp136_sustained_cue_ping",
        "self_loop",
    ],
)
def test_arrowheads_are_visible_outside_every_box(case_name, tmp_path):
    if any(shutil.which(program) is None for program in ("dot", "neato")):
        pytest.skip("Graphviz is required")
    from snnlab.viz.diagrams.routing import _bounds, _overlaps

    if case_name == "all_box_types":
        visual = case_diagram(all_box_types(), "training")
    elif case_name == "self_loop":
        net = lang.Network("self_loop")
        cell = net.population("E", size=8, neuron=lang.COBA_LIF())
        net.connect(
            cell.spikes,
            cell.excitatory,
            name="recurrent",
            synapse=lang.AMPA(),
            connection="recurrent",
        )
        visual = lang.diagram(lang.compile(net))
    else:
        visual = lang.diagram(snapshot(case_name), view="training")
    path = viz.render_diagram(visual, tmp_path / "visible.svg")
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    groups = ET.parse(path).getroot().findall(".//svg:g", namespace)

    def polygon_bounds(polygon):
        return _bounds(
            [
                tuple(map(float, point.split(",")))
                for point in polygon.get("points").split()
            ]
        )

    boxes = [
        polygon_bounds(polygon)
        for group in groups
        if "node" in group.get("class", "").split()
        for polygon in group.findall("svg:polygon", namespace)
    ]
    edges = [group for group in groups if "edge" in group.get("class", "").split()]
    assert len(edges) == len(visual.edges)
    for edge in edges:
        arrows = edge.findall("svg:polygon", namespace)
        assert arrows
        for arrow in arrows:
            assert not any(_overlaps(polygon_bounds(arrow), box) for box in boxes), (
                edge.find("svg:title", namespace).text
            )


@pytest.mark.parametrize("view", ["circuit", "training", "expanded"])
@pytest.mark.parametrize("model", ["coba", "ping"])
def test_classifier_output_retains_complete_operation_path(model, view):
    visual = lang.diagram(snapshot(f"exp124_{model}"), view=view)
    incoming = {}
    for edge in visual.edges:
        incoming.setdefault(edge.target, set()).add(edge.source)
    reachable = {"out:class_scores"}
    pending = list(reachable)
    for target in pending:
        for source in incoming.get(target, set()) - reachable:
            reachable.add(source)
            pending.append(source)
    assert {
        "drive",
        "E1",
        "E2",
        "readout",
        "scores",
        "scaled_scores",
        "normalization_counts",
        "normalized_scores",
    } <= reachable


def test_projection_and_operation_parameters_attach_to_visible_training_targets():
    visual = lang.diagram(snapshot("exp124_ping"), view="training")
    targets = {}
    for edge in visual.edges:
        if edge.role == "training":
            targets.setdefault(edge.source, set()).add(edge.target)
    assert targets == {
        "train:input": {"E1"},
        "train:interlayer": {"E2"},
        "train:readout": {"readout"},
        "train:fixed_loop": {"E1", "I1", "E2", "I2", "scaled_scores"},
    }
    nodes = {node.id: node for node in visual.nodes}
    assert nodes["I1"].accent_role == nodes["I2"].accent_role == "inhibitory"
    assert nodes["E1"].accent_role == "ink"


@pytest.mark.parametrize(
    "name", ["exp131_coupled", "exp133_coupled", "exp133_sender_receiver"]
)
def test_coupled_circuits_preserve_all_cross_projections(name):
    bundle = snapshot(name)
    visual = lang.diagram(bundle, view="expanded")
    assert len(visual.groups) == 2
    assert all(group.same_rank and not group.same_row for group in visual.groups)
    expected = {projection["id"] for projection in bundle.graph["projections"]}
    assert {edge.id for edge in visual.edges if edge.id} == expected


@pytest.mark.parametrize("view", ["circuit", "training", "expanded"])
@pytest.mark.parametrize("name", ["exp124_ping", "exp131_coupled", "exp133_coupled"])
def test_rendered_inhibitory_populations_are_directly_below_excitatory(
    name, view, tmp_path
):
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' is required")
    visual = lang.diagram(snapshot(name), view=view)
    result = subprocess.run(
        ["dot", "-Tplain"],
        input=viz.diagram_to_dot(visual),
        text=True,
        capture_output=True,
        check=True,
    )
    rows = [shlex.split(line) for line in result.stdout.splitlines()]
    positions = {
        row[1]: tuple(map(float, row[2:6])) for row in rows if row[0] == "node"
    }
    pairs = (
        [("E1", "I1"), ("E2", "I2")]
        if name == "exp124_ping"
        else [(f"PING_{letter}_E", f"PING_{letter}_I") for letter in ("A", "B")]
    )
    for excitatory, inhibitory in pairs:
        if excitatory not in positions:
            continue
        e_x, e_y, _, e_height = positions[excitatory]
        i_x, i_y, _, i_height = positions[inhibitory]
        assert e_x == pytest.approx(i_x, abs=0.01)
        assert i_y + i_height / 2 < e_y - e_height / 2
    if name == "exp124_ping":
        assert positions["I1"][1] == pytest.approx(positions["I2"][1], abs=0.01)
    graph = next(row for row in rows if row[0] == "graph")
    assert float(graph[2]) > 1.3 * float(graph[3])
    svg = viz.render_diagram(visual, tmp_path / "diagram.svg")
    root = ET.parse(svg).getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    edges = [
        group
        for group in root.findall(".//svg:g", namespace)
        if "edge" in group.get("class", "").split()
    ]
    assert len(edges) == len(visual.edges)
    assert all(group.find("svg:polygon", namespace) is not None for group in edges)
    for group in edges:
        for path in group.findall("svg:path", namespace):
            coordinates = list(
                map(float, re.findall(r"-?\d+(?:\.\d+)?", path.get("d")))
            )
            points = list(zip(coordinates[::2], coordinates[1::2]))
            for (x1, y1), (x2, y2) in zip(points, points[1:]):
                assert abs(x1 - x2) < 0.01 or abs(y1 - y2) < 0.01


@pytest.mark.parametrize("view", ["circuit", "expanded"])
@pytest.mark.parametrize(
    "name",
    [
        "exp131_coupled",
        "exp131_phase_response",
        "exp133_coupled",
        "exp133_sender_receiver",
    ],
)
def test_gallery_networks_have_landscape_layouts(name, view):
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' is required")
    visual = lang.diagram(snapshot(name), view=view)
    result = subprocess.run(
        ["dot", "-Tplain"],
        input=viz.diagram_to_dot(visual),
        text=True,
        capture_output=True,
        check=True,
    )
    graph = shlex.split(result.stdout.splitlines()[0])
    assert float(graph[2]) > 1.3 * float(graph[3])


@pytest.mark.parametrize("view", ["circuit", "training", "expanded"])
def test_grouped_fan_in_layout_separates_stages_and_keeps_e_above_i(view):
    if shutil.which("dot") is None:
        pytest.skip("Graphviz is required")
    visual = lang.diagram(snapshot("exp136_sustained_cue_ping"), view=view)
    # Rename every identifier: layout depends on topology, not experiment names.
    ids = {node.id: f"node_{index}" for index, node in enumerate(visual.nodes)}
    renamed = replace(
        visual,
        name="grouped_fan_in",
        nodes=tuple(replace(node, id=ids[node.id]) for node in visual.nodes),
        edges=tuple(
            replace(edge, source=ids[edge.source], target=ids[edge.target])
            for edge in visual.edges
        ),
        groups=tuple(
            replace(
                group,
                id=f"group_{index}",
                members=tuple(ids[member] for member in group.members),
            )
            for index, group in enumerate(visual.groups)
        ),
    )
    layout = json.loads(
        subprocess.check_output(
            ["dot", "-Tjson"], input=viz.diagram_to_dot(renamed), text=True
        )
    )
    nodes = {node["name"]: node for node in layout["objects"] if "pos" in node}
    positions = {
        name: tuple(map(float, nodes[identifier]["pos"].split(",")))
        for name, identifier in ids.items()
    }
    clusters = {node["name"]: node for node in layout["objects"] if "bb" in node}
    boxes = {
        group.id: tuple(
            map(float, clusters[f"cluster_n_group_{index}"]["bb"].split(","))
        )
        for index, group in enumerate(visual.groups)
    }
    for sender in ("S1", "S2"):
        # Whole stages have separate columns; branches have separate rows.
        assert boxes[f"cue_{sender}"][2] < boxes[sender][0]
        assert boxes[sender][2] < boxes["R"][0]
    assert boxes["S1"][1] > boxes["S2"][3]
    assert positions["S2_E"][1] < positions["R_E"][1] < positions["S1_E"][1]
    for circuit in ("S1", "S2", "R"):
        ex, ey = positions[f"{circuit}_E"]
        ix, iy = positions[f"{circuit}_I"]
        assert ex == pytest.approx(ix, abs=0.01)
        assert ey > iy
    _, _, width, height = map(float, layout["bb"].split(","))
    assert width > 1.5 * height


@pytest.mark.parametrize(
    "name,view",
    [
        ("exp124_ping", "training"),
        ("exp133_coupled", "expanded"),
        ("exp131_phase_response", "expanded"),
        ("exp136_sustained_cue_ping", "circuit"),
        ("exp136_sustained_cue_ping", "training"),
    ],
)
def test_exported_wire_labels_clear_nodes_and_wires(name, view):
    if any(shutil.which(program) is None for program in ("dot", "neato")):
        pytest.skip("Graphviz is required")
    from snnlab.viz.diagrams.routing import (
        _bounds,
        _overlaps,
        _text_boxes,
        expose_arrowheads,
        position_wire_labels,
        route_feedback_lanes,
    )

    source = viz.diagram_to_dot(lang.diagram(snapshot(name), view=view))
    layout = json.loads(
        subprocess.check_output(["dot", "-Tjson"], input=source, text=True)
    )
    positioned = subprocess.check_output(["dot", "-Tdot"], input=source, text=True)
    layout, positioned = route_feedback_lanes(layout, positioned)
    layout, positioned = expose_arrowheads(layout, positioned)
    adjusted = position_wire_labels(layout, positioned)
    exported = json.loads(
        subprocess.check_output(["neato", "-n2", "-Tjson"], input=adjusted, text=True)
    )
    obstacles = []
    for node in exported["objects"]:
        if "pos" not in node:
            continue
        x, y = map(float, node["pos"].split(","))
        w, h = float(node["width"]) * 72, float(node["height"]) * 72
        obstacles.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    labels = []
    for edge in exported.get("edges", []):
        if edge.get("style") == "invis":
            continue
        text_boxes = _text_boxes(edge.get("_ldraw_", []))
        if text_boxes:
            labels.append(
                _bounds(
                    [(box[0], box[1]) for box in text_boxes]
                    + [(box[2], box[3]) for box in text_boxes]
                )
            )
        for command in edge.get("_draw_", []):
            if command["op"] == "b":
                points = command["points"]
                obstacles.extend(
                    _bounds((a, b), 1) for a, b in zip(points, points[1:]) if a != b
                )
    assert labels
    for index, label in enumerate(labels):
        assert not any(
            _overlaps(label, obstacle) for obstacle in obstacles + labels[:index]
        )


@pytest.mark.parametrize("view", ["training", "expanded"])
@pytest.mark.parametrize(
    "name", ["exp131_coupled", "exp133_coupled", "exp133_sender_receiver"]
)
def test_feedback_uses_separated_overhead_lanes(name, view):
    if any(shutil.which(program) is None for program in ("dot", "neato")):
        pytest.skip("Graphviz is required")
    from snnlab.viz.diagrams.routing import route_feedback_lanes

    source = viz.diagram_to_dot(lang.diagram(snapshot(name), view=view))
    layout = json.loads(
        subprocess.check_output(["dot", "-Tjson"], input=source, text=True)
    )
    positioned = subprocess.check_output(["dot", "-Tdot"], input=source, text=True)
    _, positioned = route_feedback_lanes(layout, positioned)
    exported = json.loads(
        subprocess.check_output(["neato", "-n2", "-Tjson"], input=positioned, text=True)
    )
    nodes = {node["_gvid"]: node for node in exported["objects"] if "pos" in node}
    top = max(
        float(node["pos"].split(",")[1]) + float(node["height"]) * 36
        for node in nodes.values()
    )
    lanes, verticals = [], []
    for edge in exported["edges"]:
        if "feedback" not in edge.get("class", "").split():
            continue
        if (
            abs(
                float(nodes[edge["tail"]]["pos"].split(",")[0])
                - float(nodes[edge["head"]]["pos"].split(",")[0])
            )
            < 1
        ):
            continue
        segments = [
            (a, b)
            for command in edge["_draw_"]
            if command["op"] == "b"
            for a, b in zip(command["points"], command["points"][1:])
            if a != b
        ]
        overhead = [
            (a, b) for a, b in segments if abs(a[1] - b[1]) < 0.01 and a[1] > top
        ]
        assert len(overhead) == 1
        lanes.append(overhead[0][0][1])
        verticals.extend(
            (edge["id"], a[0], min(a[1], b[1]), max(a[1], b[1]))
            for a, b in segments
            if abs(a[0] - b[0]) < 0.01
        )
        assert any(command["op"] == "P" for command in edge["_hdraw_"])
    assert len(lanes) in {2, 4}
    assert all(b - a >= 31.9 for a, b in zip(sorted(lanes), sorted(lanes)[1:]))
    for index, (edge_id, x, low, high) in enumerate(verticals):
        for other_id, other_x, other_low, other_high in verticals[:index]:
            if edge_id != other_id and min(high, other_high) - max(low, other_low) > 1:
                assert abs(x - other_x) >= 17.9


@pytest.mark.parametrize("view", ["circuit", "training", "expanded"])
def test_weight_labels_are_opt_in_and_preserve_diagram_semantics(view):
    bundle = all_box_types().bundle
    original = json.dumps(bundle.graph, sort_keys=True)
    plain = lang.diagram(bundle, view=view)
    labelled = lang.diagram(bundle, view=view, weight_labels=True)
    assert all(edge.weight_label is None for edge in plain.edges)
    weighted_ids = {row["id"] for row in bundle.graph["projections"]}
    assert {edge.id for edge in labelled.edges if edge.weight_label} == weighted_ids
    for edge in labelled.edges:
        if edge.weight_label:
            assert edge.weight_label == viz.WeightLabel(
                subscript=f"{edge.source}→{edge.target}"
            )
    assert (
        replace(
            labelled,
            edges=tuple(replace(edge, weight_label=None) for edge in labelled.edges),
        )
        == plain
    )
    assert json.dumps(bundle.graph, sort_keys=True) == original


def test_custom_weight_labels_select_projections_and_validate_ids():
    bundle = all_box_types().bundle
    label = viz.WeightLabel("J", "E→I", "rec")
    visual = lang.diagram(bundle, weight_labels={"E_to_I": label})
    assert {
        edge.id: edge.weight_label for edge in visual.edges if edge.weight_label
    } == {"E_to_I": label}
    with pytest.raises(ValueError, match="weighted projection IDs"):
        lang.diagram(bundle, weight_labels={"missing": label})
    with pytest.raises(TypeError, match="WeightLabel instances"):
        lang.diagram(bundle, weight_labels={"E_to_I": "J"})
    bundle.graph["projections"][0]["parameters"] = []
    identifier = bundle.graph["projections"][0]["id"]
    visual = lang.diagram(bundle, weight_labels=True)
    assert (
        next(edge for edge in visual.edges if edge.id == identifier).weight_label
        is None
    )
    with pytest.raises(ValueError, match="weighted projection IDs"):
        lang.diagram(bundle, weight_labels={identifier: label})


@pytest.mark.parametrize("suffix", ["svg", "png", "pdf"])
def test_math_weight_labels_render_with_escaped_text(tmp_path, suffix):
    if any(shutil.which(program) is None for program in ("dot", "neato")):
        pytest.skip("Graphviz is required")
    visual = lang.diagram(
        all_box_types().bundle,
        weight_labels={"E_to_I": viz.WeightLabel("J<&", "E→I", "rec")},
    )
    dot = viz.diagram_to_dot(visual)
    assert "<B><I>J&lt;&amp;</I></B>" in dot
    assert "<SUB>" not in dot and "<SUP>" not in dot
    assert 'ROWSPAN="2"' in dot
    path = viz.render_diagram(visual, tmp_path / f"weights.{suffix}")
    if suffix == "svg":
        root = ET.parse(path).getroot()
        edge = next(
            element for element in root.iter() if element.get("id") == "e_n_E_to_I"
        )
        text = [element for element in edge.iter() if element.tag.endswith("}text")]
        assert {element.text for element in text} >= {"AMPA", "J<&", "E→I", "rec"}
        symbol = next(element for element in text if element.text == "J<&")
        assert symbol.get("font-weight") == "bold"
        assert symbol.get("font-style") == "italic"
        subscript = next(element for element in text if element.text == "E→I")
        superscript = next(element for element in text if element.text == "rec")
        assert float(subscript.get("font-size")) < float(symbol.get("font-size"))
        assert float(superscript.get("font-size")) < float(symbol.get("font-size"))
        assert (
            float(superscript.get("y"))
            < float(symbol.get("y"))
            < float(subscript.get("y"))
        )
        assert float(subscript.get("x")) == pytest.approx(float(superscript.get("x")))
        assert float(subscript.get("x")) > float(symbol.get("x"))
        assert all("baseline-shift" not in element.attrib for element in text)
    else:
        assert path.read_bytes().startswith(b"\x89PNG" if suffix == "png" else b"%PDF")


def test_long_default_weight_names_fit_grouped_classifier(tmp_path):
    if any(shutil.which(program) is None for program in ("dot", "neato")):
        pytest.skip("Graphviz is required")
    visual = lang.diagram(
        diagram_cases.representative_classifier().bundle, weight_labels=True
    )
    path = viz.render_diagram(visual, tmp_path / "classifier.svg")
    text = "".join(ET.parse(path).getroot().itertext())
    assert "layer_2_I→layer_2_E" in text
