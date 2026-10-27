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
from itertools import pairwise
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
def test_cued_digit_snapshot_preserves_separate_inputs_and_classifier(view):
    bundle = snapshot("exp155_cued_digits")
    visual = lang.diagram(bundle, view=view)
    assert len(bundle.graph["populations"]) == 6
    assert len(bundle.graph["projections"]) == 12
    assert len(bundle.graph["operations"]) == 2
    assert len(bundle.graph["outputs"]) == 1
    assert {row["id"]: row["shape"][-1] for row in bundle.graph["inputs"]} == {
        "digit_A": 64,
        "digit_B": 64,
        "cue_A": 16,
        "cue_B": 16,
    }
    assert {row["id"] for row in bundle.graph["projections"]} <= {
        edge.id for edge in visual.edges
    }
    pathways = {(edge.source, edge.target) for edge in visual.edges}
    for module in ("A", "B"):
        assert (f"digit_{module}", f"{module}_E") in pathways
        assert (f"cue_{module}", f"{module}_E") in pathways
    assert ("C_E", "sum_C_E") in pathways
    assert {node.id for node in visual.nodes if node.kind == "output"} == {
        "out:class_scores"
    }
    assert ("A_E", "C_E") in pathways
    assert ("B_E", "C_E") in pathways
    assert ("sum_C_E", "R") in pathways
    assert ("R", "out:class_scores") in pathways
    if view == "training":
        assert {node.id for node in visual.nodes if node.kind == "training"} == {
            "train:classifier",
            "train:feedback",
            "train:feedforward",
            "train:fixed",
        }
        assert ("out:class_scores", "objective:0") in pathways
        assert {
            node.id: node.badge for node in visual.nodes if node.kind == "training"
        } == {
            "train:classifier": "trainable",
            "train:feedforward": "trainable",
            "train:feedback": "frozen",
            "train:fixed": "frozen",
        }


def diagnostic_fan_in_snapshot():
    """Add data-only diagnostic branches to stress the optional readout layout."""
    bundle = snapshot("exp155_cued_digits")
    template = next(row for row in bundle.graph["operations"] if row["id"] == "sum_C_E")
    for population in bundle.graph["populations"]:
        name = population["id"]
        if name != "C_E":
            bundle.graph["operations"].append(
                {
                    **template,
                    "id": f"sum_{name}",
                    "sources": [f"{name}.spikes"],
                    "shape": ["batch", population["size"]],
                }
            )
        bundle.graph["outputs"].append(
            {"id": f"{name}_counts", "signal": f"sum_{name}.value"}
        )
    return bundle


@pytest.mark.parametrize("view", ["circuit", "training"])
@pytest.mark.parametrize("rename", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_rendered_fan_in_keeps_pairs_compact_and_readouts_aligned(
    view, rename, diagnostics, tmp_path
):
    if any(shutil.which(program) is None for program in ("dot", "neato")):
        pytest.skip("Graphviz is required")
    bundle = (
        diagnostic_fan_in_snapshot() if diagnostics else snapshot("exp155_cued_digits")
    )
    visual = lang.diagram(bundle, view=view)
    ids = {
        node.id: f"node_{index}" if rename else node.id
        for index, node in enumerate(visual.nodes)
    }
    visual = replace(
        visual,
        nodes=tuple(replace(node, id=ids[node.id]) for node in visual.nodes),
        edges=tuple(
            replace(edge, source=ids[edge.source], target=ids[edge.target])
            for edge in visual.edges
        ),
    )
    path = viz.render_diagram(visual, tmp_path / "fan-in.svg")
    ns = {"svg": "http://www.w3.org/2000/svg"}
    groups = ET.parse(path).getroot().findall(".//svg:g", ns)
    positions, boxes = {}, {}
    for group in groups:
        if "node" not in group.get("class", "").split():
            continue
        points = [
            tuple(map(float, point.split(",")))
            for point in group.find("svg:polygon", ns).get("points").split()
        ]
        x, y = zip(*points, strict=True)
        identifier = group.find("svg:title", ns).text
        positions[identifier] = ((min(x) + max(x)) / 2, (min(y) + max(y)) / 2)
        boxes[identifier] = (min(x), min(y), max(x), max(y))
    assert set(positions) == set(ids.values())
    assert len(
        [group for group in groups if "edge" in group.get("class", "").split()]
    ) == len(visual.edges)
    population_rows = {
        ids[f"{module}_{kind}"]: positions[ids[f"{module}_{kind}"]][1]
        for module in ("A", "B", "C")
        for kind in ("E", "I")
    }
    for module in ("A", "B", "C"):
        e, i = ids[f"{module}_E"], ids[f"{module}_I"]
        assert positions[e][0] == pytest.approx(positions[i][0], abs=0.01)
        assert boxes[e][3] < boxes[i][1]
        assert boxes[i][1] - boxes[e][3] < 60
        assert not any(
            positions[e][1] < y < positions[i][1]
            for node, y in population_rows.items()
            if node not in {e, i}
        )
        for kind in ("E", "I"):
            population = f"{module}_{kind}"
            if not diagnostics and population != "C_E":
                continue
            assert positions[ids[population]][1] == pytest.approx(
                positions[ids[f"sum_{population}"]][1], abs=0.01
            )
    assert positions[ids["A_E"]][0] < positions[ids["C_E"]][0]
    assert positions[ids["B_E"]][0] < positions[ids["C_E"]][0]
    for node in ("R", "out:class_scores"):
        assert positions[ids[node]][1] == pytest.approx(
            positions[ids["C_E"]][1], abs=0.01
        )
    if diagnostics:
        assert positions[ids["out:C_E_counts"]][1] > positions[ids["C_I"]][1]


@pytest.mark.parametrize("change", ["cycle", "group", "missing_readout"])
def test_fan_in_layout_falls_back_for_other_topologies(change):
    from snnlab.viz.diagrams.fan_in import position_readout_fan_in

    visual = lang.diagram(snapshot("exp155_cued_digits"))
    if change == "cycle":
        visual = replace(
            visual,
            edges=(*visual.edges, viz.DiagramEdge("C_E", "A_E", role="excitatory")),
        )
    elif change == "group":
        visual = replace(
            visual, groups=(viz.DiagramGroup("sender", "Sender", ("A_E", "A_I")),)
        )
    else:
        visual = replace(
            visual,
            edges=tuple(edge for edge in visual.edges if edge.target != "sum_C_E"),
        )
    assert position_readout_fan_in(visual, {}, viz.diagram_to_dot(visual)) is None


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
        "exp155_cued_digits",
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
            points = list(zip(coordinates[::2], coordinates[1::2], strict=True))
            for (x1, y1), (x2, y2) in pairwise(points):
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
        ("exp155_cued_digits", "circuit"),
        ("exp155_cued_digits", "training"),
    ],
)
@pytest.mark.parametrize("math_labels", [False, True])
def test_exported_wire_labels_clear_nodes_and_wires(name, view, math_labels, tmp_path):
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

    bundle = snapshot(name)
    visual = lang.diagram(
        bundle,
        view=view,
        connection_labels={
            row["id"]: rf"$J_{{{index}}}$" if math_labels else f"Path {index}"
            for index, row in enumerate(bundle.graph["projections"])
        },
    )
    source = viz.diagram_to_dot(visual)
    from snnlab.viz.diagrams.fan_in import position_readout_fan_in
    from snnlab.viz.diagrams.labels import math_labels_to_dot, prepare_math_labels

    assets = prepare_math_labels(visual, viz.DiagramTheme(), tmp_path)
    source = math_labels_to_dot(source, visual, viz.DiagramTheme(), assets)

    layout = json.loads(
        subprocess.check_output(["dot", "-Tjson"], input=source, text=True)
    )
    pinned = position_readout_fan_in(visual, layout, source)
    if pinned is not None:
        source = pinned
        layout = json.loads(
            subprocess.check_output(["neato", "-n", "-Tjson"], input=source, text=True)
        )
    positioned = subprocess.check_output(
        ["neato", "-n", "-Tdot"] if pinned else ["dot", "-Tdot"],
        input=source,
        text=True,
    )
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
                    _bounds((a, b), 1) for a, b in pairwise(points) if a != b
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
            for a, b in pairwise(command["points"])
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
    assert all(b - a >= 31.9 for a, b in pairwise(sorted(lanes)))
    for index, (edge_id, x, low, high) in enumerate(verticals):
        for other_id, other_x, other_low, other_high in verticals[:index]:
            if edge_id != other_id and min(high, other_high) - max(low, other_low) > 1:
                assert abs(x - other_x) >= 17.9


@pytest.mark.parametrize("view", ["circuit", "training", "expanded"])
def test_connection_labels_are_explicit_and_preserve_semantics(view):
    bundle = all_box_types().bundle
    original = json.dumps(bundle.graph, sort_keys=True)
    plain = lang.diagram(bundle, view=view)
    assert all(edge.label == "" for edge in plain.edges)
    labels = {"E_to_I": r"$J_{E\to I}$", "input_projection": "encoded input"}
    labelled = lang.diagram(bundle, view=view, connection_labels=labels)
    assert {edge.id: edge.label for edge in labelled.edges if edge.label} == labels
    assert (
        replace(
            labelled, edges=tuple(replace(edge, label="") for edge in labelled.edges)
        )
        == plain
    )
    assert json.dumps(bundle.graph, sort_keys=True) == original


def test_connection_labels_validate_ids_and_strings():
    bundle = all_box_types().bundle
    with pytest.raises(ValueError, match="known projection IDs"):
        lang.diagram(bundle, connection_labels={"missing": "gain"})
    with pytest.raises(TypeError, match="must be strings"):
        lang.diagram(bundle, connection_labels={"E_to_I": 3})
    with pytest.raises(TypeError, match="must be a mapping"):
        lang.diagram(bundle, connection_labels=True)
    with pytest.raises(TypeError, match="must be strings"):
        viz.Diagram(
            "invalid",
            (viz.DiagramNode("a", "A", "", ""),),
            (viz.DiagramEdge("a", "a", label=3),),
        )
    assert not hasattr(viz, "WeightLabel")


@pytest.mark.parametrize("suffix", ["svg", "png", "pdf", "dot"])
def test_connection_math_labels_render_and_survive_temporary_assets(tmp_path, suffix):
    if any(shutil.which(program) is None for program in ("dot", "neato")):
        pytest.skip("Graphviz is required")
    formula = r"$\frac{J_{E\to I}^{\mathrm{rec}}}{\tau_I}$"
    visual = lang.diagram(
        all_box_types().bundle,
        connection_labels={"E_to_I": formula, "input_projection": "u_A <&"},
    )
    path = viz.render_diagram(visual, tmp_path / f"labels.{suffix}")
    if suffix == "svg":
        root = ET.parse(path).getroot()
        ns = {"svg": "http://www.w3.org/2000/svg"}
        assert not root.findall(".//svg:image", ns)
        nested = root.findall(".//svg:svg", ns)
        assert nested and nested[0].find("svg:title", ns).text == formula
        assert nested[0].findall(".//svg:path", ns)
        identifiers = [node.get("id") for node in root.iter() if node.get("id")]
        assert len(identifiers) == len(set(identifiers))
        assert "u_A &lt;&amp;" in path.read_text()
        assert "AMPA" not in path.read_text() and "GABA" not in path.read_text()
        assert "snnviz-diagram-" not in path.read_text()
    elif suffix == "dot":
        images = re.findall(r'IMG SRC="([^"]+)"', path.read_text())
        assert images and all(Path(image).is_file() for image in images)
        result = subprocess.run(
            ["dot", "-Tpng", str(path), "-o", str(tmp_path / "from-dot.png")],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "No such file" not in result.stderr
        assert "No or improper image" not in result.stderr
    else:
        assert path.read_bytes().startswith(b"\x89PNG" if suffix == "png" else b"%PDF")


def test_connection_math_svg_is_deterministic_and_respects_canvas(tmp_path):
    if any(shutil.which(program) is None for program in ("dot", "neato")):
        pytest.skip("Graphviz is required")
    visual = lang.diagram(
        all_box_types().bundle,
        connection_labels={"E_to_I": r"$J_{EI}$", "I_to_E": r"$J_{EI}$"},
    )
    first = viz.render_diagram(visual, tmp_path / "first.svg", canvas_size=(4000, 2200))
    second = viz.render_diagram(
        visual, tmp_path / "second.svg", canvas_size=(4000, 2200)
    )
    assert first.read_bytes() == second.read_bytes()
    root = ET.parse(first).getroot()
    assert root.get("width") == "4000px" and root.get("height") == "2200px"
    identifiers = [node.get("id") for node in root.iter() if node.get("id")]
    assert len(identifiers) == len(set(identifiers))


def test_invalid_connection_math_fails_clearly(tmp_path):
    visual = lang.diagram(
        all_box_types().bundle, connection_labels={"E_to_I": r"$\notacommand$"}
    )
    with pytest.raises(ValueError, match="invalid connection label"):
        viz.render_diagram(visual, tmp_path / "invalid.svg")


def test_bundle_visualise_accepts_connection_labels(tmp_path):
    if any(shutil.which(program) is None for program in ("dot", "neato")):
        pytest.skip("Graphviz is required")
    path = all_box_types().bundle.visualise(
        tmp_path / "label.svg", connection_labels={"E_to_I": "custom_gain"}
    )
    assert "custom_gain" in path.read_text()
