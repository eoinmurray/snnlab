"""Public placement contracts across static diagram outputs."""

import copy
import json
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import replace

import pytest
from PIL import Image

from snnlab import lang, viz
from snnlab.viz.diagrams.paths import crosses

pytestmark = pytest.mark.skipif(
    not all(shutil.which(x) for x in ("dot", "neato")), reason="Graphviz required"
)


@pytest.fixture
def diagram():
    return viz.Diagram(
        name="placement",
        nodes=tuple(
            viz.DiagramNode(k, k, "8 neurons", "spiking", kind="population")
            for k in ("a", "b", "c")
        ),
        edges=(
            viz.DiagramEdge("a", "b", id="ab"),
            viz.DiagramEdge(
                "b", "c", id="bc", role="inhibitory", connection="feedback"
            ),
        ),
    )


def test_exact_pins_routes_and_static_geometry_agree(diagram, tmp_path):
    original = copy.deepcopy(diagram)
    layout = viz.DiagramLayout(
        positions={"a": (200, 250), "b": (600, 250), "c": (1000, 250)},
        edges={
            "ab": viz.EdgeLayout(
                label=viz.DiagramLabel(r"$J_{ab}$", position=0.5, offset=(0, -50))
            )
        },
    )
    scene = viz.resolve_diagram(diagram, layout=layout)
    assert diagram == original
    for key, center in layout.positions.items():
        assert scene.nodes[key].center == pytest.approx(center)
    geometry = scene.write(tmp_path / "geometry.json")
    assert json.loads(geometry.read_text()) == json.loads(json.dumps(scene.to_dict()))
    svg = viz.render_diagram(scene, tmp_path / "scene.svg")
    root = ET.parse(svg).getroot()
    for element in scene.scene["elements"]:
        if element["type"] == "rectangle":
            native = next(x for x in root.iter() if x.get("id") == element["id"])
            assert [
                float(native.get(k)) for k in ("x", "y", "width", "height")
            ] == pytest.approx([element[k] for k in ("x", "y", "width", "height")])
    assert scene.scene["files"]
    assert "snnlab-layout-" not in svg.read_text()
    ids = [x.get("id") for x in root.iter() if x.get("id")]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("suffix", ["png", "jpg", "jpeg", "pdf"])
def test_static_formats_and_scale_preserve_geometry(diagram, tmp_path, suffix):
    scene = viz.resolve_diagram(diagram)
    before = scene.to_dict()
    path = viz.render_diagram(scene, tmp_path / f"scene.{suffix}", scale=2)
    assert path.stat().st_size > 100
    if suffix != "pdf":
        with Image.open(path) as image:
            assert image.size == pytest.approx(
                (scene.canvas.width * 2, scene.canvas.height * 2), abs=1
            )
    assert scene.to_dict() == before


def test_relative_order_alignment_and_equal_spacing(diagram):
    layout = viz.DiagramLayout(
        align_y=[["a", "b", "c"]],
        constraints=[
            viz.LayoutConstraint("order_x", ("a", "b", "c"), gap=100),
            viz.LayoutConstraint("equal_spacing_x", ("a", "b", "c")),
        ],
    )
    scene = viz.resolve_diagram(diagram, layout=layout)
    a, b, c = (scene.nodes[k] for k in ("a", "b", "c"))
    assert a.center[1] == pytest.approx(b.center[1]) == pytest.approx(c.center[1])
    assert b.x - a.right >= 100 - 1e-5
    assert c.center[0] - b.center[0] == pytest.approx(b.center[0] - a.center[0])
    vertical = viz.resolve_diagram(
        diagram,
        layout=viz.DiagramLayout(below={"b": "a", "c": "b"}, align_x=[["a", "b", "c"]]),
    )
    assert vertical.nodes["b"].y >= vertical.nodes["a"].bottom + 80 - 1e-5


def test_required_conflicts_and_weighted_preferences(diagram):
    pins = {"a": (100, 200), "b": (600, 200), "c": (1100, 200)}
    with pytest.raises(ValueError, match="required constraints.*pin a"):
        viz.resolve_diagram(
            diagram, layout=viz.DiagramLayout(positions=pins, right_of={"a": "b"})
        )
    scene = viz.resolve_diagram(
        diagram,
        layout=viz.DiagramLayout(
            positions=pins,
            constraints=[
                viz.LayoutConstraint(
                    "right_of", ("a", "b"), required=False, name="prefer a after b"
                )
            ],
        ),
    )
    assert any(
        d["constraint"] == "prefer a after b" and d["violation"] > 0
        for d in scene.diagnostics
    )
    for key, point in pins.items():
        assert scene.nodes[key].center == pytest.approx(point)


def test_nested_groups_grid_defaults_and_move(diagram):
    layout = viz.DiagramLayout(
        groups={
            "outer": viz.GroupLayout(
                members=("a", "b", "c"),
                arrangement="row",
                order=("inner", "c"),
                padding=40,
            ),
            "inner": viz.GroupLayout(
                members=("a", "b"),
                parent="outer",
                arrangement="column",
                order=("b", "a"),
                node_defaults=viz.NodeLayout(min_size=(180, 180)),
                label=r"$G$",
            ),
        }
    )
    scene = viz.resolve_diagram(diagram, layout=layout)
    assert scene.nodes["a"].y > scene.nodes["b"].y
    assert scene.groups["inner"].right < scene.nodes["c"].x
    for key, group in [("inner", ("a", "b")), ("outer", ("a", "b", "c"))]:
        box = scene.groups[key]
        assert all(
            box.x <= scene.nodes[n].x and box.right >= scene.nodes[n].right
            for n in group
        )
    moved = scene.move(
        "inner",
        (scene.groups["inner"].center[0], scene.groups["inner"].center[1] + 400),
    )
    assert moved.nodes["a"].center[1] == pytest.approx(scene.nodes["a"].center[1] + 400)
    assert moved.nodes["b"].center[1] == pytest.approx(scene.nodes["b"].center[1] + 400)
    assert moved.nodes["c"].center == pytest.approx(scene.nodes["c"].center)
    assert moved.scene["files"]


def test_grid_arrangement(diagram):
    # No scientific edges are needed to test a visual container.
    diagram = replace(diagram, edges=())
    scene = viz.resolve_diagram(
        diagram,
        layout=viz.DiagramLayout(
            groups={
                "grid": viz.GroupLayout(
                    members=("a", "b", "c"), arrangement="grid", columns=2
                )
            }
        ),
    )
    assert scene.nodes["a"].center[1] == pytest.approx(scene.nodes["b"].center[1])
    assert scene.nodes["a"].center[0] == pytest.approx(scene.nodes["c"].center[0])
    assert scene.nodes["c"].y > scene.nodes["a"].bottom


@pytest.mark.parametrize("direction", ["right", "left", "up", "down"])
def test_direction_changes_initial_flow(diagram, direction):
    scene = viz.resolve_diagram(diagram, layout=viz.DiagramLayout(direction=direction))
    a, b = (scene.nodes[k].center for k in ("a", "b"))
    axis = 0 if direction in {"right", "left"} else 1
    assert (b[axis] - a[axis]) * (1 if direction in {"right", "down"} else -1) > 0


def test_group_local_coordinates_and_anchors(diagram):
    diagram = replace(diagram, edges=())
    layout = viz.DiagramLayout(
        groups={"g": viz.GroupLayout(members=("a", "b"))},
        nodes={
            "a": viz.NodeLayout(position=(40, 120), relative_to="g", anchor="top_left")
        },
    )
    scene = viz.resolve_diagram(diagram, layout=layout)
    assert scene.nodes["a"].x - scene.groups["g"].x == pytest.approx(40)
    assert scene.nodes["a"].y - scene.groups["g"].y == pytest.approx(120)


def test_named_ports_waypoints_and_obstacle_avoidance(diagram):
    diagram = replace(diagram, edges=(viz.DiagramEdge("a", "c", id="ac"),))
    layout = viz.DiagramLayout(
        positions={"a": (200, 300), "b": (600, 300), "c": (1000, 300)},
        nodes={"a": viz.NodeLayout(ports={"exit": (1, 0.25)})},
        edges={
            "ac": viz.EdgeLayout(
                source_port="exit", target_port="left", waypoints=((600, 60),)
            )
        },
    )
    scene = viz.resolve_diagram(diagram, layout=layout)
    points = scene.edges["ac"]["points"]
    assert points[0] == pytest.approx(scene.ports["a"]["exit"])
    assert (600, 60) in points
    assert not any(
        crosses(a, b, scene.nodes["b"])
        for a, b in zip(points, points[1:], strict=False)
    )
    bad = replace(layout, edges={"ac": viz.EdgeLayout(waypoints=((600, 300),))})
    with pytest.raises(ValueError, match="inside an obstacle"):
        viz.resolve_diagram(diagram, layout=bad)


@pytest.mark.parametrize("routing", ["orthogonal", "polyline", "curved"])
def test_routing_modes_lanes_and_arrow_semantics(diagram, routing, tmp_path):
    layout = viz.DiagramLayout(
        edges={"bc": viz.EdgeLayout(routing=routing, lane=50, lane_side="bottom")}
    )
    scene = viz.resolve_diagram(diagram, layout=layout)
    arrow = next(
        e
        for e in scene.scene["elements"]
        if e["type"] == "arrow" and e["data"]["id"] == "bc"
    )
    assert arrow["endArrowhead"] == "bar"
    assert arrow["strokeStyle"] == "dashed"
    assert bool(arrow["roundness"]) == (routing == "curved")
    assert max(p[1] for p in scene.edges["bc"]["points"]) > max(
        b.bottom for b in scene.nodes.values()
    )
    svg = viz.render_diagram(scene, tmp_path / f"{routing}.svg")
    assert ("Q " in svg.read_text()) == (routing == "curved")


def test_annotations_plain_math_hide_default_and_json_roundtrip(diagram, tmp_path):
    diagram = replace(diagram, edges=(replace(diagram.edges[0], label="original"),))
    layout = viz.DiagramLayout(
        nodes={
            "a": viz.NodeLayout(
                label=r"$\alpha$",
                annotations=(
                    viz.DiagramLabel(
                        "extra\n" + r"$\tau$", anchor="top", offset=(0, -20)
                    ),
                ),
            )
        },
        edges={"ab": viz.EdgeLayout(label="")},
    )
    path = layout.write(tmp_path / "layout.json")
    restored = viz.DiagramLayout.read(path)
    assert restored.to_dict() == layout.to_dict()
    scene = viz.resolve_diagram(diagram, layout=restored.to_dict())
    assert scene.scene["files"]
    sources = [e["data"].get("math_source") for e in scene.scene["elements"]]
    assert r"$\alpha$" in sources and r"$\tau$" in sources
    assert "edge:ab" not in scene.labels
    default = viz.resolve_diagram(diagram, layout=viz.DiagramLayout())
    assert "edge:ab" in default.labels


def test_move_preserves_other_positions_and_reroutes(diagram):
    scene = viz.resolve_diagram(diagram)
    moved = scene.move(
        "b", (scene.nodes["b"].center[0], scene.nodes["b"].center[1] + 400)
    )
    assert moved.nodes["a"].center == pytest.approx(scene.nodes["a"].center)
    assert moved.nodes["c"].center == pytest.approx(scene.nodes["c"].center)
    assert moved.edges["ab"]["points"] != scene.edges["ab"]["points"]
    with pytest.raises(ValueError, match="unknown node"):
        scene.move("missing", (0, 0))


def test_fixed_canvas_contain_and_error_do_not_move_pins(diagram, tmp_path):
    layout = viz.DiagramLayout(
        positions={"a": (200, 200)},
        canvas=viz.CanvasLayout(
            size=(200, 150), fit="contain", alignment=("start", "end")
        ),
    )
    scene = viz.resolve_diagram(diagram, layout=layout)
    assert (scene.canvas.width, scene.canvas.height) == (200, 150)
    assert scene.nodes["a"].center == pytest.approx((200, 200))
    path = viz.render_diagram(scene, tmp_path / "contain.png")
    assert Image.open(path).size == (200, 150)
    with pytest.raises(ValueError, match="exceed fixed canvas"):
        viz.resolve_diagram(
            diagram, layout=replace(layout, canvas=replace(layout.canvas, fit="error"))
        )


@pytest.mark.parametrize(
    "layout,match",
    [
        (viz.DiagramLayout(nodes={"missing": viz.NodeLayout()}), "unknown layout node"),
        (
            viz.DiagramLayout(edges={"missing": viz.EdgeLayout()}),
            "unknown layout connection",
        ),
        (viz.DiagramLayout(align_y=[["a", "missing"]]), "unknown IDs"),
        (
            viz.DiagramLayout(
                groups={"g": viz.GroupLayout(members=("a",), parent="missing")}
            ),
            "unknown parent",
        ),
        (
            viz.DiagramLayout(nodes={"a": viz.NodeLayout(max_size=(1, 1))}),
            "exceeds max_size",
        ),
        (
            viz.DiagramLayout(edges={"ab": viz.EdgeLayout(source_port="missing")}),
            "unknown node port",
        ),
    ],
)
def test_unknown_ids_and_impossible_sizes_fail(diagram, layout, match):
    with pytest.raises(ValueError, match=match):
        viz.resolve_diagram(diagram, layout=layout)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: viz.NodeLayout(position=(math.nan, 0)),
        lambda: viz.EdgeLayout(clearance=-1),
        lambda: viz.DiagramLabel("x", position=2),
        lambda: viz.CanvasLayout(size=(0, 100)),
        lambda: viz.NodeLayout(ports={"port": (0.5, 0.5)}),
        lambda: viz.LayoutConstraint("align_y", ("a", "a")),
        lambda: viz.DiagramLayout.from_dict({"schema": "other"}),
        lambda: viz.DiagramLayout.from_dict(
            {"schema": "snnlab.diagram-layout/v1", "typo": 1}
        ),
    ],
)
def test_bad_specifications_rejected(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()


def test_bundle_integration_conflicts_and_invalid_math(tmp_path):
    net = lang.Network("layout")
    source = net.input("drive", shape=("time", "batch", 2), signal_type="spikes")
    e = net.population("E", size=2, neuron=lang.COBA_LIF())
    net.connect(
        source,
        e.excitatory,
        name="drive_to_E",
        synapse=lang.AMPA(),
        weight=lang.Constant(0.1),
    )
    net.output("spikes", e.spikes)
    bundle = lang.compile(net)
    before = copy.deepcopy((bundle.graph, bundle.manifest))
    layout = viz.DiagramLayout(edges={"drive_to_E": viz.EdgeLayout(label=r"$J$")})
    path = bundle.visualise(tmp_path / "network.png", layout=layout)
    assert path.stat().st_size > 100
    assert before == (bundle.graph, bundle.manifest)
    with pytest.raises(ValueError, match="conflicting connection_labels"):
        bundle.visualise(
            tmp_path / "conflict.png",
            layout=layout,
            connection_labels={"drive_to_E": "other"},
        )
    invalid = viz.DiagramLayout(
        edges={"drive_to_E": viz.EdgeLayout(label=r"$\invalid_command$")}
    )
    old = path.read_bytes()
    with pytest.raises(ValueError, match="invalid connection label"):
        bundle.visualise(path, layout=invalid)
    assert path.read_bytes() == old


def test_group_gap_overrides_global_gap_and_corner_pins(diagram):
    diagram = replace(diagram, edges=())
    layout = viz.DiagramLayout(
        groups={"g": viz.GroupLayout(members=("a", "b"), gap=10)},
        nodes={
            "a": viz.NodeLayout(position=(0, 0), anchor="top_left"),
            "b": viz.NodeLayout(position=(250, 0), anchor="top_left"),
        },
    )
    scene = viz.resolve_diagram(diagram, layout=layout)
    assert scene.nodes["a"].x == pytest.approx(0)
    assert 10 <= scene.nodes["b"].x - scene.nodes["a"].right < 80
    changed = layout.with_positions({"c": (900, 900)})
    assert changed.nodes["a"].position == (0, 0)
    assert changed.nodes["c"].position == (900, 900)
    shorthand = viz.DiagramLayout(positions={"a": (100, 100)})
    assert shorthand.with_positions({"b": (500, 100)}).nodes["a"].position == (100, 100)


def test_parallel_edges_self_loop_and_semantic_styles(diagram):
    diagram = replace(
        diagram,
        edges=(
            viz.DiagramEdge("a", "b", id="ab"),
            viz.DiagramEdge(
                "b", "a", id="ba", role="inhibitory", connection="feedback"
            ),
            viz.DiagramEdge("c", "c", id="cc", role="training", frozen=True),
        ),
    )
    scene = viz.resolve_diagram(diagram)
    assert scene.edges["ab"]["points"][0] != scene.edges["ba"]["points"][-1]
    assert len(scene.edges["cc"]["points"]) >= 4
    edge = next(
        e
        for e in scene.scene["elements"]
        if e["type"] == "arrow" and e["data"]["id"] == "cc"
    )
    assert edge["strokeStyle"] == "dashed"
    assert edge["data"]["frozen"] is True


def test_preferences_weights_and_canvas_origin(diagram):
    diagram = replace(diagram, edges=())
    layout = viz.DiagramLayout(
        positions={"a": (200, 200), "c": (1800, 200)},
        constraints=(
            viz.LayoutConstraint(
                "align_y", ("a", "b"), required=False, priority=100, name="important"
            ),
            viz.LayoutConstraint(
                "below", ("b", "c"), required=False, priority=1, name="optional"
            ),
        ),
        canvas=viz.CanvasLayout(origin=(-100, -100)),
    )
    scene = viz.resolve_diagram(diagram, layout=layout)
    assert scene.nodes["a"].center[1] == pytest.approx(scene.nodes["b"].center[1])
    assert any(d["constraint"] == "optional" for d in scene.diagnostics)
    assert scene.canvas.x <= -100 and scene.canvas.y <= -100


def test_malformed_nesting_and_group_orders(diagram):
    cases = [
        {
            "x": viz.GroupLayout(members=("a", "b")),
            "y": viz.GroupLayout(members=("b", "c")),
        },
        {
            "x": viz.GroupLayout(members=("a",), parent="y"),
            "y": viz.GroupLayout(members=("a",), parent="x"),
        },
        {"x": viz.GroupLayout(members=("a", "b"), arrangement="row", order=("a",))},
        {"x": viz.GroupLayout(members=("a", "b"), order=("a", "b"))},
    ]
    for groups in cases:
        with pytest.raises(ValueError):
            viz.resolve_diagram(diagram, layout=viz.DiagramLayout(groups=groups))


def test_invalid_node_and_group_math_preserves_destination(diagram, tmp_path):
    destination = tmp_path / "drawing.svg"
    destination.write_text("existing")
    for layout in (
        viz.DiagramLayout(nodes={"a": viz.NodeLayout(label=r"$\badcommand$")}),
        viz.DiagramLayout(
            groups={"g": viz.GroupLayout(members=("a",), label=r"$\badcommand$")}
        ),
    ):
        with pytest.raises(ValueError, match="invalid connection label"):
            viz.render_diagram(diagram, destination, layout=layout)
        assert destination.read_text() == "existing"


def test_legacy_canvas_and_resolved_overrides_fail_explicitly(diagram, tmp_path):
    scene = viz.resolve_diagram(diagram)
    with pytest.raises(ValueError, match="legacy canvas/ratio"):
        viz.render_diagram(
            diagram,
            tmp_path / "x.png",
            layout=viz.DiagramLayout(),
            canvas_size=(1000, 1000),
        )
    with pytest.raises(ValueError, match="already owns"):
        viz.render_diagram(scene, tmp_path / "x.png", layout=viz.DiagramLayout())
    with pytest.raises(ValueError, match="already owns"):
        viz.render_diagram(
            scene, tmp_path / "x.png", theme=viz.DiagramTheme(ink="#000000")
        )
    with pytest.raises(ValueError, match="resolved diagram output"):
        viz.render_diagram(scene, tmp_path / "x.dot")


def test_group_pin_minimum_size_and_local_coordinates(diagram):
    diagram = replace(diagram, edges=())
    scene = viz.resolve_diagram(
        diagram,
        layout=viz.DiagramLayout(
            groups={
                "g": viz.GroupLayout(
                    members=("a", "b"),
                    position=(400, 400),
                    min_size=(600, 400),
                    max_size=(800, 600),
                )
            },
            nodes={
                "a": viz.NodeLayout(
                    position=(50, 120), relative_to="g", anchor="top_left"
                ),
                "b": viz.NodeLayout(
                    position=(320, 120), relative_to="g", anchor="top_left"
                ),
            },
        ),
    )
    box = scene.groups["g"]
    assert box.center == pytest.approx((400, 400))
    assert 600 - 1e-5 <= box.width <= 800 + 1e-5
    assert 400 - 1e-5 <= box.height <= 600 + 1e-5
    assert scene.nodes["a"].x - box.x == pytest.approx(50)


def test_routes_clear_external_annotations(diagram):
    scene = viz.resolve_diagram(
        diagram,
        layout=viz.DiagramLayout(
            positions={"a": (200, 300), "b": (600, 300), "c": (1000, 300)},
            nodes={
                "a": viz.NodeLayout(
                    annotations=(
                        viz.DiagramLabel("reserved", anchor="right", offset=(30, 0)),
                    )
                )
            },
        ),
    )
    bounds = scene.labels["node:a:0"]
    points = scene.edges["ab"]["points"]
    assert not any(
        crosses(a, b, bounds) for a, b in zip(points, points[1:], strict=False)
    )


def test_annotation_ids_do_not_collide_with_colon_node_ids(diagram):
    diagram = replace(
        diagram,
        nodes=(replace(diagram.nodes[0], id="a"), replace(diagram.nodes[1], id="a:0")),
        edges=(),
    )
    scene = viz.resolve_diagram(
        diagram,
        layout=viz.DiagramLayout(
            nodes={
                "a": viz.NodeLayout(
                    annotations=(
                        viz.DiagramLabel("extra", anchor="top", offset=(0, -20)),
                    )
                )
            }
        ),
    )
    identifiers = [e["id"] for e in scene.scene["elements"]]
    assert len(identifiers) == len(set(identifiers))
