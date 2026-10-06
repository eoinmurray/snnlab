import shlex
import shutil
import subprocess

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from snnlab.viz import (  # noqa: E402
    Diagram,
    DiagramEdge,
    DiagramGroup,
    DiagramNode,
    FigureGrid,
    FigureRect,
    FrameTimeline,
    Recording,
    RecordingError,
    Theme,
    diagram_to_dot,
    exponential_trace,
    grid_layout,
    render_diagram,
    representative_frame,
)


def test_recording_validates_shared_timeline():
    recording = Recording(0.25, {"spk_e": np.zeros((8, 3)), "v_e": np.zeros((8, 3))})
    assert recording.steps == 8
    assert recording.duration_ms == 2.0
    with pytest.raises(RecordingError):
        Recording(0.25, {"a": np.zeros((8, 1)), "b": np.zeros((7, 1))})


def test_grid_and_trace_are_backend_independent():
    assert grid_layout(5, columns=3).shape == (5, 2)
    events = np.zeros((4, 2))
    events[0, 0] = 1
    trace = exponential_trace(events, dt_ms=1, tau_ms=2)
    assert trace[1, 0] == pytest.approx(1)
    assert trace[2, 0] == pytest.approx(np.exp(-0.5))


def test_figure_grid_resolves_spans_from_top_left():
    grid = FigureGrid(
        rows=(1, 1),
        columns=(1, 2),
        bounds=FigureRect(0.0, 0.0, 1.0, 1.0),
        row_gap=0.1,
        column_gap=0.1,
    )
    grid.place("header", row=0, column=0, colspan=2)
    grid.place("left", row=1, column=0)
    grid.place("right", row=1, column=1)

    assert grid.rect("header").mpl == pytest.approx((0.0, 0.55, 1.0, 0.45))
    assert grid.rect("left").mpl == pytest.approx((0.0, 0.0, 0.3, 0.45))
    assert grid.rect("right").mpl == pytest.approx((0.4, 0.0, 0.6, 0.45))


def test_figure_grid_nests_and_reserves_regions():
    outer = FigureGrid(2, 1, bounds=(0.1, 0.1, 0.8, 0.8), row_gap=0.04)
    outer.place("content", row=0, column=0)
    outer.reserve("controls", row=1, column=0)
    nested = outer.subgrid("content", rows=1, columns=2, padding=0.1)
    nested.place("a", row=0, column=0)
    nested.place("b", row=0, column=1)

    assert nested.rect("a").x > outer.rect("content").x
    figure = outer.figure(figsize=(4, 3))
    with pytest.raises(ValueError, match="reserved region"):
        outer.add_axes(figure, "controls")
    plt.close(figure)


def test_figure_grid_rejects_overlapping_regions():
    grid = FigureGrid(2, 2)
    grid.place("wide", row=0, column=0, colspan=2)
    with pytest.raises(ValueError, match="overlaps"):
        grid.place("collision", row=0, column=1)


def test_figure_grid_uses_house_style_by_default():
    theme = Theme()
    assert theme.background == "#ffffff"
    assert theme.ink == "#1a1a1a"
    assert theme.accent == "#c8102e"
    assert theme.amber == "#e89400"


def test_timeline_and_representative_frame():
    timeline = FrameTimeline.sample(100, frames=10, dt_ms=0.25)
    assert timeline.steps[[0, -1]].tolist() == [0, 99]
    activity = np.asarray([[0, 0], [1, 0], [1, 2]])
    assert representative_frame(activity) == 2


def test_timeline_composes_slow_motion_repeats_and_holds():
    timeline = FrameTimeline.compose([(0, 9, 3), (5, 5, 2), (9, 0, 3)], dt_ms=0.25)
    assert timeline.steps.tolist() == [0, 4, 9, 5, 5, 9, 4, 0]




def test_diagram_contract_compiles_deterministically():
    diagram = Diagram(
        name="small",
        nodes=(
            DiagramNode("input", "Input", "8 channels", "spikes", kind="input"),
            DiagramNode("cell", "Cell", "E 8 · I 2", "component", kind="component"),
        ),
        edges=(DiagramEdge("input", "cell", role="excitatory"),),
        groups=(DiagramGroup("network", "Network", ("cell",)),),
    )

    assert diagram_to_dot(diagram) == diagram_to_dot(diagram)
    assert '"input" -> "cell"' in diagram_to_dot(diagram)
    assert 'subgraph "cluster_n_network"' in diagram_to_dot(diagram)
    assert 'style="filled"' in diagram_to_dot(diagram)
    assert 'fontname="Helvetica"' in diagram_to_dot(diagram)
    assert 'ratio="0.6"' in diagram_to_dot(diagram, height_to_width_ratio=0.6)

    assert "ratio=" not in diagram_to_dot(diagram)
    assert "ratio=" not in diagram_to_dot(diagram, height_to_width_ratio=None)

    with pytest.raises(ValueError, match="finite and positive"):
        diagram_to_dot(diagram, height_to_width_ratio=0)


def test_diagram_contract_rejects_unknown_references():
    with pytest.raises(ValueError, match="unknown node"):
        Diagram(
            name="broken",
            nodes=(DiagramNode("known", "Known", "", "node"),),
            edges=(DiagramEdge("known", "missing"),),
        )


def test_diagram_group_rejects_conflicting_layout():
    with pytest.raises(ValueError, match="both same_rank and same_row"):
        Diagram(
            name="conflict",
            nodes=(DiagramNode("cell", "Cell", "", "population"),),
            edges=(),
            groups=(DiagramGroup("group", "Group", ("cell",), same_rank=True, same_row=True),),
        )


def test_recurrent_population_rows_render_left_to_right(tmp_path):
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' is required for diagram rendering")
    nodes, edges, groups = [], [], []
    for name in ("a", "b"):
        nodes.extend((
            DiagramNode(f"{name}_drive", "Drive", "128 channels", "spikes", kind="input"),
            DiagramNode(f"{name}_e", "E", "80 neurons", "population", kind="population"),
            DiagramNode(f"{name}_i", "I", "20 neurons", "population", kind="population"),
        ))
        edges.extend((
            DiagramEdge(f"{name}_drive", f"{name}_e", role="excitatory"),
            DiagramEdge(f"{name}_e", f"{name}_i", role="excitatory", connection="recurrent"),
            DiagramEdge(f"{name}_i", f"{name}_e", role="inhibitory", connection="recurrent"),
        ))
        groups.append(DiagramGroup(name, name, (f"{name}_e", f"{name}_i"), same_row=True))
    edges.extend((
        DiagramEdge("a_e", "b_e", role="excitatory", connection="feedback", constraint=False),
        DiagramEdge("b_e", "a_e", role="excitatory", connection="feedback", constraint=False),
    ))
    diagram = Diagram("coupled", tuple(nodes), tuple(edges), tuple(groups))
    dot = render_diagram(diagram, tmp_path / "rows.dot")
    result = subprocess.run(["dot", "-Tplain", str(dot)], capture_output=True, text=True, check=True)
    rows = [shlex.split(line) for line in result.stdout.splitlines()]
    graph = next(row for row in rows if row[0] == "graph")
    assert float(graph[2]) > float(graph[3])
    positions = {row[1]: (float(row[2]), float(row[3])) for row in rows if row[0] == "node"}
    for name in ("a", "b"):
        drive, e, i = (positions[f"{name}_{kind}"] for kind in ("drive", "e", "i"))
        assert drive[0] < e[0] < i[0]
        assert e[1] == pytest.approx(i[1], abs=0.01)


def test_diagram_renderer_exports_svg_and_dot(tmp_path):
    diagram = Diagram(
        name="small",
        nodes=(DiagramNode("node", "Node", "one", "neutral"),),
        edges=(),
    )
    dot = render_diagram(diagram, tmp_path / "small.dot")
    assert dot.read_text() == diagram_to_dot(diagram)
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' is required for diagram rendering")
    svg = render_diagram(diagram, tmp_path / "small.svg")
    assert "n_node" in svg.read_text()


def test_diagram_aligns_external_inputs_and_population_targets(tmp_path):
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' is required for diagram rendering")
    diagram = Diagram(
        "columns",
        nodes=(
            DiagramNode("drive_e", "E afferent", "400 channels", "spikes", kind="input"),
            DiagramNode("drive_i", "I afferent", "400 channels", "spikes", kind="input"),
            DiagramNode("e", "E", "400 neurons", "population", kind="population"),
            DiagramNode("i", "I", "100 neurons", "population", kind="population"),
        ),
        edges=(
            DiagramEdge("drive_e", "e", role="excitatory"),
            DiagramEdge("drive_i", "i", role="excitatory"),
            DiagramEdge("e", "i", role="excitatory", connection="recurrent"),
            DiagramEdge("i", "e", role="inhibitory", connection="recurrent"),
        ),
        groups=(DiagramGroup("cell", "Circuit", ("e", "i"), same_rank=True),),
    )
    dot = render_diagram(diagram, tmp_path / "columns.dot")
    result = subprocess.run(["dot", "-Tplain", str(dot)], capture_output=True, text=True, check=True)
    rows = [shlex.split(line) for line in result.stdout.splitlines()]
    positions = {row[1]: (float(row[2]), float(row[3])) for row in rows if row[0] == "node"}
    assert positions["drive_e"][0] == pytest.approx(positions["drive_i"][0])
    assert positions["e"][0] == pytest.approx(positions["i"][0])
    assert positions["drive_e"][0] < positions["e"][0]
    assert positions["drive_e"][1] == pytest.approx(positions["e"][1])
    assert positions["drive_i"][1] == pytest.approx(positions["i"][1])


def test_wrapped_diagram_text_preserves_case_and_typographic_hierarchy(tmp_path):
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' is required for diagram rendering")
    import xml.etree.ElementTree as ET

    diagram = Diagram(
        "readable",
        nodes=(DiagramNode(
            "node", "Long population title", "400 units & conductance model", "spiking population"
        ),),
        edges=(),
    )
    root = ET.parse(render_diagram(diagram, tmp_path / "readable.svg")).getroot()
    text = root.findall(".//{http://www.w3.org/2000/svg}text")
    rendered = " ".join(element.text or "" for element in text)
    assert "Long population title" in rendered
    assert "400 units & conductance model" in rendered
    assert "spiking population" in rendered
    title = next(element for element in text if element.text == "Long population title")
    detail = next(element for element in text if "400 units" in (element.text or ""))
    assert title.attrib.get("font-weight") == "bold"
    assert detail.attrib.get("font-weight") != "bold"
    assert float(title.attrib["font-size"]) > float(detail.attrib["font-size"])
    assert min(float(element.attrib["font-size"]) for element in text) >= 10


def test_forward_path_and_training_annotations_remain_aligned(tmp_path):
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' is required for diagram rendering")
    # Component creation order differs from signal-flow order.
    diagram = Diagram(
        "training_layout",
        nodes=(
            DiagramNode("input", "Input", "64 channels", "spikes", kind="input"),
            DiagramNode("classifier", "Classifier", "10 units", "component", kind="component"),
            DiagramNode("hidden", "Hidden", "128 units", "component", kind="component"),
            DiagramNode("scores", "Scores", "10 classes", "output", kind="output"),
            DiagramNode("learned", "Learned", "2 tensors", "trainable", kind="training"),
            DiagramNode("frozen", "Recurrent", "2 tensors", "frozen", kind="training"),
        ),
        edges=(
            DiagramEdge("input", "hidden", role="excitatory"),
            DiagramEdge("hidden", "classifier", role="excitatory"),
            DiagramEdge("classifier", "scores", role="output"),
            DiagramEdge("learned", "classifier", role="training", constraint=False),
            DiagramEdge("learned", "hidden", role="training", constraint=False),
            DiagramEdge("frozen", "hidden", role="training", constraint=False, frozen=True),
        ),
    )
    dot = render_diagram(diagram, tmp_path / "training.dot")
    result = subprocess.run(["dot", "-Tplain", str(dot)], capture_output=True, text=True, check=True)
    rows = [shlex.split(line) for line in result.stdout.splitlines()]
    positions = {row[1]: (float(row[2]), float(row[3])) for row in rows if row[0] == "node"}
    baseline = positions["hidden"][1]
    for identifier in ("input", "classifier", "scores"):
        assert positions[identifier][1] == pytest.approx(baseline, abs=0.01)
    for annotation, anchor in (("learned", "classifier"), ("frozen", "hidden")):
        assert positions[annotation][0] == pytest.approx(positions[anchor][0], abs=0.01)
        assert positions[annotation][1] < positions[anchor][1]
    text = dot.read_text()
    assert 'arrowhead=none, style=dashed' in text
    assert 'arrowhead=vee, style=dotted' in text


def test_diagram_canvas_has_matching_svg_and_png_dimensions(tmp_path):
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' is required for diagram rendering")
    import struct
    import xml.etree.ElementTree as ET

    diagram = Diagram("canvas", (DiagramNode("cell", "Cell", "32 units", "spiking"),), ())
    svg = render_diagram(diagram, tmp_path / "canvas.svg", canvas_size=(640, 320))
    root = ET.parse(svg).getroot()
    assert (root.attrib["width"], root.attrib["height"]) == ("640px", "320px")
    assert root.attrib["viewBox"] == "0.00 0.00 320.00 160.00"
    graph = root.find("{http://www.w3.org/2000/svg}g")
    assert graph.attrib["transform"].startswith("scale(1 1)")
    png = render_diagram(diagram, tmp_path / "canvas.png", canvas_size=(640, 320))
    assert struct.unpack(">II", png.read_bytes()[16:24]) == (640, 320)
    scaled = render_diagram(diagram, tmp_path / "scaled.png", canvas_size=(640, 320), scale=2)
    assert struct.unpack(">II", scaled.read_bytes()[16:24]) == (1280, 640)
    with pytest.raises(ValueError, match="too small"):
        render_diagram(diagram, tmp_path / "clipped.svg", canvas_size=(10, 10))
    assert not (tmp_path / "clipped.svg").exists()


def test_diagram_canvas_rejects_invalid_dimensions():
    diagram = Diagram("canvas", (), ())
    for size in ((0, 320), (640, -1), (640.5, 320), (True, 320), (640,)):
        with pytest.raises(ValueError, match="two positive integers"):
            diagram_to_dot(diagram, canvas_size=size)
