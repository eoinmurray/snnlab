"""Editable graph preservation and installed-CLI rendering contracts."""

import base64
import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from snnlab import lang, viz
from tests._circuits import author_ping


@pytest.fixture
def bundle():
    net = lang.Network("Editable α & β", dt=0.1 * lang.ms)
    drive = net.input("title", shape=("time", "batch", 8), signal_type="spikes")
    cell = author_ping(
        net,
        name="local-circuit",
        n_e=8,
        n_i=2,
        source=drive,
        w_ee=lang.Constant(0.1),
    )
    net.output("direct_spikes", cell.E.spikes)
    prediction = lang.SpikeCount(source=cell.E.spikes, classes=2, name="readout")
    net.output("class_scores", prediction)
    training = lang.TrainSpec(
        objectives=[lang.CrossEntropy(prediction=prediction, target="labels")],
        parameter_groups=[
            lang.ParameterGroup(
                [p["id"] for p in net.parameters], name="learned", lr=1e-3
            )
        ],
        optimizer=lang.AdamW(),
    )
    return lang.compile(net, training=training)


@pytest.fixture
def graphviz():
    if not all(shutil.which(program) for program in ("dot", "neato")):
        pytest.skip("Graphviz is required for editable diagram export")


@pytest.mark.parametrize("view", ["circuit", "expanded", "training"])
def test_export_preserves_graph_groups_labels_and_bidirectional_bindings(
    bundle, graphviz, view, tmp_path
):
    before = copy.deepcopy((bundle.graph, bundle.training, bundle.manifest))
    labels = {bundle.graph["projections"][0]["id"]: "Excitation α"}
    visual = lang.diagram(bundle, view=view, connection_labels=labels)
    path = bundle.excalidraw_export(
        tmp_path / "nested" / "network.excalidraw",
        view=view,
        connection_labels=labels,
    )
    scene = json.loads(path.read_text())
    elements = scene["elements"]
    assert scene["type"] == "excalidraw"
    assert scene["version"] == 2
    assert not scene["files"]
    assert len({e["id"] for e in elements}) == len(elements)
    assert {e["type"] for e in elements} == {"rectangle", "arrow", "text"}
    boxes = {
        e["customData"]["snnlab"]["id"]: e
        for e in elements
        if e["type"] == "rectangle" and e["customData"]["snnlab"]["kind"] == "node"
    }
    assert set(boxes) == {n.id for n in visual.nodes}
    boundaries = [
        e
        for e in elements
        if e["type"] == "rectangle" and e["customData"]["snnlab"]["kind"] == "group"
    ]
    assert {e["customData"]["snnlab"]["id"] for e in boundaries} == {
        g.id for g in visual.groups
    }
    for group in visual.groups:
        boundary = next(
            e for e in boundaries if e["customData"]["snnlab"]["id"] == group.id
        )
        assert all(
            set(boundary["groupIds"]) <= set(boxes[n]["groupIds"])
            for n in group.members
        )
    arrows = [e for e in elements if e["type"] == "arrow"]
    assert len(arrows) == len(visual.edges)
    assert {
        e["customData"]["snnlab"]["id"]
        for e in arrows
        if e["customData"]["snnlab"]["id"]
    } == {e.id for e in visual.edges if e.id}
    assert any(e["type"] == "text" and e["text"] == "Excitation α" for e in elements)
    assert any(e["type"] == "text" and "Editable α & β" in e["text"] for e in elements)
    for arrow in arrows:
        data = arrow["customData"]["snnlab"]
        for end, node in (("startBinding", "source"), ("endBinding", "target")):
            box = boxes[data[node]]
            assert arrow[end]["elementId"] == box["id"]
            assert {"id": arrow["id"], "type": "arrow"} in box["boundElements"]
        assert arrow["endArrowhead"] == (
            "bar" if data["role"] == "inhibitory" else "arrow"
        )
        assert data["graph_digest"] == bundle.manifest["graph_digest"]
        assert data["view"] == view
    assert any(
        len(e["points"]) >= 4
        for e in arrows
        if e["customData"]["snnlab"]["source"] == e["customData"]["snnlab"]["target"]
    )
    assert (bundle.graph, bundle.training, bundle.manifest) == before


def test_export_is_deterministic_and_protects_user_edits(bundle, graphviz, tmp_path):
    path = bundle.excalidraw_export(tmp_path / "first.excalidraw")
    other = bundle.excalidraw_export(tmp_path / "second.excalidraw")
    assert path.read_bytes() == other.read_bytes()
    path.write_text("user edits")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        bundle.excalidraw_export(path)
    assert path.read_text() == "user edits"
    bundle.excalidraw_export(path, overwrite=True)
    assert path.read_bytes() == other.read_bytes()


def test_math_labels_embed_deduplicated_images_and_retain_plain_text(
    bundle, graphviz, tmp_path
):
    projection_ids = [p["id"] for p in bundle.graph["projections"]]
    fraction = r"$\frac{g_I}{\tau_I}$"
    subscript = r"$J_{E\to I}$"
    labels = {
        projection_ids[0]: f"Gain\n{fraction}\n{subscript}\nHz",
        projection_ids[1]: fraction,
    }
    path = bundle.excalidraw_export(
        tmp_path / "math.excalidraw", connection_labels=labels
    )
    scene = json.loads(path.read_text())
    images = [e for e in scene["elements"] if e["type"] == "image"]
    assert len(images) == 3
    assert len(scene["files"]) == 2
    assert {e["customData"]["snnlab"]["math_source"] for e in images} == {
        fraction,
        subscript,
    }
    for image in images:
        data = image["customData"]["snnlab"]
        file = scene["files"][image["fileId"]]
        assert data["label"] == labels[data["id"]]
        assert data["math_format"] == "mathtext"
        assert data["graph_digest"] == bundle.manifest["graph_digest"]
        assert file["id"] == image["fileId"]
        assert file["mimeType"] == "image/png"
        prefix, encoded = file["dataURL"].split(",", 1)
        assert prefix == "data:image/png;base64"
        assert base64.b64decode(encoded, validate=True).startswith(b"\x89PNG\r\n\x1a\n")
        arrow = next(
            e
            for e in scene["elements"]
            if e["type"] == "arrow" and e["customData"]["snnlab"]["id"] == data["id"]
        )
        assert image["groupIds"] == arrow["groupIds"]
        assert image["width"] > 0 and image["height"] > 0
    texts = [e["text"] for e in scene["elements"] if e["type"] == "text"]
    assert "Gain" in texts and "Hz" in texts
    assert not any("$" in text for text in texts)
    other = bundle.excalidraw_export(
        tmp_path / "again.excalidraw", connection_labels=labels
    )
    assert path.read_bytes() == other.read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "again.excalidraw",
        "math.excalidraw",
    ]


def test_invalid_math_does_not_replace_an_existing_drawing(bundle, graphviz, tmp_path):
    path = tmp_path / "finished.excalidraw"
    path.write_text("user-edited drawing")
    projection_id = bundle.graph["projections"][0]["id"]
    with pytest.raises(ValueError, match="invalid connection label"):
        bundle.excalidraw_export(
            path,
            connection_labels={projection_id: r"$\not_a_real_command{g}$"},
            overwrite=True,
        )
    assert path.read_text() == "user-edited drawing"


def test_export_validation_does_not_require_rendering_tools(bundle, tmp_path):
    with pytest.raises(ValueError, match=".excalidraw"):
        bundle.excalidraw_export(tmp_path / "network.png")
    with pytest.raises(ValueError, match="view"):
        bundle.excalidraw_export(tmp_path / "network.excalidraw", view="missing")
    with pytest.raises(ValueError, match="known projection"):
        bundle.excalidraw_export(
            tmp_path / "network.excalidraw", connection_labels={"missing": "label"}
        )


def test_missing_graphviz_has_actionable_error(bundle, monkeypatch, tmp_path):
    def missing(*args, **kwargs):
        raise FileNotFoundError("dot")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(FileNotFoundError, match="install Graphviz separately"):
        bundle.excalidraw_export(tmp_path / "network.excalidraw")
    assert not (tmp_path / "network.excalidraw").exists()


def test_render_validation(tmp_path):
    source = tmp_path / "source.excalidraw"
    with pytest.raises(FileNotFoundError, match="drawing not found"):
        viz.render_excalidraw(source, tmp_path / "out.png")
    source.write_text("{}")
    with pytest.raises(ValueError, match=".png"):
        viz.render_excalidraw(source, tmp_path / "out.svg")
    for scale in (0, 17, True, 1.5):
        with pytest.raises(ValueError, match="scale"):
            viz.render_excalidraw(source, tmp_path / "out.png", scale=scale)


@pytest.mark.parametrize("failure", ["missing", "browser", "timeout", "no_png"])
def test_render_errors_preserve_source_and_previous_image(
    failure, monkeypatch, tmp_path
):
    source = tmp_path / "user edited.excalidraw"
    output = tmp_path / "finished.png"
    source.write_text('{"elements": [], "userEdit": "preserve"}')
    output.write_bytes(b"previous image")
    before = source.read_bytes()

    def run(args, **kwargs):
        assert args[:3] == ["excalidraw", "render", str(source.resolve())]
        if failure == "missing":
            raise FileNotFoundError("excalidraw")
        if failure == "browser":
            return subprocess.CompletedProcess(
                args, 1, "", "No installed browser found"
            )
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 60)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises((FileNotFoundError, RuntimeError)) as error:
        viz.render_excalidraw(source, output)
    assert {
        "missing": "bun install -g",
        "browser": "browser",
        "timeout": "timed out",
        "no_png": "PNG",
    }[failure] in str(error.value)
    assert source.read_bytes() == before
    assert output.read_bytes() == b"previous image"
    assert not list(tmp_path.glob(".snnlab-excalidraw-*"))


@pytest.mark.integration
def test_real_cli_renders_user_edited_document_without_changing_it(
    bundle, graphviz, tmp_path
):
    if not shutil.which("excalidraw"):
        pytest.skip("Installed Excalidraw CLI and browser are required")
    original = bundle.excalidraw_export(
        tmp_path / "original" / "network.excalidraw",
        connection_labels={
            bundle.graph["projections"][0]["id"]: r"$\frac{g_I}{\tau_I}$"
        },
    )
    scene = json.loads(original.read_text())
    assert any(e["type"] == "image" for e in scene["elements"])
    # Keep only the document: rendering must not depend on exported image files.
    source = tmp_path / "relocated.excalidraw"
    shutil.move(original, source)
    shutil.rmtree(original.parent)
    first = viz.render_excalidraw(source, tmp_path / "first.png")
    scene = json.loads(source.read_text())
    title = next(
        e
        for e in scene["elements"]
        if e["type"] == "text" and "Editable α" in e["text"]
    )
    title["text"] = title["originalText"] = "User-edited network"
    source.write_text(json.dumps(scene))
    edited = source.read_bytes()
    second = viz.render_excalidraw(source, tmp_path / "second.png")
    assert source.read_bytes() == edited
    assert first.read_bytes() != second.read_bytes()
    assert second.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert Path(second).stat().st_size > 1000
