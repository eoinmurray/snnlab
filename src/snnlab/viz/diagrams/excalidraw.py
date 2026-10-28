"""Editable Excalidraw scenes and local rendering through the installed CLI."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from .dot import _svg_id, diagram_to_dot
from .geometry import ResolvedDiagram
from .labels import math_labels_to_dot, prepare_math_labels
from .layout import graphviz, position_diagram
from .model import Diagram, DiagramTheme
from .native import _binding, _element, _id
from .spec import DiagramLayout

_SOURCE = "https://github.com/eoinmurray/snnlab"
_SCALE = 2
_DEFAULT_THEME = DiagramTheme()


def _text_elements(commands, identifier, height, *, groups=(), data=None):
    """Keep Graphviz's typography as editable text grouped with its owner."""
    size, colour, lines = 11.0, "#24292F", []
    for command in commands:
        if command.get("op") == "F":
            size = command["size"]
        elif command.get("op") == "c":
            colour = command["color"][:7]
        elif command.get("op") == "T":
            x, y = command["pt"]
            width = command["width"]
            if command["align"] == "c":
                x -= width / 2
            elif command["align"] == "r":
                x -= width
            lines.append(
                _element(
                    "text",
                    _id("text", f"{identifier}:{len(lines)}"),
                    x=x * _SCALE,
                    y=(height - y - size) * _SCALE,
                    width=width * _SCALE,
                    height=size * 1.25 * _SCALE,
                    groups=groups,
                    data=data,
                    strokeColor=colour,
                    text=command["text"],
                    originalText=command["text"],
                    fontSize=size * _SCALE,
                    fontFamily=2,
                    lineHeight=1.25,
                    textAlign="left",
                    verticalAlign="top",
                    containerId=None,
                    autoResize=True,
                )
            )
    return lines


def _math_elements(edge, row, identifier, height, assets, files, *, groups, data):
    """Embed measured MathText rows at their Graphviz-cleared label positions."""
    sources = [line for line in edge.label.splitlines() if line in assets]
    if not sources:
        return []
    cells = [
        command["points"]
        for command in row.get("_ldraw_", [])
        if command.get("op") in {"p", "P"}
    ]
    if len(cells) != len(sources):
        raise RuntimeError(f"Graphviz omitted a math label cell for {edge.label!r}")
    images = []
    for index, (source, cell) in enumerate(zip(sources, cells, strict=True)):
        asset = assets[source]
        png = asset.png.read_bytes()
        file_id = hashlib.sha256(png).hexdigest()
        files.setdefault(
            file_id,
            {
                "id": file_id,
                "mimeType": "image/png",
                "dataURL": "data:image/png;base64,"
                + base64.b64encode(png).decode("ascii"),
                "created": 1,
                "lastRetrieved": 1,
            },
        )
        xs, ys = zip(*cell, strict=True)
        center_x, center_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        images.append(
            _element(
                "image",
                _id("math", f"{identifier}:{index}"),
                x=(center_x - asset.width / 2) * _SCALE,
                y=(height - center_y - asset.height / 2) * _SCALE,
                width=asset.width * _SCALE,
                height=asset.height * _SCALE,
                groups=groups,
                data={**data, "math_source": source, "math_format": "mathtext"},
                strokeColor="transparent",
                fileId=file_id,
                status="saved",
                scale=[1, 1],
                crop=None,
            )
        )
    return images


def _rectangle(row, identifier, height, *, groups, data):
    if "pos" in row:
        cx, cy = map(float, row["pos"].split(","))
        width, box_height = float(row["width"]) * 72, float(row["height"]) * 72
        left, top = cx - width / 2, height - cy - box_height / 2
    else:
        left, bottom, right, upper = map(float, row["bb"].split(","))
        width, box_height = right - left, upper - bottom
        top = height - upper
    return _element(
        "rectangle",
        identifier,
        x=left * _SCALE,
        y=top * _SCALE,
        width=width * _SCALE,
        height=box_height * _SCALE,
        groups=groups,
        data=data,
        strokeColor=row.get("color", "#D8DEE3")[:7],
        backgroundColor=row.get("fillcolor", "transparent"),
        strokeWidth=float(row.get("penwidth", 1)) * _SCALE,
    )


def _points(row, height):
    points = []
    for command in row.get("_draw_", []):
        if command["op"] in {"b", "B", "L"}:
            for x, y in command["points"]:
                point = [x * _SCALE, (height - y) * _SCALE]
                if not points or points[-1] != point:
                    points.append(point)
    # Graphviz stops the spline at the arrow's base; native arrows need its tip.
    for token in row.get("pos", "").split():
        if token.startswith("e,"):
            _, x, y = token.split(",")
            points.append([float(x) * _SCALE, (height - float(y)) * _SCALE])
    simplified = []
    for point in points:
        if simplified and point == simplified[-1]:
            continue
        while len(simplified) >= 2:
            a, b = simplified[-2:]
            if not (a[0] == b[0] == point[0] or a[1] == b[1] == point[1]):
                break
            simplified.pop()
        simplified.append(point)
    if len(simplified) < 2:
        raise RuntimeError("Graphviz produced an edge without a usable route")
    return simplified


def _arrow(edge, row, identifier, boxes, height, *, groups, data):
    points = _points(row, height)
    x, y = points[0]
    arrowhead = {"inhibitory": "bar", "modulatory": "diamond"}.get(edge.role, "arrow")
    arrow = _element(
        "arrow",
        identifier,
        x=x,
        y=y,
        width=max(p[0] for p in points) - min(p[0] for p in points),
        height=max(p[1] for p in points) - min(p[1] for p in points),
        groups=groups,
        data=data,
        strokeColor=row["color"][:7],
        strokeWidth=edge.pen_width * _SCALE,
        strokeStyle=row.get("style", "solid"),
        points=[[px - x, py - y] for px, py in points],
        startArrowhead=None,
        endArrowhead=arrowhead,
        startBinding=_binding(boxes[edge.source], points[0]),
        endBinding=_binding(boxes[edge.target], points[-1]),
        elbowed=False,
    )
    for node in {edge.source, edge.target}:
        boxes[node]["boundElements"].append({"id": identifier, "type": "arrow"})
    return arrow


def _scene(diagram: Diagram, layout: dict, graph_digest: str, assets) -> dict:
    height = float(layout["bb"].split(",")[3])
    provenance = {
        "graph_digest": graph_digest,
        "view": diagram.metadata.get("view", "custom"),
    }
    groups, boundaries, texts, files = {}, [], [], {}
    for group in diagram.groups:
        group_id = _id("group", group.id)
        groups[group.id] = group_id
        row = next(
            r for r in layout["objects"] if r["name"] == "cluster_" + _svg_id(group.id)
        )
        data = {**provenance, "kind": "group", "id": group.id}
        boundaries.append(
            _rectangle(
                row, _id("boundary", group.id), height, groups=(group_id,), data=data
            )
        )
        texts.extend(
            _text_elements(
                row.get("_ldraw_", []), group_id, height, groups=(group_id,), data=data
            )
        )
    memberships = {
        node.id: tuple(groups[g.id] for g in diagram.groups if node.id in g.members)
        for node in diagram.nodes
    }
    rows = {r["name"]: r for r in layout.get("objects", []) if "pos" in r}
    boxes = {}
    for node in diagram.nodes:
        row = rows[node.id]
        node_groups = (_id("node_group", node.id), *memberships[node.id])
        data = {
            **provenance,
            "kind": "node",
            "id": node.id,
            "node_kind": node.kind,
            "title": node.title,
            "detail": node.detail,
            "badge": node.badge,
        }
        boxes[node.id] = _rectangle(
            row, _id("node", node.id), height, groups=node_groups, data=data
        )
        texts.extend(
            _text_elements(
                row.get("_ldraw_", []),
                boxes[node.id]["id"],
                height,
                groups=node_groups,
                data=data,
            )
        )
    edge_rows = {r["id"]: r for r in layout.get("edges", []) if "id" in r}
    arrows, occurrences = [], defaultdict(int)
    for index, edge in enumerate(diagram.edges):
        key = edge.id or f"{edge.source}:{edge.target}:{edge.role}:{edge.connection}"
        occurrence = occurrences[key]
        occurrences[key] += 1
        identifier = _id("edge", f"{key}:{occurrence}")
        dot_id = "e_" + _svg_id(edge.id) if edge.id is not None else f"e_layout_{index}"
        row = edge_rows[dot_id]
        edge_groups = (
            _id("edge_group", identifier),
            *(g for g in memberships[edge.source] if g in memberships[edge.target]),
        )
        data = {
            **provenance,
            "kind": "edge",
            "id": edge.id,
            "source": edge.source,
            "target": edge.target,
            "role": edge.role,
            "connection": edge.connection,
            "label": edge.label,
        }
        arrows.append(
            _arrow(edge, row, identifier, boxes, height, groups=edge_groups, data=data)
        )
        texts.extend(
            _text_elements(
                row.get("_ldraw_", []),
                identifier,
                height,
                groups=edge_groups,
                data=data,
            )
        )
        texts.extend(
            _math_elements(
                edge,
                row,
                identifier,
                height,
                assets,
                files,
                groups=edge_groups,
                data=data,
            )
        )
    texts.extend(
        _text_elements(layout.get("_ldraw_", []), "title", height, data=provenance)
    )
    return {
        "type": "excalidraw",
        "version": 2,
        "source": _SOURCE,
        "elements": [*boundaries, *arrows, *boxes.values(), *texts],
        "appState": {
            "viewBackgroundColor": layout.get("bgcolor", "#FFFFFF"),
            "gridSize": None,
            "exportBackground": True,
        },
        "files": files,
    }


def export_excalidraw(
    diagram: Diagram | ResolvedDiagram,
    path: str | Path,
    *,
    graph_digest: str | None = None,
    overwrite: bool = False,
    layout: DiagramLayout | dict | None = None,
    theme: DiagramTheme = _DEFAULT_THEME,
) -> Path:
    """Write a deterministic scene with editable text and bound native arrows."""
    output = Path(path)
    if output.suffix.lower() != ".excalidraw":
        raise ValueError("Excalidraw export path must end in .excalidraw")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"drawing already exists: {output}; pass overwrite=True to replace it"
        )
    if layout is not None or isinstance(diagram, ResolvedDiagram):
        from .resolved import resolve_diagram

        if isinstance(diagram, ResolvedDiagram):
            if layout is not None or theme != DiagramTheme() and theme != diagram.theme:
                raise ValueError("a resolved diagram already owns its layout and theme")
            resolved = diagram
        else:
            resolved = resolve_diagram(diagram, layout=layout, theme=theme)
        import copy

        scene = copy.deepcopy(resolved.scene)
        if graph_digest is not None:
            for element in scene["elements"]:
                element["customData"]["snnlab"]["graph_digest"] = graph_digest
        scene["snnlabGeometry"] = resolved.to_dict()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w" if overwrite else "x", encoding="utf-8") as stream:
            json.dump(scene, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        return output
    dot = diagram_to_dot(diagram, theme=theme)
    with tempfile.TemporaryDirectory(prefix="snnlab-excalidraw-") as temporary:
        assets = prepare_math_labels(diagram, theme, Path(temporary), scale=2)
        dot = math_labels_to_dot(dot, diagram, theme, assets)
        positioned = position_diagram(diagram, dot, Path(temporary))
        layout = json.loads(
            graphviz(["neato", "-n2", "-Tjson"], source=positioned.read_text())
        )
        scene = _scene(diagram, layout, graph_digest, assets)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(scene, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return output


def render_excalidraw(source: str | Path, path: str | Path, *, scale: int = 1) -> Path:
    """Render the saved drawing to PNG using an already installed Excalidraw CLI."""
    source, output = Path(source), Path(path)
    if source.suffix.lower() != ".excalidraw":
        raise ValueError("Excalidraw source path must end in .excalidraw")
    if not source.is_file():
        raise FileNotFoundError(f"Excalidraw drawing not found: {source}")
    if output.suffix.lower() != ".png":
        raise ValueError("Excalidraw image output must end in .png")
    if type(scale) is not int or not 1 <= scale <= 16:
        raise ValueError("Excalidraw scale must be an integer between 1 and 16")
    frame_args, render_scale = _saved_canvas_options(source, scale)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".snnlab-excalidraw-", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "render.png"
        try:
            result = subprocess.run(
                [
                    "excalidraw",
                    "render",
                    str(source.resolve()),
                    "--out",
                    str(staged.resolve()),
                    "--scale",
                    render_scale,
                    *frame_args,
                    "--max-width",
                    "16384",
                    "--max-height",
                    "16384",
                    "--fonts-url",
                    "none",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "Excalidraw CLI is not installed or not on PATH; install it separately "
                "with `bun install -g @excalidraw/cli`. Rendering also requires an installed "
                "Chrome, Chromium, Edge or Firefox browser."
            ) from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "Excalidraw rendering timed out after 60 seconds"
            ) from error
        if result.returncode:
            raise RuntimeError(
                f"Excalidraw rendering failed: {result.stderr.strip() or result.stdout.strip()}"
            )
        if not staged.is_file():
            raise RuntimeError("Excalidraw CLI did not produce a PNG image")
        with staged.open("rb") as stream:
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                raise RuntimeError("Excalidraw CLI did not produce a PNG image")
        staged.replace(output)
    return output


def _saved_canvas_options(source, scale):
    try:
        scene = json.loads(source.read_text())
    except (ValueError, UnicodeError):
        return [], str(scale)
    options = scene.get("snnlabExport") if isinstance(scene, dict) else None
    if not options:
        return [], str(scale)
    if not isinstance(options, dict):
        raise ValueError("invalid saved Excalidraw canvas settings")
    frame_id, factor = options.get("frame_id"), options.get("scale")
    if not isinstance(frame_id, str) or not any(
        e.get("id") == frame_id
        and e.get("type") == "frame"
        and not e.get("isDeleted", False)
        for e in scene.get("elements", [])
    ):
        raise ValueError(
            "saved canvas frame is missing; restore it or remove snnlabExport to render the whole scene"
        )
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, float))
        or not math.isfinite(factor)
        or not 0 < factor * scale <= 16
    ):
        raise ValueError(
            "saved canvas requires an Excalidraw render scale outside (0, 16]"
        )
    return ["--frame-id", frame_id], format(factor * scale, ".12g")
