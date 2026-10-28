"""Resolve one measured presentation scene for static and editable exports."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import textwrap
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from .dot import _box_colours, diagram_to_dot
from .geometry import Box, ResolvedDiagram
from .labels import _render_math
from .layout import graphviz, position_diagram
from .model import Diagram, DiagramTheme
from .native import _binding, _element, _id
from .paths import SIDES, along, route, union
from .placement import ancestors, group_specs, node_specs, overlaps, place
from .spec import DiagramLabel, DiagramLayout, EdgeLayout, merged

_DEFAULT_THEME = DiagramTheme()


class _Labels:
    def __init__(self, theme, directory):
        from matplotlib.backends.backend_agg import RendererAgg

        self.renderer = RendererAgg(1, 1, 72)
        self.theme, self.directory = theme, directory
        self.assets, self.files, self.vectors = {}, {}, {}

    def rows(self, text, size, colour, columns=26):
        from matplotlib.font_manager import FontProperties

        result = []
        if not text:
            return result
        for paragraph in text.split("\n"):
            lines = (
                [paragraph]
                if "$" in paragraph
                else textwrap.wrap(paragraph, width=columns, break_long_words=False)
                or [""]
            )
            for line in lines:
                if "$" in line:
                    key = line, size, colour
                    if key not in self.assets:
                        theme = replace(
                            self.theme, secondary_size=size + 1, muted=colour
                        )
                        self.assets[key] = _render_math(
                            line, theme, self.directory, scale=2
                        )
                    asset = self.assets[key]
                    png = asset.png.read_bytes()
                    file_id = hashlib.sha256(png).hexdigest()
                    self.files[file_id] = {
                        "id": file_id,
                        "mimeType": "image/png",
                        "dataURL": "data:image/png;base64,"
                        + base64.b64encode(png).decode("ascii"),
                        "created": 1,
                        "lastRetrieved": 1,
                    }
                    self.vectors[file_id] = asset.svg
                    result.append(
                        {
                            "text": line,
                            "width": asset.width,
                            "height": asset.height + 4,
                            "size": size,
                            "colour": colour,
                            "file_id": file_id,
                            "image_height": asset.height,
                        }
                    )
                else:
                    width, _, _ = self.renderer.get_text_width_height_descent(
                        line,
                        FontProperties(family=self.theme.font_name, size=size),
                        False,
                    )
                    result.append(
                        {
                            "text": line,
                            "width": width,
                            "height": size * 1.3,
                            "size": size,
                            "colour": colour,
                        }
                    )
        return result

    @staticmethod
    def size(rows):
        return max((r["width"] for r in rows), default=0), sum(
            r["height"] for r in rows
        )

    @staticmethod
    def emit(rows, box, identifier, *, groups=(), data=None, align="left"):
        elements = []
        y = box.y
        for i, row in enumerate(rows):
            if not row["text"] and "file_id" not in row:
                y += row["height"]
                continue
            x = (
                box.x
                + (box.width - row["width"])
                * {"left": 0, "center": 0.5, "right": 1}[align]
            )
            extra = (
                {
                    "fileId": row["file_id"],
                    "status": "saved",
                    "scale": [1, 1],
                    "crop": None,
                }
                if "file_id" in row
                else {
                    "text": row["text"],
                    "originalText": row["text"],
                    "fontSize": row["size"],
                    "fontFamily": 2,
                    "lineHeight": 1.3,
                    "textAlign": "left",
                    "verticalAlign": "top",
                    "containerId": None,
                    "autoResize": True,
                }
            )
            metadata = dict(data or {})
            if "file_id" in row:
                metadata.update(math_source=row["text"], math_format="mathtext")
            elements.append(
                _element(
                    "image" if "file_id" in row else "text",
                    _id("label", f"{identifier}:{i}"),
                    x=x,
                    y=y + (2 if "file_id" in row else 0),
                    width=row["width"],
                    height=row.get("image_height", row["height"]),
                    groups=groups,
                    data=metadata,
                    strokeColor=row["colour"],
                    **extra,
                )
            )
            y += row["height"]
        return elements


def _card(node, settings, labels, theme):
    columns = settings.wrap_columns or theme.wrap_columns
    _, accent = _box_colours(node, theme)
    fields = [
        (node.kind.upper(), 20, accent),
        (
            (settings.label.text if settings.label is not None else node.title),
            theme.label_size * 2,
            theme.ink,
        ),
    ]
    fields.extend(
        (line, theme.secondary_size * 2, theme.muted)
        for line in node.detail.splitlines()
    )
    fields = fields[: 4 if node.kind == "population" else 3]
    rows = []
    for text, size, colour in fields:
        if not text:
            continue
        if rows:
            rows.append(
                {"text": "", "width": 0, "height": 8, "size": 1, "colour": colour}
            )
        rows.extend(labels.rows(text, size, colour, columns))
    width, height = labels.size(rows)
    pad = 24 if settings.padding is None else settings.padding
    size = max(width + pad * 2, 80), max(height + pad * 2, 60)
    if settings.min_size:
        size = tuple(max(a, b) for a, b in zip(size, settings.min_size, strict=True))
    if settings.max_size and any(
        a > b + 1e-6 for a, b in zip(size, settings.max_size, strict=True)
    ):
        raise ValueError(
            f"node {node.id!r} content exceeds max_size; increase it or wrap the label"
        )
    return rows, size, pad


def _initial(diagram, direction, directory):
    dot = diagram_to_dot(diagram)
    path = position_diagram(diagram, dot, directory)
    data = json.loads(graphviz(["neato", "-n2", "-Tjson"], source=path.read_text()))
    height = float(data["bb"].split(",")[3])
    result = {}
    for row in data.get("objects", []):
        if "pos" not in row:
            continue
        x, y = map(float, row["pos"].split(","))
        point = (x * 2, (height - y) * 2)
        if direction == "left":
            point = (-point[0], point[1])
        elif direction == "down":
            point = (point[1], point[0])
        elif direction == "up":
            point = (point[1], -point[0])
        result[row["name"]] = point
    return result


def _edge_keys(diagram):
    keys = [
        e.id if e.id is not None else f"edge:{i}" for i, e in enumerate(diagram.edges)
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "layout requires unique connection IDs (anonymous connections use edge:<index>)"
        )
    return keys


def _label_box(rows, point, settings, labels):
    w, h = labels.size(rows)
    x, y = point[0] + settings.offset[0], point[1] + settings.offset[1]
    return Box(
        x - w * {"left": 1, "right": 0}.get(settings.anchor, 0.5),
        y - h * {"top": 1, "bottom": 0}.get(settings.anchor, 0.5),
        w,
        h,
    )


def _place_label(rows, points, settings, labels, obstacles, wires, name):
    fractions = (
        [settings.position]
        if settings.position is not None
        else [0.5, 0.35, 0.65, 0.2, 0.8]
    )
    explicit = settings.position is not None or settings.offset != (0, 0)
    width, height = labels.size(rows)
    shifts = (
        [(0, 0)]
        if explicit
        else [
            (dx * factor, dy * factor)
            for factor in (1, 2, 3)
            for dx, dy in (
                (-(width / 2 + 12), 0),
                (width / 2 + 12, 0),
                (0, -(height / 2 + 12)),
                (0, height / 2 + 12),
            )
        ]
    )
    for fraction in fractions:
        point = along(points, fraction)
        for dx, dy in shifts:
            candidate = _label_box(
                rows, (point[0] + dx, point[1] + dy), settings, labels
            )
            if not settings.avoid_collisions or not any(
                overlaps(candidate.expanded(4), b) for b in obstacles
            ):
                # A label must clear wires as well as boxes unless explicitly placed.
                from .paths import crosses

                if not explicit and any(
                    crosses(a, b, candidate.expanded(2))
                    for path in wires
                    for a, b in zip(path, path[1:], strict=False)
                ):
                    continue
                return candidate
    raise ValueError(
        f"no collision-free placement for label {name!r}; set its position/offset, change the layout or use avoid_collisions=False"
    )


def _canvas(spec, content):
    padded = content.expanded(spec.padding)
    if spec.size is None:
        bounds = union([padded, Box(*spec.origin, 1, 1)])
        return bounds, bounds
    canvas = Box(*spec.origin, *spec.size)
    fits = (
        padded.x >= canvas.x - 1e-5
        and padded.y >= canvas.y - 1e-5
        and padded.right <= canvas.right + 1e-5
        and padded.bottom <= canvas.bottom + 1e-5
    )
    if fits:
        return canvas, canvas
    if spec.fit == "error":
        raise ValueError(f"content bounds {padded} exceed fixed canvas {canvas}")
    if spec.fit == "expand":
        bounds = union([padded, canvas])
        return bounds, bounds
    factor = min(canvas.width / padded.width, canvas.height / padded.height)
    view_w, view_h = canvas.width / factor, canvas.height / factor
    ax, ay = ({"start": 0, "center": 0.5, "end": 1}[x] for x in spec.alignment)
    return canvas, Box(
        padded.x - (view_w - padded.width) * ax,
        padded.y - (view_h - padded.height) * ay,
        view_w,
        view_h,
    )


def _edge_style(edge, theme):
    colours = {
        "inhibitory": theme.inhibitory,
        "signal": theme.signal,
        "output": theme.output_line,
        "training": theme.muted if edge.frozen else theme.training_line,
    }
    style = (
        "dashed"
        if edge.role == "modulatory"
        or edge.connection == "feedback"
        or edge.role == "training"
        and edge.frozen
        else "dotted"
        if edge.role == "training"
        else "solid"
    )
    return colours.get(edge.role, theme.ink), style


def _resolve(diagram, spec, theme, directory):
    known, keys = _validate_ids(diagram, spec)
    groups = group_specs(diagram, spec)
    nodes = node_specs(diagram, spec, groups)
    labels = _Labels(theme, directory)
    cards = {n.id: _card(n, nodes[n.id], labels, theme) for n in diagram.nodes}
    group_rows = {
        k: labels.rows(
            g.label.text if g.label is not None else k,
            theme.secondary_size * 2,
            theme.muted,
        )
        for k, g in groups.items()
    }
    initial = _initial(diagram, spec.direction or spec.canvas.direction, directory)
    boxes, diagnostics = place(
        diagram,
        spec,
        groups,
        nodes,
        {k: c[1] for k, c in cards.items()},
        initial,
        {k: labels.size(v) for k, v in group_rows.items()},
    )
    elements, label_bounds = [], {}
    provenance = {"view": diagram.metadata.get("view", "custom")}
    group_ids = {k: _id("group", k) for k in groups}
    for k in sorted(groups, key=lambda k: len(ancestors(k, groups))):
        box, group = boxes[k], groups[k]
        memberships = tuple(group_ids[g] for g in [k, *ancestors(k, groups)])
        data = {**provenance, "kind": "group", "id": k}
        elements.append(
            _element(
                "rectangle",
                _id("boundary", k),
                **asdict(box),
                groups=memberships,
                data=data,
                strokeColor=theme.line,
                backgroundColor="transparent",
            )
        )
        pad = 24 if group.padding is None else group.padding
        w, h = labels.size(group_rows[k])
        point = (
            box.x + pad + w / 2,
            box.bottom - pad - h / 2
            if group.title_position == "bottom"
            else box.y + pad + h / 2,
        )
        label_spec = group.label or DiagramLabel(k)
        bounds = _label_box(group_rows[k], point, label_spec, labels)
        if label_spec.position is not None:
            raise ValueError(
                "group labels use title_position and offset; position is an edge fraction"
            )
        if (
            bounds.x < box.x
            or bounds.right > box.right
            or bounds.y < box.y
            or bounds.bottom > box.bottom
        ):
            raise ValueError(f"group {k!r} label offset exceeds its boundary")
        if label_spec.avoid_collisions and any(
            overlaps(bounds, boxes[n]) for n in known
        ):
            raise ValueError(f"group {k!r} label overlaps a node")
        label_bounds[f"group:{k}"] = bounds
        elements.extend(
            labels.emit(
                group_rows[k], bounds, f"group:{k}", groups=memberships, data=data
            )
        )
    node_elements, annotations = [], []
    native_boxes, ports = {}, {}
    for node in diagram.nodes:
        k, box, settings = node.id, boxes[node.id], nodes[node.id]
        fill, border = _box_colours(node, theme)
        memberships = (
            _id("node_group", k),
            *(group_ids[g] for g, group in groups.items() if k in group.members),
        )
        data = {
            **provenance,
            "kind": "node",
            "id": k,
            "node_kind": node.kind,
            "title": node.title,
            "detail": node.detail,
            "badge": node.badge,
            "classes": list(node.classes),
        }
        native_boxes[k] = _element(
            "rectangle",
            _id("node", k),
            **asdict(box),
            groups=memberships,
            data=data,
            strokeColor=border,
            backgroundColor=fill,
        )
        rows, _, pad = cards[k]
        bounds = Box(
            box.x + pad, box.y + pad, box.width - pad * 2, box.height - pad * 2
        )
        node_elements.extend(
            labels.emit(
                rows,
                bounds,
                f"node:{k}",
                groups=memberships,
                data=data,
                align=settings.content_align or "left",
            )
        )
        if settings.label is not None and (
            settings.label.position is not None
            or settings.label.offset != (0, 0)
            or settings.label.anchor != "center"
        ):
            raise ValueError(
                "node title labels use content_align; use annotations for external label placement"
            )
        for i, annotation in enumerate(settings.annotations or ()):
            if annotation.position is not None:
                raise ValueError(
                    "node annotations use anchor and offset; position is an edge fraction"
                )
            annotation_rows = labels.rows(
                annotation.text, theme.secondary_size * 2, theme.muted
            )
            u, v = SIDES.get(annotation.anchor, (0.5, 0.5))
            point = box.x + box.width * u, box.y + box.height * v
            bounds = _label_box(annotation_rows, point, annotation, labels)
            if annotation.avoid_collisions and any(
                overlaps(bounds, obstacle)
                for obstacle in [*(boxes[n] for n in known), *label_bounds.values()]
            ):
                raise ValueError(f"node {k!r} annotation overlaps a node or label")
            label_bounds[f"node:{k}:{i}"] = bounds
            annotations.extend(
                labels.emit(
                    annotation_rows,
                    bounds,
                    f"annotation:{k}:{i}",
                    groups=memberships,
                    data={**data, "annotation": i},
                )
            )
        ports[k] = {
            name: (box.x + p[0] * box.width, box.y + p[1] * box.height)
            for name, p in {**SIDES, **(settings.ports or {})}.items()
        }
    # Route all wires before placing labels so annotations can clear every wire.
    counts = Counter(tuple(sorted((e.source, e.target))) for e in diagram.edges)
    occurrences, edge_geometry, edge_elements, edge_specs = defaultdict(int), {}, [], {}
    for key, edge in zip(keys, diagram.edges, strict=True):
        settings = merged(spec.defaults.edge, spec.edges.get(key, EdgeLayout()))
        edge_specs[key] = settings
        pair = tuple(sorted((edge.source, edge.target)))
        shift = (
            occurrences[pair] - (counts[pair] - 1) / 2
        ) * spec.defaults.parallel_gap
        occurrences[pair] += 1
        points, source_side, target_side = route(
            edge,
            settings,
            boxes,
            nodes,
            shift=shift,
            extra_obstacles=tuple(label_bounds.values()),
        )
        identifier = _id("edge", key)
        memberships = (
            _id("edge_group", key),
            *(
                group_ids[g]
                for g, group in groups.items()
                if edge.source in group.members and edge.target in group.members
            ),
        )
        edge_label = (
            settings.label if settings.label is not None else DiagramLabel(edge.label)
        )
        data = {
            **provenance,
            "kind": "edge",
            "id": key,
            "source": edge.source,
            "target": edge.target,
            "role": edge.role,
            "connection": edge.connection,
            "label": edge_label.text,
            "frozen": edge.frozen,
            "classes": list(edge.classes),
        }
        x, y = points[0]
        element = _element(
            "arrow",
            identifier,
            x=x,
            y=y,
            width=max(p[0] for p in points) - min(p[0] for p in points),
            height=max(p[1] for p in points) - min(p[1] for p in points),
            groups=memberships,
            data=data,
            points=[[px - x, py - y] for px, py in points],
            strokeColor=_edge_style(edge, theme)[0],
            strokeWidth=edge.pen_width * 2,
            strokeStyle=_edge_style(edge, theme)[1],
            startArrowhead=None,
            endArrowhead={"inhibitory": "bar", "modulatory": "diamond"}.get(
                edge.role, "arrow"
            ),
            startBinding=_binding(native_boxes[edge.source], points[0]),
            endBinding=_binding(native_boxes[edge.target], points[-1]),
            elbowed=False,
            roundness={"type": 2} if settings.routing == "curved" else None,
        )
        for k in {edge.source, edge.target}:
            native_boxes[k]["boundElements"].append({"id": identifier, "type": "arrow"})
        edge_elements.append(element)
        edge_geometry[key] = {
            "source": edge.source,
            "target": edge.target,
            "points": points,
            "routing": settings.routing or "orthogonal",
            "source_port": settings.source_port or source_side,
            "target_port": settings.target_port or target_side,
        }
    occupied = [boxes[k] for k in known] + list(label_bounds.values())
    wires = [e["points"] for e in edge_geometry.values()]
    for key, edge, element in zip(keys, diagram.edges, edge_elements, strict=True):
        settings = edge_specs[key]
        text = (
            settings.label if settings.label is not None else DiagramLabel(edge.label)
        )
        if not text.text:
            continue
        rows = labels.rows(
            text.text,
            (theme.secondary_size - 1) * 2,
            theme.muted,
            min(14, theme.wrap_columns),
        )
        bounds = _place_label(
            rows, edge_geometry[key]["points"], text, labels, occupied, wires, key
        )
        occupied.append(bounds.expanded(4))
        label_bounds[f"edge:{key}"] = bounds
        node_elements.extend(
            labels.emit(
                rows,
                bounds,
                f"edge:{key}",
                groups=element["groupIds"],
                data=element["customData"]["snnlab"],
            )
        )
    title = diagram.title or diagram.name.replace("_", " ")
    title = title[:1].upper() + title[1:]
    content = union(
        [
            *boxes.values(),
            *label_bounds.values(),
            *(
                Box(p[0], p[1], 0, 0)
                for e in edge_geometry.values()
                for p in e["points"]
            ),
        ]
    )
    if title:
        rows = labels.rows(title, theme.title_size * 2, theme.ink, 100)
        w, h = labels.size(rows)
        bounds = Box(content.x, content.y - h - 24, w, h)
        elements.extend(labels.emit(rows, bounds, "title", data=provenance))
        label_bounds["title"] = bounds
        content = union([content, bounds])
    canvas, viewport = _canvas(spec.canvas, content)
    frame_id = _id("frame", "canvas")
    drawing_elements = [
        *elements,
        *edge_elements,
        *native_boxes.values(),
        *node_elements,
        *annotations,
    ]
    frame = _canvas_frame(drawing_elements, viewport, provenance, frame_id)
    scene = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://github.com/eoinmurray/snnlab",
        "elements": [frame, *drawing_elements],
        "appState": {
            "viewBackgroundColor": theme.background,
            "gridSize": None,
            "exportBackground": True,
        },
        "files": labels.files,
        "snnlabLayout": spec.to_dict(),
        "snnlabExport": {"frame_id": frame_id, "scale": canvas.width / viewport.width},
    }
    return ResolvedDiagram(
        diagram,
        spec,
        theme,
        {n.id: boxes[n.id] for n in diagram.nodes},
        {k: boxes[k] for k in groups},
        edge_geometry,
        label_bounds,
        ports,
        canvas,
        viewport,
        tuple(diagnostics),
        scene,
        labels.vectors,
    )


def resolve_diagram(
    diagram: Diagram,
    *,
    layout: DiagramLayout | dict | None = None,
    theme: DiagramTheme = _DEFAULT_THEME,
) -> ResolvedDiagram:
    """Measure content, solve placement, route connections and place annotations."""
    if not isinstance(diagram, Diagram):
        raise TypeError("resolve_diagram requires a Diagram")
    if not diagram.nodes:
        raise ValueError("cannot resolve an empty diagram")
    spec = (
        DiagramLayout()
        if layout is None
        else DiagramLayout.from_dict(layout)
        if isinstance(layout, dict)
        else layout
    )
    if not isinstance(spec, DiagramLayout):
        raise TypeError("layout must be DiagramLayout or a versioned mapping")
    with tempfile.TemporaryDirectory(prefix="snnlab-layout-") as temporary:
        return _resolve(diagram, spec, theme, Path(temporary))


def _validate_ids(diagram, spec):
    known = {n.id for n in diagram.nodes}
    if unknown := (spec.nodes.keys() | spec.positions.keys()) - known:
        raise ValueError(f"unknown layout node IDs: {sorted(unknown)}")
    keys = _edge_keys(diagram)
    if unknown := spec.edges.keys() - set(keys):
        raise ValueError(f"unknown layout connection IDs: {sorted(unknown)}")
    return known, keys


def _canvas_frame(drawing_elements, viewport, provenance, frame_id):
    for element in drawing_elements:
        element["frameId"] = frame_id
    return _element(
        "frame",
        frame_id,
        **asdict(viewport),
        data={**provenance, "kind": "canvas", "id": "canvas"},
        name="Canvas",
        strokeColor="transparent",
        backgroundColor="transparent",
    )
