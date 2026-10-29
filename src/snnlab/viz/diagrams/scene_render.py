"""Render resolved diagram scenes to static image formats."""

from __future__ import annotations

import base64
import io
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def _path(points, curved):
    """Return line/quadratic commands; rounding stays within the route corridor."""
    commands = [("M", tuple(points[0]))]
    for i, point in enumerate(points[1:-1], 1):
        previous, following = points[i - 1], points[i + 1]
        if curved:
            radius = min(
                6, math.dist(previous, point) / 3, math.dist(point, following) / 3
            )

            def near(other, point=point, radius=radius):
                length = math.dist(other, point)
                return (
                    tuple(
                        point[j] + (other[j] - point[j]) * radius / length
                        for j in (0, 1)
                    )
                    if length
                    else tuple(point)
                )

            commands.extend(
                [("L", near(previous)), ("Q", near(following), tuple(point))]
            )
        else:
            commands.append(("L", tuple(point)))
    commands.append(("L", tuple(points[-1])))
    return commands


def _arrow_points(element):
    return [(element["x"] + p[0], element["y"] + p[1]) for p in element["points"]]


def _svg(resolved, path):
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)

    def tag(name):
        return f"{{{ns}}}{name}"

    canvas, view = resolved.canvas, resolved.viewport
    root = ET.Element(
        tag("svg"),
        {
            "width": str(canvas.width),
            "height": str(canvas.height),
            "viewBox": f"{view.x} {view.y} {view.width} {view.height}",
        },
    )
    ET.SubElement(
        root,
        tag("rect"),
        {
            "x": str(view.x),
            "y": str(view.y),
            "width": str(view.width),
            "height": str(view.height),
            "fill": resolved.theme.background,
        },
    )
    for element in resolved.scene["elements"]:
        attributes = {
            "id": element["id"],
            "data-snnlab-id": str(element["data"].get("id", "")),
        }
        kind = element["type"]
        if kind == "rectangle":
            attributes.update(
                {k: str(element[k]) for k in ("x", "y", "width", "height")}
            )
            attributes.update(
                fill=element["backgroundColor"]
                if element["backgroundColor"] != "transparent"
                else "none",
                stroke=element["strokeColor"],
                **{"stroke-width": str(element["strokeWidth"])},
            )
            ET.SubElement(root, tag("rect"), attributes)
        elif kind == "text":
            attributes.update(
                x=str(element["x"]),
                y=str(element["y"] + element["fontSize"]),
                fill=element["strokeColor"],
                **{
                    "font-size": str(element["fontSize"]),
                    "font-family": resolved.theme.font_name,
                },
            )
            ET.SubElement(root, tag("text"), attributes).text = element["text"]
        elif kind == "image":
            _svg_math(resolved, element, attributes, root)
        elif kind == "arrow":
            points = _arrow_points(element)
            commands = _path(points, element.get("roundness") is not None)
            parts = []
            for command in commands:
                if command[0] == "Q":
                    parts.append(
                        f"Q {command[2][0]} {command[2][1]} {command[1][0]} {command[1][1]}"
                    )
                else:
                    parts.append(f"{command[0]} {command[1][0]} {command[1][1]}")
            attributes.update(
                d=" ".join(parts),
                fill="none",
                stroke=element["strokeColor"],
                **{
                    "stroke-width": str(element["strokeWidth"]),
                    "stroke-linejoin": "round",
                },
            )
            if element["strokeStyle"] != "solid":
                attributes["stroke-dasharray"] = (
                    "8 6" if element["strokeStyle"] == "dashed" else "2 4"
                )
            ET.SubElement(root, tag("path"), attributes)
            a, b = points[-2:]
            length = math.dist(a, b)
            ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
            x, y = b
            left, right = (
                (x - ux * 14 - uy * 7, y - uy * 14 + ux * 7),
                (x - ux * 14 + uy * 7, y - uy * 14 - ux * 7),
            )
            head = element["endArrowhead"]
            if head == "bar":
                d = f"M {x - uy * 8} {y + ux * 8} L {x + uy * 8} {y - ux * 8}"
            elif head == "diamond":
                middle = x - ux * 7, y - uy * 7
                d = f"M {x} {y} L {middle[0] - uy * 5} {middle[1] + ux * 5} L {x - ux * 14} {y - uy * 14} L {middle[0] + uy * 5} {middle[1] - ux * 5} Z"
            else:
                d = f"M {left[0]} {left[1]} L {x} {y} L {right[0]} {right[1]}"
            ET.SubElement(
                root,
                tag("path"),
                {
                    "d": d,
                    "fill": "none",
                    "stroke": element["strokeColor"],
                    "stroke-width": str(element["strokeWidth"]),
                    "stroke-linecap": "round",
                },
            )
    path.write_text(ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n")


def _matplotlib(resolved, path, scale):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.image import imread
    from matplotlib.patches import PathPatch, Polygon, Rectangle
    from matplotlib.path import Path as MplPath

    canvas, view = resolved.canvas, resolved.viewport
    figure = Figure(
        figsize=(canvas.width / 72, canvas.height / 72),
        dpi=72 * scale,
        facecolor=resolved.theme.background,
    )
    FigureCanvasAgg(figure)
    unit_scale = canvas.width / view.width
    axis = figure.add_axes(
        (0, 0, 1, 1), xlim=(view.x, view.right), ylim=(view.bottom, view.y)
    )
    axis.set_axis_off()
    for e in resolved.scene["elements"]:
        kind = e["type"]
        if kind == "rectangle":
            axis.add_patch(
                Rectangle(
                    (e["x"], e["y"]),
                    e["width"],
                    e["height"],
                    facecolor="none"
                    if e["backgroundColor"] == "transparent"
                    else e["backgroundColor"],
                    edgecolor=e["strokeColor"],
                    linewidth=e["strokeWidth"] * unit_scale,
                    zorder=2,
                )
            )
        elif kind == "text":
            axis.text(
                e["x"],
                e["y"],
                e["text"],
                fontsize=e["fontSize"] * unit_scale,
                fontfamily=resolved.theme.font_name,
                color=e["strokeColor"],
                va="top",
                ha="left",
                parse_math=False,
                zorder=4,
            )
        elif kind == "image":
            data = resolved.scene["files"][e["fileId"]]["dataURL"].split(",", 1)[1]
            pixels = imread(io.BytesIO(base64.b64decode(data)), format="png")
            axis.imshow(
                pixels,
                extent=(e["x"], e["x"] + e["width"], e["y"] + e["height"], e["y"]),
                aspect="auto",
                zorder=4,
            )
        elif kind == "arrow":
            points = _arrow_points(e)
            vertices, codes = [], []
            for command in _path(points, e.get("roundness") is not None):
                if command[0] == "Q":
                    vertices.extend([command[2], command[1]])
                    codes.extend([MplPath.CURVE3, MplPath.CURVE3])
                else:
                    vertices.append(command[1])
                    codes.append(
                        MplPath.MOVETO if command[0] == "M" else MplPath.LINETO
                    )
            axis.add_patch(
                PathPatch(
                    MplPath(vertices, codes),
                    facecolor="none",
                    edgecolor=e["strokeColor"],
                    linewidth=e["strokeWidth"] * unit_scale,
                    linestyle={"solid": "-", "dashed": "--", "dotted": ":"}[
                        e["strokeStyle"]
                    ],
                    zorder=1,
                )
            )
            a, b = points[-2:]
            length = math.dist(a, b)
            ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
            x, y = b
            if e["endArrowhead"] == "bar":
                head = [(x - uy * 8, y + ux * 8), (x + uy * 8, y - ux * 8)]
            elif e["endArrowhead"] == "diamond":
                head = [
                    (x, y),
                    (x - ux * 7 - uy * 5, y - uy * 7 + ux * 5),
                    (x - ux * 14, y - uy * 14),
                    (x - ux * 7 + uy * 5, y - uy * 7 - ux * 5),
                    (x, y),
                ]
            else:
                head = [
                    (x - ux * 14 - uy * 7, y - uy * 14 + ux * 7),
                    (x, y),
                    (x - ux * 14 + uy * 7, y - uy * 14 - ux * 7),
                ]
            axis.add_patch(
                Polygon(
                    head,
                    closed=False,
                    fill=False,
                    edgecolor=e["strokeColor"],
                    linewidth=e["strokeWidth"] * unit_scale,
                    zorder=3,
                )
            )
    figure.savefig(path, dpi=72 * scale, facecolor=resolved.theme.background)
    figure.clear()


def render_resolved(resolved, path, *, scale=1):
    output = Path(path)
    if type(scale) is not int or scale < 1:
        raise ValueError("diagram scale must be a positive integer")
    if output.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        raise ValueError(
            "resolved diagram output must be .svg, .png, .jpg, .jpeg or .pdf"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".snnlab-render-", dir=output.parent
    ) as directory:
        staged = Path(directory) / output.name
        if output.suffix.lower() == ".svg":
            _svg(resolved, staged)
        else:
            _matplotlib(resolved, staged, scale)
        staged.replace(output)
    return output


def _svg_math(resolved, element, attributes, root):
    vector = ET.fromstring(resolved.math_vectors[element["fileId"]])
    ids = {
        n.get("id"): element["id"] + "_" + n.get("id")
        for n in vector.iter()
        if n.get("id")
    }
    for node in vector.iter():
        if node.get("id"):
            node.set("id", ids[node.get("id")])
        for key, value in list(node.attrib.items()):
            if value.startswith("#") and value[1:] in ids:
                node.set(key, "#" + ids[value[1:]])
            elif value.startswith("url(#") and value[5:-1] in ids:
                node.set(key, "url(#" + ids[value[5:-1]] + ")")
    vector.attrib.update(attributes)
    vector.attrib.update({k: str(element[k]) for k in ("x", "y", "width", "height")})
    root.append(vector)
