"""Render optional connection mathematics without a system TeX installation."""

from __future__ import annotations

import hashlib
import html
import io
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .dot import _label
from .model import Diagram, DiagramTheme


@dataclass(frozen=True)
class MathLabel:
    png: Path
    svg: str
    width: float
    height: float


def _render_math(text, theme, directory, scale):
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser

    digest = hashlib.sha256(
        repr((text, theme.secondary_size, theme.muted)).encode()
    ).hexdigest()[:16]
    png = directory / f"math-{digest}.png"
    prop = FontProperties(family="DejaVu Sans", size=theme.secondary_size - 1)
    with rc_context(
        {
            "text.usetex": False,
            "svg.fonttype": "path",
            "svg.hashsalt": digest,
            "mathtext.fontset": "dejavusans",
        }
    ):
        try:
            width, height, depth, _, _ = MathTextParser("path").parse(
                text, dpi=72, prop=prop
            )
        except ValueError as error:
            raise ValueError(f"invalid connection label {text!r}: {error}") from error
        if width <= 0 or height <= 0:
            raise ValueError(f"connection math label has no visible content: {text!r}")
        figure = Figure(figsize=(width / 72, height / 72))
        figure.text(0, depth / height, text, fontproperties=prop, color=theme.muted)
        figure.savefig(png, dpi=288 * scale, format="png", transparent=True)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="svg", transparent=True, metadata={"Date": None})
    return MathLabel(png, buffer.getvalue().decode(), float(width), float(height))


def prepare_math_labels(
    diagram: Diagram, theme: DiagramTheme, directory: Path, *, scale=1
):
    """Materialise only explicitly supplied lines containing math delimiters."""
    directory = directory.resolve()
    texts = dict.fromkeys(
        line
        for edge in diagram.edges
        for line in edge.label.splitlines()
        if "$" in line
    )
    if texts:
        directory.mkdir(parents=True, exist_ok=True)
    return {text: _render_math(text, theme, directory, scale) for text in texts}


def math_labels_to_dot(dot, diagram, theme, assets):
    """Use measured image cells so Graphviz includes maths in label clearance."""
    columns = min(14, theme.wrap_columns)
    for text in dict.fromkeys(edge.label for edge in diagram.edges):
        if not any(line in assets for line in text.splitlines()):
            continue
        rows = []
        for line in text.splitlines():
            if line in assets:
                asset = assets[line]
                # The transparent border exposes the image rectangle in JSON
                # even when Graphviz lacks a PNG-to-JSON drawing plugin.
                rows.append(
                    f'<TR><TD WIDTH="{asset.width + 2:g}" HEIGHT="{asset.height + 2:g}" '
                    'FIXEDSIZE="TRUE" BORDER="1" COLOR="#FFFFFF00">'
                    f'<IMG SRC="{html.escape(str(asset.png), quote=True)}" SCALE="TRUE"/></TD></TR>'
                )
            else:
                rows.append(f"<TR><TD>{_label(line, columns)}</TD></TR>")
        table = (
            '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
            + "".join(rows)
            + "</TABLE>"
        )
        dot = dot.replace(f"xlabel=<{_label(text, columns)}>", f"xlabel=<{table}>")
    return dot


def embed_math_svg(path: Path, assets):
    """Replace temporary PNG references with self-contained vector mathematics."""
    if not assets:
        return
    namespace = "http://www.w3.org/2000/svg"
    href = "{http://www.w3.org/1999/xlink}href"
    ET.register_namespace("", namespace)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.parse(path).getroot()
    by_path = {str(asset.png): asset for asset in assets.values()}
    occurrence = 0
    for parent in list(root.iter()):
        for index, element in enumerate(list(parent)):
            if (
                element.tag != f"{{{namespace}}}image"
                or element.get(href) not in by_path
            ):
                continue
            asset = by_path[element.get(href)]
            vector = ET.fromstring(asset.svg)
            prefix = f"math_{occurrence}_"
            occurrence += 1
            identifiers = {
                node.get("id"): prefix + node.get("id")
                for node in vector.iter()
                if node.get("id")
            }
            for node in vector.iter():
                if node.get("id"):
                    node.set("id", identifiers[node.get("id")])
                for key, value in node.attrib.items():
                    if value.startswith("#") and value[1:] in identifiers:
                        node.set(key, "#" + identifiers[value[1:]])
                    elif value.startswith("url(#") and value[5:-1] in identifiers:
                        node.set(key, f"url(#{identifiers[value[5:-1]]})")
            for key in ("x", "y", "width", "height", "preserveAspectRatio"):
                if key in element.attrib:
                    vector.set(key, element.get(key))
            title = ET.Element(f"{{{namespace}}}title")
            title.text = next(
                text for text, candidate in assets.items() if candidate is asset
            )
            vector.insert(0, title)
            parent.remove(element)
            parent.insert(index, vector)
    svg = ET.tostring(root, encoding="unicode", xml_declaration=True)
    path.write_text("\n".join(line.rstrip() for line in svg.split("\n")) + "\n")
