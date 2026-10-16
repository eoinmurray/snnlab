"""Graphviz process execution and diagram file export."""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

from .dot import diagram_to_dot
from .model import Diagram, DiagramTheme
from .routing import expose_arrowheads, position_wire_labels, route_feedback_lanes


def render_diagram(
    diagram: Diagram,
    path: str | Path,
    *,
    scale: int = 1,
    theme: DiagramTheme = DiagramTheme(),
    height_to_width_ratio: float | None = None,
    canvas_size: tuple[int, int] | None = None,
) -> Path:
    """Render a diagram as SVG, PNG, PDF, or its deterministic DOT source."""

    if not isinstance(scale, int) or scale < 1:
        raise ValueError("diagram scale must be a positive integer")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dot = diagram_to_dot(
        diagram,
        theme=theme,
        height_to_width_ratio=height_to_width_ratio,
        canvas_size=canvas_size,
    )
    suffix = output.suffix.lower()
    if suffix not in {".svg", ".png", ".pdf", ".dot"}:
        raise ValueError("diagram output must be .svg, .png, .pdf, or .dot")
    if suffix == ".dot":
        output.write_text(dot)
        return output
    with tempfile.TemporaryDirectory(prefix="snnviz-diagram-") as temporary:
        dot_path = Path(temporary) / "diagram.dot"
        dot_path.write_text(dot)
        layout_path = Path(temporary) / "layout.json"
        positioned_path = Path(temporary) / "positioned.dot"
        # Separate invocations avoid renderer-dependent coordinate mutations when
        # Graphviz writes multiple formats from a single in-memory graph.
        for format_name, destination in (
            ("json", layout_path),
            ("dot", positioned_path),
        ):
            layout = subprocess.run(
                ["dot", f"-T{format_name}", "-o", str(destination), str(dot_path)],
                capture_output=True,
                text=True,
            )
            if layout.returncode:
                raise RuntimeError(f"Graphviz failed: {layout.stderr.strip()}")
        layout_data, positioned_dot = route_feedback_lanes(
            json.loads(layout_path.read_text()), positioned_path.read_text()
        )
        layout_data, positioned_dot = expose_arrowheads(layout_data, positioned_dot)
        positioned_path.write_text(position_wire_labels(layout_data, positioned_dot))
        if canvas_size is not None:
            natural_layout = subprocess.run(
                ["neato", "-n2", "-Tplain"],
                input=re.sub(
                    r'\bviewport="[^"]*"\s*,?', "", positioned_path.read_text()
                ),
                capture_output=True,
                text=True,
            )
            if natural_layout.returncode:
                raise RuntimeError(f"Graphviz failed: {natural_layout.stderr.strip()}")
            graph = natural_layout.stdout.splitlines()[0].split()
            natural = tuple(
                math.ceil((float(value) + 0.6) * 144) for value in graph[2:4]
            )
            if any(
                required > available
                for required, available in zip(natural, canvas_size)
            ):
                raise ValueError(
                    f"diagram canvas is too small; natural layout needs at least {natural[0]} × {natural[1]} pixels"
                )
        args = [
            "neato",
            "-n2",
            f"-T{suffix[1:]}",
            str(positioned_path),
            "-o",
            str(output),
        ]
        if suffix == ".png":
            args.insert(1, f"-Gdpi={144 * scale}")
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Graphviz failed: {result.stderr.strip()}")
    if canvas_size is not None and suffix == ".svg":
        # Match SVG's intrinsic pixel dimensions to the PNG, preserving viewBox.
        svg = output.read_text()
        svg = re.sub(
            r'<svg\s+width="[^"]+"\s+height="[^"]+"',
            f'<svg width="{canvas_size[0]}px" height="{canvas_size[1]}px"',
            svg,
            count=1,
        )
        output.write_text(svg)
    return output
