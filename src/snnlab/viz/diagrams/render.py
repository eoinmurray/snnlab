"""Graphviz process execution and diagram file export."""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from pathlib import Path

from .dot import diagram_to_dot
from .geometry import ResolvedDiagram
from .labels import embed_math_svg, math_labels_to_dot, prepare_math_labels
from .layout import position_diagram
from .model import Diagram, DiagramTheme
from .spec import DiagramLayout

_DEFAULT_THEME = DiagramTheme()


def _finish_svg(output, assets, canvas_size):
    embed_math_svg(output, assets)
    if canvas_size is not None:
        # Match SVG's intrinsic pixel dimensions to the PNG, preserving viewBox.
        svg = re.sub(
            r'(<svg\b[^>]*\bwidth=")[^"]+("[^>]*\bheight=")[^"]+(")',
            lambda match: (
                f"{match[1]}{canvas_size[0]}px{match[2]}{canvas_size[1]}px{match[3]}"
            ),
            output.read_text(),
            count=1,
        )
        output.write_text(svg)


def render_diagram(
    diagram: Diagram | ResolvedDiagram,
    path: str | Path,
    *,
    scale: int = 1,
    theme: DiagramTheme = _DEFAULT_THEME,
    height_to_width_ratio: float | None = None,
    canvas_size: tuple[int, int] | None = None,
    layout: DiagramLayout | dict | None = None,
) -> Path:
    """Render a diagram as SVG, PNG, PDF, or its deterministic DOT source."""

    from .resolved import resolve_diagram
    from .scene_render import render_resolved

    if layout is not None or isinstance(diagram, ResolvedDiagram):
        return _render_with_layout(
            diagram, path, scale, theme, layout, height_to_width_ratio, canvas_size
        )
    if Path(path).suffix.lower() in {".jpg", ".jpeg"}:
        if height_to_width_ratio is not None or canvas_size is not None:
            raise ValueError("use layout.canvas for JPEG canvas settings")
        from .spec import DiagramLayout

        return render_resolved(
            resolve_diagram(diagram, layout=DiagramLayout(), theme=theme),
            path,
            scale=scale,
        )
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
        assets = prepare_math_labels(
            diagram, theme, output.parent / f"{output.stem}.labels"
        )
        dot = math_labels_to_dot(dot, diagram, theme, assets)
        output.write_text(dot)
        return output
    with tempfile.TemporaryDirectory(prefix="snnviz-diagram-") as temporary:
        assets = prepare_math_labels(diagram, theme, Path(temporary), scale=scale)
        dot = math_labels_to_dot(dot, diagram, theme, assets)
        positioned_path = position_diagram(
            diagram, dot, Path(temporary), pin=height_to_width_ratio is None
        )
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
                for required, available in zip(natural, canvas_size, strict=True)
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
    if suffix == ".svg":
        _finish_svg(output, assets, canvas_size)
    return output


def _render_with_layout(diagram, path, scale, theme, layout, ratio, canvas_size):
    from .resolved import resolve_diagram
    from .scene_render import render_resolved

    if ratio is not None or canvas_size is not None:
        raise ValueError(
            "use layout.canvas instead of legacy canvas/ratio options with DiagramLayout"
        )
    if isinstance(diagram, ResolvedDiagram):
        if layout is not None or theme != _DEFAULT_THEME and theme != diagram.theme:
            raise ValueError("a resolved diagram already owns its layout and theme")
        resolved = diagram
    else:
        resolved = resolve_diagram(diagram, layout=layout, theme=theme)
    return render_resolved(resolved, path, scale=scale)
