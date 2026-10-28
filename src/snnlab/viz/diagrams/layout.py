"""Shared Graphviz positioning for static and editable diagram exports."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .fan_in import position_readout_fan_in
from .routing import expose_arrowheads, position_wire_labels, route_feedback_lanes


def graphviz(args: list[str], *, source: str) -> str:
    try:
        result = subprocess.run(args, input=source, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Graphviz executable {args[0]!r} is not installed or not on PATH; "
            "install Graphviz separately with your operating system package manager."
        ) from error
    if result.returncode:
        raise RuntimeError(f"Graphviz failed: {result.stderr.strip()}")
    return result.stdout


def position_diagram(diagram, dot: str, directory: Path, *, pin: bool = True) -> Path:
    """Return routed DOT, retaining the existing static renderer's layout order."""
    layout = json.loads(graphviz(["dot", "-Tjson"], source=dot))
    positioned = graphviz(["dot", "-Tdot"], source=dot)
    pinned = position_readout_fan_in(diagram, layout, dot) if pin else None
    if pinned is not None:
        layout = json.loads(graphviz(["neato", "-n", "-Tjson"], source=pinned))
        positioned = graphviz(["neato", "-n", "-Tdot"], source=pinned)
    layout, positioned = route_feedback_lanes(layout, positioned)
    layout, positioned = expose_arrowheads(layout, positioned)
    path = directory / "positioned.dot"
    path.write_text(position_wire_labels(layout, positioned), encoding="utf-8")
    return path
