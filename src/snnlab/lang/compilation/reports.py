"""Human-readable graph and training summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from snnlab.lang.compilation.diagnostics import Diagnostic


def text_report(
    graph: Mapping[str, Any],
    training: Mapping[str, Any] | None,
    diagnostics: list[Diagnostic],
) -> str:
    params = graph["parameters"]
    count = sum(_shape_product(p["shape"]) for p in params)
    state_scalars = sum(p["size"] for p in graph["populations"])
    projection_edges = sum(
        _shape_product(
            next(p["shape"] for p in params if p["id"] == projection["parameters"][0])
        )
        for projection in graph["projections"]
    )
    selected = {
        p
        for group in (training or {}).get("parameter_groups", [])
        if not group["frozen"]
        for p in group["parameters"]
    }
    recurrent = sorted(
        p["id"]
        for p in graph["projections"]
        if p["connection"] in {"recurrent", "feedback"}
    )
    lines = [
        f"# snnlang report — {graph['name']}",
        "",
        f"Populations: {len(graph['populations'])} ({sum(p['size'] for p in graph['populations']):,} units)",
        f"Projections: {len(graph['projections'])}",
        f"Operations: {len(graph['operations'])}",
        f"Parameters: {len(params)} tensors / {count:,} scalars",
        f"Estimated state: {state_scalars:,} scalars per sample and timestep",
        f"Estimated dense projection edges: {projection_edges:,}",
        f"Trainable this recipe: {len(selected)} tensors",
        f"Outputs: {', '.join(o['id'] for o in graph['outputs']) or 'none'}",
        f"Recurrent paths: {', '.join(recurrent) or 'none'}",
        f"Diagnostics: {sum(d.severity == 'error' for d in diagnostics)} errors, {sum(d.severity == 'warning' for d in diagnostics)} warnings",
        "",
        "## Populations",
    ]
    lines += [
        f"- {p['id']}: {p['size']} × {p['neuron']['kind']} ({'spiking' if p['spiking'] else 'non-spiking'})"
        for p in graph["populations"]
    ]
    lines += ["", "## Projections"]
    lines += [
        f"- {p['id']}: {p['source']} → {p['target']} [{p['connection']}, {p['polarity']}]"
        for p in graph["projections"]
    ]
    lines += ["", "## Parameters"]
    lines += [
        f"- {p['id']}: {p['shape']} {p['unit']} ({'selected' if p['id'] in selected else 'frozen/unselected'})"
        for p in params
    ]
    if diagnostics:
        lines += ["", "## Diagnostics"] + [f"- {d.line()}" for d in diagnostics]
    return "\n".join(lines) + "\n"


def _shape_product(shape: list[Any]) -> int:
    result = 1
    for value in shape:
        if isinstance(value, int):
            result *= value
    return result
