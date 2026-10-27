"""Align converging recurrent circuits and their per-population readouts."""

from __future__ import annotations

from collections import defaultdict

from .dot import _population_columns, _q
from .model import Diagram


def _circuit_stages(diagram, pairs):
    representatives = {member: e for e, i in pairs.items() for member in (e, i)}
    parents = {e: set() for e in pairs}
    for edge in diagram.edges:
        if edge.role == "training" or edge.connection != "feedforward":
            continue
        source, target = (
            representatives.get(edge.source),
            representatives.get(edge.target),
        )
        if source is not None and target is not None and source != target:
            parents[target].add(source)
    if not any(len(sources) > 1 for sources in parents.values()):
        return None
    stages, centers = {}, {}
    pending = list(pairs)
    roots = [e for e in pairs if not parents[e]]
    while pending:
        ready = [e for e in pending if parents[e] <= stages.keys()]
        if not ready:
            return None
        for e in ready:
            stages[e] = 1 + max((stages[p] for p in parents[e]), default=-1)
            centers[e] = (
                sum(centers[p] for p in parents[e]) / len(parents[e])
                if parents[e]
                else roots.index(e)
            )
            pending.remove(e)
    return stages, centers


def _readout_paths(diagram, pairs):
    nodes = {node.id: node for node in diagram.nodes}
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for edge in diagram.edges:
        if edge.role == "training" or edge.connection != "feedforward":
            continue
        outgoing[edge.source].append(edge.target)
        incoming[edge.target].append(edge.source)
    paths, side_outputs = {}, {}
    for population in (member for pair in pairs.items() for member in pair):
        entries = [
            target
            for target in outgoing[population]
            if nodes[target].kind == "operation"
        ]
        if len(entries) > 1:
            return None
        if not entries:
            paths[population] = []
            continue
        path = []
        target = entries[0]
        while target is not None:
            if target in path or len(incoming[target]) != 1:
                return None
            path.append(target)
            children = outgoing[target]
            continuing = [child for child in children if nodes[child].kind != "output"]
            outputs = [child for child in children if nodes[child].kind == "output"]
            if len(continuing) > 1 or len(outputs) > 1:
                return None
            if continuing and outputs:
                if population in side_outputs:
                    return None
                side_outputs[population] = outputs[0]
            elif outputs:
                path.append(outputs[0])
                # An objective can follow the official classifier output.
                path.extend(outgoing[outputs[0]])
            target = continuing[0] if continuing else None
        paths[population] = path
    flattened = [member for path in paths.values() for member in path]
    if len(flattened) != len(set(flattened)):
        return None
    covered = set(flattened) | set(side_outputs.values())
    covered.update(member for pair in pairs.items() for member in pair)
    if any(
        node.kind not in {"input", "training"} and node.id not in covered
        for node in diagram.nodes
    ):
        return None
    return paths, side_outputs, outgoing


def _readout_grid(
    diagram, pairs, stages, centers, paths, side_outputs, outgoing, input_ids
):
    columns, rows = dict.fromkeys(input_ids, 0), {}
    readout_column = max(stages.values()) + 2
    row = 0
    for e in sorted(pairs, key=lambda e: (centers[e], stages[e])):
        for population in (e, pairs[e]):
            columns[population] = stages[e] + 1
            rows[population] = row
            for offset, member in enumerate(paths[population]):
                columns[member] = readout_column + offset
                rows[member] = row
            row += 1
        for population in (e, pairs[e]):
            if population in side_outputs:
                member = side_outputs[population]
                columns[member] = readout_column + 1
                rows[member] = row
                row += 1
    inputs_by_row = defaultdict(list)
    for input_id in input_ids:
        input_rows = [rows[target] for target in outgoing[input_id] if target in rows]
        if not input_rows:
            return None
        inputs_by_row[(min(input_rows) + max(input_rows)) / 2].append(input_id)
    for center, members in inputs_by_row.items():
        for index, input_id in enumerate(members):
            rows[input_id] = center + (index - (len(members) - 1) / 2) * 0.9
    for node in diagram.nodes:
        if node.kind != "training":
            continue
        # Training links are excluded from the forward dependency graph.
        targets = [
            edge.target
            for edge in diagram.edges
            if edge.source == node.id and edge.role == "training"
        ]
        if not targets or any(target not in columns for target in targets):
            return None
        columns[node.id] = max(columns[target] for target in targets) + (
            len(targets) == 1
        )
        rows[node.id] = rows[targets[0]] + 1 if len(targets) == 1 else row
        while any(
            columns[other] == columns[node.id] and rows[other] == rows[node.id]
            for other in columns
            if other != node.id
        ):
            rows[node.id] += 1
    return columns, rows


def position_readout_fan_in(diagram: Diagram, layout: dict, dot: str) -> str | None:
    """Return pinned DOT for an ungrouped E/I fan-in with readout paths.

    Place each diagnostic on its population's row, reserving an extra row when
    a reduction also feeds a classifier. Ordinary dot ranking otherwise tears
    local loops apart to optimise crossings to the many terminal outputs.
    Unsupported topologies retain the standard Graphviz layout.
    """
    if diagram.groups:
        return None
    pairs, _ = _population_columns(diagram, {})
    population_ids = {node.id for node in diagram.nodes if node.kind == "population"}
    if len(pairs) < 3 or population_ids != {
        member for pair in pairs.items() for member in pair
    }:
        return None
    circuits = _circuit_stages(diagram, pairs)
    readouts = _readout_paths(diagram, pairs)
    inputs = [node.id for node in diagram.nodes if node.kind == "input"]
    if circuits is None or readouts is None or not inputs:
        return None
    stages, centers = circuits
    paths, side_outputs, outgoing = readouts
    grid = _readout_grid(
        diagram, pairs, stages, centers, paths, side_outputs, outgoing, inputs
    )
    if grid is None:
        return None
    columns, rows = grid
    sizes = {
        node["name"]: (float(node["width"]) * 72, float(node["height"]) * 72)
        for node in layout["objects"]
        if "pos" in node
    }
    row_gap = max(height for _, height in sizes.values()) + 36
    for index, input_id in enumerate(inputs):
        if any(
            abs(rows[input_id] - rows[other]) * row_gap
            < (sizes[input_id][1] + sizes[other][1]) / 2 + 12
            for other in inputs[:index]
        ):
            return None
    widths = {
        column: max(sizes[node][0] for node in columns if columns[node] == column)
        for column in set(columns.values())
    }
    x, previous = {}, None
    for column in sorted(widths):
        x[column] = (
            widths[column] / 2
            if previous is None
            else x[previous] + (widths[previous] + widths[column]) / 2 + 90
        )
        previous = column
    lines = [line for line in dot.splitlines()[:-1] if "style=invis" not in line]
    for node in diagram.nodes:
        lines.append(
            f'{_q(node.id)} [pos="{x[columns[node.id]]:g},{-rows[node.id] * row_gap:g}!"];'
        )
    return "\n".join((*lines, "}")) + "\n"
