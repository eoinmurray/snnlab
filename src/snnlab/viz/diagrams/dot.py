"""Deterministic Graphviz DOT formatting without a rendering backend."""

from __future__ import annotations

import html
import math
import textwrap
from itertools import pairwise

from .model import Diagram, DiagramEdge, DiagramNode, DiagramTheme, WeightLabel

_DEFAULT_THEME = DiagramTheme()


def _q(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _svg_id(value: str) -> str:
    return "n_" + "".join(char if char.isalnum() else "_" for char in value)


def _colour(theme: DiagramTheme, role: str) -> str:
    values = {
        "ink": theme.ink,
        "muted": theme.muted,
        "line": theme.line,
        "neutral": theme.neutral,
        "output": theme.output,
        "modulatory": theme.modulatory,
        "training": theme.training,
        "inhibitory": theme.inhibitory,
        "signal": theme.signal,
        "output_line": theme.output_line,
        "training_line": theme.training_line,
    }
    try:
        return values[role]
    except KeyError as error:
        raise ValueError(f"unknown diagram colour role: {role}") from error


def _label(value: str, columns: int) -> str:
    """Wrap display text without dropping words or shrinking its type."""
    text = "".join(
        html.escape(line) + '<BR ALIGN="LEFT"/>'
        for paragraph in value.replace("_", " ").splitlines()
        for line in textwrap.wrap(paragraph, width=columns, break_long_words=False)
    )
    return (
        '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5">'
        f'<TR><TD ALIGN="LEFT">{text}</TD></TR></TABLE>'
    )


def _matrix_label(matrix: WeightLabel, size: float) -> str:
    symbol = (
        f'<FONT FACE="Times New Roman" POINT-SIZE="{size:g}">'
        f"<B><I>{html.escape(matrix.symbol)}</I></B></FONT>"
    )
    if not matrix.subscript and not matrix.superscript:
        return symbol
    script_size = size * 0.7

    def script(text: str) -> str:
        return (
            f'<TD ALIGN="LEFT"><FONT FACE="Times New Roman" '
            f'POINT-SIZE="{script_size:g}">{html.escape(text) or " "}</FONT></TD>'
        )

    # Explicit table cells give all exporters the same script sizes and positions.
    # Graphviz's SVG SUB/SUP output relies on browser-dependent baseline-shift.
    return (
        '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        f'<TR><TD ROWSPAN="2" VALIGN="MIDDLE">{symbol}</TD>'
        f"{script(matrix.superscript)}</TR><TR>{script(matrix.subscript)}</TR></TABLE>"
    )


def _edge_label(edge: DiagramEdge, columns: int, size: float) -> str:
    text = _label(edge.label, columns)
    if edge.weight_label is None:
        return text
    matrix = _matrix_label(edge.weight_label, size)
    return (
        text.removesuffix("</TABLE>")
        + f'<TR><TD ALIGN="LEFT">{matrix}</TD></TR></TABLE>'
    )


def _box_colours(node: DiagramNode, theme: DiagramTheme) -> tuple[str, str]:
    fills = {
        "neutral": theme.neutral,
        "output": theme.output,
        "training": theme.training,
    }
    fill = fills.get(node.kind) or getattr(theme, f"{node.kind}_fill")
    border = getattr(theme, f"{node.kind}_border")
    if node.kind == "population" and node.accent_role == "inhibitory":
        border = theme.inhibitory
    return fill, border


def _card(node: DiagramNode, theme: DiagramTheme) -> str:
    _, accent = _box_colours(node, theme)
    rows = []
    box_type = {"training": "Training group", "neutral": "Neutral / external"}.get(
        node.kind, node.kind.capitalize()
    )
    fields = [
        (
            box_type.upper(),
            max(10, theme.secondary_size - 1),
            accent,
            False,
        ),
        (node.title, theme.label_size, theme.ink, True),
    ]
    fields.extend(
        (line, theme.secondary_size, theme.muted, False)
        for index, line in enumerate(node.detail.splitlines())
    )
    standard_badges = {
        "input",
        "population",
        "operation",
        "output",
        "training",
        "objective",
        "spiking",
        "analogue",
        "spikes",
        "continuous",
        "mask",
        "frozen",
        "trainable",
        "annotation",
    }
    if node.badge and node.badge not in standard_badges:
        fields.append((node.badge, theme.secondary_size, accent, False))
    row_limit = 4 if node.kind == "population" else 3
    visible_fields = [field for field in fields if field[0]][:row_limit]
    for text, size, colour, bold in visible_fields:
        if rows:
            rows.append('<TR><TD HEIGHT="5"></TD></TR>')
        for paragraph in text.replace("_", " ").splitlines():
            for line in textwrap.wrap(
                paragraph, width=theme.wrap_columns, break_long_words=False
            ):
                label = html.escape(line)
                if bold:
                    label = f"<B>{label}</B>"
                rows.append(
                    f'<TR><TD ALIGN="LEFT" HEIGHT="{math.ceil(size * 1.2)}">'
                    f'<FONT COLOR="{colour}" POINT-SIZE="{size:g}">{label}</FONT>'
                    "</TD></TR>"
                )
    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        + "".join(rows)
        + "</TABLE>>"
    )


def _forward_spine(diagram: Diagram) -> tuple[str, ...]:
    """Find a longest acyclic forward path to prioritise its alignment."""
    eligible = [
        edge
        for edge in diagram.edges
        if edge.constraint
        and edge.connection == "feedforward"
        and edge.role != "training"
        and edge.source != edge.target
    ]
    incoming = {node.id: 0 for node in diagram.nodes}
    outgoing: dict[str, list[str]] = {node.id: [] for node in diagram.nodes}
    for edge in eligible:
        incoming[edge.target] += 1
        outgoing[edge.source].append(edge.target)
    paths = {node.id: (node.id,) for node in diagram.nodes}
    queue = [node.id for node in diagram.nodes if incoming[node.id] == 0]
    for source in queue:
        for target in outgoing[source]:
            candidate = (*paths[source], target)
            if len(candidate) > len(paths[target]):
                paths[target] = candidate
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
    return max((paths[node] for node in queue), key=len, default=())


def _group_entry_links(diagram: Diagram) -> tuple[tuple[str, str], ...]:
    """Separate input-bearing circuit groups when their dependencies form a DAG.

    Without these layout-only links, a downstream input shares a rank with its
    upstream population. Overlapping cluster extents then force a staircase.
    Restrict this layout to disjoint, fully grouped input/population circuits;
    explicit row/column hints and annotation layouts retain their own rules.
    """
    nodes = {node.id: node for node in diagram.nodes}
    membership, entries = {}, {}
    for group in diagram.groups:
        if group.same_rank or group.same_row:
            return ()
        inputs = [member for member in group.members if nodes[member].kind == "input"]
        if len(inputs) != 1:
            return ()
        entries[group.id] = inputs[0]
        for member in group.members:
            if member in membership or nodes[member].kind not in {
                "input",
                "population",
            }:
                return ()
            membership[member] = group.id
    if set(membership) != set(nodes):
        return ()
    outgoing = {group.id: set() for group in diagram.groups}
    incoming = dict.fromkeys(outgoing, 0)
    links = set()
    for edge in diagram.edges:
        source, target = membership[edge.source], membership[edge.target]
        if source == target:
            continue
        if edge.connection != "feedforward" or not edge.constraint:
            return ()
        if target not in outgoing[source]:
            outgoing[source].add(target)
            incoming[target] += 1
        links.add((edge.source, entries[target]))
    queue = [group for group, count in incoming.items() if count == 0]
    for source in queue:
        for target in sorted(outgoing[source]):
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
    return tuple(sorted(links)) if len(queue) == len(outgoing) else ()


def _append_nodes(diagram, theme, lines):
    known_kinds = {
        "population",
        "output",
        "objective",
        "training",
        "input",
        "operation",
        "neutral",
    }
    for node in diagram.nodes:
        if node.kind not in known_kinds:
            raise ValueError(f"unknown diagram node kind: {node.kind}")
        fill, border = _box_colours(node, theme)
        if border.startswith("#") and len(border) == 7:
            border += "80"
        classes = " ".join(("node", *node.classes))
        margin = f"{node.margin[0]:g},{node.margin[1]:g}"
        lines.append(
            f"{_q(node.id)} [id={_q(_svg_id(node.id))}, class={_q(classes)}, "
            f'label={_card(node, theme)}, shape=box, style="filled", '
            f'fillcolor="{fill}", color="{border}", penwidth={node.pen_width:g}, margin="{margin}", width=0, height=0, nojustify=true];'
        )
    # External sources share the entry column; downstream inputs retain their rank.


def _population_columns(diagram, column_membership):
    population_ids = {node.id for node in diagram.nodes if node.kind == "population"}
    explicit_rows = {
        member for group in diagram.groups if group.same_row for member in group.members
    }
    excitatory_pairs = {
        (edge.source, edge.target)
        for edge in diagram.edges
        if edge.role == "excitatory" and edge.connection in {"recurrent", "feedback"}
    }
    population_columns = {}
    for edge in diagram.edges:
        if (
            edge.role == "inhibitory"
            and edge.connection in {"recurrent", "feedback"}
            and edge.source != edge.target
            and {edge.source, edge.target} <= population_ids
            and not {edge.source, edge.target} & explicit_rows
            and (edge.target, edge.source) in excitatory_pairs
            and edge.source not in population_columns.values()
            and edge.target not in population_columns
        ):
            population_columns[edge.target] = edge.source
            column_membership[edge.target] = column_membership[edge.source] = (
                edge.target
            )
    return population_columns, explicit_rows


def _coupled_population_columns(diagram, population_columns, explicit_rows):
    # Feedback-coupled circuits read left to right, with each local E/I column intact.
    {node.id: index for index, node in enumerate(diagram.nodes)}
    nodes_by_id = {node.id: node for node in diagram.nodes}
    representatives = {
        member: e for e, i in population_columns.items() for member in (e, i)
    }
    coupled_columns = set()
    for edge in diagram.edges:
        if edge.connection != "feedback":
            continue
        source = representatives.get(edge.source, edge.source)
        target = representatives.get(edge.target, edge.target)
        if (
            source != target
            and all(
                nodes_by_id[identifier].kind in {"population"}
                for identifier in (source, target)
            )
            and not {source, target} & explicit_rows
        ):
            coupled_columns.add(tuple(sorted((source, target))))
    return nodes_by_id, coupled_columns


def _append_groups(
    *, diagram, lines, theme, entry_links, population_columns, row_membership
):
    for group in diagram.groups:
        lines.append(
            f"subgraph {_q('cluster_' + _svg_id(group.id))} {{ label={_q(group.label)}; "
            f'color="{theme.line}"; fontcolor="{theme.muted}"; fontname={_q(theme.font_name)}; '
            f'fontsize={theme.secondary_size:g}; penwidth=0.8; style="rounded"; margin=18; labeljust="l";'
        )
        if group.same_rank:
            lines.append("rank=same;")
        lines.extend(f"{_q(member)};" for member in group.members)
        if entry_links:
            for excitatory, inhibitory in population_columns.items():
                if {excitatory, inhibitory} <= set(group.members):
                    lines.append(
                        f"{{ rank=same; {_q(excitatory)}; {_q(inhibitory)}; "
                        f"{_q(inhibitory)} -> {_q(excitatory)} [style=invis, weight=10000]; }}"
                    )
        if group.same_row:
            # Invisible ordering edges arrange the row without adding scientific links.
            row_membership.update((member, group.id) for member in group.members)
            for source, target in pairwise(group.members):
                lines.append(
                    f"{_q(source)} -> {_q(target)} "
                    "[style=invis, weight=100, constraint=true];"
                )
        lines.append("}")


def _append_layout_constraints(*, diagram, theme, lines, entry_links):
    targets = {edge.target for edge in diagram.edges}
    grouped = {member for group in diagram.groups for member in group.members}
    sources = [
        node.id
        for node in diagram.nodes
        if node.kind == "input" and node.id not in targets and node.id not in grouped
    ]
    outputs = [
        node.id
        for node in diagram.nodes
        if node.kind == "output"
        and not any(edge.source == node.id for edge in diagram.edges)
    ]
    if len(outputs) > 1:
        lines.append(
            "{ rank=same; " + " ".join(f"{_q(node)};" for node in outputs) + " }"
        )
    row_membership = {}
    column_membership = {
        member: group.id
        for group in diagram.groups
        if group.same_rank
        for member in group.members
    }
    # Reciprocal E/I populations form a column even without an authored group.
    population_columns, explicit_rows = _population_columns(
        diagram=diagram, column_membership=column_membership
    )

    for excitatory, inhibitory in population_columns.items():
        if entry_links:
            continue
        grouped_column = any(
            group.same_rank and {excitatory, inhibitory} <= set(group.members)
            for group in diagram.groups
        )
        lines.append(f"{{ rank=same; {_q(excitatory)}; {_q(inhibitory)}; }}")
        lines.append(
            f"{_q(inhibitory)} -> {_q(excitatory)} "
            f"[style=invis, weight=10000, constraint={'true' if grouped_column else 'false'}];"
        )
    nodes_by_id, coupled_columns = _coupled_population_columns(
        diagram=diagram,
        population_columns=population_columns,
        explicit_rows=explicit_rows,
    )

    if len(sources) > 1 and not coupled_columns:
        lines.append(
            "{ rank=same; " + " ".join(f"{_q(node)};" for node in sources) + " }"
        )
    for source, target in sorted(coupled_columns):
        entries = [
            edge.source
            for edge in diagram.edges
            if edge.target == target and nodes_by_id[edge.source].kind == "input"
        ]
        entry = entries[0] if len(entries) == 1 else target
        lines.append(
            f"{_q(source)} -> {_q(entry)} [style=invis, weight=10000, minlen=2];"
        )
        if source in population_columns and target in population_columns:
            lines.append(
                f"{_q(population_columns[source])} -> {_q(population_columns[target])} "
                "[style=invis, weight=10000, minlen=2];"
            )
    # Keep successive local inhibitory populations on a common baseline.
    for edge in diagram.edges:
        if (
            edge.connection == "feedforward"
            and edge.source in population_columns
            and edge.target in population_columns
            and not entry_links
        ):
            lines.append(
                f"{_q(population_columns[edge.source])} -> {_q(population_columns[edge.target])} "
                "[style=invis, weight=1000, constraint=true];"
            )
    _append_groups(
        diagram=diagram,
        lines=lines,
        theme=theme,
        entry_links=entry_links,
        population_columns=population_columns,
        row_membership=row_membership,
    )

    for source, entry in entry_links:
        lines.append(f"{_q(source)} -> {_q(entry)} [style=invis, weight=100];")
    return (
        row_membership,
        column_membership,
        nodes_by_id,
        coupled_columns,
        population_columns,
    )


def _append_edge_ports(
    *, edge, attributes, column_membership, coupled_columns, row_membership, node_order
):
    if edge.source == edge.target:
        port = "n"
        attributes.extend([f"tailport={port}", f"headport={port}"])
    elif edge.connection in {"feedback", "recurrent"} and (
        edge.source in column_membership
        and column_membership.get(edge.target) == column_membership[edge.source]
    ):
        if edge.role == "inhibitory":
            attributes.extend(["tailport=nw", "headport=sw"])
        else:
            attributes.extend(["tailport=se", "headport=ne"])
    elif edge.connection == "feedback" and not coupled_columns:
        if (
            edge.source in row_membership
            and edge.target in row_membership
            and row_membership[edge.source] != row_membership[edge.target]
        ):
            # Coupled rows exchange signals through the gap between circuits.
            forward = node_order[edge.source] < node_order[edge.target]
            attributes.extend(
                [
                    f"tailport={'s' if forward else 'n'}",
                    f"headport={'n' if forward else 's'}",
                ]
            )
        else:
            # Opposite directions use opposite sides, rather than sharing a U-turn.
            if node_order[edge.source] < node_order[edge.target]:
                attributes.extend(["tailport=e", "headport=w"])
            else:
                attributes.extend(["tailport=n", "headport=n"])
    elif edge.role == "training":
        attributes.extend(["tailport=n", "headport=s"])


def _append_edges(
    *,
    diagram,
    theme,
    lines,
    spine,
    spine_order,
    node_order,
    row_membership,
    column_membership,
    nodes_by_id,
    coupled_columns,
):
    for edge_index, edge in enumerate(diagram.edges):
        colour = theme.ink
        arrow = "normal"
        style = "solid"
        if edge.role == "inhibitory":
            colour, arrow = theme.inhibitory, "normaltee"
        elif edge.role == "modulatory":
            arrow, style = "diamond", "dashed"
        elif edge.role == "signal":
            colour, arrow = theme.signal, "vee"
        elif edge.role == "output":
            colour = theme.output_line
        elif edge.role == "training":
            colour, arrow, style = (
                theme.muted if edge.frozen else theme.training_line,
                "vee",
                "dashed" if edge.frozen else "dotted",
            )
        elif edge.role != "excitatory":
            raise ValueError(f"unknown diagram edge role: {edge.role}")
        if edge.connection == "feedback":
            style = "dashed"
            if edge.role == "inhibitory":
                arrow = "normaltee"
        attributes = []
        if edge.id is not None:
            attributes.append(f"id={_q('e_' + _svg_id(edge.id))}")
        else:
            attributes.append(f"id={_q('e_layout_' + str(edge_index))}")
        classes = " ".join(
            dict.fromkeys(("edge", edge.role, edge.connection, *edge.classes))
        )
        attributes.append(f"class={_q(classes)}")
        internal_row = (
            edge.source in row_membership
            and row_membership.get(edge.target) == row_membership[edge.source]
        )
        internal_column = (
            edge.source in column_membership
            and column_membership.get(edge.target) == column_membership[edge.source]
            and edge.source != edge.target
        )
        attributes.extend(
            [
                f'color="{colour}"',
                f"arrowhead={arrow}",
                f"style={style}",
                (
                    f"xlabel=<{_edge_label(edge, min(14, theme.wrap_columns), theme.secondary_size)}>"
                    if edge.label or edge.weight_label is not None
                    else 'label=""'
                ),
                f"constraint={'true' if edge.constraint and not internal_row and not internal_column else 'false'}",
                f"penwidth={edge.pen_width:g}",
                "weight=10000"
                if nodes_by_id[edge.source].kind == "input"
                else "weight=100"
                if (edge.source, edge.target) in spine
                else "weight=1",
            ]
        )
        _append_edge_ports(
            edge=edge,
            attributes=attributes,
            column_membership=column_membership,
            coupled_columns=coupled_columns,
            row_membership=row_membership,
            node_order=node_order,
        )

        lines.append(
            f"{_q(edge.source)} -> {_q(edge.target)} [{', '.join(attributes)}];"
        )


def diagram_to_dot(
    diagram: Diagram,
    *,
    theme: DiagramTheme = _DEFAULT_THEME,
    height_to_width_ratio: float | None = None,
    canvas_size: tuple[int, int] | None = None,
) -> str:
    """Compile a structured diagram into deterministic Graphviz DOT."""

    if height_to_width_ratio is not None and (
        not math.isfinite(height_to_width_ratio) or height_to_width_ratio <= 0
    ):
        raise ValueError("diagram height-to-width ratio must be finite and positive")
    if canvas_size is not None and (
        len(canvas_size) != 2
        or any(type(value) is not int or value <= 0 for value in canvas_size)
    ):
        raise ValueError("diagram canvas size must contain two positive integers")
    title = diagram.title or diagram.name.replace("_", " ")
    title = title[:1].upper() + title[1:]
    ratio = (
        "" if height_to_width_ratio is None else f', ratio="{height_to_width_ratio:g}"'
    )
    # Graphviz uses points; PNG exports use 144 DPI, or two pixels per point.
    viewport = (
        ""
        if canvas_size is None
        else f', viewport="{canvas_size[0] / 2:g},{canvas_size[1] / 2:g},1"'
    )
    incoming_nodes = {edge.target for edge in diagram.edges}
    input_count = sum(
        node.kind == "input" and node.id not in incoming_nodes for node in diagram.nodes
    )
    entry_links = _group_entry_links(diagram)
    rank_spacing = "1.3" if entry_links else "4.5" if input_count >= 3 else "1.1"
    matrix_labels = [
        edge.weight_label for edge in diagram.edges if edge.weight_label is not None
    ]
    nodesep = "1.15" if matrix_labels else "0.65"
    if matrix_labels:
        longest = max(
            len(label.symbol) + len(label.subscript) + len(label.superscript)
            for label in matrix_labels
        )
        label_spacing = longest * (theme.secondary_size - 1) * 0.6 / 72 + 0.6
        rank_spacing = f"{max(float(rank_spacing), 2.0, label_spacing):g}"
    # Local cluster ranking preserves E/I order in mixed input/population groups.
    newrank = "false" if entry_links else "true"
    lines = [
        f"digraph {_q(diagram.name)} {{",
        f'graph [rankdir=LR{ratio}{viewport}, bgcolor="{theme.background}", pad="0.3", nodesep="{nodesep}", ranksep="{rank_spacing}",',
        f'  splines=ortho, outputorder=edgesfirst, fontname={_q(theme.font_name)}, fontcolor="{theme.ink}",',
        f"  label=<<B>{html.escape(title)}</B>>, labelloc=t, labeljust=l, fontsize={theme.title_size:g}, compound=true, newrank={newrank}];",
        f'node [shape=plain, fontname={_q(theme.font_name)}, fontcolor="{theme.ink}"];',
        f'edge [fontname={_q(theme.font_name)}, fontsize={theme.secondary_size - 1:g}, fontcolor="{theme.muted}", color="{theme.ink}", penwidth=1.1, arrowsize=0.65];',
    ]
    _append_nodes(diagram=diagram, theme=theme, lines=lines)

    (
        row_membership,
        column_membership,
        nodes_by_id,
        coupled_columns,
        population_columns,
    ) = _append_layout_constraints(
        diagram=diagram, theme=theme, lines=lines, entry_links=entry_links
    )

    path = _forward_spine(diagram)
    spine_order = {identifier: index for index, identifier in enumerate(path)}
    node_order = {node.id: index for index, node in enumerate(diagram.nodes)}
    for node in diagram.nodes:
        if node.kind != "training":
            continue
        targets = [
            edge.target
            for edge in diagram.edges
            if edge.source == node.id and edge.role == "training"
        ]
        if targets:
            anchor = max(
                targets,
                key=lambda identifier: (
                    spine_order.get(identifier, -1),
                    node_order[identifier],
                ),
            )
            anchor = population_columns.get(anchor, anchor)
            lines.append(f"{{ rank=same; {_q(anchor)}; {_q(node.id)}; }}")
            lines.append(f"{_q(anchor)} -> {_q(node.id)} [style=invis, weight=50];")
    spine = set(pairwise(path))
    _append_edges(
        diagram=diagram,
        theme=theme,
        lines=lines,
        spine=spine,
        spine_order=spine_order,
        node_order=node_order,
        row_membership=row_membership,
        column_membership=column_membership,
        nodes_by_id=nodes_by_id,
        coupled_columns=coupled_columns,
    )

    lines.append("}")
    return "\n".join(lines) + "\n"
