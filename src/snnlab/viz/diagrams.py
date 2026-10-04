"""Renderer-owned contracts and Graphviz export for structural diagrams."""

from __future__ import annotations

import html
import math
import re
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DiagramTheme:
    """Shared colours and article-scale typography for structural diagrams."""

    background: str = "#FFFFFF"
    ink: str = "#28323C"
    muted: str = "#6B7680"
    line: str = "#D8DEE3"
    neutral: str = "#FAFBFC"
    output: str = "#F4F7FA"
    modulatory: str = "#92764B"
    training: str = "#F4F8F7"
    inhibitory: str = "#AA5B63"
    signal: str = "#7A8690"
    output_line: str = "#58768F"
    training_line: str = "#56857D"

    title_size: float = 18
    label_size: float = 14
    secondary_size: float = 11
    wrap_columns: int = 26
    font_name: str = "Helvetica"


@dataclass(frozen=True)
class DiagramNode:
    """One semantic node, independent of the source graph schema."""

    id: str
    title: str
    detail: str
    badge: str
    kind: str = "neutral"
    accent_role: str = "ink"
    classes: tuple[str, ...] = ()
    pen_width: float = 1.0
    margin: tuple[float, float] = (0.18, 0.14)


@dataclass(frozen=True)
class DiagramEdge:
    """A directed semantic relationship between two diagram nodes."""

    source: str
    target: str
    role: str = "signal"
    label: str = ""
    connection: str = "feedforward"
    id: str | None = None
    classes: tuple[str, ...] = ()
    constraint: bool = True
    pen_width: float = 1.1
    frozen: bool = False


@dataclass(frozen=True)
class DiagramGroup:
    """A labelled visual boundary around existing nodes."""

    id: str
    label: str
    members: tuple[str, ...]
    same_rank: bool = False
    same_row: bool = False


@dataclass(frozen=True)
class Diagram:
    """Renderer-neutral structural diagram ready for composition."""

    name: str
    nodes: tuple[DiagramNode, ...]
    edges: tuple[DiagramEdge, ...]
    groups: tuple[DiagramGroup, ...] = ()
    title: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [node.id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("diagram node ids must be unique")
        known = set(ids)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(
                    f"diagram edge references unknown node: {edge.source} -> {edge.target}"
                )
        for group in self.groups:
            if group.same_rank and group.same_row:
                raise ValueError(
                    "diagram groups cannot request both same_rank and same_row"
                )
            unknown = set(group.members) - known
            if unknown:
                raise ValueError(
                    f"diagram group {group.id} references unknown nodes: "
                    + ", ".join(sorted(unknown))
                )


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
    return "".join(
        html.escape(line) + '<BR ALIGN="LEFT"/>'
        for paragraph in value.replace("_", " ").splitlines()
        for line in textwrap.wrap(paragraph, width=columns, break_long_words=False)
    )


def _card(node: DiagramNode, theme: DiagramTheme) -> str:
    accent = (
        theme.muted if "frozen" in node.classes else _colour(theme, node.accent_role)
    )
    rows = []
    for text, size, colour, bold in (
        (node.title, theme.label_size, theme.ink, True),
        (node.detail, theme.secondary_size, theme.muted, False),
        (node.badge, theme.secondary_size - 1, accent, False),
    ):
        if not text:
            continue
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


def diagram_to_dot(
    diagram: Diagram,
    *,
    theme: DiagramTheme = DiagramTheme(),
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
    lines = [
        f"digraph {_q(diagram.name)} {{",
        f'graph [rankdir=LR{ratio}{viewport}, bgcolor="{theme.background}", pad="0.3", nodesep="0.35", ranksep="0.65",',
        f'  splines=spline, outputorder=edgesfirst, fontname={_q(theme.font_name)}, fontcolor="{theme.ink}",',
        f"  label=<<B>{html.escape(title)}</B>>, labelloc=t, labeljust=l, fontsize={theme.title_size:g}, compound=true, newrank=true];",
        f'node [shape=plain, fontname={_q(theme.font_name)}, fontcolor="{theme.ink}"];',
        f'edge [fontname={_q(theme.font_name)}, fontsize={theme.secondary_size - 1:g}, fontcolor="{theme.muted}", color="{theme.ink}", penwidth=1.1, arrowsize=0.65];',
    ]
    known_kinds = {
        "component",
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
        border = (
            theme.line
            if node.accent_role == "ink"
            else _colour(theme, node.accent_role)
        )
        fill = _colour(
            theme, node.kind if node.kind in {"output", "training"} else "neutral"
        )
        if "frozen" in node.classes:
            border, fill = theme.line, theme.neutral
        classes = " ".join(("node", *node.classes))
        margin = f"{node.margin[0]:g},{node.margin[1]:g}"
        lines.append(
            f"{_q(node.id)} [id={_q(_svg_id(node.id))}, class={_q(classes)}, "
            f'label={_card(node, theme)}, shape=box, style="filled", '
            f'fillcolor="{fill}", color="{border}", penwidth={node.pen_width:g}, margin="{margin}", width=1.4, height=0.8];'
        )
    # External sources share the entry column; downstream inputs retain their rank.
    targets = {edge.target for edge in diagram.edges}
    grouped = {member for group in diagram.groups for member in group.members}
    sources = [
        node.id
        for node in diagram.nodes
        if node.kind == "input" and node.id not in targets and node.id not in grouped
    ]
    if len(sources) > 1:
        lines.append(
            "{ rank=same; " + " ".join(f"{_q(node)};" for node in sources) + " }"
        )
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
    for group in diagram.groups:
        lines.append(
            f"subgraph {_q('cluster_' + _svg_id(group.id))} {{ label={_q(group.label)}; "
            f'color="{theme.line}"; fontcolor="{theme.muted}"; fontname={_q(theme.font_name)}; '
            f'fontsize={theme.secondary_size:g}; penwidth=0.8; style="rounded"; margin=18; labeljust="l";'
        )
        if group.same_rank:
            lines.append("rank=same;")
        lines.extend(f"{_q(member)};" for member in group.members)
        if group.same_row:
            # Invisible ordering edges arrange the row without adding scientific links.
            row_membership.update((member, group.id) for member in group.members)
            for source, target in zip(group.members, group.members[1:]):
                lines.append(
                    f"{_q(source)} -> {_q(target)} "
                    "[style=invis, weight=100, constraint=true];"
                )
        lines.append("}")
    # Place training annotations below the last component they describe.
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
            lines.append(f"{{ rank=same; {_q(anchor)}; {_q(node.id)}; }}")
            lines.append(f"{_q(anchor)} -> {_q(node.id)} [style=invis, weight=50];")
    spine = set(zip(path, path[1:]))
    for edge in diagram.edges:
        colour = theme.ink
        arrow = "normal"
        style = "solid"
        if edge.role == "inhibitory":
            colour, arrow = theme.inhibitory, "tee"
        elif edge.role == "modulatory":
            arrow, style = "diamond", "dashed"
        elif edge.role == "signal":
            colour, arrow = theme.signal, "vee"
        elif edge.role == "output":
            colour = theme.output_line
        elif edge.role == "training":
            colour, arrow, style = (
                theme.muted if edge.frozen else theme.training_line,
                "none" if edge.frozen else "vee",
                "dashed" if edge.frozen else "dotted",
            )
        elif edge.role != "excitatory":
            raise ValueError(f"unknown diagram edge role: {edge.role}")
        if edge.connection == "feedback":
            style = "dashed"
        attributes = []
        if edge.id is not None:
            attributes.append(f"id={_q('e_' + _svg_id(edge.id))}")
        classes = " ".join(("edge", *edge.classes))
        if edge.classes:
            attributes.append(f"class={_q(classes)}")
        internal_row = (
            edge.source in row_membership
            and row_membership.get(edge.target) == row_membership[edge.source]
        )
        attributes.extend(
            [
                f'color="{colour}"',
                f"arrowhead={arrow}",
                f"style={style}",
                (
                    f"label=<{_label(edge.label, min(14, theme.wrap_columns))}>"
                    if edge.label
                    else 'label=""'
                ),
                f"constraint={'true' if edge.constraint and not internal_row else 'false'}",
                f"penwidth={edge.pen_width:g}",
                "weight=100" if (edge.source, edge.target) in spine else "weight=1",
            ]
        )
        if edge.source == edge.target:
            port = "n"
            attributes.extend([f"tailport={port}", f"headport={port}"])
        elif edge.connection == "feedback":
            attributes.extend(["tailport=s", "headport=s"])
        elif edge.role == "training":
            attributes.extend(["tailport=n", "headport=s"])
        lines.append(
            f"{_q(edge.source)} -> {_q(edge.target)} [{', '.join(attributes)}];"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


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
        if canvas_size is not None:
            dot_path.write_text(
                diagram_to_dot(
                    diagram, theme=theme, height_to_width_ratio=height_to_width_ratio
                )
            )
            layout = subprocess.run(
                ["dot", "-Tplain", str(dot_path)], capture_output=True, text=True
            )
            if layout.returncode:
                raise RuntimeError(f"Graphviz failed: {layout.stderr.strip()}")
            graph = layout.stdout.splitlines()[0].split()
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
        dot_path.write_text(dot)
        args = ["dot", f"-T{suffix[1:]}", str(dot_path), "-o", str(output)]
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
