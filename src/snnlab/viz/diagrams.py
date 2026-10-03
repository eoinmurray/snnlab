"""Renderer-owned contracts and Graphviz export for structural diagrams."""

from __future__ import annotations

import html
import math
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DiagramTheme:
    """Shared colours and article-scale typography for structural diagrams."""

    background: str = "#FFFFFF"
    ink: str = "#1A1A1A"
    muted: str = "#5F5F5F"
    line: str = "#E7E5DF"
    neutral: str = "#FFFFFF"
    output: str = "#FFFFFF"
    modulatory: str = "#E89400"
    training: str = "#FFFFFF"
    inhibitory: str = "#C8102E"
    signal: str = "#5F5F5F"
    output_line: str = "#E89400"
    training_line: str = "#00B4D8"

    title_size: float = 26
    label_size: float = 16
    secondary_size: float = 13
    wrap_columns: int = 14


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
    pen_width: float = 1.4
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
    pen_width: float = 1.7
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
                raise ValueError("diagram groups cannot request both same_rank and same_row")
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
        for line in textwrap.wrap(value.upper().replace("_", " "), width=columns)
    )


def _card(node: DiagramNode, theme: DiagramTheme) -> str:
    accent = _colour(theme, node.accent_role)
    rows = []
    for text, size, colour in (
        (node.title, theme.label_size, accent),
        (node.detail, theme.secondary_size, theme.ink),
        (node.badge, theme.secondary_size, accent),
    ):
        if not text:
            continue
        if rows:
            rows.append('<TR><TD HEIGHT="6"></TD></TR>')
        for line in textwrap.wrap(text.upper().replace("_", " "), width=theme.wrap_columns):
            rows.append(
                f'<TR><TD ALIGN="LEFT" HEIGHT="{math.ceil(size * 1.2)}">'
                f'<FONT COLOR="{colour}" POINT-SIZE="{size:g}"><B>{html.escape(line)}</B></FONT>'
                '</TD></TR>'
            )
    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        + "".join(rows) + "</TABLE>>"
    )


def diagram_to_dot(
    diagram: Diagram,
    *,
    theme: DiagramTheme = DiagramTheme(),
    height_to_width_ratio: float | None = 0.58,
) -> str:
    """Compile a structured diagram into deterministic Graphviz DOT."""

    if height_to_width_ratio is not None and (
        not math.isfinite(height_to_width_ratio) or height_to_width_ratio <= 0
    ):
        raise ValueError("diagram height-to-width ratio must be finite and positive")
    title = (diagram.title or diagram.name.replace("_", " ")).upper()
    ratio = (
        ""
        if height_to_width_ratio is None
        else f', ratio="{height_to_width_ratio:g}"'
    )
    lines = [
        f"digraph {_q(diagram.name)} {{",
        f'graph [rankdir=LR{ratio}, bgcolor="{theme.background}", pad="0.4", nodesep="0.40", ranksep="0.40",',
        f'  splines=spline, outputorder=edgesfirst, fontname="Menlo", fontcolor="{theme.ink}",',
        f"  label=<<B>{html.escape(title)}</B>>, labelloc=t, labeljust=l, fontsize={theme.title_size:g}, compound=true, newrank=true];",
        f'node [shape=plain, fontname="Menlo", fontcolor="{theme.ink}"];',
        f'edge [fontname="Menlo", fontsize={theme.secondary_size:g}, fontcolor="{theme.ink}", color="{theme.ink}", penwidth=2.0, arrowsize=0.8];',
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
        border = _colour(theme, node.accent_role)
        classes = " ".join(("node", *node.classes))
        margin = f"{node.margin[0]:g},{node.margin[1]:g}"
        lines.append(
            f"{_q(node.id)} [id={_q(_svg_id(node.id))}, class={_q(classes)}, "
            f'label={_card(node, theme)}, shape=box, style="filled", '
            f'fillcolor="{theme.background}", color="{border}", penwidth={node.pen_width:g}, margin="{margin}"];'
        )
    # External sources share the entry column; downstream inputs retain their rank.
    targets = {edge.target for edge in diagram.edges}
    grouped = {member for group in diagram.groups for member in group.members}
    sources = [
        node.id for node in diagram.nodes
        if node.kind == "input" and node.id not in targets and node.id not in grouped
    ]
    if len(sources) > 1:
        lines.append("{ rank=same; " + " ".join(f"{_q(node)};" for node in sources) + " }")
    row_membership = {}
    for group in diagram.groups:
        lines.append(
            f"subgraph {_q('cluster_' + _svg_id(group.id))} {{ label=<<B>{html.escape(group.label.upper())}</B>>; "
            f'color="{theme.line}"; fontcolor="{theme.ink}"; fontname="Menlo"; '
            f'fontsize={theme.label_size:g}; penwidth=1.2; style="solid"; margin=16; labeljust="l";'
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
                    '[style=invis, weight=100, constraint=true];'
                )
        lines.append("}")
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
                theme.training_line,
                "none" if edge.frozen else "vee",
                "dotted",
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
                (f'label=<<B>{_label(edge.label, theme.wrap_columns)}</B>>' if edge.label else 'label=""'),
                f"constraint={'true' if edge.constraint and not internal_row else 'false'}",
                f"penwidth={edge.pen_width:g}",
            ]
        )
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
    height_to_width_ratio: float | None = 0.58,
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
        args = ["dot", f"-T{suffix[1:]}", str(dot_path), "-o", str(output)]
        if suffix == ".png":
            args.insert(1, f"-Gdpi={144 * scale}")
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Graphviz failed: {result.stderr.strip()}")
    return output
