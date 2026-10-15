"""Renderer-neutral structural diagrams and their visual theme."""

from __future__ import annotations

from dataclasses import dataclass, field


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
