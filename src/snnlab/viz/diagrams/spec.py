"""Versioned, renderer-independent diagram presentation specifications."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

SCHEMA = "snnlab.diagram-layout/v1"


def _finite(value, name, *, minimum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _pair(value, name, *, minimum=None):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must have two coordinates")
    for coordinate in value:
        _finite(coordinate, name, minimum=minimum)
    return tuple(value)


def _choice(value, name, choices):
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}")


def _ids(value, name):
    if not isinstance(value, (tuple, list)) or any(
        not isinstance(x, str) or not x for x in value
    ):
        raise TypeError(f"{name} must be a sequence of nonempty IDs")
    return tuple(value)


def _decode(cls, value):
    if isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"expected {cls.__name__} or a mapping")
    unknown = value.keys() - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**value)


@dataclass(frozen=True)
class DiagramLabel:
    """Plain or MathText rows, plus placement in logical canvas units."""

    text: str
    position: float | None = None
    offset: tuple[float, float] = (0, 0)
    anchor: str = "center"
    avoid_collisions: bool = True

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise TypeError("label text must be a string")
        if self.position is not None:
            _finite(self.position, "label position", minimum=0)
            if self.position > 1:
                raise ValueError("label position must be between 0 and 1")
        object.__setattr__(self, "offset", _pair(self.offset, "label offset"))
        _choice(
            self.anchor, "label anchor", ("center", "top", "bottom", "left", "right")
        )
        if type(self.avoid_collisions) is not bool:
            raise TypeError("avoid_collisions must be a boolean")


def label(value):
    return (
        DiagramLabel(value) if isinstance(value, str) else _decode(DiagramLabel, value)
    )


@dataclass(frozen=True)
class CanvasLayout:
    direction: str = "right"
    padding: float = 32
    origin: tuple[float, float] = (0, 0)
    size: tuple[float, float] | None = None
    fit: str = "expand"
    alignment: tuple[str, str] = ("center", "center")

    def __post_init__(self):
        _choice(self.direction, "direction", ("right", "left", "down", "up"))
        _finite(self.padding, "canvas padding", minimum=0)
        object.__setattr__(self, "origin", _pair(self.origin, "canvas origin"))
        if self.size is not None:
            object.__setattr__(self, "size", _pair(self.size, "canvas size", minimum=1))
        _choice(self.fit, "canvas fit", ("expand", "error", "contain"))
        if len(self.alignment) != 2:
            raise ValueError("canvas alignment needs horizontal and vertical values")
        for value in self.alignment:
            _choice(value, "canvas alignment", ("start", "center", "end"))
        object.__setattr__(self, "alignment", tuple(self.alignment))


@dataclass(frozen=True)
class NodeLayout:
    position: tuple[float, float] | None = None
    relative_to: str | None = None
    anchor: str | None = None
    min_size: tuple[float, float] | None = None
    max_size: tuple[float, float] | None = None
    wrap_columns: int | None = None
    content_align: str | None = None
    padding: float | None = None
    clearance: float | None = None
    label: DiagramLabel | str | None = None
    annotations: tuple[DiagramLabel | str, ...] | None = None
    ports: dict[str, tuple[float, float]] | None = None

    def __post_init__(self):
        for name in ("position", "min_size", "max_size"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _pair(value, name, minimum=None if name == "position" else 1),
                )
        if self.anchor is not None:
            _choice(
                self.anchor,
                "node anchor",
                ("center", "top_left", "top_right", "bottom_left", "bottom_right"),
            )
        if self.content_align is not None:
            _choice(
                self.content_align, "content alignment", ("left", "center", "right")
            )
        if self.wrap_columns is not None and (
            type(self.wrap_columns) is not int or self.wrap_columns < 1
        ):
            raise ValueError("wrap_columns must be a positive integer")
        for name in ("padding", "clearance"):
            if getattr(self, name) is not None:
                _finite(getattr(self, name), name, minimum=0)
        if self.relative_to is not None and self.position is None:
            raise ValueError("relative_to requires a position")
        if self.label is not None:
            object.__setattr__(self, "label", label(self.label))
        if self.annotations is not None:
            object.__setattr__(
                self, "annotations", tuple(label(x) for x in self.annotations)
            )
        if self.ports is not None:
            ports = {
                name: _pair(p, f"port {name}", minimum=0)
                for name, p in self.ports.items()
            }
            for name, p in ports.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or name in {"top", "bottom", "left", "right", "auto"}
                ):
                    raise ValueError(f"invalid or reserved port name: {name!r}")
                if max(p) > 1 or not any(x in (0, 1) for x in p):
                    raise ValueError(
                        f"port {name} must lie on a normalized box boundary"
                    )
            object.__setattr__(self, "ports", ports)


@dataclass(frozen=True)
class GroupLayout:
    members: tuple[str, ...] | None = None
    parent: str | None = None
    position: tuple[float, float] | None = None
    arrangement: str | None = None
    order: tuple[str, ...] | None = None
    columns: int | None = None
    gap: float | None = None
    padding: float | None = None
    alignment: str | None = None
    title_position: str | None = None
    label: DiagramLabel | str | None = None
    min_size: tuple[float, float] | None = None
    max_size: tuple[float, float] | None = None
    node_defaults: NodeLayout | None = None

    def __post_init__(self):
        if self.members is not None:
            object.__setattr__(self, "members", _ids(self.members, "group members"))
        if self.order is not None:
            object.__setattr__(self, "order", _ids(self.order, "group order"))
        if self.position is not None:
            object.__setattr__(self, "position", _pair(self.position, "group position"))
        for name in ("min_size", "max_size"):
            if getattr(self, name) is not None:
                object.__setattr__(
                    self, name, _pair(getattr(self, name), name, minimum=1)
                )
        if self.arrangement is not None:
            _choice(
                self.arrangement, "group arrangement", ("auto", "row", "column", "grid")
            )
        if self.alignment is not None:
            _choice(self.alignment, "group alignment", ("start", "center", "end"))
        if self.title_position is not None:
            _choice(self.title_position, "title position", ("top", "bottom"))
        if self.columns is not None and (
            type(self.columns) is not int or self.columns < 1
        ):
            raise ValueError("grid columns must be a positive integer")
        for name in ("gap", "padding"):
            if getattr(self, name) is not None:
                _finite(getattr(self, name), name, minimum=0)
        if self.label is not None:
            object.__setattr__(self, "label", label(self.label))
        if self.node_defaults is not None:
            object.__setattr__(
                self, "node_defaults", _decode(NodeLayout, self.node_defaults)
            )


@dataclass(frozen=True)
class EdgeLayout:
    label: DiagramLabel | str | None = None
    source_port: str | None = None
    target_port: str | None = None
    routing: str | None = None
    waypoints: tuple[tuple[float, float], ...] | None = None
    clearance: float | None = None
    lane: float | None = None
    lane_side: str | None = None

    def __post_init__(self):
        if self.label is not None:
            object.__setattr__(self, "label", label(self.label))
        if self.routing is not None:
            _choice(self.routing, "edge routing", ("orthogonal", "polyline", "curved"))
        if self.waypoints is not None:
            object.__setattr__(
                self, "waypoints", tuple(_pair(p, "waypoint") for p in self.waypoints)
            )
        for name in ("clearance", "lane"):
            if getattr(self, name) is not None:
                _finite(getattr(self, name), name, minimum=0)
        if self.lane_side is not None:
            _choice(self.lane_side, "lane side", ("top", "bottom", "left", "right"))
        if self.lane_side is not None and self.lane is None:
            raise ValueError("lane_side requires a lane distance")
        for name in ("source_port", "target_port"):
            if getattr(self, name) is not None and (
                not isinstance(getattr(self, name), str) or not getattr(self, name)
            ):
                raise ValueError(f"{name} must be a nonempty port name")


@dataclass(frozen=True)
class LayoutDefaults:
    node: NodeLayout = field(default_factory=NodeLayout)
    group: GroupLayout = field(default_factory=GroupLayout)
    edge: EdgeLayout = field(default_factory=EdgeLayout)
    gap: float = 80
    parallel_gap: float = 16

    def __post_init__(self):
        for name, cls in (
            ("node", NodeLayout),
            ("group", GroupLayout),
            ("edge", EdgeLayout),
        ):
            object.__setattr__(self, name, _decode(cls, getattr(self, name)))
        for name in ("gap", "parallel_gap"):
            _finite(getattr(self, name), name, minimum=0)
        if self.node.position is not None or self.node.relative_to is not None:
            raise ValueError("default nodes cannot have positions")
        if any(
            getattr(self.group, name) is not None
            for name in ("members", "parent", "position", "order")
        ):
            raise ValueError(
                "default groups cannot specify membership, order or position"
            )


@dataclass(frozen=True)
class LayoutConstraint:
    kind: str
    nodes: tuple[str, ...]
    gap: float | None = None
    required: bool = True
    priority: float = 1
    name: str | None = None

    def __post_init__(self):
        _choice(
            self.kind,
            "constraint kind",
            (
                "above",
                "below",
                "left_of",
                "right_of",
                "align_x",
                "align_y",
                "order_x",
                "order_y",
                "equal_spacing_x",
                "equal_spacing_y",
                "non_overlap",
            ),
        )
        object.__setattr__(self, "nodes", _ids(self.nodes, "constraint nodes"))
        minimum = 3 if self.kind.startswith("equal_spacing") else 2
        if len(self.nodes) < minimum or len(set(self.nodes)) != len(self.nodes):
            raise ValueError(f"{self.kind} requires at least {minimum} distinct IDs")
        if (
            self.kind in {"above", "below", "left_of", "right_of"}
            and len(self.nodes) != 2
        ):
            raise ValueError(f"{self.kind} requires exactly two IDs")
        if self.gap is not None:
            _finite(self.gap, "constraint gap", minimum=0)
        _finite(self.priority, "constraint priority", minimum=0.000001)
        if type(self.required) is not bool:
            raise TypeError("constraint required must be a boolean")


def merged(*specs):
    """Later non-None fields override earlier settings, including empty labels."""
    cls = type(specs[0])
    values = {}
    for spec in specs:
        values.update(
            {
                f.name: getattr(spec, f.name)
                for f in fields(cls)
                if getattr(spec, f.name) is not None
            }
        )
    return cls(**values)


@dataclass(frozen=True)
class DiagramLayout:
    canvas: CanvasLayout = field(default_factory=CanvasLayout)
    defaults: LayoutDefaults = field(default_factory=LayoutDefaults)
    nodes: dict[str, NodeLayout] = field(default_factory=dict)
    groups: dict[str, GroupLayout] = field(default_factory=dict)
    edges: dict[str, EdgeLayout] = field(default_factory=dict)
    constraints: tuple[LayoutConstraint, ...] = ()
    non_overlap: bool = True
    direction: str | None = None
    gap: float | None = None
    positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    below: dict[str, str] = field(default_factory=dict)
    above: dict[str, str] = field(default_factory=dict)
    left_of: dict[str, str] = field(default_factory=dict)
    right_of: dict[str, str] = field(default_factory=dict)
    align_x: tuple[tuple[str, ...], ...] = ()
    align_y: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "canvas", _decode(CanvasLayout, self.canvas))
        object.__setattr__(self, "defaults", _decode(LayoutDefaults, self.defaults))
        for name, cls in (
            ("nodes", NodeLayout),
            ("groups", GroupLayout),
            ("edges", EdgeLayout),
        ):
            mapping = getattr(self, name)
            if not isinstance(mapping, dict) or any(
                not isinstance(k, str) or not k for k in mapping
            ):
                raise TypeError(
                    f"{name} must map nonempty semantic IDs to specifications"
                )
            object.__setattr__(
                self, name, {k: _decode(cls, v) for k, v in mapping.items()}
            )
        object.__setattr__(
            self,
            "constraints",
            tuple(_decode(LayoutConstraint, c) for c in self.constraints),
        )
        if self.direction is not None:
            _choice(self.direction, "direction", ("right", "left", "down", "up"))
            if (
                self.canvas.direction != "right"
                and self.canvas.direction != self.direction
            ):
                raise ValueError("direction conflicts with canvas.direction")
        if self.gap is not None:
            _finite(self.gap, "gap", minimum=0)
        if type(self.non_overlap) is not bool:
            raise TypeError("non_overlap must be a boolean")
        object.__setattr__(
            self,
            "positions",
            {k: _pair(p, "node position") for k, p in self.positions.items()},
        )
        for name in ("above", "below", "left_of", "right_of"):
            mapping = getattr(self, name)
            if not isinstance(mapping, dict) or any(
                not isinstance(k, str) or not k or not isinstance(v, str) or not v
                for k, v in mapping.items()
            ):
                raise TypeError(f"{name} must map node/group IDs to node/group IDs")
        for name in ("align_x", "align_y"):
            object.__setattr__(
                self, name, tuple(_ids(ids, name) for ids in getattr(self, name))
            )

    def all_constraints(self):
        result = list(self.constraints)
        for kind in ("above", "below", "left_of", "right_of"):
            result.extend(
                LayoutConstraint(kind, (a, b)) for a, b in getattr(self, kind).items()
            )
        for kind in ("align_x", "align_y"):
            result.extend(LayoutConstraint(kind, ids) for ids in getattr(self, kind))
        return tuple(result)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps({"schema": SCHEMA, **asdict(self)}, allow_nan=False)
        )

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop("schema", None) != SCHEMA:
            raise ValueError(f"layout schema must be {SCHEMA!r}")
        return _decode(cls, value)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")
        return path

    @classmethod
    def read(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def with_positions(self, positions):
        nodes = dict(self.nodes)
        for identifier, position in {**self.positions, **positions}.items():
            nodes[identifier] = replace(
                nodes.get(identifier, NodeLayout()),
                position=position,
                relative_to=None,
                anchor="center",
            )
        return replace(self, nodes=nodes, positions={})


def check_legacy_labels(layout, connection_labels):
    if isinstance(layout, dict):
        layout = DiagramLayout.from_dict(layout)
    if layout is not None and not isinstance(layout, DiagramLayout):
        raise TypeError("layout must be DiagramLayout or a versioned mapping")
    if layout is not None and connection_labels:
        for identifier, text in connection_labels.items():
            settings = merged(
                layout.defaults.edge, layout.edges.get(identifier, EdgeLayout())
            )
            if settings.label is not None and settings.label.text != text:
                raise ValueError(
                    f"conflicting connection_labels and layout annotation for {identifier!r}"
                )
    return layout
