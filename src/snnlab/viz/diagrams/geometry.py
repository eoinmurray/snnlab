"""Backend-independent resolved diagram geometry and inspection contracts."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .model import Diagram, DiagramTheme
from .spec import DiagramLayout


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self):
        return self.x + self.width / 2, self.y + self.height / 2

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    def expanded(self, padding):
        return Box(
            self.x - padding,
            self.y - padding,
            self.width + 2 * padding,
            self.height + 2 * padding,
        )


@dataclass(frozen=True)
class ResolvedDiagram:
    """Inspectable geometry and a rendering scene; positions use top-left canvas axes."""

    diagram: Diagram
    layout: DiagramLayout
    theme: DiagramTheme
    nodes: dict[str, Box]
    groups: dict[str, Box]
    edges: dict[str, dict]
    labels: dict[str, Box]
    ports: dict[str, dict]
    canvas: Box
    viewport: Box
    diagnostics: tuple[dict, ...]
    scene: dict
    math_vectors: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "schema": "snnlab.resolved-diagram/v1",
            "units": "logical canvas units; x right, y down; positions are box centres",
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
            "groups": {k: asdict(v) for k, v in self.groups.items()},
            "edges": copy.deepcopy(self.edges),
            "labels": {k: asdict(v) for k, v in self.labels.items()},
            "ports": copy.deepcopy(self.ports),
            "canvas": asdict(self.canvas),
            "viewport": asdict(self.viewport),
            "diagnostics": list(self.diagnostics),
            "layout": self.layout.to_dict(),
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
        return path

    def move(self, identifier: str, position: tuple[float, float]) -> ResolvedDiagram:
        from .placement import ancestors, group_specs
        from .resolved import resolve_diagram

        if identifier in self.nodes:
            positions = {k: b.center for k, b in self.nodes.items()}
            positions[identifier] = position
        elif identifier in self.groups:
            settings = group_specs(self.diagram, self.layout)
            dx, dy = (position[i] - self.groups[identifier].center[i] for i in (0, 1))
            positions = {
                k: (b.center[0] + dx, b.center[1] + dy)
                if k in settings[identifier].members
                else b.center
                for k, b in self.nodes.items()
            }
            overrides = dict(self.layout.groups)
            for k, group in settings.items():
                if (
                    k == identifier or identifier in ancestors(k, settings)
                ) and group.position is not None:
                    overrides[k] = replace(
                        group,
                        position=(group.position[0] + dx, group.position[1] + dy),
                    )
            spec = replace(self.layout, groups=overrides)
            return resolve_diagram(
                self.diagram, layout=spec.with_positions(positions), theme=self.theme
            )
        else:
            raise ValueError(f"unknown node or group {identifier!r}")
        return resolve_diagram(
            self.diagram, layout=self.layout.with_positions(positions), theme=self.theme
        )
