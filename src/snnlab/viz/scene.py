"""An intentionally thin composition layer over Matplotlib."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import matplotlib.pyplot as plt

from .contracts import Recording
from .figure_grid import FigureGrid


class Panel(Protocol):
    def draw(self, ax: Any, recording: Recording) -> Any: ...

    def update(self, frame: int) -> object: ...


@dataclass
class Scene:
    """Compose reusable and bespoke panels without hiding Matplotlib."""

    recording: Recording
    figure: Any = None
    layout: FigureGrid | None = None
    panels: list[Panel] = field(default_factory=list)
    callbacks: list[Callable[[int, "Scene"], object]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        if self.figure is None:
            self.figure = plt.figure()

    def add(
        self,
        panel: Panel,
        *,
        axis: Any = None,
        region: str | None = None,
    ) -> Any:
        """Draw a panel on an axis, a named region, or a default axis."""

        if axis is not None and region is not None:
            raise ValueError("provide either an axis or a figure-grid region")
        if region is not None:
            if self.layout is None:
                raise ValueError("a figure-grid region requires a Scene layout")
            ax = self.layout.add_axes(self.figure, region)
        else:
            ax = axis if axis is not None else self.figure.add_subplot(1, 1, 1)
        panel.draw(ax, self.recording)
        self.panels.append(panel)
        return ax

    def on_frame(self, callback: Callable[[int, "Scene"], object]) -> None:
        self.callbacks.append(callback)

    def update(self, frame: int) -> tuple[object, ...]:
        artists = [panel.update(frame) for panel in self.panels]
        artists.extend(
            callback(frame, self) for callback in self.callbacks
        )
        return tuple(artists)
