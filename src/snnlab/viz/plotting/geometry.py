"""Backend-independent figure geometry and point layouts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FigureRect:
    """A rectangle in normalized figure coordinates."""

    x: float
    y: float
    width: float
    height: float

    @property
    def mpl(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.width, self.height

    def inset(
        self,
        padding: float | tuple[float, float, float, float],
    ) -> FigureRect:
        """Inset by relative left, bottom, right and top fractions."""

        if isinstance(padding, (int, float)):
            left = bottom = right = top = float(padding)
        else:
            left, bottom, right, top = padding
        if min(left, bottom, right, top) < 0:
            raise ValueError("figure-grid padding must be non-negative")
        width = self.width * (1 - left - right)
        height = self.height * (1 - bottom - top)
        if width <= 0 or height <= 0:
            raise ValueError("figure-grid padding consumes the region")
        return FigureRect(
            self.x + self.width * left,
            self.y + self.height * bottom,
            width,
            height,
        )


@dataclass(frozen=True)
class FigureRegion:
    """A named rectangular selection of grid tracks."""

    name: str
    row: int
    column: int
    rowspan: int = 1
    colspan: int = 1
    reserved: bool = False


def _tracks(value: int | Sequence[float], label: str) -> tuple[float, ...]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"figure-grid {label} must be positive")
        return (1.0,) * value
    tracks = tuple(float(item) for item in value)
    if not tracks or any(item <= 0 for item in tracks):
        raise ValueError(f"figure-grid {label} must contain positive weights")
    return tracks


def grid_layout(
    count: int,
    *,
    columns: int,
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    if count < 0:
        raise ValueError("count must be non-negative")
    if columns <= 0:
        raise ValueError("columns must be positive")
    if count == 0:
        return np.empty((0, 2), dtype=float)
    rows = int(np.ceil(count / columns))
    x, y = np.meshgrid(np.linspace(*x_range, columns), np.linspace(*y_range, rows))
    return np.c_[x.ravel()[:count], y.ravel()[:count]]
