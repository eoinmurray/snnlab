"""Deterministic geometry helpers with no plotting-backend dependency."""

from __future__ import annotations

import numpy as np


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
    x, y = np.meshgrid(
        np.linspace(*x_range, columns), np.linspace(*y_range, rows)
    )
    return np.c_[x.ravel()[:count], y.ravel()[:count]]
