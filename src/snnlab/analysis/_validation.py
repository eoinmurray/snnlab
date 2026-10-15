"""Private validation for dense spike measurements."""

from __future__ import annotations

from numbers import Integral

import numpy as np


def _raster(spikes):
    values = np.asarray(spikes)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("spikes must have shape (time, cells) with at least one cell")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("spikes must contain only binary 0/1 events")
    return values


def _positive_ms(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive (milliseconds)")
    return value


def _population_size(values, n_cells):
    if n_cells is None:
        return values.shape[1]
    if isinstance(n_cells, bool) or not isinstance(n_cells, Integral) or n_cells <= 0:
        raise ValueError("n_cells must be a positive integer or None")
    return int(n_cells)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)
