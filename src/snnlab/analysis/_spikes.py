"""Basic population activity measurements for one dense spike presentation."""

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


def firing_rate(spikes, dt, *, n_cells=None):
    """Mean population firing rate per neuron in Hz for a binary (time, cells) raster.

    ``dt`` is the timestep in ms. By default the denominator uses the number
    of raster columns; ``n_cells`` explicitly overrides that denominator.
    An empty time axis has no duration and raises ValueError.
    """
    values = _raster(spikes)
    dt = _positive_ms(dt, "dt")
    cells = _population_size(values, n_cells)
    if values.shape[0] == 0:
        raise ValueError("firing_rate requires at least one timestep")
    duration_s = values.shape[0] * dt / 1000.0
    return float(values.sum() / (cells * duration_s))


def population_spike_count_cv(spikes, dt, *, bin_ms=2.0):
    """Population-count standard deviation / mean across complete time bins.

    This is not a per-neuron inter-spike-interval CV. ``dt`` and ``bin_ms``
    use ms; each bin uses max(1, int(bin_ms / dt)) steps (floor rounding).
    An incomplete trailing bin is discarded. Fewer than two complete bins
    or silent activity returns 0.0. Uses population standard deviation (ddof=0)
    and a 1e-9 floor on the mean, preserving historical simulator reporting.
    """
    # Import locally so the basic raster validators remain reusable by binning.
    from ._activity import binned_spike_counts

    bins = binned_spike_counts(spikes, dt, bin_ms=bin_ms)
    if len(bins["counts"]) <= 1:
        return 0.0
    counts = bins["counts"].sum(axis=1).astype(np.asarray(spikes).sum().dtype)
    return float(counts.std() / max(counts.mean(), 1e-9))


def active_fraction(spikes, *, n_cells=None):
    """Fraction of cells emitting at least one spike in a binary (time, cells) raster.

    Defaults to the raster's column count; ``n_cells`` explicitly overrides
    the denominator. An empty time axis returns 0.0. A denominator smaller
    than the observed active-cell count can produce a value above one.
    """
    values = _raster(spikes)
    cells = _population_size(values, n_cells)
    return float((values.sum(axis=0) > 0).sum()) / cells
