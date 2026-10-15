"""Population activity, rates and per-cell spike counts."""

from __future__ import annotations

import numpy as np

from ._sampling import _bin, _smooth, _window
from ._validation import _population_size, _positive_ms, _raster


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


def per_neuron_firing_rates(spikes, dt, *, window=None):
    """Return per-cell Hz over a half-open step window; empty duration is undefined."""
    values, start, end = _window(spikes, window)
    dt = _positive_ms(dt, "dt")
    rates = (
        values.sum(axis=0) / (len(values) * dt / 1000)
        if len(values)
        else np.full(values.shape[1], np.nan)
    )
    return {
        "rates_hz": rates,
        "status": "ok" if len(values) else "empty_window",
        "metadata": {
            "dt_ms": dt,
            "window": [start, end],
            "duration_ms": len(values) * dt,
        },
    }


def binned_spike_counts(
    spikes,
    dt,
    *,
    bin_ms=2.0,
    window=None,
    rounding="floor",
    trailing="drop",
    alignment="left",
):
    """Per-cell integer counts in bins anchored at window start, with explicit labels."""
    result = _bin(spikes, dt, bin_ms, window, rounding, trailing)
    dt = result["metadata"]["dt_ms"]
    if alignment not in {"left", "center", "right"}:
        raise ValueError("alignment must be left, center or right")
    result["times_ms"] = dt * (
        result["start_steps"]
        if alignment == "left"
        else result["end_steps"]
        if alignment == "right"
        else (result["start_steps"] + result["end_steps"]) / 2
    )
    result["metadata"]["alignment"] = alignment
    result["status"] = "ok" if len(result["counts"]) else "no_bins"
    return result


def population_rate_trace(
    spikes,
    dt,
    *,
    bin_ms=2.0,
    window=None,
    rounding="floor",
    trailing="drop",
    alignment="left",
    smoothing_ms=0.0,
):
    """Mean Hz per cell in each bin, optionally Gaussian smoothed with zero padding."""
    result = binned_spike_counts(
        spikes,
        dt,
        bin_ms=bin_ms,
        window=window,
        rounding=rounding,
        trailing=trailing,
        alignment=alignment,
    )
    sigma = float(smoothing_ms)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("smoothing_ms must be finite and non-negative")
    if (
        sigma
        and len(result["duration_ms"])
        and np.any(result["duration_ms"] != result["metadata"]["realized_bin_ms"])
    ):
        raise ValueError("smoothing requires equal-duration bins; drop the partial bin")
    rate = (
        result["counts"].sum(axis=1)
        / result["metadata"]["cells"]
        / (result["duration_ms"] / 1000)
    )
    result["rate_hz"] = _smooth(rate, sigma / result["metadata"]["realized_bin_ms"])
    result["metadata"].update(
        smoothing_ms=sigma,
        smoothing="gaussian",
        smoothing_truncate=4.0,
        smoothing_boundary="zero",
    )
    return result
