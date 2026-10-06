"""Population burst detection, separate cycle construction and occupancy counting."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ._activity import _integer, _smooth, _window
from ._spikes import _positive_ms, _raster


def detect_population_bursts(
    spikes,
    dt,
    *,
    window=None,
    smoothing_ms=1.0,
    height=None,
    relative_height=0.05,
    prominence=None,
    min_separation_ms=1.0,
):
    """Find population count peaks; thresholds are spikes per simulation step."""
    values, start, end = _window(spikes, window)
    dt = _positive_ms(dt, "dt")
    separation = _positive_ms(min_separation_ms, "min_separation_ms")
    sigma = float(smoothing_ms)
    relative = float(relative_height)
    if (
        not np.isfinite(sigma)
        or sigma < 0
        or not np.isfinite(relative)
        or not 0 <= relative <= 1
    ):
        raise ValueError("invalid smoothing or relative_height")
    height = None if height is None else float(height)
    prominence = None if prominence is None else float(prominence)
    for value in (height, prominence):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError("height/prominence must be finite and nonnegative")
    # A one-step minimum reproduces the historical Gaussian estimator when enabled.
    sigma_steps = max(1.0, sigma / dt) if sigma else 0.0
    smooth = _smooth(values.sum(axis=1).astype(np.float32), sigma_steps)
    threshold = (
        float(height)
        if height is not None
        else relative * float(smooth.max())
        if len(smooth)
        else 0.0
    )
    distance = max(1, int(separation / dt))
    peaks, properties = find_peaks(
        smooth, height=threshold, prominence=prominence, distance=distance
    )
    return {
        "peak_steps": peaks.astype(np.int64) + start,
        "peak_times_ms": (peaks + start) * dt,
        "peak_heights": properties["peak_heights"],
        "smoothed_counts": smooth,
        "status": "ok" if len(peaks) else "no_peaks",
        "metadata": {
            "dt_ms": dt,
            "window": [start, end],
            "smoothing_ms": sigma,
            "realized_sigma_steps": sigma_steps,
            "smoothing_truncate": 4.0,
            "smoothing_boundary": "zero",
            "height": threshold,
            "relative_height": relative,
            "prominence": prominence,
            "requested_min_separation_ms": separation,
            "min_separation_steps": distance,
        },
    }


def cycle_boundaries(peak_steps, steps_count, *, policy="midpoints", start_step=0):
    """Return absolute half-open edges: midpoint cycles or complete peak-to-peak cycles."""
    length, start = (
        _integer(steps_count, "steps_count"),
        _integer(start_step, "start_step"),
    )
    peaks = np.asarray(peak_steps)
    if (
        peaks.ndim != 1
        or (len(peaks) and not np.issubdtype(peaks.dtype, np.integer))
        or np.any(np.diff(peaks) <= 0)
        or np.any(peaks < start)
        or np.any(peaks >= start + length)
    ):
        raise ValueError(
            "peak_steps must be strictly increasing integer steps inside the window"
        )
    if policy not in {"midpoints", "peak_to_peak"}:
        raise ValueError("policy must be midpoints or peak_to_peak")
    peaks = peaks.astype(np.int64)
    edges = (
        np.concatenate(([start], (peaks[:-1] + peaks[1:]) // 2, [start + length]))
        if policy == "midpoints" and len(peaks)
        else peaks.copy()
        if policy == "peak_to_peak" and len(peaks) >= 2
        else np.empty(0, dtype=np.int64)
    )
    if len(edges) and np.any(np.diff(edges) <= 0):
        raise ValueError("boundary policy produces an empty cycle")
    return {
        "edges": edges,
        "cycle_count": max(0, len(edges) - 1),
        "status": "ok" if len(edges) > 1 else "no_cycles",
        "metadata": {
            "policy": policy,
            "window": [start, start + length],
            "includes_partial_edge_cycles": policy == "midpoints",
        },
    }


def cycle_spike_counts(spikes, boundaries, *, start_step=0):
    """Count per-cell events in supplied absolute half-open cycle edges."""
    values = _raster(spikes)
    start = _integer(start_step, "start_step")
    edges = np.asarray(boundaries)
    if (
        edges.ndim != 1
        or (len(edges) and not np.issubdtype(edges.dtype, np.integer))
        or np.any(np.diff(edges) <= 0)
        or np.any(edges < start)
        or np.any(edges > start + len(values))
    ):
        raise ValueError(
            "boundaries must be increasing integer steps within the raster window"
        )
    if len(edges) == 1:
        raise ValueError("supply no edges or at least two cycle boundaries")
    indices = edges.astype(np.int64) - start
    cumulative = np.concatenate(
        (
            np.zeros((1, values.shape[1]), dtype=np.int64),
            np.cumsum(values, axis=0, dtype=np.int64),
        )
    )
    counts = cumulative[indices[1:]] - cumulative[indices[:-1]]
    return {
        "counts": counts,
        "status": "ok" if len(counts) else "no_cycles",
        "metadata": {
            "start_step": start,
            "boundaries": edges.tolist(),
            "axis_order": ["cycle", "cell"],
        },
    }


def cycle_occupancy_distribution(counts, *, max_count=None):
    """Opportunity-pooled occupancy histogram; optional final overflow bucket."""
    values = np.asarray(counts)
    if (
        values.ndim != 2
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(values < 0)
    ):
        raise ValueError("counts must be a nonnegative integer (cycles, cells) array")
    maximum = (
        int(values.max())
        if values.size and max_count is None
        else 0
        if max_count is None
        else _integer(max_count, "max_count")
    )
    histogram = np.bincount(
        np.minimum(values.ravel(), maximum).astype(np.int64), minlength=maximum + 1
    )
    opportunities = values.size
    return {
        "spike_counts": np.arange(maximum + 1),
        "opportunity_counts": histogram,
        "fractions": histogram / opportunities
        if opportunities
        else np.full(maximum + 1, np.nan),
        "opportunities": opportunities,
        "status": "ok" if opportunities else "no_opportunities",
        "metadata": {
            "aggregation": "opportunity_pooled",
            "cycles": values.shape[0],
            "cells": values.shape[1],
            "max_count": maximum,
            "last_bucket_includes_overflow": max_count is not None,
        },
    }
