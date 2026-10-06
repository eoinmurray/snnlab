"""Windowed activity, shared binning and per-neuron interval measurements."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.signal import convolve

from ._spikes import _positive_ms, _raster


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _window(spikes, window):
    values = _raster(spikes)
    start, end = (0, len(values)) if window is None else window
    start, end = _integer(start, "window start"), _integer(end, "window end")
    if not 0 <= start <= end <= len(values):
        raise ValueError("window must be half-open step bounds within the raster")
    return values[start:end], start, end


def _smooth(values, sigma_steps):
    if sigma_steps == 0 or not len(values):
        return values.copy()
    radius = int(np.ceil(4 * sigma_steps))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_steps) ** 2)
    kernel /= kernel.sum()
    return convolve(values, kernel, mode="same", method="direct")


def _bin(spikes, dt, bin_ms, window, rounding, trailing):
    values, start, end = _window(spikes, window)
    dt, bin_ms = _positive_ms(dt, "dt"), _positive_ms(bin_ms, "bin_ms")
    ratio = bin_ms / dt
    if rounding == "floor":
        width = max(1, int(ratio))
    elif rounding == "nearest":
        width = max(1, int(round(ratio)))
    elif rounding == "exact":
        width = int(round(ratio))
        if width < 1 or not np.isclose(ratio, width, rtol=0, atol=1e-9):
            raise ValueError("exact bin width must be an integer multiple of dt")
    else:
        raise ValueError("rounding must be floor, nearest or exact")
    if trailing not in {"drop", "include", "error"}:
        raise ValueError("trailing must be drop, include or error")
    remainder = len(values) % width
    if trailing == "error" and remainder:
        raise ValueError("window has an incomplete trailing bin")
    stop = len(values) if trailing == "include" else len(values) - remainder
    starts = np.arange(0, stop, width, dtype=np.int64)
    ends = np.minimum(starts + width, len(values))
    counts = (
        np.add.reduceat(values[:stop], starts, axis=0, dtype=np.int64)
        if len(starts)
        else np.empty((0, values.shape[1]), dtype=np.int64)
    )
    return {
        "counts": counts,
        "start_steps": starts + start,
        "end_steps": ends + start,
        "duration_ms": (ends - starts) * dt,
        "metadata": {
            "dt_ms": dt,
            "window": [start, end],
            "requested_bin_ms": bin_ms,
            "cells": values.shape[1],
            "bin_steps": width,
            "realized_bin_ms": width * dt,
            "rounding": rounding,
            "trailing": trailing,
            "discarded_steps": remainder if trailing == "drop" else 0,
        },
    }


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


def inter_spike_intervals(spikes, dt, *, window=None, trial_boundaries=None):
    """Per-cell intervals in ms, never crossing supplied presentation boundaries."""
    values, start, end = _window(spikes, window)
    dt = _positive_ms(dt, "dt")
    boundaries = (
        np.array([0, len(spikes)], dtype=np.int64)
        if trial_boundaries is None
        else np.asarray(trial_boundaries)
    )
    if (
        boundaries.ndim != 1
        or len(boundaries) < 2
        or not np.issubdtype(boundaries.dtype, np.integer)
        or boundaries[0] != 0
        or boundaries[-1] != len(spikes)
        or np.any(np.diff(boundaries) <= 0)
    ):
        if not (len(spikes) == 0 and trial_boundaries is None):
            raise ValueError(
                "trial_boundaries must strictly partition the full raster from 0 to T"
            )
    intervals, trial_ids, spike_counts = (
        [],
        [],
        np.zeros((len(boundaries) - 1, values.shape[1]), dtype=np.int64),
    )
    for cell in range(values.shape[1]):
        pieces, ids = [], []
        for trial, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            events = (
                np.flatnonzero(
                    values[max(a, start) - start : max(min(b, end) - start, 0), cell]
                )
                if max(a, start) < min(b, end)
                else np.array([], dtype=int)
            )
            spike_counts[trial, cell] = len(events)
            pieces.append(np.diff(events) * dt)
            ids.append(np.full(max(0, len(events) - 1), trial, dtype=np.int64))
        intervals.append(np.concatenate(pieces))
        trial_ids.append(np.concatenate(ids))
    return {
        "intervals_ms": tuple(intervals),
        "trial_indices": tuple(trial_ids),
        "spike_counts": spike_counts,
        "status": "ok" if any(len(x) for x in intervals) else "no_intervals",
        "metadata": {
            "dt_ms": dt,
            "window": [start, end],
            "trial_boundaries": boundaries.tolist(),
        },
    }


def isi_cv(
    spikes,
    dt,
    *,
    window=None,
    trial_boundaries=None,
    min_spikes=3,
    ddof=0,
    aggregation="none",
):
    """Per-cell ISI CV, NaN for insufficient within-trial spikes; optional mean/median."""
    min_spikes, ddof = _integer(min_spikes, "min_spikes", 2), _integer(ddof, "ddof")
    if aggregation not in {"none", "mean", "median"}:
        raise ValueError("aggregation must be none, mean or median")
    result = inter_spike_intervals(
        spikes, dt, window=window, trial_boundaries=trial_boundaries
    )
    cvs = np.full(len(result["intervals_ms"]), np.nan)
    interval_counts = np.zeros(len(cvs), dtype=np.int64)
    for cell, (intervals, ids) in enumerate(
        zip(result["intervals_ms"], result["trial_indices"])
    ):
        kept = intervals[result["spike_counts"][ids, cell] >= min_spikes]
        interval_counts[cell] = len(kept)
        if len(kept) > ddof:
            cvs[cell] = kept.std(ddof=ddof) / kept.mean()
    valid = np.isfinite(cvs)
    aggregate = (
        None
        if aggregation == "none" or not valid.any()
        else float(
            np.mean(cvs[valid]) if aggregation == "mean" else np.median(cvs[valid])
        )
    )
    return {
        "cv": cvs,
        "valid": valid,
        "interval_counts": interval_counts,
        "aggregate": aggregate,
        "status": "ok" if valid.any() else "insufficient_spikes",
        "metadata": {
            **result["metadata"],
            "min_spikes_per_trial": min_spikes,
            "ddof": ddof,
            "aggregation": aggregation,
        },
    }


def spike_count_correlations(
    spikes,
    dt,
    *,
    bin_ms=10.0,
    window=None,
    cells=None,
    rounding="floor",
    trailing="drop",
    constant="nan",
    aggregation="none",
):
    """Pearson correlations of per-cell bin counts; constant cells are explicit."""
    if constant not in {"nan", "omit", "error"} or aggregation not in {
        "none",
        "mean",
        "median",
    }:
        raise ValueError("invalid constant or aggregation policy")
    result = binned_spike_counts(
        spikes, dt, bin_ms=bin_ms, window=window, rounding=rounding, trailing=trailing
    )
    counts = result["counts"]
    ids = (
        np.arange(counts.shape[1], dtype=np.int64)
        if cells is None
        else np.asarray(cells)
    )
    if (
        ids.ndim != 1
        or len(ids) == 0
        or not np.issubdtype(ids.dtype, np.integer)
        or len(np.unique(ids)) != len(ids)
        or np.any(ids < 0)
        or np.any(ids >= counts.shape[1])
    ):
        raise ValueError("cells must be unique in-range integer indices")
    counts = counts[:, ids].astype(float)
    valid = (
        np.std(counts, axis=0) > 0
        if len(counts) >= 2
        else np.zeros(len(ids), dtype=bool)
    )
    if constant == "error" and not valid.all():
        raise ValueError("insufficient bins or constant neuron counts")
    excluded = ids[~valid]
    if constant == "omit":
        counts, ids = counts[:, valid], ids[valid]
        valid = np.ones(len(ids), dtype=bool)
    matrix = np.full((len(ids), len(ids)), np.nan)
    if valid.any():
        x = counts[:, valid] - counts[:, valid].mean(axis=0)
        norms = np.linalg.norm(x, axis=0)
        matrix[np.ix_(valid, valid)] = np.clip(
            (x.T @ x) / np.outer(norms, norms), -1, 1
        )
    pairs = matrix[np.triu_indices(len(ids), 1)]
    pairs = pairs[np.isfinite(pairs)]
    aggregate = (
        None
        if aggregation == "none" or not len(pairs)
        else float(np.mean(pairs) if aggregation == "mean" else np.median(pairs))
    )
    return {
        "correlations": matrix,
        "cells": ids,
        "excluded_cells": excluded,
        "pair_count": len(pairs),
        "aggregate": aggregate,
        "status": "ok" if len(pairs) else "no_valid_pairs",
        "metadata": {
            **result["metadata"],
            "constant": constant,
            "aggregation": aggregation,
            "selected_cells": ids.tolist(),
        },
    }
