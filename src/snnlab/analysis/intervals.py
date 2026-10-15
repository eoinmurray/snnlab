"""Within-presentation per-neuron intervals and irregularity."""

from __future__ import annotations

import numpy as np

from ._sampling import _window
from ._validation import _integer, _positive_ms


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
