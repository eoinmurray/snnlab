"""Correlations between per-neuron binned spike counts."""

from __future__ import annotations

import numpy as np

from .activity import binned_spike_counts


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
