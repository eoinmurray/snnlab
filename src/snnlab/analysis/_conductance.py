"""Geometric descriptors of paired conductance trajectories."""

from __future__ import annotations

import numpy as np


def conductance_loop_score(g_e, g_i):
    """Bounded geometric score for a consistently rotating E/I conductance loop.

    The trajectory is centred and closed, then scored as the isoperimetric
    quotient ``4πA/L²`` multiplied by signed-rotation coherence. The result is
    dimensionless, invariant to uniform conductance scaling, and lies in [0, 1].
    A circular, consistently traversed loop approaches one; an irregular path
    with cancelling rotation approaches zero.
    """
    x = np.asarray(g_e, dtype=float).reshape(-1)
    y = np.asarray(g_i, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 3:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    x = np.r_[x, x[0]]
    y = np.r_[y, y[0]]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    signed_twice_area = float(cross.sum())
    absolute_sweep = float(np.abs(cross).sum())
    perimeter = float(np.hypot(np.diff(x), np.diff(y)).sum())
    if absolute_sweep <= 0 or perimeter <= 0:
        return 0.0
    area = 0.5 * abs(signed_twice_area)
    compactness = min(1.0, 4.0 * np.pi * area / (perimeter * perimeter))
    direction_coherence = abs(signed_twice_area) / absolute_sweep
    return float(np.clip(compactness * direction_coherence, 0.0, 1.0))


def rolling_conductance_loop_score(
    g_e,
    g_i,
    dt,
    *,
    window_ms=40.0,
    stride_ms=5.0,
    normalize_percentiles=(10.0, 95.0),
):
    """Rolling conductance-loop score plus an explicitly run-relative [0, 1] view."""
    x = np.asarray(g_e, dtype=float).reshape(-1)
    y = np.asarray(g_i, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("g_e and g_i must have equal length")
    window_steps = max(3, int(round(float(window_ms) / float(dt))))
    stride_steps = max(1, int(round(float(stride_ms) / float(dt))))
    ends = np.arange(window_steps, x.size + 1, stride_steps)
    raw = np.array(
        [
            conductance_loop_score(
                x[end - window_steps : end], y[end - window_steps : end]
            )
            for end in ends
        ]
    )
    times_ms = ends.astype(float) * float(dt)
    lo, hi = np.percentile(raw, normalize_percentiles) if raw.size else (0.0, 0.0)
    normalized = np.clip((raw - lo) / max(float(hi - lo), 1e-12), 0.0, 1.0)
    return {
        "times_ms": times_ms,
        "raw": raw,
        "normalized": normalized,
        "normalization_low": float(lo),
        "normalization_high": float(hi),
    }
