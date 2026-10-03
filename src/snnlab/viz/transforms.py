"""Reusable numerical transforms, independent of Matplotlib."""

from __future__ import annotations

import numpy as np


def exponential_trace(
    events: np.ndarray, *, dt_ms: float, tau_ms: float
) -> np.ndarray:
    events = np.asarray(events)
    if events.ndim != 2:
        raise ValueError("events must have shape (time, units)")
    if dt_ms <= 0 or tau_ms <= 0:
        raise ValueError("dt_ms and tau_ms must be positive")
    trace = np.zeros(events.shape, dtype=np.float32)
    decay = np.exp(-dt_ms / tau_ms)
    for step in range(1, len(events)):
        trace[step] = trace[step - 1] * decay + events[step - 1]
    return trace


def projection_activity(
    weights: np.ndarray,
    source_trace: np.ndarray,
    *,
    scale: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Return per-edge activity without imposing a visual representation."""

    weight = np.asarray(weights)
    trace = np.asarray(source_trace)
    source, target = np.nonzero(weight)
    values = trace[:, source] * weight[source, target]
    scale_array = np.asarray(scale)
    return values * (
        scale_array[:, None] if scale_array.ndim else float(scale_array)
    )


def representative_frame(
    *signals: np.ndarray, candidates: np.ndarray | None = None
) -> int:
    """Select the candidate with greatest simultaneous aggregate activity."""

    if not signals:
        raise ValueError("at least one signal is required")
    length = len(np.asarray(signals[0]))
    if any(len(np.asarray(signal)) != length for signal in signals):
        raise ValueError("signals must share a time dimension")
    indices = (
        np.arange(length)
        if candidates is None
        else np.asarray(candidates, dtype=int)
    )
    score = np.zeros(len(indices), dtype=float)
    for signal in signals:
        values = np.asarray(signal)[indices]
        score += values.reshape(len(indices), -1).sum(axis=1)
    return int(indices[int(np.argmax(score))])
