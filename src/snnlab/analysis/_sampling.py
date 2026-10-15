"""Private windowing, binning and smoothing rules."""

from __future__ import annotations

import numpy as np

from ._validation import _integer, _positive_ms, _raster


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
    from scipy.signal import convolve

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
