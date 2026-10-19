"""Validated sample index and vectorized binary binning for event snapshots."""

from __future__ import annotations

import math

import numpy as np
import torch

from snnlab.sim.inputs.encoders import transfer_compact


class EventIndex:
    """Index an immutable snapshot once, then visit only requested samples."""

    def __init__(self, arrays, *, sample_count, channels, duration_ms, dt_ms):
        keys = ("event_sample", "event_time_ms", "event_channel")
        missing = [key for key in keys if key not in arrays]
        if missing:
            raise ValueError(f"event dataset snapshot is missing keys {missing}")
        samples, times, event_channels = (np.asarray(arrays[key]) for key in keys)
        if (
            any(value.ndim != 1 for value in (samples, times, event_channels))
            or not (len(samples) == len(times) == len(event_channels))
            or not np.issubdtype(samples.dtype, np.integer)
            or not np.issubdtype(event_channels.dtype, np.integer)
            or not np.issubdtype(times.dtype, np.floating)
        ):
            raise ValueError(
                "event dataset coordinates must be equal-length sample/channel integer and time floating arrays"
            )
        raw_steps = duration_ms / dt_ms
        if (
            not math.isfinite(raw_steps)
            or duration_ms <= 0
            or not math.isclose(raw_steps, round(raw_steps), abs_tol=1e-9)
        ):
            raise ValueError(
                "event-bin encoder duration must be a positive integer number of timesteps"
            )
        if (
            not np.isfinite(times).all()
            or np.any(samples < 0)
            or np.any(samples >= sample_count)
            or np.any(event_channels < 0)
            or np.any(event_channels >= channels)
            or np.any(times < 0)
            or np.any(times >= duration_ms)
        ):
            raise ValueError("event dataset coordinates are out of snapshot bounds")
        self.steps_count = int(round(raw_steps))
        self.channels = channels
        order = np.argsort(samples, kind="stable")
        # Python's old float(time_ms) / dt_ms binning used double precision.
        self.steps = np.minimum(
            (times[order].astype(np.float64) / dt_ms).astype(np.int64),
            self.steps_count - 1,
        )
        self.event_channels = event_channels[order].astype(np.int64)
        self.offsets = np.concatenate(
            (
                [0],
                np.cumsum(
                    np.bincount(samples.astype(np.int64), minlength=sample_count)
                ),
            )
        )

    def encode(self, selected, *, device="cpu"):
        starts = self.offsets[selected]
        lengths = self.offsets[selected + 1] - starts
        retained = int(lengths.sum())
        batches = np.repeat(np.arange(len(selected)), lengths)
        positions = np.repeat(
            starts - (np.cumsum(lengths) - lengths), lengths
        ) + np.arange(retained)
        linear = (
            self.steps[positions] * len(selected) + batches
        ) * self.channels + self.event_channels[positions]
        unique = np.unique(linear)
        spikes = torch.zeros(
            (self.steps_count, len(selected), self.channels),
            device=device,
            dtype=torch.float32,
        )
        spikes.view(-1)[transfer_compact(torch.from_numpy(unique), device)] = 1
        return spikes, retained, retained - len(unique)
