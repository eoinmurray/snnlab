"""Timeline sampling and backend encoding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation


@dataclass(frozen=True)
class FrameTimeline:
    steps: np.ndarray
    dt_ms: float

    @classmethod
    def sample(
        cls, steps: int, *, frames: int, dt_ms: float
    ) -> "FrameTimeline":
        if steps <= 0 or frames <= 0 or frames > steps:
            raise ValueError("require 0 < frames <= steps")
        return cls(np.linspace(0, steps - 1, frames, dtype=int), dt_ms)

    @classmethod
    def compose(
        cls,
        segments: list[tuple[int, int, int]],
        *,
        dt_ms: float,
    ) -> "FrameTimeline":
        """Compose paced, repeated, or held inclusive step ranges.

        Each segment is ``(start_step, end_step, frame_count)``. A reversed
        range plays backward and equal endpoints create a hold. This gives a
        composition full pacing control without coupling it to a plot type.
        """

        if not segments:
            raise ValueError("at least one timeline segment is required")
        sampled = []
        for start, end, frames in segments:
            if start < 0 or end < 0 or frames <= 0:
                raise ValueError("timeline steps must be non-negative and frames positive")
            sampled.append(np.linspace(start, end, frames, dtype=int))
        return cls(np.concatenate(sampled), dt_ms)

    def time_ms(self, frame: int) -> float:
        return float(self.steps[frame] * self.dt_ms)


def save_animation(
    figure,
    update: Callable[[int], object],
    output: str | Path,
    *,
    frames: int,
    fps: int = 25,
    bitrate: int = 3800,
) -> FuncAnimation:
    animation = FuncAnimation(
        figure, update, frames=frames, interval=1000 / fps, blit=False
    )
    animation.save(
        Path(output), writer=FFMpegWriter(fps=fps, bitrate=bitrate)
    )
    return animation
