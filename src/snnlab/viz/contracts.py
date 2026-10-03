"""Typed contracts between simulation output and visual composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class RecordingError(ValueError):
    """Raised when an input cannot support truthful rendering."""


@dataclass(frozen=True)
class Recording:
    """An open-ended, renderer-neutral view of one time-series recording."""

    dt_ms: float
    signals: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: Path | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt_ms) or self.dt_ms <= 0:
            raise RecordingError("dt_ms must be finite and positive")
        lengths = {
            np.asarray(value).shape[0]
            for value in self.signals.values()
            if np.asarray(value).ndim
        }
        if not lengths:
            raise RecordingError("recording contains no time-varying signals")
        if len(lengths) != 1:
            raise RecordingError(
                f"time-varying signals disagree on length: {sorted(lengths)}"
            )

    @property
    def steps(self) -> int:
        return next(
            np.asarray(value).shape[0]
            for value in self.signals.values()
            if np.asarray(value).ndim
        )

    @property
    def duration_ms(self) -> float:
        return self.steps * self.dt_ms

    def require(self, *names: str) -> tuple[np.ndarray, ...]:
        missing = [name for name in names if name not in self.signals]
        if missing:
            raise RecordingError(
                "recording lacks required signals: " + ", ".join(missing)
            )
        return tuple(np.asarray(self.signals[name]) for name in names)
