"""Physical-time conversion shared by simulation entry points.

Trial durations use whole steps rounded downward, except for floating-point
representations of an integer ratio. Refractory durations must be exactly
representable on the timestep grid; zero disables the refractory hold.
"""

from __future__ import annotations

import math


def _step_ratio(duration_ms: float, dt_ms: float) -> float:
    duration_ms, dt_ms = float(duration_ms), float(dt_ms)
    if not math.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be finite and positive")
    if not math.isfinite(duration_ms) or duration_ms < 0:
        raise ValueError("duration_ms must be finite and nonnegative")
    ratio = duration_ms / dt_ms
    if not math.isfinite(ratio):
        raise ValueError("duration/timestep ratio must be finite")
    return ratio


def _is_integer_ratio(ratio: float, nearest: int) -> bool:
    # Relative tolerance scales only with floating-point error in the ratio,
    # not with any scientifically meaningful fraction of an integration step.
    return math.isclose(ratio, nearest, rel_tol=1e-12, abs_tol=1e-12)


def duration_steps(duration_ms: float, dt_ms: float) -> int:
    """Return the number of complete steps, correcting near-integer roundoff."""
    ratio = _step_ratio(duration_ms, dt_ms)
    nearest = round(ratio)
    return nearest if _is_integer_ratio(ratio, nearest) else math.floor(ratio)


def refractory_steps(duration_ms: float, dt_ms: float) -> int:
    """Resolve an exact, nonnegative refractory duration into internal steps."""
    ratio = _step_ratio(duration_ms, dt_ms)
    if duration_ms == 0:
        return 0
    nearest = round(ratio)
    if nearest < 1 or not _is_integer_ratio(ratio, nearest):
        raise ValueError(
            f"refractory duration {duration_ms:g} ms is not a positive whole "
            f"number of steps at dt={dt_ms:g} ms; choose a compatible timestep "
            "or duration (refractory rounding is not supported)"
        )
    return nearest


def duration_metadata(duration_ms: float, dt_ms: float) -> dict:
    """Record the requested trial duration and the duration actually integrated."""
    steps = duration_steps(duration_ms, dt_ms)
    return {
        "nominal_duration_ms": float(duration_ms),
        "duration_steps": steps,
        "realized_duration_ms": steps * float(dt_ms),
    }


def refractory_metadata(
    refractory_e_ms: float,
    refractory_i_ms: float,
    dt_ms: float,
) -> dict:
    """Record physical E/I durations and their validated runtime counters."""
    e_steps = refractory_steps(refractory_e_ms, dt_ms)
    i_steps = refractory_steps(refractory_i_ms, dt_ms)
    return {
        "refractory_e_ms": float(refractory_e_ms),
        "refractory_i_ms": float(refractory_i_ms),
        "refractory_e_steps": e_steps,
        "refractory_i_steps": i_steps,
        "realized_refractory_e_ms": e_steps * float(dt_ms),
        "realized_refractory_i_ms": i_steps * float(dt_ms),
    }
