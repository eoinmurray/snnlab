"""Continuous-time controls with explicit restart boundaries."""

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

from .model import magnitude


@dataclass(frozen=True)
class PiecewiseConstant:
    """Right-continuous schedule: len(values) == len(times) + 1; times in ms."""

    times: object
    values: object

    def __post_init__(self):
        times = tuple(magnitude(t, "ms") for t in self.times)
        values = tuple(float(v) for v in self.values)
        if (
            len(values) != len(times) + 1
            or np.any(np.diff(times) <= 0)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "schedule needs finite values and strictly increasing times"
            )
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)

    def __call__(self, time):
        return self.values[bisect_right(self.times, time)]

    @property
    def breakpoints(self):
        return self.times


def control_spec(control):
    if hasattr(control, "times") and hasattr(control, "values"):
        return {
            "kind": type(control).__name__,
            "times_ms": list(control.times),
            "values": list(control.values),
        }
    return {
        "kind": "runtime_callback",
        "name": getattr(control, "__qualname__", type(control).__name__),
        "breakpoints_ms": list(getattr(control, "breakpoints", ())),
    }


def graph_control(spec, dt):
    """Translate portable graph schedules; discrete switches retain arrival grid."""
    from snnlab._shared.gains import _transition_step, validate

    from .analysis import PiecewiseLinear

    validate(spec)
    kind = spec["kind"]
    if kind == "constant_gain":
        return PiecewiseConstant((), (spec["value"],))
    if kind == "step_gain":
        return PiecewiseConstant(
            (_transition_step(spec["at_ms"], dt) * dt,), (spec["before"], spec["after"])
        )
    if kind == "piecewise_gain":
        # Several authored transitions may map to the same simulation timestep.
        values = {
            _transition_step(t, dt) * dt: value
            for t, value in zip(spec["times_ms"], spec["values"][1:], strict=True)
        }
        return PiecewiseConstant(tuple(values), (spec["values"][0], *values.values()))
    if kind == "ramp_gain":
        return PiecewiseLinear(
            (spec["start_ms"], spec["end_ms"]), (spec["before"], spec["after"])
        )
    raise ValueError(
        f"unsupported graph modulation translation: {kind}; use an explicit control"
    )
