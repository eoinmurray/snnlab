"""Explicit pre-integration histories, in named state units and milliseconds."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from .model import magnitude


@dataclass(frozen=True)
class History:
    """Callback on [start, 0] returning every named state; declare all corners.

    Values must be finite in the model's units. The callback must be continuous,
    piecewise smooth and agree with the initial state at zero. ``description``
    records its mathematical specification; callback code is never serialized.
    """

    function: object
    start: object
    description: str
    breakpoints: object = ()

    def __post_init__(self):
        start = magnitude(self.start, "ms")
        points = tuple(magnitude(t, "ms") for t in self.breakpoints)
        if start >= 0 or not callable(self.function) or not self.description.strip():
            raise ValueError("history needs a negative start, callable and description")
        if any(t <= start or t >= 0 for t in points) or np.any(np.diff(points) <= 0):
            raise ValueError(
                "history breakpoints must increase strictly inside (start, 0)"
            )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "breakpoints", points)

    def __call__(self, time):
        return self.function(time)

    def specification(self, model):
        return {
            "kind": "callback",
            "interval_ms": [self.start, 0],
            "description": self.description,
            "breakpoints_ms": list(self.breakpoints),
            "units": {n: model.units[n] for n in model.states},
        }


@dataclass(frozen=True)
class ConstantHistory:
    """A named constant state for the entire required prehistory interval."""

    state: Mapping
    breakpoints: tuple = field(default=(), init=False)

    def __post_init__(self):
        if not isinstance(self.state, Mapping):
            raise ValueError("constant history requires named states")
        object.__setattr__(self, "state", MappingProxyType(dict(self.state)))

    def __call__(self, time):
        return self.state

    def specification(self, model):
        maximum = max(item.delay_ms for item in model.delayed.values())
        return {
            "kind": "constant",
            "interval_ms": [-maximum, 0],
            "state": dict(zip(model.states, model.vector(self.state))),
            "units": {n: model.units[n] for n in model.states},
        }


def validate_history(model, history, initial):
    if not isinstance(history, (History, ConstantHistory)):
        raise ValueError("explicit delays require History or ConstantHistory")
    maximum = max(item.delay_ms for item in model.delayed.values())
    if isinstance(history, History) and history.start > -maximum:
        raise ValueError(f"history must cover [-{maximum:g}, 0] ms")

    def at(time):
        result = history(time)
        if not isinstance(result, Mapping):
            raise ValueError("history must return every named state")
        return model.vector(result)

    # These probes catch common malformed histories; every later lookup is also
    # validated. Finite sampling cannot prove an arbitrary callback continuous.
    for time in sorted({*np.linspace(-maximum, 0, 17), *history.breakpoints}):
        at(time)
    if not np.allclose(at(0), initial, rtol=0, atol=1e-12):
        raise ValueError(
            "history at zero must agree with initial state (atol=1e-12 native units)"
        )
    return at
