"""Declarative, portable schedules for projection weight gains."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any

from snnlab._shared import gains
from snnlab.lang._specs import Spec
from snnlab.lang.units import Hz, Quantity, ms


def _quantity(value: Quantity, unit: str) -> float:
    if not isinstance(value, Quantity) or value.unit != unit:
        raise ValueError(f"schedule requires a quantity in {unit}")
    return value.value


def _gain(kind: str, **values: Any) -> Spec:
    spec = Spec(kind, values)
    gains.validate(spec.json())
    return spec


def ConstantGain(value: float = 1.0) -> Spec:
    """Apply a fixed nonnegative multiplier."""
    return _gain("constant_gain", value=value)


def StepGain(*, at: Quantity = 0 * ms, before: float = 1.0, after: float = 0.0) -> Spec:
    """Switch from before to after at the first timestep at or after at."""
    return _gain("step_gain", at_ms=_quantity(at, "ms"), before=before, after=after)


def RampGain(
    *,
    start: Quantity = 0 * ms,
    end: Quantity = 100 * ms,
    before: float = 0.0,
    after: float = 1.0,
) -> Spec:
    """Interpolate linearly between start and end; hold endpoint gains outside."""
    return _gain(
        "ramp_gain",
        start_ms=_quantity(start, "ms"),
        end_ms=_quantity(end, "ms"),
        before=before,
        after=after,
    )


def SineGain(
    *,
    frequency: Quantity = 40 * Hz,
    depth: float = 0.5,
    offset: float = 1.0,
    phase: float = 0.0,
) -> Spec:
    """Return offset + depth*sin(2*pi*frequency*t + phase), phase in radians."""
    return _gain(
        "sine_gain",
        frequency_hz=_quantity(frequency, "Hz"),
        depth=depth,
        offset=offset,
        phase=phase,
    )


def PiecewiseGain(*, times: Sequence[Quantity], values: Sequence[float]) -> Spec:
    """Hold successive gains, switching at each strictly increasing time."""
    return _gain(
        "piecewise_gain",
        times_ms=[_quantity(time, "ms") for time in times],
        values=list(values),
    )


def CustomGain(
    function: str | Callable, *, config: dict[str, Any] | None = None
) -> Spec:
    """Use an importable function(t_ms, config) returning a nonnegative scalar.

    Supply a module:function reference or that same importable callable. Lambdas,
    closures and functions defined in __main__ cannot be persisted and fail here.
    """
    return _gain(
        "function_gain",
        function=gains.function_reference(function),
        config=copy.deepcopy({} if config is None else config),
    )
