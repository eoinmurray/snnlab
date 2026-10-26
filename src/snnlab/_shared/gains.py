"""Tensor-free projection gain validation and sampling.

Schedules are pure functions of absolute arrival time in milliseconds. Values
are nonnegative, dimensionless scalars, independent of batch and weight state.
"""

from __future__ import annotations

import copy
import importlib
import json
import math
from collections.abc import Mapping
from itertools import pairwise
from numbers import Real

from snnlab._shared.timing import _is_integer_ratio, _step_ratio


def _transition_step(time_ms, dt_ms):
    ratio = _step_ratio(time_ms, dt_ms)
    nearest = round(ratio)
    return nearest if _is_integer_ratio(ratio, nearest) else math.ceil(ratio)


def _number(value, name, *, nonnegative=True):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite scalar number")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def import_function(reference):
    """Load a portable callback without unpickling or evaluating source."""
    if not isinstance(reference, str):
        raise ValueError("gain function reference must be module:function")
    module, separator, attribute = reference.partition(":")
    if (
        not separator
        or module == "__main__"
        or not all(part.isidentifier() for part in module.split("."))
        or not all(part.isidentifier() for part in attribute.split("."))
    ):
        raise ValueError("gain function must be an importable module:function")
    try:
        function = importlib.import_module(module)
        for part in attribute.split("."):
            function = getattr(function, part)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"cannot load gain function {reference!r}: {error}") from error
    if not callable(function):
        raise ValueError(f"gain function {reference!r} is not callable")
    return function


def function_reference(function):
    if isinstance(function, str):
        import_function(function)
        return function
    module = getattr(function, "__module__", "")
    name = getattr(function, "__qualname__", "")
    reference = f"{module}:{name}"
    if import_function(reference) is not function:
        raise ValueError("gain function must be an importable module function")
    return reference


def _json_config(value):
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("gain config keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            raise ValueError("gain config must contain only JSON values")

    if not isinstance(value, dict):
        raise ValueError("gain config must be a JSON object")
    check(value)
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("gain config must contain finite JSON values") from error


def validate(spec):
    if not isinstance(spec, Mapping):
        raise ValueError("projection modulation must be a gain specification")
    fields = {
        "constant_gain": {"value"},
        "step_gain": {"at_ms", "before", "after"},
        "ramp_gain": {"start_ms", "end_ms", "before", "after"},
        "sine_gain": {"frequency_hz", "depth", "offset", "phase"},
        "piecewise_gain": {"times_ms", "values"},
        "function_gain": {"function", "config"},
    }
    kind = spec.get("kind")
    if not isinstance(kind, str) or kind not in fields:
        raise ValueError(f"unsupported projection gain kind {kind!r}")
    if set(spec) != fields[kind] | {"kind"}:
        raise ValueError(f"invalid fields for {kind}; expected {sorted(fields[kind])}")
    if kind == "function_gain":
        _json_config(spec["config"])
        import_function(spec["function"])
    elif kind == "piecewise_gain":
        times, values = spec["times_ms"], spec["values"]
        if (
            not isinstance(times, list)
            or not isinstance(values, list)
            or len(values) != len(times) + 1
        ):
            raise ValueError(
                "piecewise gain needs one more value than transition times"
            )
        for time in times:
            _number(time, "transition time")
        if any(right <= left for left, right in pairwise(times)):
            raise ValueError("piecewise gain times must be strictly increasing")
        for value in values:
            _number(value, "gain")
    else:
        for name in fields[kind]:
            _number(spec[name], name, nonnegative=name != "phase")
        if kind == "ramp_gain" and spec["end_ms"] <= spec["start_ms"]:
            raise ValueError("ramp end must be after its start")
        if kind == "sine_gain" and spec["depth"] > spec["offset"]:
            raise ValueError(
                "sine depth must not exceed offset (gains must stay nonnegative)"
            )


def sample(spec, *, start_step, steps, dt_ms):
    """Sample once per timestep on the host, before tensor execution begins."""
    from bisect import bisect_right

    validate(spec)
    kind = spec["kind"]
    function = import_function(spec["function"]) if kind == "function_gain" else None
    transitions = (
        [_transition_step(time, dt_ms) for time in spec["times_ms"]]
        if kind == "piecewise_gain"
        else []
    )
    at_step = _transition_step(spec["at_ms"], dt_ms) if kind == "step_gain" else None
    values = []
    for step in range(start_step, start_step + steps):
        time = step * dt_ms
        if kind == "constant_gain":
            value = spec["value"]
        elif kind == "step_gain":
            value = spec["after"] if step >= at_step else spec["before"]
        elif kind == "ramp_gain":
            fraction = min(
                1.0,
                max(
                    0.0, (time - spec["start_ms"]) / (spec["end_ms"] - spec["start_ms"])
                ),
            )
            value = (1 - fraction) * spec["before"] + fraction * spec["after"]
        elif kind == "sine_gain":
            value = spec["offset"] + spec["depth"] * math.sin(
                math.tau * spec["frequency_hz"] * (time / 1000) + spec["phase"]
            )
        elif kind == "piecewise_gain":
            value = spec["values"][bisect_right(transitions, step)]
        else:
            # Isolate configuration so accidental mutation cannot alter graph
            # identity or make later samples depend on previous calls.
            value = function(time, copy.deepcopy(spec["config"]))
        values.append(_number(value, f"gain at t={time:g} ms"))
    return tuple(values)
