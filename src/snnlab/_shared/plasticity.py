"""Shared, tensor-free contracts for plasticity rules.

``lang`` validates specs and graphs with these functions, and ``sim`` builds its
runtime rules from them. A bundle loaded from disk is therefore checked exactly
like a network built in Python.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

STDP_SIMPLE = "stdp_simple"
PLASTICITY_KINDS = frozenset({STDP_SIMPLE})
STDP_SIMPLE_PAIRINGS = ("nearest", "all_to_all")
_STDP_SIMPLE_KEYS = frozenset(
    {
        "kind",
        "update",
        "bounds",
        "pairing",
        "a_plus",
        "a_minus",
        "tau_plus",
        "tau_minus",
        "w_min",
        "w_max",
        "freeze_zero_weights",
    }
)
_LEARNABLE_CONNECTIVITY = frozenset({None, "one_to_one"})


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"stdp_simple {name} must be a finite number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"stdp_simple {name} must be finite, got {value!r}")
    return value


def validate_stdp_simple(
    *,
    a_plus: Any,
    a_minus: Any,
    tau_plus_ms: Any,
    tau_minus_ms: Any,
    w_max: Any,
    w_min: Any,
    pairing: Any,
) -> dict[str, Any]:
    """Validate stdp_simple parameters and return them as plain floats."""
    values = {
        "a_plus": _finite("a_plus", a_plus),
        "a_minus": _finite("a_minus", a_minus),
        "tau_plus_ms": _finite("tau_plus", tau_plus_ms),
        "tau_minus_ms": _finite("tau_minus", tau_minus_ms),
        "w_max": _finite("w_max", w_max),
        "w_min": _finite("w_min", w_min),
    }
    if values["a_plus"] < 0 or values["a_minus"] < 0:
        raise ValueError("stdp_simple amplitudes a_plus and a_minus must be >= 0")
    if values["tau_plus_ms"] <= 0 or values["tau_minus_ms"] <= 0:
        raise ValueError(
            "stdp_simple time constants tau_plus and tau_minus must be > 0"
        )
    if not 0 <= values["w_min"] < values["w_max"]:
        raise ValueError(
            "stdp_simple bounds need 0 <= w_min < w_max, got "
            f"w_min={values['w_min']}, w_max={values['w_max']}"
        )
    if pairing not in STDP_SIMPLE_PAIRINGS:
        raise ValueError(
            f"stdp_simple pairing must be one of {STDP_SIMPLE_PAIRINGS}, got {pairing!r}"
        )
    values["pairing"] = pairing
    return values


def _time_ms(name: str, value: Any) -> float:
    if not isinstance(value, Mapping) or value.get("unit") != "ms":
        raise ValueError(f"stdp_simple {name} must be a time in ms, got {value!r}")
    return _finite(name, value.get("value"))


def stdp_simple_from_json(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a serialized stdp_simple spec and return its runtime values."""
    if not isinstance(spec, Mapping):
        raise ValueError("plasticity must be a serialized spec mapping")
    kind = spec.get("kind")
    if kind not in PLASTICITY_KINDS:
        raise ValueError(f"unknown plasticity kind {kind!r}")
    unknown = sorted(set(spec) - _STDP_SIMPLE_KEYS)
    if unknown:
        raise ValueError(f"stdp_simple spec has unknown fields {unknown}")
    if spec.get("update") != "additive" or spec.get("bounds") != "hard":
        raise ValueError("stdp_simple is additive with hard bounds")
    freeze = spec.get("freeze_zero_weights", False)
    if not isinstance(freeze, bool):
        raise ValueError("stdp_simple freeze_zero_weights must be boolean")
    values = validate_stdp_simple(
        a_plus=spec.get("a_plus"),
        a_minus=spec.get("a_minus"),
        tau_plus_ms=_time_ms("tau_plus", spec.get("tau_plus")),
        tau_minus_ms=_time_ms("tau_minus", spec.get("tau_minus")),
        w_max=spec.get("w_max"),
        w_min=spec.get("w_min", 0.0),
        pairing=spec.get("pairing", "nearest"),
    )
    values["freeze_zero_weights"] = freeze
    return values


def _spiking_source(graph: Mapping[str, Any], source: str) -> bool:
    owner, _, port = source.partition(".")
    for row in graph.get("inputs", []):
        if row["id"] == owner:
            return port == "value" and row.get("signal_type") == "spikes"
    for row in graph.get("populations", []):
        if row["id"] == owner:
            return port == "spikes" and row.get("spiking") is True
    return False


def unsupported_reasons(graph: Mapping[str, Any], row: Mapping[str, Any]) -> list[str]:
    """Why a projection cannot carry stdp_simple; empty when it can."""
    reasons = []
    polarity = row.get("polarity")
    if polarity != "excitatory":
        reasons.append(
            f"it targets the {polarity} receptor; stdp_simple applies to excitatory "
            "receptors only"
        )
    if not _spiking_source(graph, row.get("source", "")):
        reasons.append("its source does not emit spikes")
    target = row.get("target", "").partition(".")[0]
    population = next(
        (p for p in graph.get("populations", []) if p["id"] == target), None
    )
    if population is not None and population.get("spiking") is not True:
        reasons.append("its target population does not spike")
    if "modulation" in row:
        reasons.append("it uses weight modulation")
    connectivity = (row.get("connectivity") or {}).get("kind")
    if connectivity not in _LEARNABLE_CONNECTIVITY:
        reasons.append(f"it uses {connectivity} connectivity, whose weights are shared")
    parameters = set(row.get("parameters", []))
    sharing = sorted(
        other["id"]
        for other in graph.get("projections", [])
        if other.get("id") != row.get("id")
        and parameters & set(other.get("parameters", []))
    )
    sharing += sorted(
        op["id"]
        for op in graph.get("operations", [])
        if parameters & set(op.get("parameters", []))
    )
    if sharing:
        reasons.append(f"it shares its weights with {sharing}")
    return reasons
