"""Shared, tensor-free contracts for built-in neurons and physical timing."""

from __future__ import annotations

import copy
import math

from snnlab._shared.timing import refractory_steps

ADAPTIVE_KINDS = frozenset({"cuba_alif", "coba_alif", "cuba_adex", "coba_adex"})
CURRENT_KINDS = frozenset({"cuba_lif", "lif", "cuba_alif", "cuba_adex"})
SPIKING_KINDS = ADAPTIVE_KINDS | {"cuba_lif", "coba_lif"}
REFRACTORY_TIMING = "following_steps"
LEGACY_REFRACTORY_TIMING = "legacy_coba"


def resolve_refractory(spec, dt_ms, *, legacy=False):
    """Resolve built-in timing without modifying authenticated graph data."""
    refractory_steps(0, dt_ms)
    if "refractory_policy" in spec:
        raise ValueError(
            "refractory_policy is no longer supported; only exact durations are supported"
        )
    if "refractory" in spec and "refractory_steps" in spec:
        raise ValueError("conflicting refractory and refractory_steps representations")
    if "refractory_steps" in spec:
        if not legacy:
            raise ValueError(
                "refractory_steps is no longer supported; use refractory in ms"
            )
        count = spec["refractory_steps"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("legacy refractory_steps must be a non-negative integer")
        duration = count * dt_ms
    elif "refractory" in spec:
        quantity = spec["refractory"]
        if (
            not isinstance(quantity, dict)
            or quantity.get("unit") != "ms"
            or isinstance(quantity.get("value"), bool)
            or not isinstance(quantity.get("value"), (int, float))
            or not math.isfinite(quantity["value"])
            or quantity["value"] < 0
        ):
            raise ValueError("refractory must be a non-negative finite ms quantity")
        duration = quantity["value"]
        count = refractory_steps(duration, dt_ms)
    elif legacy:
        if spec["kind"] == "coba_lif":
            duration = 3.0 if spec["tau_mem"]["value"] >= 15 else 1.5
            # Only archived graphs retain their historical implicit rounding.
            count = max(1, round(duration / dt_ms))
        else:
            duration, count = 0.0, 0
    else:
        raise ValueError("built-in neuron requires refractory as an ms quantity")
    legacy_coba = legacy and spec["kind"] == "coba_lif"
    held_steps = max(count - 1, 0) if legacy_coba else count
    return {
        "requested_refractory_ms": float(duration),
        "realized_refractory_ms": count * dt_ms,
        "refractory_steps": count,
        "held_steps": held_steps,
        "held_duration_ms": held_steps * dt_ms,
        "timing": LEGACY_REFRACTORY_TIMING if legacy_coba else REFRACTORY_TIMING,
    }


def migrate_refractory(graph):
    """Copy historical counts to physical durations at the source timestep."""
    migrated = copy.deepcopy(graph)
    timing = graph.get("refractory_timing")
    if timing not in (None, REFRACTORY_TIMING, LEGACY_REFRACTORY_TIMING):
        raise ValueError("unsupported refractory_timing contract")
    legacy = timing != REFRACTORY_TIMING
    dt_ms = float(graph["timebase"]["dt"]["value"])
    for population in migrated.get("populations", []):
        spec = population["neuron"]
        if spec.get("kind") in SPIKING_KINDS:
            resolved = resolve_refractory(spec, dt_ms, legacy=legacy)
            if "refractory" not in spec:
                spec["refractory"] = {
                    "value": resolved["realized_refractory_ms"],
                    "unit": "ms",
                }
            spec.pop("refractory_steps", None)
    migrated["refractory_timing"] = (
        LEGACY_REFRACTORY_TIMING if legacy else REFRACTORY_TIMING
    )
    return migrated


def state_units(spec):
    kind = spec.get("kind")
    if kind in ADAPTIVE_KINDS:
        return {"adaptation": "mV" if kind.endswith("alif") else "nA"}
    return {}


def _validate_time_constants(spec, kind, label):
    for key in ("tau_mem", "tau_adaptation"):
        if key == "tau_adaptation" and kind not in ADAPTIVE_KINDS:
            continue
        quantity = spec.get(key)
        if (
            not isinstance(quantity, dict)
            or quantity.get("unit") != "ms"
            or isinstance(quantity.get("value"), bool)
            or not isinstance(quantity.get("value"), (int, float))
            or not math.isfinite(quantity["value"])
            or quantity["value"] <= 0
        ):
            raise ValueError(f"{label} {key} must be positive finite ms")


def _number_validator(spec, label):
    def number(key, *, positive=False, nonnegative=False):
        value = spec.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{label} {key} must be finite")
        if positive and value <= 0:
            raise ValueError(f"{label} {key} must be positive and finite")
        if nonnegative and value < 0:
            raise ValueError(f"{label} {key} must be non-negative and finite")

    return number


def validate(spec, *, legacy=False):
    """Validate the explicit membrane and adaptation parameter contracts."""
    kind = spec.get("kind")
    if kind not in SPIKING_KINDS:
        return
    label = kind.upper()

    if "refractory_policy" in spec or ("refractory_steps" in spec and not legacy):
        raise ValueError(
            f"{label} use refractory in ms; refractory_steps and refractory_policy are no longer supported"
        )
    if "refractory" in spec and "refractory_steps" in spec:
        raise ValueError(f"{label} conflicting refractory representations")
    if kind == "coba_lif":
        return

    number = _number_validator(spec=spec, label=label)

    _validate_time_constants(spec=spec, kind=kind, label=label)

    for key in ("capacitance_nf", "voltage_grad_dampen"):
        number(key, positive=True)
    for key in ("resting_mv", "threshold_mv", "reset_mv", "initial_voltage_mv"):
        number(key)
    if kind.endswith("adex"):
        number("delta_t_mv", positive=True)
        number("spike_mv")
        number("a_us", nonnegative=True)
        number("b_na", nonnegative=True)
        number("initial_adaptation_na")
        if spec["spike_mv"] <= max(spec["threshold_mv"], spec["reset_mv"]):
            raise ValueError(f"{label} spike_mv must exceed threshold_mv and reset_mv")
    else:
        if spec["reset_mv"] >= spec["threshold_mv"]:
            raise ValueError(f"{label} reset_mv must be below threshold_mv")
        if kind.endswith("alif"):
            number("adaptation_increment_mv", nonnegative=True)
            number("initial_adaptation_mv", nonnegative=True)
    if kind.startswith("coba_"):
        number("excitatory_reversal_mv")
        number("inhibitory_reversal_mv")
