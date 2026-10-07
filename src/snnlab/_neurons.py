"""Shared, tensor-free contracts for built-in adaptive neurons."""

from __future__ import annotations

import math

ADAPTIVE_KINDS = frozenset({"cuba_alif", "coba_alif", "cuba_adex", "coba_adex"})
CURRENT_KINDS = frozenset({"cuba_lif", "lif", "cuba_alif", "cuba_adex"})


def state_units(spec):
    kind = spec.get("kind")
    if kind in ADAPTIVE_KINDS:
        return {"adaptation": "mV" if kind.endswith("alif") else "nA"}
    return {}


def validate(spec):
    """Validate the explicit membrane and adaptation parameter contracts."""
    kind = spec.get("kind")
    if kind != "cuba_lif" and kind not in ADAPTIVE_KINDS:
        return
    label = kind.upper()

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
    for key in ("capacitance_nf", "voltage_grad_dampen"):
        number(key, positive=True)
    for key in ("resting_mv", "threshold_mv", "reset_mv", "initial_voltage_mv"):
        number(key)
    count = spec.get("refractory_steps", 0)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{label} refractory_steps must be a non-negative integer")
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
