"""Plasticity rules that a projection can carry (``Network.connect(plasticity=...)``)."""

from __future__ import annotations

from typing import Any

from snnlab._shared.plasticity import STDP_SIMPLE, validate_stdp_simple
from snnlab.lang._specs import Spec
from snnlab.lang.units import Quantity


def STDPSimple(
    *,
    a_plus: float,
    a_minus: float,
    tau_plus: Quantity,
    tau_minus: Quantity,
    w_max: float,
    w_min: float = 0.0,
    pairing: str = "nearest",
    freeze_zero_weights: bool = False,
) -> Spec:
    """Pair-based, additive STDP with hard bounds, for one projection.

    A presynaptic trace x (time constant ``tau_plus``) and a postsynaptic trace
    y (time constant ``tau_minus``) record recent spikes. A presynaptic spike
    depresses by ``a_minus * y``, clipped at ``w_min``; a postsynaptic spike
    then potentiates by ``a_plus * x``, clipped at ``w_max``. The step size does
    not depend on the weight (additive), and the bounds are hard.

    Args:
        a_plus: Potentiation amplitude, in the projection's weight units (>= 0).
        a_minus: Depression amplitude, in the projection's weight units (>= 0).
        tau_plus: Presynaptic trace time constant, such as ``16.8 * lang.ms``.
        tau_minus: Postsynaptic trace time constant, such as ``33.7 * lang.ms``.
        w_max: Upper bound, in weight units.
        w_min: Lower bound, in weight units; 0 <= w_min < w_max.
        pairing: ``"nearest"`` (symmetric nearest-spike) or ``"all_to_all"``.
        freeze_zero_weights: Keep weights that start at exactly zero fixed.

    The rule applies to projections onto an excitatory receptor, from a
    spiking source to a spiking target, with weights of their own.
    """
    for name, tau in (("tau_plus", tau_plus), ("tau_minus", tau_minus)):
        if not isinstance(tau, Quantity) or tau.unit != "ms":
            raise ValueError(
                f"stdp_simple {name} must be a time such as 20 * lang.ms, got {tau!r}"
            )
    if not isinstance(freeze_zero_weights, bool):
        raise ValueError("stdp_simple freeze_zero_weights must be boolean")
    values: dict[str, Any] = validate_stdp_simple(
        a_plus=a_plus,
        a_minus=a_minus,
        tau_plus_ms=tau_plus.value,
        tau_minus_ms=tau_minus.value,
        w_max=w_max,
        w_min=w_min,
        pairing=pairing,
    )
    return Spec(
        STDP_SIMPLE,
        {
            "update": "additive",
            "bounds": "hard",
            "pairing": values["pairing"],
            "a_plus": values["a_plus"],
            "a_minus": values["a_minus"],
            "tau_plus": tau_plus,
            "tau_minus": tau_minus,
            "w_min": values["w_min"],
            "w_max": values["w_max"],
            "freeze_zero_weights": freeze_zero_weights,
        },
    )
