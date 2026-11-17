"""stdp_simple: pair-based, additive STDP with hard bounds.

Each synapse j -> i has a weight w_ij. Each presynaptic neuron j keeps a trace
x_j and each postsynaptic neuron i keeps a trace y_i. On a time grid of step
dt, one call to :func:`step` does, in this order:

1. Decay both traces: x <- x exp(-dt / tau_plus), y <- y exp(-dt / tau_minus).
2. Depression at presynaptic spikes, clipped at the lower bound:
   w <- max(w_min, w - a_minus y_i).
3. Potentiation at postsynaptic spikes, clipped at the upper bound:
   w <- min(w_max, w + a_plus x_j).
4. Add this step's spikes to the traces. With ``"nearest"`` pairing a spike
   sets its trace to 1; with ``"all_to_all"`` it adds 1.

Clipping is sequential: depression is clipped before potentiation is added.
When a synapse is depressed and potentiated in the same step, the result can
differ from clipping the combined update once.

Because steps 2 and 3 read the traces before step 4, spikes in the same step do
not pair with each other. They still pair with earlier spikes through the
traces, and they enter the traces for later pairs. Presynaptic spikes are
passed at arrival, after any delay.

Dense weights use the runtime layout ``(batch, source, target)``: every trial
in a batch has its own weights. One-to-one projections join source ``i`` to
target ``i`` only; :func:`step_one_to_one` updates their ``(batch, n)``
weights channel by channel and gives the diagonal of the dense update. Both
steps are functional and never modify their inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from snnlab._shared.plasticity import (
    STDP_SIMPLE,
    STDP_SIMPLE_PAIRINGS,
    validate_stdp_simple,
)

PAIRINGS = STDP_SIMPLE_PAIRINGS


@dataclass(frozen=True, kw_only=True)
class STDPSimple:
    """Parameters of one stdp_simple rule.

    Amplitudes and bounds are in the projection's weight units; time constants
    are in ms. ``w_min`` and ``pairing`` have defaults; the rest are required.
    The rule is always additive with hard bounds, so those are fixed class
    attributes, not arguments.
    """

    a_plus: float
    a_minus: float
    tau_plus_ms: float
    tau_minus_ms: float
    w_max: float
    w_min: float = 0.0
    pairing: str = "nearest"

    kind: ClassVar[str] = STDP_SIMPLE
    update: ClassVar[str] = "additive"
    bounds: ClassVar[str] = "hard"

    def __post_init__(self) -> None:
        values = validate_stdp_simple(
            a_plus=self.a_plus,
            a_minus=self.a_minus,
            tau_plus_ms=self.tau_plus_ms,
            tau_minus_ms=self.tau_minus_ms,
            w_max=self.w_max,
            w_min=self.w_min,
            pairing=self.pairing,
        )
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "update": self.update,
            "bounds": self.bounds,
            "pairing": self.pairing,
            "a_plus": self.a_plus,
            "a_minus": self.a_minus,
            "tau_plus_ms": self.tau_plus_ms,
            "tau_minus_ms": self.tau_minus_ms,
            "w_min": self.w_min,
            "w_max": self.w_max,
        }


@dataclass(frozen=True)
class STDPState:
    """Plasticity state carried between steps and between calls."""

    pre_trace: torch.Tensor  # x_j, shape (batch, source)
    post_trace: torch.Tensor  # y_i, shape (batch, target)


def init_state(
    rule: STDPSimple,
    *,
    batch: int,
    n_source: int,
    n_target: int,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> STDPState:
    """Return a state with both traces at zero."""
    del rule  # every stdp_simple rule starts from empty traces
    return STDPState(
        pre_trace=torch.zeros(batch, n_source, dtype=dtype, device=device),
        post_trace=torch.zeros(batch, n_target, dtype=dtype, device=device),
    )


def step(
    rule: STDPSimple,
    state: STDPState,
    weights: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    *,
    dt_ms: float,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, STDPState]:
    """Advance one time step and return ``(new_weights, new_state)``.

    ``weights`` is ``(batch, source, target)``; ``pre`` is ``(batch, source)``
    and ``post`` is ``(batch, target)``, with 0/1 entries. ``mask`` is a
    boolean ``(source, target)`` tensor; entries that are ``False`` never change.
    """
    pre, post, x, y = _decay(rule, state, weights.dtype, pre, post, dt_ms)
    new = _clip(
        rule,
        weights,
        rule.a_minus * pre[:, :, None] * y[:, None, :],
        rule.a_plus * x[:, :, None] * post[:, None, :],
        mask,
    )
    return new, _jump(rule, x, y, pre, post)


def step_one_to_one(
    rule: STDPSimple,
    state: STDPState,
    weights: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    *,
    dt_ms: float,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, STDPState]:
    """Advance one step of a one-to-one projection, channel by channel.

    Synapse ``i`` joins source ``i`` to target ``i``, so ``weights``, ``pre``
    and ``post`` are all ``(batch, n)``, and ``mask`` is a boolean ``(n,)``
    tensor. The result equals the diagonal of :func:`step` with an identity
    mask.
    """
    pre, post, x, y = _decay(rule, state, weights.dtype, pre, post, dt_ms)
    new = _clip(rule, weights, rule.a_minus * pre * y, rule.a_plus * x * post, mask)
    return new, _jump(rule, x, y, pre, post)


def _decay(rule, state, dtype, pre, post, dt_ms):
    """Cast the spikes and decay both traces by one step."""
    if not math.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError(f"dt_ms must be positive and finite, got {dt_ms!r}")
    x = state.pre_trace * math.exp(-dt_ms / rule.tau_plus_ms)
    y = state.post_trace * math.exp(-dt_ms / rule.tau_minus_ms)
    return pre.to(dtype), post.to(dtype), x, y


def _clip(rule, weights, depression, potentiation, mask):
    """Apply depression, clip at w_min, then potentiation, clip at w_max."""
    new = torch.clamp(weights - depression, min=rule.w_min)
    new = torch.clamp(new + potentiation, max=rule.w_max)
    if mask is not None:
        new = torch.where(mask, new, weights)
    return new


def _jump(rule, x, y, pre, post):
    """Add this step's spikes to the traces."""
    if rule.pairing == "nearest":
        x = torch.where(pre > 0, torch.ones_like(x), x)
        y = torch.where(post > 0, torch.ones_like(y), y)
    else:
        x = x + pre
        y = y + post
    return STDPState(pre_trace=x, post_trace=y)
