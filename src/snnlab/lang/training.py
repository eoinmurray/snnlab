"""Data-only standard training recipe declarations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .core import ParameterRef, Quantity, Signal, Spec


class SignalLike(Protocol):
    id: str


@dataclass(frozen=True)
class Objective:
    kind: str
    prediction: str
    target: str
    weight: float = 1.0


def CrossEntropy(
    *, prediction: SignalLike | str, target: str, weight: float = 1.0
) -> Objective:
    value = prediction if isinstance(prediction, str) else prediction.id
    return Objective("cross_entropy", value, target, weight)


@dataclass(frozen=True)
class ParameterGroup:
    parameters: Sequence[ParameterRef | str]
    name: str
    lr: float
    frozen: bool = False

    def ids(self) -> list[str]:
        return [p.id if isinstance(p, ParameterRef) else p for p in self.parameters]


@dataclass(frozen=True)
class Regularizer:
    kind: str
    signals: tuple[str, ...]
    strength: float
    config: dict[str, Any] = field(default_factory=dict)


def UpperRatePenalty(
    *, signal: Signal, threshold: float, strength: float
) -> Regularizer:
    return SpikeBudgetPenalty(
        signals=(signal,), ceiling_hz=threshold, strength=strength
    )


def SpikeBudgetPenalty(
    *, signals: Sequence[Signal | str], ceiling_hz: float, strength: float
) -> Regularizer:
    ids = tuple(signal if isinstance(signal, str) else signal.id for signal in signals)
    return Regularizer(
        "spike_budget",
        ids,
        strength,
        {
            "ceiling": {"value": ceiling_hz, "unit": "Hz"},
            "penalty": "squared_hinge",
            "aggregation": "mean_presentations_then_layers_of_population_mean_rate",
        },
    )


@dataclass(frozen=True)
class Optimizer:
    kind: str
    config: dict[str, Any] = field(default_factory=dict)


def AdamW(**config: Any) -> Optimizer:
    return Optimizer("adamw", config)


def FastSigmoid(*, slope: float = 1.0) -> Spec:
    """Fast-sigmoid surrogate used by the collection's spike backward pass."""
    return Spec("fast_sigmoid", {"slope": slope})


@dataclass(frozen=True)
class StopGradient:
    signal: str

    @classmethod
    def at(cls, signal: Signal) -> "StopGradient":
        return cls(signal.id)


@dataclass
class TrainSpec:
    objectives: Sequence[Objective]
    parameter_groups: Sequence[ParameterGroup]
    optimizer: Optimizer
    regularizers: Sequence[Regularizer] = ()
    stop_gradients: Sequence[StopGradient] = ()
    epochs: int = 1
    gradient_clip: float | None = None
    surrogate: Spec | None = None
    presentation_duration: Quantity | None = None
