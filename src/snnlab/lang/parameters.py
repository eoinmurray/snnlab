"""Weight initializer and parameter constraint declarations."""

from __future__ import annotations

from typing import Any

from snnlab.lang._specs import Spec, _custom


def CustomInitializer(definition: str, **config: Any) -> Spec:
    return _custom("initializer", definition, config)


def CustomConstraint(definition: str, **config: Any) -> Spec:
    return _custom("constraint", definition, config)


def Normal(mean: float, std: float) -> Spec:
    """Compatibility name for the collection's lower-clamped normal law."""
    return LowerClampedNormal(mean, std)


def LowerClampedNormal(
    mean: float,
    std: float,
    *,
    initial_zero_fraction: float = 0.0,
    zeroing: str = "bernoulli",
) -> Spec:
    return Spec(
        "lower_clamped_normal",
        {
            "mean": mean,
            "std": std,
            "initial_zero_fraction": initial_zero_fraction,
            "zeroing": zeroing,
        },
    )


def SignedNormal(mean: float, std: float) -> Spec:
    return Spec("signed_normal", {"mean": mean, "std": std})


def Uniform(low: float, high: float) -> Spec:
    return Spec("uniform", {"low": low, "high": high})


def Zeros() -> Spec:
    return Spec("zeros")


def Constant(value: float) -> Spec:
    return Spec("constant", {"value": value})


def NonNegative() -> Spec:
    return Spec("non_negative")
