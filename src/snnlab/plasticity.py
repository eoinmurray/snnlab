"""Opt-in, data-only local learning rules for simulation (never training)."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType

from snnlab.lang.units import Quantity, ms


def _quantity(value, unit=None, *, positive=False):
    if not isinstance(value, Quantity) or (unit is not None and value.unit != unit):
        raise ValueError(f"expected a quantity in {unit or 'the weight unit'}")
    if value.unit not in {"ms", "Hz", "uS", "nA", "1"}:
        raise ValueError(f"unsupported plasticity unit {value.unit!r}")
    if not math.isfinite(value.value) or (positive and value.value <= 0):
        raise ValueError(
            "plasticity quantities must be finite and time constants positive"
        )
    return value.value


def _bounds(lower, upper):
    _quantity(lower)
    _quantity(upper, lower.unit)
    if (
        lower.unit not in {"uS", "nA", "1"}
        or lower.value < 0
        or lower.value >= upper.value
    ):
        raise ValueError("plasticity requires 0 <= w_min < w_max in a weight unit")


def _rule_json(rule, kind):
    return {
        "kind": kind,
        **{
            f.name: value.json()
            if isinstance(value := getattr(rule, f.name), Quantity)
            else value
            for f in fields(rule)
        },
    }


def _amplitudes(rule, names):
    _bounds(rule.w_min, rule.w_max)
    for name in names:
        if _quantity(getattr(rule, name), rule.w_min.unit) < 0:
            raise ValueError("STDP amplitudes must be nonnegative")


@dataclass(frozen=True, kw_only=True)
class PairSTDP:
    """Bounded all-pairs STDP; simultaneous spikes do not pair.

    Amplitudes and bounds use the projection weight unit. Traces decay before
    updates, then receive this step's spikes. Updates affect the next timestep.
    Multiplicative mode scales increments by normalized distance to each bound.
    """

    a_plus: Quantity
    a_minus: Quantity
    w_min: Quantity
    w_max: Quantity
    tau_pre: Quantity = 20 * ms
    tau_post: Quantity = 20 * ms
    weight_dependence: str = "additive"

    def __post_init__(self):
        _amplitudes(self, ("a_plus", "a_minus"))
        _quantity(self.tau_pre, "ms", positive=True)
        _quantity(self.tau_post, "ms", positive=True)
        if self.weight_dependence not in {"additive", "multiplicative"}:
            raise ValueError("weight_dependence must be additive or multiplicative")

    def json(self):
        data = _rule_json(self, "pair_stdp")
        # Preserve compatibility with existing additive-rule trajectory states.
        if self.weight_dependence == "additive":
            data.pop("weight_dependence")
        return data


@dataclass(frozen=True, kw_only=True)
class TripletSTDP:
    """All-pairs four-trace Pfister–Gerstner rule with explicit parameters."""

    a2_plus: Quantity
    a2_minus: Quantity
    a3_plus: Quantity
    a3_minus: Quantity
    tau_pre: Quantity
    tau_post: Quantity
    tau_x: Quantity
    tau_y: Quantity
    w_min: Quantity
    w_max: Quantity

    def __post_init__(self):
        _amplitudes(self, ("a2_plus", "a2_minus", "a3_plus", "a3_minus"))
        for name in ("tau_pre", "tau_post", "tau_x", "tau_y"):
            _quantity(getattr(self, name), "ms", positive=True)

    def json(self):
        return _rule_json(self, "triplet_stdp")


@dataclass(frozen=True, kw_only=True)
class InhibitorySTDP:
    """Vogels-style symmetric STDP on nonnegative inhibitory magnitudes."""

    eta: Quantity
    target_rate: Quantity
    w_min: Quantity
    w_max: Quantity
    tau: Quantity = 20 * ms

    def __post_init__(self):
        _amplitudes(self, ("eta",))
        _quantity(self.tau, "ms", positive=True)
        if _quantity(self.target_rate, "Hz") < 0:
            raise ValueError("target_rate must be nonnegative")

    def json(self):
        return _rule_json(self, "inhibitory_stdp")


@dataclass(frozen=True, kw_only=True)
class RewardModulatedSTDP(PairSTDP):
    """Pair eligibility plus a signed, dimensionless scalar reward input.

    Eligibility receives the pair increment after decaying. Weight increments
    are reward * eligibility * dt / tau_weight at every step, including silence.
    """

    reward_input: str
    tau_eligibility: Quantity
    tau_weight: Quantity

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.reward_input, str) or not self.reward_input:
            raise ValueError("reward_input must name a graph input")
        _quantity(self.tau_eligibility, "ms", positive=True)
        _quantity(self.tau_weight, "ms", positive=True)

    def json(self):
        return _rule_json(self, "reward_modulated_stdp")


@dataclass(frozen=True, kw_only=True)
class CustomPlasticity:
    """A registered step(ctx) -> (weight_delta, tensor_state) rule."""

    definition: str
    w_min: Quantity
    w_max: Quantity
    config: Mapping | None = None

    def __post_init__(self):
        from snnlab import extensions as E

        _bounds(self.w_min, self.w_max)
        config = json.loads(json.dumps(dict(self.config or {}), allow_nan=False))
        object.__setattr__(self, "config", MappingProxyType(config))
        E.resolve("plasticity", self.json())

    def json(self):
        return {
            "kind": "custom_plasticity",
            "definition": self.definition,
            "config": copy.deepcopy(dict(self.config)),
            "w_min": self.w_min.json(),
            "w_max": self.w_max.json(),
        }


@dataclass(frozen=True, kw_only=True)
class Plasticity:
    """Projection-local learning for one trajectory with batch_size=1."""

    rules: Mapping[
        str,
        PairSTDP
        | TripletSTDP
        | InhibitorySTDP
        | RewardModulatedSTDP
        | CustomPlasticity,
    ]

    def __post_init__(self):
        rules = dict(self.rules)
        if not rules or any(not isinstance(k, str) or not k for k in rules):
            raise ValueError("plasticity requires named projection rules")
        if any(
            not isinstance(v, (PairSTDP, TripletSTDP, InhibitorySTDP, CustomPlasticity))
            for v in rules.values()
        ):
            raise TypeError(
                "plasticity rules must be built-in or CustomPlasticity rules"
            )
        object.__setattr__(self, "rules", MappingProxyType(rules))

    def json(self):
        from snnlab import extensions as E

        data = {
            "schema": "snnlab.plasticity/v1",
            "rules": {name: rule.json() for name, rule in sorted(self.rules.items())},
        }
        return E.with_requirements(data)

    @classmethod
    def from_json(cls, data):
        """Restore JSON specifications and importable custom implementations."""
        from snnlab import extensions as E

        if data.get("schema") != "snnlab.plasticity/v1":
            raise ValueError("unsupported plasticity schema")
        E.restore(data)
        rules = {}
        constructors = {
            "pair_stdp": PairSTDP,
            "triplet_stdp": TripletSTDP,
            "inhibitory_stdp": InhibitorySTDP,
            "reward_modulated_stdp": RewardModulatedSTDP,
            "custom_plasticity": CustomPlasticity,
        }
        for name, saved in data["rules"].items():
            row = dict(saved)
            kind = row.pop("kind")
            if kind not in constructors:
                raise ValueError(f"unknown plasticity rule {kind!r}")
            for key, value in row.items():
                if (
                    key != "config"
                    and isinstance(value, dict)
                    and set(value) == {"value", "unit"}
                ):
                    row[key] = Quantity(**value)
            rules[name] = constructors[kind](**row)
        return cls(rules=rules)
