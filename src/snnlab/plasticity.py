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
    if value.unit not in {"ms", "Hz", "mV", "uS", "nA", "1"}:
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
class NonlinearSTDP(PairSTDP):
    """Gütig power-law pair STDP; exponents independently control both bounds."""

    mu_plus: float = 1.0
    mu_minus: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.weight_dependence != "additive":
            raise ValueError(
                "NonlinearSTDP uses mu_plus/mu_minus, not weight_dependence"
            )
        for value in (self.mu_plus, self.mu_minus):
            if not math.isfinite(value) or value < 0:
                raise ValueError("STDP exponents must be finite and nonnegative")

    def json(self):
        data = _rule_json(self, "nonlinear_stdp")
        data.pop("weight_dependence")
        return data


@dataclass(frozen=True, kw_only=True)
class VoltageSTDP:
    """Clopath-style voltage rule without metaplastic homeostasis.

    Amplitudes are weight quantities per numerical mV (LTD), or per numerical
    mV squared per ms (LTP). spike_voltage explicitly supplies the excursion
    absent from reset-based neurons; this is a discrete LIF adaptation.
    """

    a_plus: Quantity
    a_minus: Quantity
    theta_minus: Quantity
    theta_plus: Quantity
    voltage_rest: Quantity
    spike_voltage: Quantity
    tau_pre: Quantity
    tau_minus: Quantity
    tau_plus: Quantity
    w_min: Quantity
    w_max: Quantity

    def __post_init__(self):
        _amplitudes(self, ("a_plus", "a_minus"))
        for name in ("tau_pre", "tau_minus", "tau_plus"):
            _quantity(getattr(self, name), "ms", positive=True)
        for name in ("theta_minus", "theta_plus", "voltage_rest", "spike_voltage"):
            _quantity(getattr(self, name), "mV")
        if (
            not self.voltage_rest.value
            <= self.theta_minus.value
            < self.theta_plus.value
        ):
            raise ValueError("require voltage_rest <= theta_minus < theta_plus")
        if self.spike_voltage.value <= self.theta_plus.value:
            raise ValueError("spike_voltage must exceed theta_plus")

    def json(self):
        return _rule_json(self, "voltage_stdp")


@dataclass(frozen=True, kw_only=True)
class CalciumPlasticity:
    """Deterministic (zero-noise) Graupner–Brunel bistable calcium rule.

    Calcium and efficacy are dimensionless. Efficacy is the normalized weight;
    the projection transmission delay is also the presynaptic calcium delay.
    """

    w_min: Quantity
    w_max: Quantity
    tau_calcium: Quantity = 20 * ms
    tau_weight: Quantity = 150000 * ms
    c_pre: float = 1.0
    c_post: float = 2.0
    theta_depression: float = 1.0
    theta_potentiation: float = 1.3
    gamma_depression: float = 200.0
    gamma_potentiation: float = 321.808
    rho_star: float = 0.5

    def __post_init__(self):
        _bounds(self.w_min, self.w_max)
        _quantity(self.tau_calcium, "ms", positive=True)
        _quantity(self.tau_weight, "ms", positive=True)
        for name in ("c_pre", "c_post", "gamma_depression", "gamma_potentiation"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not (0 < self.theta_depression < self.theta_potentiation < math.inf):
            raise ValueError("require 0 < theta_depression < theta_potentiation")
        if not 0 < self.rho_star < 1:
            raise ValueError("rho_star must lie strictly between zero and one")

    def json(self):
        return _rule_json(self, "calcium_plasticity")


@dataclass(frozen=True, kw_only=True)
class SynapticScaling:
    """Van Rossum proportional-integral rate homeostasis, composable with rules.

    beta is per second per Hz; gamma is per second squared per Hz. Rates are Hz
    and integration uses seconds. The rate starts at target_rate; the integral
    starts at zero.
    """

    target_rate: Quantity
    w_min: Quantity
    w_max: Quantity
    tau_rate: Quantity = 100000 * ms
    beta: float = 4e-5
    gamma: float = 1e-7

    def __post_init__(self):
        _bounds(self.w_min, self.w_max)
        _quantity(self.target_rate, "Hz", positive=True)
        _quantity(self.tau_rate, "ms", positive=True)
        for value in (self.beta, self.gamma):
            if not math.isfinite(value) or value < 0:
                raise ValueError("scaling gains must be finite and nonnegative")

    def json(self):
        return _rule_json(self, "synaptic_scaling")


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
        | VoltageSTDP
        | CalciumPlasticity
        | CustomPlasticity,
    ]
    scaling: Mapping[str, SynapticScaling] | None = None

    def __post_init__(self):
        rules = dict(self.rules)
        scaling = dict(self.scaling or {})
        if not (rules or scaling) or any(
            not isinstance(k, str) or not k for k in rules.keys() | scaling.keys()
        ):
            raise ValueError("plasticity requires named projection rules")
        if any(
            not isinstance(
                v,
                (
                    PairSTDP,
                    TripletSTDP,
                    InhibitorySTDP,
                    VoltageSTDP,
                    CalciumPlasticity,
                    CustomPlasticity,
                ),
            )
            for v in rules.values()
        ):
            raise TypeError(
                "plasticity rules must be built-in or CustomPlasticity rules"
            )
        object.__setattr__(self, "rules", MappingProxyType(rules))
        if any(not isinstance(v, SynapticScaling) for v in scaling.values()):
            raise TypeError("scaling must contain SynapticScaling specifications")
        for name in rules.keys() & scaling.keys():
            if (rules[name].w_min, rules[name].w_max) != (
                scaling[name].w_min,
                scaling[name].w_max,
            ):
                raise ValueError("composed plasticity and scaling bounds must agree")
        object.__setattr__(self, "scaling", MappingProxyType(scaling))

    def json(self):
        from snnlab import extensions as E

        data = {
            "schema": "snnlab.plasticity/v1",
            "rules": {name: rule.json() for name, rule in sorted(self.rules.items())},
        }
        if self.scaling:
            data["scaling"] = {
                name: rule.json() for name, rule in sorted(self.scaling.items())
            }
        return E.with_requirements(data)

    @classmethod
    def from_json(cls, data):
        """Restore JSON specifications and importable custom implementations."""
        from snnlab import extensions as E

        if data.get("schema") != "snnlab.plasticity/v1":
            raise ValueError("unsupported plasticity schema")
        E.restore(data)
        constructors = {
            "pair_stdp": PairSTDP,
            "nonlinear_stdp": NonlinearSTDP,
            "voltage_stdp": VoltageSTDP,
            "calcium_plasticity": CalciumPlasticity,
            "synaptic_scaling": SynapticScaling,
            "triplet_stdp": TripletSTDP,
            "inhibitory_stdp": InhibitorySTDP,
            "reward_modulated_stdp": RewardModulatedSTDP,
            "custom_plasticity": CustomPlasticity,
        }

        def restore(saved):
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
            return constructors[kind](**row)

        rules = {name: restore(saved) for name, saved in data["rules"].items()}
        scaling = {
            name: restore(saved) for name, saved in data.get("scaling", {}).items()
        }
        return cls(rules=rules, scaling=scaling)
