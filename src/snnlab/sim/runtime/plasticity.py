"""Private local-learning engine; evolving weights are dynamic trajectory state."""

from __future__ import annotations

from collections import Counter

import torch

from snnlab import extensions as E
from snnlab.plasticity import (
    CustomPlasticity,
    InhibitorySTDP,
    Plasticity,
    RewardModulatedSTDP,
    VoltageSTDP,
)
from snnlab.sim.graph import extensions as X
from snnlab.sim.runtime import plasticity_rules


def _validate_rule_projection(rule, projection):
    if isinstance(rule, InhibitorySTDP) and projection.polarity != "inhibitory":
        raise ValueError(
            f"{projection.id}: InhibitorySTDP requires an inhibitory projection"
        )


def _reward_inputs(request, executor, values):
    inputs = {row["id"]: row for row in executor.plan.graph["inputs"]}
    rewards = {}
    for name, rule in request.rules.items():
        if not isinstance(rule, RewardModulatedSTDP):
            continue
        row = inputs.get(rule.reward_input)
        if (
            row is None
            or row["unit"] != "1"
            or row["signal_type"] in {"spikes", "mask"}
            or row["shape"] != ["time", "batch", 1]
        ):
            raise ValueError(
                f"{name}: reward_input requires a dimensionless continuous [time, batch, 1] graph input"
            )
        value = (values or {}).get(rule.reward_input)
        if (
            value is None
            or value.ndim != 3
            or tuple(value.shape[1:]) != (1, 1)
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                f"{name}: reward input must contain finite scalar samples for batch_size=1"
            )
        rewards[name] = value
    return rewards


class PlasticitySession:
    """Own private weights/traces until a simulation successfully completes."""

    def __init__(self, request, executor, *, batch, saved=None, inputs=None):
        if not isinstance(request, Plasticity):
            raise TypeError("plasticity must be Plasticity")
        if batch != 1:
            raise ValueError("plasticity requires batch_size=1")
        if getattr(executor, "_stream_metadata", None) is not None:
            raise ValueError("plasticity is unavailable in prepared streaming")
        self.request = request
        self.rules = dict(request.scaling) | dict(request.rules)
        self.executor = executor
        self.projections = {p.id: p for p in executor.plan.projections}
        self.states = {}
        self.weights = {}
        self.definitions = {}
        self.rewards = _reward_inputs(request, executor, inputs)
        self.dt_ms = executor.plan.dt_ms
        graph = executor.plan.graph
        self.rows = {p["id"]: p for p in graph["projections"]}
        parameters = {p["id"]: p for p in graph["parameters"]}
        owners = Counter(p.parameter for p in executor.plan.projections)
        for op in graph.get("operations", []):
            owners.update(op.get("parameters", []))
        populations = {p["id"]: p for p in graph["populations"]}
        inputs = {p["id"]: p for p in graph["inputs"]}
        for name, rule in self.rules.items():
            if name not in self.projections:
                raise ValueError(f"unknown plasticity projection {name!r}")
            p = self.projections[name]
            row = self.rows[name]
            _validate_rule_projection(rule, p)
            parameter = parameters[p.parameter]
            source, _, port = p.source.partition(".")
            spike_source = (source in populations and port == "spikes") or (
                source in inputs and inputs[source]["signal_type"] == "spikes"
            )
            target = populations[p.target.partition(".")[0]]
            if (
                isinstance(rule, VoltageSTDP)
                and target["neuron"]["kind"] == "leaky_integrator"
            ):
                raise ValueError(f"{name}: VoltageSTDP requires membrane voltage in mV")
            if (
                not p.enabled
                or p.continuous
                or not spike_source
                or not target["spiking"]
            ):
                raise ValueError(
                    f"{name}: plasticity requires an enabled spike-to-spike projection"
                )
            if row.get("connectivity") is not None or owners[p.parameter] != 1:
                raise ValueError(f"{name}: plasticity requires dense, unshared weights")
            if parameter["initializer"].get("initial_zero_fraction", 0):
                raise ValueError(
                    f"{name}: sparse initialization is unsupported by plasticity"
                )
            if parameter["unit"] != rule.w_min.unit:
                raise ValueError(
                    f"{name}: plasticity quantities require {parameter['unit']}"
                )
            weight = executor.parameter_map()[p.parameter].detach().clone()
            if saved is None:
                self._validate_weight(name, weight, rule)
            self.weights[p.parameter] = weight
            self.states[name] = self._initial_state(name, rule, weight)
            if name in request.scaling:
                rate = weight.new_full(
                    (1, weight.shape[1]), request.scaling[name].target_rate.value
                )
                self.states[name].update(
                    scaling_rate=rate, scaling_integral=torch.zeros_like(rate)
                )
        if saved is not None:
            self._restore(saved)

    def _initial_state(self, name, rule, weight):
        if not isinstance(rule, CustomPlasticity):
            state = plasticity_rules.initial_state(rule, weight)
        else:
            definition = E.resolve("plasticity", rule.json())
            if definition.initialize is None:
                raise ValueError("custom plasticity requires an initializer")
            self.definitions[name] = definition
            state = definition.initialize(
                E.PlasticityInitContext(
                    weight.clone(), self.dt_ms, rule.json()["config"]
                )
            )
            state = X.checked_state(state, None, definition.name)
            if set(state) != set(definition.state_units):
                raise ValueError(
                    f"{name}: plasticity state must match registered state_units"
                )
            for key, value in state.items():
                X.checked_tensor(
                    value,
                    shape=value.shape,
                    device=weight.device,
                    dtype=weight.dtype,
                    name=f"{name}.{key}",
                )
        if (
            name in self.request.scaling
            and {"scaling_rate", "scaling_integral"} & state.keys()
        ):
            raise ValueError(
                f"{name}: scaling_rate and scaling_integral are reserved state names"
            )
        return {k: v.detach().clone() for k, v in state.items()}

    def _validate_weight(self, name, value, rule):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name}: plastic weights must be finite")
        p = self.projections[name]
        active = torch.ones_like(value, dtype=torch.bool)
        if p.parameter in self.executor._excluded_autapse_parameters:
            active.fill_diagonal_(False)
            if torch.any(value.diagonal() != 0):
                raise ValueError(f"{name}: excluded autapses must remain zero")
        if torch.any(
            (value[active] < rule.w_min.value) | (value[active] > rule.w_max.value)
        ):
            raise ValueError(
                f"{name}: initial/restored weights are outside plasticity bounds"
            )

    def tensors(self):
        result = {}
        for name, state in self.states.items():
            prefix = f"plasticity/{name}/"
            result[prefix + "weight"] = self.weights[self.projections[name].parameter]
            result.update(
                {prefix + "trace/" + key: value for key, value in state.items()}
            )
        return result

    def _restore(self, saved):
        expected = self.tensors()
        actual = {k: v for k, v in saved.items() if k.startswith("plasticity/")}
        if set(actual) != set(expected):
            raise ValueError("runtime plasticity tensor keys differ from rule state")
        restored = {}
        for key, template in expected.items():
            value = actual[key]
            # Portable runtime states may be on CPU; dtype and shape remain exact.
            value = value.detach().to(template.device).clone()
            restored[key] = X.checked_tensor(
                value,
                shape=template.shape,
                device=template.device,
                dtype=template.dtype,
                name=key,
            )
        for name, state in self.states.items():
            prefix = f"plasticity/{name}/"
            weight = restored[prefix + "weight"]
            self._validate_weight(name, weight, self.rules[name])
            self.weights[self.projections[name].parameter] = weight
            self.states[name] = {
                key: restored[prefix + "trace/" + key] for key in state
            }

    @torch.no_grad()
    def step(self, arrivals, spikes, step_index, voltages=None):
        for name, rule in self.rules.items():
            p = self.projections[name]
            weight = self.weights[p.parameter]
            pre = arrivals[name]
            post = spikes[p.target.partition(".")[0]]
            if torch.any((pre != 0) & (pre != 1)) or torch.any(
                (post != 0) & (post != 1)
            ):
                raise ValueError(f"{name}: plasticity requires binary spikes")
            state = self.states[name]
            scaling_state = {
                key: value
                for key, value in state.items()
                if name in self.request.scaling
                and key in {"scaling_rate", "scaling_integral"}
            }
            state = {
                key: value for key, value in state.items() if key not in scaling_state
            }
            if not isinstance(rule, CustomPlasticity):
                delta, state = plasticity_rules.step(
                    rule,
                    weight,
                    pre,
                    post,
                    state,
                    self.dt_ms,
                    self.rewards[name][step_index] if name in self.rewards else None,
                    voltages[p.target.partition(".")[0]]
                    if voltages is not None
                    else None,
                )
            else:
                definition = self.definitions[name]
                # Callbacks cannot mutate the engine, neural state or source inputs.
                delta, updated = definition.function(
                    E.PlasticityContext(
                        weight.clone(),
                        pre.clone(),
                        post.clone(),
                        {k: v.clone() for k, v in state.items()},
                        self.dt_ms,
                        rule.json()["config"],
                    )
                )
                state = X.checked_state(updated, state, definition.name)
            if name in self.request.scaling:
                scaling_delta, rate, integral = plasticity_rules.scaling_step(
                    self.request.scaling[name],
                    weight,
                    post,
                    scaling_state["scaling_rate"],
                    scaling_state["scaling_integral"],
                    self.dt_ms,
                )
                delta = delta + scaling_delta
                state.update(scaling_rate=rate, scaling_integral=integral)
            X.checked_tensor(
                delta,
                shape=weight.shape,
                device=weight.device,
                dtype=weight.dtype,
                name=f"{name}.delta",
            )
            candidate = (weight + delta).clamp(rule.w_min.value, rule.w_max.value)
            constraint = self.executor._constraints.get(p.parameter)
            if constraint is not None:
                candidate = constraint(candidate)
            candidate = self.executor._mask_autapses(p.parameter, candidate)
            self._validate_weight(name, candidate, rule)
            weight.copy_(candidate)
            self.states[name] = {k: v.detach().clone() for k, v in state.items()}

    @torch.no_grad()
    def commit(self):
        for parameter, value in self.weights.items():
            self.executor.parameter_map()[parameter].copy_(value)
