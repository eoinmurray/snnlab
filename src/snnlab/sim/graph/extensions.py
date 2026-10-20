"""Tensor contracts and runtime helpers for graph extensions."""

from __future__ import annotations

import copy
import math
from typing import Mapping

import torch

from snnlab import extensions as E
from snnlab._shared import neurons as N
from snnlab.sim.graph.validation import _PENDING, check_finite


def checked_tensor(value, *, shape, device, dtype, name, check_values=True):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must return a torch.Tensor")
    if (
        tuple(value.shape) != tuple(shape)
        or value.device != torch.device(device)
        or value.dtype != dtype
    ):
        raise ValueError(
            f"{name} tensor must have shape={tuple(shape)}, device={device}, dtype={dtype}; got shape={tuple(value.shape)}, device={value.device}, dtype={value.dtype}"
        )
    if check_values:
        check_finite(value, name)
    return value


def checked_state(value, template, name, *, required=(), check_values=True):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must return a mapping of tensor state")
    if template is not None and set(value) != set(template):
        raise ValueError(f"{name} state keys changed during execution")
    if set(required) - set(value):
        raise ValueError(
            f"{name} missing required state {sorted(set(required) - set(value))}"
        )
    for key, tensor in value.items():
        if not isinstance(key, str) or not key or "/" in key or "." in key:
            raise ValueError(f"{name} state keys must be simple strings")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name}.{key} must be a finite tensor")
        if template is not None:
            previous = template[key]
            checked_tensor(
                tensor,
                shape=previous.shape,
                device=previous.device,
                dtype=previous.dtype,
                name=f"{name}.{key}",
                check_values=False,
            )
    if check_values and _PENDING.get() is not None:
        for key, tensor in value.items():
            check_finite(tensor, f"{name}.{key}")
    elif check_values:
        # Structural validation above retains useful errors. On valid states,
        # reduce finite flags per device before asking Python for a scalar.
        by_device = {}
        for key, tensor in value.items():
            by_device.setdefault(tensor.device, []).append(
                (key, torch.isfinite(tensor).all())
            )
        for flags in by_device.values():
            if not torch.stack([flag for _, flag in flags]).all():
                for key, flag in flags:
                    if not flag:
                        raise ValueError(f"{name}.{key} must be a finite tensor")
    return dict(value)


def make_spike(
    *,
    slope,
    custom=None,
    check_values=True,
    gradient_scale=1.0,
    threshold_inclusive=True,
    compile_mode=None,
):
    """Resolve a surrogate and create its autograd dispatch once per executor."""
    if custom is None:
        from snnlab.sim.graph import dynamics as M

        return lambda value: M.fast_sigmoid_spike(
            value, slope, gradient_scale, threshold_inclusive
        )
    definition = E.resolve("surrogate", custom)

    config = copy.deepcopy(custom.get("config", {}))

    def derivative(value):
        return definition.function(value, config)

    # Strict diagnostics cross an eager autograd boundary; the derivative's
    # numerical work can still compile independently of the host value checks.
    if compile_mode is not None:
        from snnlab.sim.graph.kernels import prepare_kernel

        derivative = prepare_kernel(derivative, compile_mode)

    class RegisteredSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return (x > 0).to(x.dtype)

        @staticmethod
        def backward(ctx, gradient):
            (x,) = ctx.saved_tensors
            values = derivative(x)
            checked_tensor(
                values,
                shape=x.shape,
                device=x.device,
                dtype=x.dtype,
                name=definition.name,
                check_values=check_values,
            )
            return gradient * values

    if compile_mode is not None and check_values:
        return torch.compiler.disable(RegisteredSpike.apply)
    return RegisteredSpike.apply


def spike(value, *, slope, custom=None):
    """Compatibility helper; executors retain the callable from make_spike."""
    return make_spike(slope=slope, custom=custom)(value)


def current_lif(
    state, excitatory, inhibitory, *, dt_ms, config, spike_function, ref_steps=None
):
    if ref_steps is None:
        ref_steps = N.resolve_refractory(config, dt_ms)["refractory_steps"]
    voltage, refractory = state["voltage"], state["refractory"]
    tau = float(config["tau_mem"]["value"])
    capacitance = float(config["capacitance_nf"])
    rest = float(config["resting_mv"])
    beta = math.exp(-dt_ms / tau)
    update = (rest - voltage) * (1 - beta) + (
        excitatory - inhibitory
    ) * tau / capacitance * (1 - beta)
    dampen = float(config.get("voltage_grad_dampen", 1.0))
    update = update.detach() + (update - update.detach()) / dampen
    active = refractory == 0
    candidate = torch.where(
        active, voltage + update, torch.full_like(voltage, float(config["reset_mv"]))
    )
    spikes = spike_function(candidate - float(config["threshold_mv"])) * active.to(
        voltage.dtype
    )
    next_voltage = torch.where(
        spikes.bool(), torch.full_like(voltage, float(config["reset_mv"])), candidate
    )
    next_ref = torch.where(
        spikes.bool(),
        torch.full_like(refractory, ref_steps),
        (refractory - 1).clamp(min=0),
    )
    return next_voltage, spikes, next_ref


def adaptive_neuron(
    state, excitatory, inhibitory, *, dt_ms, config, spike_function, ref_steps=None
):
    """ALIF/AdEx with exponential Euler and left-endpoint nonlinear terms.

    Linear membrane dynamics are integrated exactly with frozen synaptic drive
    and adaptation. AdEx's exponential term and subthreshold adaptation use the
    previous voltage. Adaptation decays even during refractory steps, then gets
    its per-spike increment. All state is per batch item and per cell.
    """
    if ref_steps is None:
        ref_steps = N.resolve_refractory(config, dt_ms)["refractory_steps"]
    voltage, refractory = state["voltage"], state["refractory"]
    adaptation = state["adaptation"]
    kind = config["kind"]
    adex = kind.endswith("adex")
    capacitance = float(config["capacitance_nf"])
    tau = float(config["tau_mem"]["value"])
    leak = capacitance / tau
    rest = float(config["resting_mv"])
    current_adaptation = adaptation if adex else 0.0
    if kind.startswith("cuba_"):
        beta = math.exp(-dt_ms / tau)
        gain = tau / capacitance * (1 - beta)
        update = (rest - voltage) * (1 - beta) + (
            excitatory - inhibitory - current_adaptation
        ) * gain
    else:
        total = leak + excitatory + inhibitory
        decay_fraction = -torch.expm1(-dt_ms * total / capacitance)
        gain = decay_fraction / total
        equilibrium = (
            leak * rest
            + excitatory * float(config["excitatory_reversal_mv"])
            + inhibitory * float(config["inhibitory_reversal_mv"])
            - current_adaptation
        ) / total
        update = (equilibrium - voltage) * decay_fraction
    if adex:
        delta = float(config["delta_t_mv"])
        # Work in log space and bound only extreme, already suprathreshold
        # increments. This avoids exp overflow without changing subthreshold
        # dynamics. The cap is far above any biophysical voltage excursion.
        log_gain = math.log(leak) + math.log(delta)
        log_gain = log_gain + (
            torch.log(gain) if isinstance(gain, torch.Tensor) else math.log(gain)
        )
        max_log = math.log(torch.finfo(voltage.dtype).max) / 2
        update = update + torch.exp(
            ((voltage - float(config["threshold_mv"])) / delta + log_gain).clamp(
                max=max_log
            )
        )
        threshold = float(config["spike_mv"])
        target = float(config["a_us"]) * (voltage - rest)
        increment = float(config["b_na"])
    else:
        threshold = float(config["threshold_mv"]) + adaptation
        target = 0.0
        increment = float(config["adaptation_increment_mv"])
    dampen = float(config["voltage_grad_dampen"])
    update = update.detach() + (update - update.detach()) / dampen
    active = refractory == 0
    reset = torch.full_like(voltage, float(config["reset_mv"]))
    candidate = torch.where(active, voltage + update, reset)
    spikes = spike_function(candidate - threshold) * active.to(voltage.dtype)
    rho = math.exp(-dt_ms / float(config["tau_adaptation"]["value"]))
    next_adaptation = rho * adaptation + (1 - rho) * target + increment * spikes
    return {
        "voltage": torch.where(spikes.bool(), reset, candidate),
        "refractory": torch.where(
            spikes.bool(),
            torch.full_like(refractory, ref_steps),
            (refractory - 1).clamp(min=0),
        ),
        "adaptation": next_adaptation,
    }, spikes


def apply_constraint(parameter, spec):
    if not spec:
        return parameter
    if spec["kind"] == "non_negative":
        return parameter.clamp(min=0)
    definition = E.resolve("constraint", spec)
    value = definition.function(parameter, spec.get("config", {}))
    return checked_tensor(
        value,
        shape=parameter.shape,
        device=parameter.device,
        dtype=parameter.dtype,
        name=definition.name,
    )


def classification(objective):
    return objective.get("kind") == "cross_entropy" or (
        objective.get("kind") == "custom_objective"
        and E.resolve("objective", objective).classification
    )
