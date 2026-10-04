"""Tensor contracts and runtime helpers for graph extensions."""

from __future__ import annotations

import math
from typing import Mapping

import torch

from snnlab import extensions as E


def checked_tensor(value, *, shape, device, dtype, name):
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
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} tensor must be finite")
    return value


def checked_state(value, template, name, *, required=()):
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
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError(f"{name}.{key} must be a finite tensor")
        if template is not None:
            previous = template[key]
            checked_tensor(
                tensor,
                shape=previous.shape,
                device=previous.device,
                dtype=previous.dtype,
                name=f"{name}.{key}",
            )
    return dict(value)


def spike(value, *, slope, custom=None):
    if custom is None:
        from . import models as M

        return M.fast_sigmoid_spike(value, slope)
    definition = E.resolve("surrogate", custom)

    class RegisteredSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return (x > 0).to(x.dtype)

        @staticmethod
        def backward(ctx, gradient):
            (x,) = ctx.saved_tensors
            derivative = definition.function(x, custom.get("config", {}))
            checked_tensor(
                derivative,
                shape=x.shape,
                device=x.device,
                dtype=x.dtype,
                name=definition.name,
            )
            return gradient * derivative

    return RegisteredSpike.apply(value)


def current_lif(state, excitatory, inhibitory, *, dt_ms, config, spike_function):
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
        torch.full_like(refractory, int(config["refractory_steps"])),
        (refractory - 1).clamp(min=0),
    )
    return next_voltage, spikes, next_ref


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
