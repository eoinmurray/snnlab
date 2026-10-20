"""Importable custom callbacks; kept outside the saved network bundle."""

import math

import torch


# 1. Define a neuron using tensor state and a normal Python step function.
def validate_adaptive(config):
    for name in ("tau_mem_ms", "tau_adapt_ms"):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(config["adaptation_na"]) or config["adaptation_na"] < 0:
        raise ValueError("adaptation_na must be finite and nonnegative")


def adaptive_initial_state(context):
    shape, device, dtype = context.shape, context.device, context.dtype
    return {
        "voltage": torch.full(shape, -65.0, device=device, dtype=dtype),
        "adaptation": torch.zeros(shape, device=device, dtype=dtype),
    }


def adaptive_step(context):
    voltage = context.state["voltage"]
    adaptation = context.state["adaptation"]
    beta = math.exp(-context.dt_ms / context.config["tau_mem_ms"])
    # State retains its dtype even when an upstream CUDA matmul uses BF16.
    current = (
        context.excitatory.to(voltage.dtype)
        - context.inhibitory.to(voltage.dtype)
        - adaptation
    )
    voltage = (
        -65
        + (voltage + 65) * beta
        + current * context.config["tau_mem_ms"] * (1 - beta)
    )
    spikes = context.spike(voltage + 50)
    # Broadcast the reset scalar; no per-step full-sized reset allocation.
    voltage = torch.where(spikes.bool(), -65.0, voltage)
    adaptation = adaptation * math.exp(-context.dt_ms / context.config["tau_adapt_ms"])
    adaptation = adaptation + spikes * context.config["adaptation_na"]
    return {**context.state, "voltage": voltage, "adaptation": adaptation}, spikes


# 2. Define a synapse using the already weight-projected drive.
def validate_exponential(config):
    if not math.isfinite(config["tau_ms"]) or config["tau_ms"] <= 0:
        raise ValueError("tau_ms must be finite and positive")


def exponential_initial_state(context):
    return {
        "value": torch.zeros(context.shape, device=context.device, dtype=context.dtype)
    }


def exponential_step(context):
    previous = context.state["value"]
    # Configuration is fixed per prepared kernel; the scalar specializes.
    decay = math.exp(-context.dt_ms / context.config["tau_ms"])
    drive = context.drive.to(previous.dtype)
    return {**context.state, "value": previous * decay + drive}


# 3. Define an initialization distribution with ordinary PyTorch.
def clipped_normal(shape, config, *, device, dtype):
    values = torch.randn(shape, device=device, dtype=dtype)
    return (values * config["std"] + config["mean"]).clamp(min=config["minimum"])
