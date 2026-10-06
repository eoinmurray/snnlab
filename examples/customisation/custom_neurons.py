"""Importable custom callbacks; kept outside the saved network bundle."""

import math

import torch


# 1. Define a neuron using tensor state and a normal Python step function.
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
    current = context.excitatory - context.inhibitory - adaptation
    voltage = (
        -65
        + (voltage + 65) * beta
        + current * context.config["tau_mem_ms"] * (1 - beta)
    )
    spikes = context.spike(voltage + 50)
    voltage = torch.where(spikes.bool(), torch.full_like(voltage, -65), voltage)
    adaptation = adaptation * math.exp(-context.dt_ms / context.config["tau_adapt_ms"])
    adaptation = adaptation + spikes * context.config["adaptation_na"]
    return {**context.state, "voltage": voltage, "adaptation": adaptation}, spikes


# 2. Define an initialization distribution with ordinary PyTorch.
def clipped_normal(shape, config, *, device, dtype):
    values = torch.randn(shape, device=device, dtype=dtype)
    return (values * config["std"] + config["mean"]).clamp(min=config["minimum"])
