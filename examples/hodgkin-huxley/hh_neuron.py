"""Classic single-compartment Hodgkin-Huxley dynamics as importable callbacks.

Uses the modern absolute-voltage convention (rest near -65 mV), with the
original squid-axon rates at 6.3 degrees C and no temperature rescaling.
Source: Hodgkin & Huxley (1952), https://doi.org/10.1113/jphysiol.1952.sp004764

A 1,000 um^2 patch converts the usual density parameters to snnlab's units:
1 uF/cm^2 -> 0.01 nF; (120, 36, 0.3) mS/cm^2 -> (1.2, 0.36, 0.003) uS.
Thus 0.1 nA injected current corresponds to 10 uA/cm^2. Intrinsic channel
conductances are present even though this example accepts current input.
"""

import math

import torch

CAPACITANCE_NF = 0.01
G_NA_US, G_K_US, G_LEAK_US = 1.2, 0.36, 0.003
E_NA_MV, E_K_MV, E_LEAK_MV = 50.0, -77.0, -54.387
MAX_STEP_MS = 0.025


def _vtrap(x):
    """x / (1 - exp(-x)), with its removable singularity handled smoothly."""
    small = x.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1 + x / 2 + x.square() / 12, safe / -torch.expm1(-safe))


def rates(voltage):
    """Return (alpha, beta) rates in 1/ms for the m, h and n gates."""
    return (
        (_vtrap((voltage + 40) / 10), 4 * torch.exp(-(voltage + 65) / 18)),
        (0.07 * torch.exp(-(voltage + 65) / 20), torch.sigmoid((voltage + 35) / 10)),
        (0.1 * _vtrap((voltage + 55) / 10), 0.125 * torch.exp(-(voltage + 65) / 80)),
    )


def initial_state(context):
    voltage = torch.full(
        context.shape, -65.0, device=context.device, dtype=context.dtype
    )
    state = {"voltage": voltage}
    for gate, (alpha, beta) in zip(("m", "h", "n"), rates(voltage)):
        state[gate] = alpha / (alpha + beta)
    return state


def _derivative(values, current):
    voltage, m, h, n = values
    ionic = (
        G_NA_US * m**3 * h * (voltage - E_NA_MV)
        + G_K_US * n**4 * (voltage - E_K_MV)
        + G_LEAK_US * (voltage - E_LEAK_MV)
    )
    gates = tuple(
        alpha * (1 - gate) - beta * gate
        for gate, (alpha, beta) in zip((m, h, n), rates(voltage))
    )
    return ((current - ionic) / CAPACITANCE_NF, *gates)


def step(context):
    """RK4 with fixed substeps; input is held constant within the network step.

    Spikes mark upward 0 mV crossings, never a voltage reset. The unused
    runtime refractory state is preserved. At most one event is returned per
    network step, so choose a network timestep that resolves spike timing.
    """
    keys = ("voltage", "m", "h", "n")
    values = tuple(context.state[key] for key in keys)
    current = (context.excitatory - context.inhibitory).to(values[0].dtype)
    substeps = max(1, math.ceil(context.dt_ms / MAX_STEP_MS))
    dt = context.dt_ms / substeps
    crossed = torch.zeros_like(values[0], dtype=torch.bool)
    for _ in range(substeps):
        k1 = _derivative(values, current)
        k2 = _derivative(tuple(y + dt / 2 * k for y, k in zip(values, k1)), current)
        k3 = _derivative(tuple(y + dt / 2 * k for y, k in zip(values, k2)), current)
        k4 = _derivative(tuple(y + dt * k for y, k in zip(values, k3)), current)
        updated = tuple(
            y + dt / 6 * (a + 2 * b + 2 * c + d)
            for y, a, b, c, d in zip(values, k1, k2, k3, k4)
        )
        crossed = crossed | ((values[0] < 0) & (updated[0] >= 0))
        values = updated
    return {**context.state, **dict(zip(keys, values))}, crossed.to(values[0].dtype)


def current_step(context):
    """Memoryless current injection: projected drive is already in nA."""
    return {"value": context.drive.to(context.state["value"].dtype)}
