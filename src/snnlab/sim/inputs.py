"""Synthetic input generation for PING networks.

Provides dt-invariant drive generation with Börgers-style step + OU noise.
All drive values are calibrated relative to DT_CAL=0.1 ms.
"""

from __future__ import annotations

import numpy as np
import torch

# ── Calibration reference ──
DT_CAL = 0.1  # T_E values were calibrated at this dt
DT_REF = 0.01  # reference resolution for noise generation (dt-stability)
TAU_AMPA = 2.0  # AMPA decay for scaling


def drive_scale(dt):
    """Compute the dt-invariant scaling factor for conductance injection.

    Ensures steady-state ge is the same regardless of dt:
        ge_ss = (T_E * scale) / (1 - exp(-dt/tau))  =  constant
    """
    return (1 - np.exp(-dt / TAU_AMPA)) / (1 - np.exp(-DT_CAL / TAU_AMPA))


def make_step_drive(
    n_e,
    t_steps,
    dt,
    t_e_async,
    t_e_ping,
    step_on_ms,
    step_off_ms,
    sigma_e=0.05,
    noise_sigma=0.001,
    noise_tau=3.0,
    seed=42,
    noise_seed=None,
):
    """Börgers-style tonic drive with step + independent OU noise per neuron.

    seed controls per-neuron heterogeneity (X_i).
    noise_seed controls OU noise; defaults to seed if not set.

    Returns:
        ext_g_sim: (t_steps, n_e) tensor — dt-scaled, feed directly to network
        ext_g_raw: (t_steps, n_e) ndarray — physical values for display
    """
    rng_het = np.random.RandomState(seed)
    X_i = rng_het.randn(n_e)

    rng_noise = np.random.RandomState(noise_seed if noise_seed is not None else seed)
    decay = np.exp(-dt / noise_tau)
    noise_scale = noise_sigma * np.sqrt(1 - decay**2)
    eta = np.zeros((t_steps, n_e))
    for t in range(1, t_steps):
        eta[t] = eta[t - 1] * decay + noise_scale * rng_noise.randn(n_e)

    ext_g_raw = np.zeros((t_steps, n_e))
    for t in range(t_steps):
        t_ms = t * dt
        T_E = t_e_ping if step_on_ms <= t_ms < step_off_ms else t_e_async
        drive = T_E * (1.0 + sigma_e * X_i) + eta[t]
        ext_g_raw[t] = np.clip(drive, 0, None)

    scale = drive_scale(dt)
    ext_g_sim = ext_g_raw * scale
    return torch.tensor(ext_g_sim, dtype=torch.float32), ext_g_raw


def make_reference_noise(n_e, sim_ms, noise_sigma=0.001, noise_tau=3.0, seed=42):
    """Generate OU noise and heterogeneity at reference resolution (DT_REF).

    Used by dt-stability to ensure identical noise across dt values.

    Returns:
        X_i: (n_e,) per-neuron heterogeneity factors
        eta_ref: (t_steps_ref, n_e) OU noise at DT_REF resolution
    """
    t_steps_ref = int(sim_ms / DT_REF)
    rng = np.random.RandomState(seed)
    X_i = rng.randn(n_e)
    decay = np.exp(-DT_REF / noise_tau)
    noise_scale = noise_sigma * np.sqrt(1 - decay**2)
    eta = np.zeros((t_steps_ref, n_e))
    for t in range(1, t_steps_ref):
        eta[t] = eta[t - 1] * decay + noise_scale * rng.randn(n_e)
    return X_i, eta


def make_step_drive_from_ref(
    n_e,
    dt,
    t_e_async,
    t_e_ping,
    step_on_ms,
    step_off_ms,
    sim_ms,
    X_i,
    eta_ref,
    sigma_e=0.05,
):
    """Build drive at target dt by interpolating from reference noise.

    Used by dt-stability for dt-invariant noise across different dt values.

    Returns:
        ext_g_sim: (t_steps, n_e) ndarray — dt-scaled for simulation
        ext_g_raw: (t_steps, n_e) ndarray — physical values for display
    """
    t_steps = int(sim_ms / dt)
    t_ms_target = np.arange(t_steps) * dt
    ref_indices = np.clip((t_ms_target / DT_REF).astype(int), 0, len(eta_ref) - 1)
    eta = eta_ref[ref_indices]

    ext_g_raw = np.zeros((t_steps, n_e))
    for t in range(t_steps):
        T_E = t_e_ping if step_on_ms <= t_ms_target[t] < step_off_ms else t_e_async
        drive = T_E * (1.0 + sigma_e * X_i) + eta[t]
        ext_g_raw[t] = np.clip(drive, 0, None)

    scale = drive_scale(dt)
    ext_g_sim = ext_g_raw * scale
    return ext_g_sim, ext_g_raw



