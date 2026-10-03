"""Coverage for inputs.make_step_drive_from_ref — the dt-invariant drive path.

test_inputs.py covers drive_scale / make_step_drive / make_reference_noise; this
file fills the remaining gap: building a drive at a target dt by interpolating
from reference-resolution OU noise (used by the dt-stability sweep).
"""

from __future__ import annotations

import numpy as np

from snnlab.sim.inputs import (
    drive_scale,
    make_reference_noise,
    make_step_drive_from_ref,
)


def test_from_ref_shapes_and_scaling():
    n_e, sim_ms, dt = 8, 50.0, 0.25
    X_i, eta_ref = make_reference_noise(n_e, sim_ms, seed=1)
    ext_g_sim, ext_g_raw = make_step_drive_from_ref(
        n_e,
        dt=dt,
        t_e_async=0.3,
        t_e_ping=1.2,
        step_on_ms=10.0,
        step_off_ms=40.0,
        sim_ms=sim_ms,
        X_i=X_i,
        eta_ref=eta_ref,
    )
    t_steps = int(sim_ms / dt)
    assert ext_g_sim.shape == (t_steps, n_e)
    assert ext_g_raw.shape == (t_steps, n_e)
    # sim drive is the raw drive scaled by the dt-invariant factor
    np.testing.assert_allclose(ext_g_sim, ext_g_raw * drive_scale(dt), rtol=1e-6)


def test_from_ref_step_window_is_higher_than_baseline():
    n_e, sim_ms, dt = 4, 60.0, 0.5
    X_i, eta_ref = make_reference_noise(n_e, sim_ms, seed=7)
    _, raw = make_step_drive_from_ref(
        n_e,
        dt=dt,
        t_e_async=0.2,
        t_e_ping=2.0,
        step_on_ms=20.0,
        step_off_ms=40.0,
        sim_ms=sim_ms,
        X_i=X_i,
        eta_ref=eta_ref,
        sigma_e=0.0,  # kill heterogeneity so the step is the only signal
    )
    t_ms = np.arange(int(sim_ms / dt)) * dt
    inside = raw[(t_ms >= 20.0) & (t_ms < 40.0)].mean()
    outside = raw[(t_ms < 20.0) | (t_ms >= 40.0)].mean()
    assert inside > outside


def test_from_ref_clips_negative_drive_to_zero():
    n_e, sim_ms, dt = 4, 20.0, 0.5
    X_i, eta_ref = make_reference_noise(n_e, sim_ms, seed=3)
    _, raw = make_step_drive_from_ref(
        n_e,
        dt=dt,
        t_e_async=-5.0,  # negative baseline → clipped to 0
        t_e_ping=-5.0,
        step_on_ms=0.0,
        step_off_ms=0.0,
        sim_ms=sim_ms,
        X_i=X_i,
        eta_ref=eta_ref,
    )
    assert np.all(raw >= 0.0)
