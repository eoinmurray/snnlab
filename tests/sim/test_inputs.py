import numpy as np
import pytest
import torch

from snnlab.sim.inputs import (
    DT_CAL,
    TAU_AMPA,
    drive_scale,
    make_reference_noise,
    make_step_drive,
)


class TestDriveScale:
    def test_identity_at_calibration_dt(self):
        assert drive_scale(DT_CAL) == pytest.approx(1.0)

    def test_monotonic_in_dt(self):
        dts = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
        scales = [drive_scale(dt) for dt in dts]
        assert scales == sorted(scales)

    def test_matches_analytic_form(self):
        dt = 0.5
        expected = (1 - np.exp(-dt / TAU_AMPA)) / (1 - np.exp(-DT_CAL / TAU_AMPA))
        assert drive_scale(dt) == pytest.approx(expected)


class TestMakeStepDrive:
    def test_output_shapes_and_types(self):
        n_e, t_steps, dt = 8, 100, 0.25
        sim, raw = make_step_drive(n_e, t_steps, dt, 0.0, 1.0, 20.0, 60.0)
        assert isinstance(sim, torch.Tensor)
        assert sim.shape == (t_steps, n_e)
        assert raw.shape == (t_steps, n_e)
        assert sim.dtype == torch.float32

    def test_step_boundaries(self):
        """Raw drive should be ~t_e_async outside the step window and
        ~t_e_ping inside it (averaged across neurons)."""
        n_e, t_steps, dt = 64, 400, 0.25  # 100 ms sim
        t_async, t_ping = 0.5, 2.0
        step_on, step_off = 25.0, 75.0
        _, raw = make_step_drive(
            n_e,
            t_steps,
            dt,
            t_async,
            t_ping,
            step_on,
            step_off,
            sigma_e=0.0,
            noise_sigma=0.0,  # disable heterogeneity + noise
        )
        t_ms = np.arange(t_steps) * dt
        pre = raw[t_ms < step_on].mean()
        during = raw[(t_ms >= step_on) & (t_ms < step_off)].mean()
        post = raw[t_ms >= step_off].mean()
        assert pre == pytest.approx(t_async, abs=1e-5)
        assert during == pytest.approx(t_ping, abs=1e-5)
        assert post == pytest.approx(t_async, abs=1e-5)

    def test_reproducible_under_same_seed(self):
        args = (16, 200, 0.25, 0.0, 1.0, 20.0, 60.0)
        a, _ = make_step_drive(*args, seed=123)
        b, _ = make_step_drive(*args, seed=123)
        assert torch.allclose(a, b)

    def test_different_seeds_differ(self):
        args = (16, 200, 0.25, 0.0, 1.0, 20.0, 60.0)
        a, _ = make_step_drive(*args, seed=1)
        b, _ = make_step_drive(*args, seed=2)
        assert not torch.allclose(a, b)

    def test_dt_invariant_steady_state(self):
        """Simulated ge steady-state should match across dt values.

        ge_{t+1} = decay * ge_t + ext_g_sim_t
        ge_ss = ext_g_sim / (1 - decay)   with decay = exp(-dt/tau_ampa)
        The dt-invariance claim: ge_ss(dt) is independent of dt.
        """
        n_e = 4
        t_async, t_ping = 0.0, 1.0
        ge_ss_values = []
        for dt in [0.1, 0.25, 0.5, 1.0]:
            sim_ms = 400.0
            t_steps = int(sim_ms / dt)
            # Hold at t_ping the whole time (step covers entire sim).
            sim, _ = make_step_drive(
                n_e,
                t_steps,
                dt,
                t_async,
                t_ping,
                0.0,
                sim_ms + 1.0,
                sigma_e=0.0,
                noise_sigma=0.0,
            )
            decay = np.exp(-dt / TAU_AMPA)
            inp = sim[-1, 0].item()  # steady-state input value
            ge_ss = inp / (1 - decay)
            ge_ss_values.append(ge_ss)
        # All dt values should give the same steady-state ge.
        for v in ge_ss_values[1:]:
            assert v == pytest.approx(ge_ss_values[0], rel=1e-5)


class TestReferenceNoise:
    def test_shapes_and_seed(self):
        n_e, sim_ms = 8, 50.0
        X1, eta1 = make_reference_noise(n_e, sim_ms, seed=7)
        X2, eta2 = make_reference_noise(n_e, sim_ms, seed=7)
        assert X1.shape == (n_e,)
        assert eta1.shape == (int(sim_ms / 0.01), n_e)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(eta1, eta2)
