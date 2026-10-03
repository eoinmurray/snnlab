"""Exponential-Euler COBA membrane update.

Tests the new `lif_step_expeuler` primitive that replaces the forward-Euler
`lif_step` for biophysical models. Under a zero-order hold on g_e, g_i over
one step of length dt:

    g_tot   = g_L + g_e + g_i
    tau_eff = C_m / g_tot
    v_inf   = (g_L*E_L + g_e*E_e + g_i*E_i) / g_tot
    v_{t+1} = v_inf + (v_t - v_inf) * exp(-dt / tau_eff)

These tests are the acceptance contract for the new primitive; they fail
until `lif_step_expeuler` is implemented.
"""

import math

import pytest
import torch

from snnlab.sim import models as M
from snnlab.sim.models import spike_biophysical

pytest.importorskip("models")  # noqa
# The symbol under test — implemented in a follow-up commit.
lif_step_expeuler = pytest.importorskip(
    "models", reason="lif_step_expeuler not yet implemented"
).__dict__.get("lif_step_expeuler")

pytestmark = pytest.mark.skipif(
    lif_step_expeuler is None,
    reason="lif_step_expeuler not yet implemented — TDD stub",
)


def _fresh_state(v0=None, B=1, N=1, dtype=torch.float32):
    v = torch.full((B, N), v0 if v0 is not None else M.E_L, dtype=dtype)
    ref = torch.zeros((B, N), dtype=torch.long)
    return v, ref


def _step(v, ref, *, g_e=0.0, g_i=None, C_m=None, g_L=None, ref_steps=None, dt=None):
    """Thin adapter so tests don't carry the full call signature."""
    C_m = M.C_m_E if C_m is None else C_m
    g_L = M.g_L_E if g_L is None else g_L
    ref_steps = M.ref_steps_E if ref_steps is None else ref_steps
    g_e_t = torch.as_tensor(g_e, dtype=v.dtype).broadcast_to(v.shape)
    g_i_t = (
        None
        if g_i is None
        else torch.as_tensor(g_i, dtype=v.dtype).broadcast_to(v.shape)
    )
    kwargs = {}
    if dt is not None:
        kwargs["dt_override"] = dt
    return lif_step_expeuler(
        v, ref, g_e_t, g_i_t, C_m, g_L, ref_steps, spike_biophysical, **kwargs
    )


class TestPassiveDecay:
    def test_resting_zero_input_is_stationary(self):
        """v=E_L, no conductance input → voltage doesn't move, no spike."""
        v, ref = _fresh_state()
        v2, s, ref2 = _step(v, ref)
        torch.testing.assert_close(v2, v)
        assert s.item() == 0.0
        assert ref2.item() == 0

    def test_passive_decay_matches_closed_form(self):
        """With g_e=g_i=0 and v != E_L, one step follows
            v_{t+1} = E_L + (v_t - E_L) * exp(-dt * g_L / C_m)
        exactly (this is the exp-Euler / exact solution for the homogeneous
        ODE C dv/dt = -g_L (v - E_L))."""
        v0 = -55.0  # 10 mV above E_L
        v, ref = _fresh_state(v0=v0)
        v2, _, _ = _step(v, ref)
        expected = M.E_L + (v0 - M.E_L) * math.exp(-M.dt * M.g_L_E / M.C_m_E)
        assert v2.item() == pytest.approx(expected, abs=1e-6)

    def test_passive_decay_is_dt_invariant(self):
        """The headline win: N steps at dt equal 1 step at N*dt, exactly.
        Forward Euler does not satisfy this; exp-Euler does."""
        v0 = -55.0
        # Fine: N steps at small dt (float64 to see exact-integrator equality).
        v_fine, ref = _fresh_state(v0=v0, dtype=torch.float64)
        N = 10
        dt_fine = 0.1
        for _ in range(N):
            v_fine, _, ref = _step(v_fine, ref, dt=dt_fine)
        v_coarse, ref_c = _fresh_state(v0=v0, dtype=torch.float64)
        v_coarse, _, _ = _step(v_coarse, ref_c, dt=N * dt_fine)
        assert v_fine.item() == pytest.approx(v_coarse.item(), abs=1e-10)


class TestConductanceDrive:
    def test_steady_state_with_constant_g_e(self):
        """Under constant g_e with no g_i, running many steps drives v toward
            v_inf = (g_L*E_L + g_e*E_e) / (g_L + g_e)
        which sits below V_th (no spikes) for g_e small enough."""
        g_e = 0.01  # uS, below rheobase
        v_inf = (M.g_L_E * M.E_L + g_e * M.E_e) / (M.g_L_E + g_e)
        assert v_inf < M.V_th, "test precondition: subthreshold drive"
        v, ref = _fresh_state()
        for _ in range(2000):  # >> tau_eff
            v, s, ref = _step(v, ref, g_e=g_e)
            assert s.item() == 0.0, "should not spike subthreshold"
        assert v.item() == pytest.approx(v_inf, abs=1e-3)

    def test_tau_eff_governs_approach(self):
        """From v = E_L, under constant g_e, after one dt the fraction of the
        gap to v_inf closed is exactly (1 - exp(-dt / tau_eff))."""
        g_e = 0.02
        g_tot = M.g_L_E + g_e
        tau_eff = M.C_m_E / g_tot
        v_inf = (M.g_L_E * M.E_L + g_e * M.E_e) / g_tot
        v, ref = _fresh_state(dtype=torch.float64)
        v2, _, _ = _step(v, ref, g_e=g_e)
        expected = v_inf + (M.E_L - v_inf) * math.exp(-M.dt / tau_eff)
        assert v2.item() == pytest.approx(expected, abs=1e-10)

    def test_inhibition_pulls_v_below_E_L(self):
        """With only g_i active, v_inf < E_L — the exp-Euler step must reflect
        this (forward Euler does too, but we want to confirm the ZOH on g_i
        is wired up)."""
        g_i = 0.05
        v_inf = (M.g_L_E * M.E_L + g_i * M.E_i) / (M.g_L_E + g_i)
        assert v_inf < M.E_L
        v, ref = _fresh_state()
        for _ in range(2000):
            v, _, ref = _step(v, ref, g_e=0.0, g_i=g_i)
        assert v.item() == pytest.approx(v_inf, abs=1e-3)


class TestLimits:
    def test_dt_to_zero_matches_forward_euler(self):
        """In the dt → 0 limit, exp-Euler and forward Euler agree to O(dt^2).
        The *dv* predicted by each integrator has relative error
        ≈ (dt/tau_eff)/2; pick dt small enough that this is tiny."""
        g_e = 0.01
        v0 = -55.0
        dt_tiny = 0.001  # ms
        v_e, ref_e = _fresh_state(v0=v0, dtype=torch.float64)
        v_e, _, _ = _step(v_e, ref_e, g_e=g_e, dt=dt_tiny)

        # Forward-Euler reference reconstructed directly (old lif_step is
        # hard-wired to module-level M.dt):
        expected_fwd = v0 + (dt_tiny / M.C_m_E) * (
            -M.g_L_E * (v0 - M.E_L) + g_e * (M.E_e - v0)
        )

        dv_exp = v_e.item() - v0
        dv_fwd = expected_fwd - v0
        rel = abs(dv_exp - dv_fwd) / abs(dv_fwd)
        # Leading error is (dt/tau_eff)/2 ≈ 3e-5 at dt=0.001, tau_eff≈17 ms
        assert rel < 1e-4, f"exp-Euler diverged from fwd-Euler at dt={dt_tiny}"
