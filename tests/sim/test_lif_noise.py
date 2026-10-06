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

lif_step_expeuler = M.lif_step_expeuler


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


class TestMembraneNoise:
    def _noiseless_passive_step(self, v0):
        return M.E_L + (v0 - M.E_L) * math.exp(-M.dt * M.g_L_E / M.C_m_E)

    def test_zero_noise_is_deterministic_closed_form(self):
        """v_noise_std=0 leaves the deterministic exp-Euler step untouched."""
        v0 = -55.0
        v, ref = _fresh_state(v0=v0, dtype=torch.float64)
        v2, _, _ = lif_step_expeuler(
            v,
            ref,
            torch.zeros_like(v),
            None,
            M.C_m_E,
            M.g_L_E,
            M.ref_steps_E,
            spike_biophysical,
            v_noise_std=0.0,
        )
        assert v2.item() == pytest.approx(self._noiseless_passive_step(v0), abs=1e-9)

    def test_noise_is_reproducible_under_seed(self):
        """Same seed → identical membrane noise (the Wiener increment is drawn
        from the global torch RNG)."""

        def once():
            torch.manual_seed(1234)
            v, ref = _fresh_state(v0=-60.0, N=64)
            v2, _, _ = lif_step_expeuler(
                v,
                ref,
                torch.zeros_like(v),
                None,
                M.C_m_E,
                M.g_L_E,
                M.ref_steps_E,
                spike_biophysical,
                v_noise_std=1.0,
            )
            return v2

        torch.testing.assert_close(once(), once())

    def test_noise_perturbs_but_is_zero_mean(self):
        """Membrane noise deflects individual neurons off the deterministic
        value, but its ensemble mean is ≈ 0 (zero-mean Wiener increment)."""
        torch.manual_seed(7)
        n = 40000
        v, ref = _fresh_state(v0=-60.0, N=n, dtype=torch.float64)
        v2, _, _ = lif_step_expeuler(
            v,
            ref,
            torch.zeros_like(v),
            None,
            M.C_m_E,
            M.g_L_E,
            M.ref_steps_E,
            spike_biophysical,
            v_noise_std=1.0,
        )
        dev = v2 - self._noiseless_passive_step(-60.0)
        assert dev.abs().mean().item() > 0.0  # noise actually applied
        assert abs(dev.mean().item()) < 0.05  # but zero-mean

    def test_stationary_std_matches_v_noise_std_and_is_dt_invariant(self):
        """The docstring's scaling claim: the sqrt(2 dt / tau_leak) factor makes
        the stationary subthreshold std ≈ v_noise_std, roughly independent of dt."""
        target = 2.0

        def stationary_std(dt_step, n=4000, steps=1500):
            torch.manual_seed(42)
            v = torch.full((1, n), M.E_L, dtype=torch.float64)
            ref = torch.zeros((1, n), dtype=torch.long)
            g_e = torch.zeros_like(v)
            for _ in range(steps):
                v, _, ref = lif_step_expeuler(
                    v,
                    ref,
                    g_e,
                    None,
                    M.C_m_E,
                    M.g_L_E,
                    M.ref_steps_E,
                    spike_biophysical,
                    dt_override=dt_step,
                    v_noise_std=target,
                )
            return v.std().item()

        assert stationary_std(0.1) == pytest.approx(target, rel=0.1)
        assert stationary_std(0.5) == pytest.approx(target, rel=0.15)
