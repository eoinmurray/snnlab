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

import pytest
import torch

from snnlab.sim import models as M
from snnlab.sim.models import coba_current, spike_biophysical

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


class TestZOHConductanceInvariance:
    def test_dt_invariant_under_held_conductance(self):
        """The ZOH exp-Euler property WITH drive: holding g_e constant, N steps
        at dt equal 1 step at N*dt exactly (constant-coefficient linear ODE has
        an exact solution that composes). Forward Euler lacks this."""
        g_e = 0.01  # subthreshold: v_inf ≈ -54 mV < V_th, so no reset intervenes
        v_fine, ref = _fresh_state(dtype=torch.float64)
        N, dt_fine = 8, 0.05
        for _ in range(N):
            v_fine, s, ref = _step(v_fine, ref, g_e=g_e, dt=dt_fine)
            assert s.item() == 0.0
        v_coarse, ref_c = _fresh_state(dtype=torch.float64)
        v_coarse, _, _ = _step(v_coarse, ref_c, g_e=g_e, dt=N * dt_fine)
        assert v_fine.item() == pytest.approx(v_coarse.item(), abs=1e-10)


class TestCobaCurrent:
    """The conductance-based synaptic current I = g_e(E_e - v) [+ g_i(E_i - v)],
    the driving term the exp-Euler step integrates in closed form."""

    def test_excitatory_only_closed_form(self):
        """With no inhibition, I = g_e * (E_e - v)."""
        g_e = torch.tensor([[0.03]])
        v = torch.tensor([[-60.0]])
        expected = 0.03 * (M.E_e - (-60.0))
        assert coba_current(g_e, v).item() == pytest.approx(expected, abs=1e-6)

    def test_excitatory_and_inhibitory_sum(self):
        """Both conductances present: I = g_e(E_e - v) + g_i(E_i - v)."""
        g_e, g_i = 0.03, 0.05
        v = -55.0
        expected = g_e * (M.E_e - v) + g_i * (M.E_i - v)
        got = coba_current(
            torch.tensor([[g_e]]), torch.tensor([[v]]), torch.tensor([[g_i]])
        )
        assert got.item() == pytest.approx(expected, abs=1e-6)

    def test_current_vanishes_at_reversal_potentials(self):
        """At v = E_e the excitatory drive is zero; adding v = E_i zeroes the
        inhibitory drive too, so the net current is zero."""
        g_e = torch.tensor([[0.04]])
        g_i = torch.tensor([[0.06]])
        # Excitatory term alone vanishes at v = E_e.
        assert coba_current(g_e, torch.full_like(g_e, M.E_e)).item() == pytest.approx(
            0.0, abs=1e-6
        )
        # Both terms vanish only where each reversal is met; check the inhibitory
        # term is zero at v = E_i with g_e = 0.
        i_at_ei = coba_current(torch.zeros_like(g_i), torch.full_like(g_i, M.E_i), g_i)
        assert i_at_ei.item() == pytest.approx(0.0, abs=1e-6)

    def test_signs_are_depolarising_and_hyperpolarising(self):
        """Below E_e excitation is an inward (positive, depolarising) current;
        above E_i inhibition is an outward (negative, hyperpolarising) one."""
        v = torch.tensor([[M.E_L]])  # -65, between E_i (-80) and E_e (0)
        assert coba_current(torch.tensor([[0.05]]), v).item() > 0.0
        i_inhib_only = coba_current(torch.zeros_like(v), v, torch.tensor([[0.05]]))
        assert i_inhib_only.item() < 0.0

    def test_batched_matches_per_element(self):
        """Vectorised over a (B, N) tensor with per-element g_e, g_i, v."""
        g_e = torch.tensor([[0.01, 0.02, 0.03]])
        g_i = torch.tensor([[0.00, 0.04, 0.05]])
        v = torch.tensor([[-70.0, -60.0, -50.0]])
        got = coba_current(g_e, v, g_i)
        expected = g_e * (M.E_e - v) + g_i * (M.E_i - v)
        torch.testing.assert_close(got, expected)
