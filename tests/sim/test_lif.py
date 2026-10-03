"""LIF integrator + exp_synapse primitives: the math underwriting every run."""

import math

import pytest
import torch

from snnlab.sim import models as M
from snnlab.sim.models import exp_synapse, lif_step, spike_biophysical


def _fresh_state(B=1, N=1):
    v = torch.full((B, N), M.E_L)
    ref = torch.zeros((B, N), dtype=torch.long)
    return v, ref


class TestLifStep:
    def test_resting_with_zero_input_is_stationary(self):
        """At v=E_L with I=0, leak term is zero → voltage doesn't move, no spike."""
        v, ref = _fresh_state()
        v2, s, ref2 = lif_step(
            v,
            torch.zeros_like(v),
            ref,
            M.C_m_E,
            M.g_L_E,
            M.ref_steps_E,
            spike_biophysical,
        )
        assert torch.allclose(v2, v)
        assert s.item() == 0.0
        assert ref2.item() == 0

    def test_subthreshold_step_voltage_increase_matches_euler(self):
        """For a single Euler step dv = (dt/C) * (-g_L*(v-E_L) + I)."""
        v, ref = _fresh_state()
        I = torch.tensor([[0.5]])  # nA
        v2, s, _ = lif_step(
            v, I, ref, M.C_m_E, M.g_L_E, M.ref_steps_E, spike_biophysical
        )
        expected_dv = (M.dt / M.C_m_E) * (-M.g_L_E * (M.E_L - M.E_L) + 0.5)
        assert v2.item() == pytest.approx(M.E_L + expected_dv, abs=1e-5)
        assert s.item() == 0.0

    def test_constant_current_fires_at_analytic_interval(self):
        """For constant input I > I_rheobase, LIF fires periodically with ISI
        T_spike = tau_m * ln(I / (I - g_L*(V_th - E_L))).

        Drive hard enough that the ISI is ~8 ms; verify spike count over
        100 ms matches 100 / T_spike within ±1 spike.
        """
        I_rheo = M.g_L_E * (M.V_th - M.E_L)  # threshold current
        I_val = 3.0 * I_rheo  # well above rheobase
        tau_m = M.C_m_E / M.g_L_E
        T_spike_ms = tau_m * math.log(I_val / (I_val - I_rheo))

        v, ref = _fresh_state()
        I = torch.full_like(v, I_val)
        steps = int(100.0 / M.dt)
        spikes = 0
        for _ in range(steps):
            v, s, ref = lif_step(
                v, I, ref, M.C_m_E, M.g_L_E, M.ref_steps_E, spike_biophysical
            )
            spikes += int(s.item())

        # With refractory, ISI is effectively T_spike + ref_ms_E
        expected_isi = T_spike_ms + M.ref_ms_E
        expected_count = 100.0 / expected_isi
        assert abs(spikes - expected_count) <= 1, (
            f"got {spikes} spikes, expected ≈ {expected_count:.1f}"
        )

    def test_refractory_holds_v_at_reset(self):
        """After a spike, v is pinned to V_reset for ref_steps regardless of input."""
        # Pre-arm: voltage just under threshold, huge input → spike next step.
        v = torch.full((1, 1), M.V_th - 0.01)
        ref = torch.zeros((1, 1), dtype=torch.long)
        big_I = torch.tensor([[1000.0]])
        v, s, ref = lif_step(
            v, big_I, ref, M.C_m_E, M.g_L_E, M.ref_steps_E, spike_biophysical
        )
        assert s.item() == 1.0
        assert v.item() == pytest.approx(M.V_reset)
        assert ref.item() == M.ref_steps_E

        # For the next ref_steps - 1 steps, massive input must not produce
        # spikes and v must remain at V_reset.
        for _ in range(M.ref_steps_E - 1):
            v, s, ref = lif_step(
                v, big_I, ref, M.C_m_E, M.g_L_E, M.ref_steps_E, spike_biophysical
            )
            assert s.item() == 0.0, "neuron spiked during refractory period"
            assert v.item() == pytest.approx(M.V_reset)


class TestExpSynapse:
    def test_pure_decay_with_no_spike(self):
        """No spikes in → g' = g * decay."""
        g = torch.tensor([[1.0, 2.0, 3.0]])
        s = torch.zeros((1, 4))
        W = torch.zeros((4, 3))
        decay = 0.9
        g_next = exp_synapse(g, s, W, decay)
        torch.testing.assert_close(g_next, g * decay)

    def test_impulse_then_decay(self):
        """Decay-then-add: an impulse spike through identity W lands at its
        full weight (1.0), then subsequent steps decay as decay, decay^2, ..."""
        g = torch.zeros((1, 3))
        s = torch.ones((1, 3))  # single impulse, broadcast via I
        W = torch.eye(3)
        decay = 0.8

        g1 = exp_synapse(g, s, W, decay)
        torch.testing.assert_close(g1, torch.full((1, 3), 1.0))

        # Subsequent steps: zero spikes, only decay
        s_zero = torch.zeros_like(s)
        g2 = exp_synapse(g1, s_zero, W, decay)
        g3 = exp_synapse(g2, s_zero, W, decay)
        torch.testing.assert_close(g2, torch.full((1, 3), decay))
        torch.testing.assert_close(g3, torch.full((1, 3), decay**2))
