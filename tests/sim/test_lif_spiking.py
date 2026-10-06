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


class TestRefractory:
    def test_refractory_pins_v_at_reset(self):
        """Behavioural: after a spike, v stays at V_reset for ref_steps
        regardless of incoming conductance."""
        v = torch.full((1, 1), M.V_th - 0.01)
        ref = torch.zeros((1, 1), dtype=torch.long)
        # Force spike via huge conductance
        v, s, ref = _step(v, ref, g_e=100.0)
        assert s.item() == 1.0
        assert v.item() == pytest.approx(M.V_reset)
        assert ref.item() == M.ref_steps_E
        for _ in range(M.ref_steps_E - 1):
            v, s, ref = _step(v, ref, g_e=100.0)
            assert s.item() == 0.0, "spiked during refractory period"
            assert v.item() == pytest.approx(M.V_reset)


class TestFiring:
    def test_constant_conductance_fires_periodically(self):
        """Under constant suprathreshold g_e, neuron fires repeatedly. We
        don't pin the ISI (it differs slightly from forward Euler) — only
        that spikes appear and recurs at a stable rate."""
        # Enough drive that v_inf exceeds V_th:
        g_e_rheo = M.g_L_E * (M.V_th - M.E_L) / (M.E_e - M.V_th)
        g_e = 3.0 * g_e_rheo
        v, ref = _fresh_state()
        spikes = 0
        steps = int(200.0 / M.dt)
        for _ in range(steps):
            v, s, ref = _step(v, ref, g_e=g_e)
            spikes += int(s.item())
        assert spikes >= 5, f"expected periodic firing, got {spikes} spikes"


class TestCombinedDrive:
    def test_v_inf_with_both_conductances_closed_form(self):
        """One step with BOTH g_e and g_i active must hit the three-term
        closed form: v_inf = (g_L E_L + g_e E_e + g_i E_i) / (g_L + g_e + g_i)."""
        g_e, g_i = 0.03, 0.02
        g_tot = M.g_L_E + g_e + g_i
        v_inf = (M.g_L_E * M.E_L + g_e * M.E_e + g_i * M.E_i) / g_tot
        tau_eff = M.C_m_E / g_tot
        v, ref = _fresh_state(dtype=torch.float64)
        v2, _, _ = _step(v, ref, g_e=g_e, g_i=g_i)
        expected = v_inf + (M.E_L - v_inf) * math.exp(-M.dt / tau_eff)
        assert v2.item() == pytest.approx(expected, abs=1e-10)

    def test_v_inf_sits_between_reversal_potentials(self):
        """With mixed E/I drive, v_inf is a conductance-weighted average of the
        three reversal potentials, so it lies within [E_i, E_e]."""
        g_e, g_i = 0.04, 0.06
        v, ref = _fresh_state(v0=-60.0, dtype=torch.float64)
        for _ in range(4000):  # >> tau_eff → converged
            v, s, ref = _step(v, ref, g_e=g_e, g_i=g_i)
            assert s.item() == 0.0
        g_tot = M.g_L_E + g_e + g_i
        v_inf = (M.g_L_E * M.E_L + g_e * M.E_e + g_i * M.E_i) / g_tot
        assert M.E_i <= v_inf <= M.E_e
        assert v.item() == pytest.approx(v_inf, abs=1e-3)


class TestThresholdOffset:
    def test_positive_offset_suppresses_spike(self):
        """threshold_offset raises the effective threshold (v - V_th - offset):
        a voltage that spikes at offset 0 must NOT spike under a large offset."""
        v0 = M.V_th + 1.0  # just above threshold
        v = torch.full((1, 1), v0)
        ref = torch.zeros((1, 1), dtype=torch.long)
        off0 = torch.zeros_like(v)
        _, s_on, _ = lif_step_expeuler(
            v.clone(),
            ref.clone(),
            torch.zeros_like(v),
            None,
            M.C_m_E,
            M.g_L_E,
            M.ref_steps_E,
            spike_biophysical,
            threshold_offset=off0,
        )
        assert s_on.item() == 1.0
        off_hi = torch.full_like(v, 5.0)  # raise threshold by 5 mV
        _, s_off, _ = lif_step_expeuler(
            v.clone(),
            ref.clone(),
            torch.zeros_like(v),
            None,
            M.C_m_E,
            M.g_L_E,
            M.ref_steps_E,
            spike_biophysical,
            threshold_offset=off_hi,
        )
        assert s_off.item() == 0.0


class TestVoltageClamps:
    def test_v_max_caps_the_membrane(self):
        """V_max hard-caps the post-update voltage. With strong depolarising
        drive but V_max below threshold, v pins at V_max and never spikes."""
        V_max = M.V_th - 5.0  # -55 mV, below threshold
        v, ref = _fresh_state()
        g_e = torch.full_like(v, 50.0)  # v_inf ≈ E_e = 0, drives hard up
        for _ in range(50):
            v, s, ref = lif_step_expeuler(
                v,
                ref,
                g_e,
                None,
                M.C_m_E,
                M.g_L_E,
                M.ref_steps_E,
                spike_biophysical,
                V_max=V_max,
            )
            assert s.item() == 0.0
            assert v.item() <= V_max + 1e-6
        assert v.item() == pytest.approx(V_max, abs=1e-4)  # pinned at the cap
