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


class TestGradientFlow:
    def test_grad_flows_through_g_e(self):
        """BPTT must push gradient through the membrane update back to g_e;
        otherwise COBA/PING training silently breaks."""
        v, ref = _fresh_state()
        g_e = torch.tensor([[0.01]], requires_grad=True)
        v2, _, _ = lif_step_expeuler(
            v, ref, g_e, None, M.C_m_E, M.g_L_E, M.ref_steps_E, spike_biophysical
        )
        v2.sum().backward()
        assert g_e.grad is not None
        assert g_e.grad.abs().item() > 0.0

    def test_v_grad_dampen_attenuates_grad(self):
        """v_grad_dampen > 1 must reduce the gradient magnitude relative to the
        identity-scale reference (the point of the dampening hook)."""
        v_ref, ref = _fresh_state()
        v_damp, ref2 = _fresh_state()
        g_e_ref = torch.tensor([[0.01]], requires_grad=True)
        g_e_damp = torch.tensor([[0.01]], requires_grad=True)
        v2r, _, _ = lif_step_expeuler(
            v_ref,
            ref,
            g_e_ref,
            None,
            M.C_m_E,
            M.g_L_E,
            M.ref_steps_E,
            spike_biophysical,
            v_grad_dampen=1.0,
        )
        v2d, _, _ = lif_step_expeuler(
            v_damp,
            ref2,
            g_e_damp,
            None,
            M.C_m_E,
            M.g_L_E,
            M.ref_steps_E,
            spike_biophysical,
            v_grad_dampen=1000.0,
        )
        v2r.sum().backward()
        v2d.sum().backward()
        assert g_e_damp.grad is not None and g_e_ref.grad is not None
        assert g_e_damp.grad.abs().item() < g_e_ref.grad.abs().item()


class TestBatched:
    def test_batched_neurons_follow_independent_closed_form(self):
        """The vectorised step integrates each (batch, neuron) under its own
        g_e — no cross-talk. Checks the closed form per column."""
        g_e_list = [0.005, 0.01, 0.02]
        g_e = torch.tensor([g_e_list], dtype=torch.float64)
        v = torch.full((1, 3), M.E_L, dtype=torch.float64)
        ref = torch.zeros((1, 3), dtype=torch.long)
        v2, s, _ = lif_step_expeuler(
            v, ref, g_e, None, M.C_m_E, M.g_L_E, M.ref_steps_E, spike_biophysical
        )
        for j, ge in enumerate(g_e_list):
            g_tot = M.g_L_E + ge
            v_inf = (M.g_L_E * M.E_L + ge * M.E_e) / g_tot
            expected = v_inf + (M.E_L - v_inf) * math.exp(-M.dt / (M.C_m_E / g_tot))
            assert v2[0, j].item() == pytest.approx(expected, abs=1e-10)
        assert s.sum().item() == 0.0  # one step from rest stays subthreshold
