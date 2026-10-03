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


class TestExpEulerContract:
    """Integrator invariants not pinned above: the None-vs-zero g_i fast path,
    the hard voltage floor, dt_override's independence from the module dt, the
    full refractory arm→release cycle, and unconditional (no-overshoot) stability.
    """

    def test_g_i_none_matches_explicit_zero_g_i(self):
        """The `g_i is None` fast path must be identical to passing a zero g_i
        tensor — same v, same spike, same refractory. Guards the branch in
        lif_step_expeuler that swaps g_sum / g_E_drive when g_i is absent."""
        v0 = -58.0
        va, ra = _fresh_state(v0=v0)
        vb, rb = _fresh_state(v0=v0)
        v_none, s_none, ref_none = _step(va, ra, g_e=0.03, g_i=None)
        v_zero, s_zero, ref_zero = _step(vb, rb, g_e=0.03, g_i=0.0)
        torch.testing.assert_close(v_none, v_zero)
        assert s_none.item() == s_zero.item()
        torch.testing.assert_close(ref_none, ref_zero)

    def test_v_floor_clamps_membrane(self):
        """Strong inhibition drives v_inf ≈ E_i in one step; the hard V_floor
        caps the membrane below that. Mirror of the V_max cap test."""
        v = torch.full((1, 1), M.E_L)
        ref = torch.zeros((1, 1), dtype=torch.long)
        g_e = torch.zeros((1, 1))
        g_i = torch.full((1, 1), 100.0)  # v_inf ≈ E_i = -80, decay ≈ 0
        floor = -70.0
        v2, s, _ = lif_step_expeuler(
            v,
            ref,
            g_e,
            g_i,
            M.C_m_E,
            M.g_L_E,
            M.ref_steps_E,
            spike_biophysical,
            V_floor=floor,
        )
        assert v2.item() == pytest.approx(floor)  # clamped up from ≈ -80
        assert s.item() == 0.0

    def test_dt_override_ignores_module_dt(self):
        """The step integrates over dt_override, never the module-global M.dt —
        the Dynamo-safe design that keeps dt out of a graph-break. A wrong M.dt
        must not change the result (conftest restores M.dt afterwards)."""
        v0 = -55.0
        dt_use = 0.3
        M.dt = 999.0  # deliberately wrong; must be ignored
        v, ref = _fresh_state(v0=v0, dtype=torch.float64)
        v2, _, _ = _step(v, ref, dt=dt_use)  # passive (g_e=g_i=0)
        expected = M.E_L + (v0 - M.E_L) * math.exp(-dt_use * M.g_L_E / M.C_m_E)
        assert v2.item() == pytest.approx(expected, abs=1e-10)

    def test_refractory_releases_after_exactly_ref_steps(self):
        """A spike arms the refractory counter to ref_steps; the neuron is
        locked for exactly that many steps, then free to fire again."""
        v = torch.full((1, 1), M.V_th - 0.01)
        ref = torch.zeros((1, 1), dtype=torch.long)
        v, s, ref = _step(v, ref, g_e=100.0)  # force a spike
        assert s.item() == 1.0
        assert ref.item() == M.ref_steps_E
        # Sit with no drive: counter ticks down, no re-fire, until it hits 1.
        for _ in range(M.ref_steps_E - 1):
            v, s, ref = _step(v, ref, g_e=0.0)
            assert s.item() == 0.0
        assert ref.item() == 1
        # Next step clears the lock (ref→0, can_spike) and strong drive fires again.
        v, s, ref = _step(v, ref, g_e=100.0)
        assert s.item() == 1.0
        assert ref.item() == M.ref_steps_E

    def test_no_overshoot_at_large_dt(self):
        """Unconditional stability: even at dt ≫ tau_eff the membrane approaches
        v_inf monotonically from below and never overshoots it — the property
        forward Euler loses (it would oscillate/diverge at this step size)."""
        g_e = 0.01  # subthreshold: v_inf below V_th, so no spikes interrupt
        g_tot = M.g_L_E + g_e
        v_inf = (M.g_L_E * M.E_L + g_e * M.E_e) / g_tot
        assert v_inf < M.V_th
        tau_eff = M.C_m_E / g_tot
        big_dt = 50.0 * tau_eff  # far past forward Euler's stability limit
        v, ref = _fresh_state(dtype=torch.float64)  # starts at E_L, below v_inf
        prev = v.item()
        for _ in range(5):
            v, s, ref = _step(v, ref, g_e=g_e, dt=big_dt)
            val = v.item()
            assert prev <= val <= v_inf + 1e-9, "overshoot past v_inf"
            assert s.item() == 0.0
            prev = val
        assert v.item() == pytest.approx(v_inf, abs=1e-3)
