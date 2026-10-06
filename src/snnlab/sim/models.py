"""Numerical neuron, synapse and spike-generation primitives.

Graph execution supplies its timestep and biophysics explicitly.
"""

from __future__ import annotations

import torch

from snnlab.sim.timing import refractory_steps

dt: float = 0.25
tau_m_E = 20.0
tau_m_ratio = 4.0
C_m_E = 1.0
_CM_RATIO = 2.0
g_L_E = C_m_E / tau_m_E
tau_m_I = tau_m_E / tau_m_ratio
C_m_I = C_m_E / _CM_RATIO
g_L_I = C_m_I / tau_m_I
E_L = -65.0
E_e = 0.0
E_i = -80.0
V_th = -50.0
V_reset = -65.0
V_floor = -200.0
ref_ms_E = 3.0
_REF_RATIO = 2.0
ref_ms_I = ref_ms_E / _REF_RATIO
SURROGATE_SLOPE = 5.0
V_GRAD_DAMPEN = 80.0
COBA_INTEGRATOR = "expeuler"
ref_steps_E = refractory_steps(ref_ms_E, dt)
ref_steps_I = refractory_steps(ref_ms_I, dt)


def fast_sigmoid_spike(u, slope):
    """Fast-sigmoid surrogate spike.

    Forward: Heaviside(u). Backward gradient: slope / (1 + slope·|u|)^2 —
    equivalent to the snntorch FastSigmoid surrogate that the library path
    uses, so slope=1 is a pure update-rule comparison against snntorch-library,
    not a surrogate comparison.

    Implementation is a detach-style straight-through estimator: the forward
    value is the hard step, but the gradient flows through the smooth proxy
    p(u) = slope·u / (1 + slope·|u|), whose derivative is exactly the
    fast-sigmoid kernel. Pure tensor ops — torch.compile-friendly with no
    custom autograd.Function graph break.
    """
    hard = (u >= 0).float()
    proxy = slope * u / (1.0 + slope * u.abs())
    # Parenthesise so (proxy - proxy.detach()) collapses to bitwise zero
    # in the forward pass; otherwise (hard + proxy) then - proxy.detach()
    # loses fp32 precision when |proxy| is close to 1 (value drifts to
    # 0.9999994), which poisons downstream int(s.item()) spike counts.
    return hard.detach() + (proxy - proxy.detach())


def spike_biophysical(v, threshold_offset=0.0):
    # mV-scale membrane: slope=1 keeps gradient support at the ~mV width of
    # typical threshold crossings. `threshold_offset` shifts the effective
    # threshold up — used by ALIF where each neuron's threshold rises with
    # its own recent firing.
    return fast_sigmoid_spike(v - V_th - threshold_offset, SURROGATE_SLOPE)


def _scale_grad(x, scale):
    """Return x unchanged in forward, but multiply gradient by scale in backward."""
    return x * scale + x.detach() * (1.0 - scale)


def exp_synapse(g, spikes, W, decay):
    """Exponential synapse: decay, then add the undecayed spike kick.

    Canonical exponential synapse — a presynaptic spike makes g jump by its
    full weight W (its peak conductance), then decays as exp(-dt/tau). So W is
    the per-spike conductance increment, independent of dt. (Decay-then-add;
    cf. the older add-then-decay form, which scaled every kick by one extra
    factor of `decay`.)
    """
    return g * decay + spikes @ W


def lif_step(
    v,
    I_total,
    ref,
    C_m,
    g_L,
    ref_steps,
    spike_fn,
    V_floor=V_floor,
    V_max=None,
    v_grad_dampen=1.0,
):
    """One LIF timestep: voltage update, spike decision, then reset.
    Returns (v, s, ref)."""
    dv = (dt / C_m) * (-g_L * (v - E_L) + I_total)
    if v_grad_dampen != 1.0:
        dv = _scale_grad(dv, 1.0 / v_grad_dampen)
    v = v + dv
    v = v.clamp(min=V_floor) if V_max is None else v.clamp(min=V_floor, max=V_max)
    ref = (ref - 1).clamp(min=0)
    can_spike = ref == 0
    s = spike_fn(v) * can_spike.float()
    spiked_or_ref = s.bool() | (~can_spike)
    v = torch.where(spiked_or_ref, torch.full_like(v, V_reset), v)
    ref = torch.where(s.bool(), torch.full_like(ref, ref_steps), ref)
    return v, s, ref


def lif_step_expeuler(
    v,
    ref,
    g_e,
    g_i,
    C_m,
    g_L,
    ref_steps,
    spike_fn,
    v_grad_dampen=1.0,
    dt_override=None,
    V_floor=V_floor,
    V_max=None,
    threshold_offset=None,
    v_noise_std=0.0,
):
    """COBA LIF step under exponential Euler with a zero-order hold on g_e, g_i.

    Closed-form integration of
        C_m dv/dt = -g_L (v - E_L) - g_e (v - E_e) - g_i (v - E_i)
    over one step of length `dt`, holding g_e and g_i constant. Yields
        g_tot   = g_L + g_e + g_i
        tau_eff = C_m / g_tot
        v_inf   = (g_L*E_L + g_e*E_e + g_i*E_i) / g_tot
        v_{t+1} = v_inf + (v_t - v_inf) * exp(-dt / tau_eff)
    which is dt-invariant under N-vs-1 step in the passive case, unlike the
    forward-Euler `lif_step` above. Returns (v, s, ref).

    The kwarg is `dt_override` (not `dt`) so the module-level `dt` is
    accessible without `globals()['dt']`, which is a Dynamo graph-break.
    """
    dt_step = dt if dt_override is None else dt_override
    if g_i is None:
        g_sum = g_e
        g_E_drive = g_e * E_e
    else:
        g_sum = g_e + g_i
        g_E_drive = g_e * E_e + g_i * E_i
    g_tot = g_L + g_sum
    v_inf = (g_L * E_L + g_E_drive) / g_tot
    decay = torch.exp(-dt_step / (C_m / g_tot))
    dv = (v_inf - v) * (1.0 - decay)
    if v_grad_dampen != 1.0:
        dv = _scale_grad(dv, 1.0 / v_grad_dampen)
    v = v + dv
    if v_noise_std > 0.0:
        # Diffusive membrane noise: zero-mean Wiener increment on v, injected
        # before the spike decision so it jitters threshold-crossing *timing*
        # (the quantity a gamma clock can resynchronise). Scaled by
        # sqrt(2*dt/tau_leak) so the stationary subthreshold std ≈ v_noise_std
        # (mV) in the passive limit, independent of dt — unlike the old
        # per-step g_E noise whose power scaled with the step count.
        tau_leak = C_m / g_L
        v = v + v_noise_std * (2.0 * dt_step / tau_leak) ** 0.5 * torch.randn_like(v)
    v = v.clamp(min=V_floor) if V_max is None else v.clamp(min=V_floor, max=V_max)
    ref = (ref - 1).clamp(min=0)
    can_spike = ref == 0
    if threshold_offset is None:
        s = spike_fn(v) * can_spike.float()
    else:
        s = spike_fn(v, threshold_offset) * can_spike.float()
    spiked_or_ref = s.bool() | (~can_spike)
    v = torch.where(spiked_or_ref, torch.full_like(v, V_reset), v)
    ref = torch.where(s.bool(), torch.full_like(ref, ref_steps), ref)
    return v, s, ref


def coba_current(g_e, v, g_i=None):
    """COBA synaptic current: g_e*(E_e - v) [+ g_i*(E_i - v)]."""
    I = g_e * (E_e - v)
    if g_i is not None:
        I = I + g_i * (E_i - v)
    return I


def poisson_spikes(rate_hz, shape, dt, generator, device=None):
    """Bernoulli spike tensor at per-step probability rate_hz * dt / 1000.

    Each entry fires independently with probability (Hz × ms / 1000) per step.
    Generation runs on the generator's device (usually CPU); pass `device` to
    move the result. Single source of truth for the uniform-Poisson drive/input
    streams used by graph-based applications.
    """
    p = rate_hz * dt / 1000.0
    spk = (torch.rand(*shape, generator=generator) < p).to(torch.float32)
    return spk.to(device) if device is not None else spk


def init_lif_state(B, N, device, randomize=False, ref_mean=0.0, ref_std=0.0):
    """Initialise (v, ref) for a LIF population.
    If randomize=True, scatter initial voltages uniformly between E_L and V_th
    so neurons start at different phases (Börgers-style asynchronous init).
    ref_mean/ref_std: if nonzero, sample initial refractory from N(mean,std)
    clamped to [0, inf) so neurons come out of refractory at staggered times.
    """
    if randomize:
        v = E_L + (V_th - E_L) * torch.rand(B, N, device=device)
    else:
        v = torch.full((B, N), E_L, device=device)
    if ref_std > 0:
        ref = (
            (torch.randn(B, N, device=device) * ref_std + ref_mean).clamp(min=0).long()
        )
    else:
        ref = torch.zeros(B, N, device=device, dtype=torch.long)
    return v, ref


def init_conductance(B, N, device):
    """Initialise a conductance variable to zero."""
    return torch.zeros(B, N, device=device)


def e_step_coba(
    v,
    ref,
    g_e,
    g_i=None,
    ref_steps=None,
    threshold_offset=None,
    v_noise_std=0.0,
    C_m=None,
    g_L=None,
):
    """One E-neuron LIF step with COBA driving force."""
    if ref_steps is None:
        ref_steps = refractory_steps(ref_ms_E, dt, policy="nearest")
    C_m = C_m_E if C_m is None else C_m
    g_L = g_L_E if g_L is None else g_L
    if COBA_INTEGRATOR == "expeuler":
        return lif_step_expeuler(
            v,
            ref,
            g_e,
            g_i,
            C_m,
            g_L,
            ref_steps,
            spike_biophysical,
            v_grad_dampen=V_GRAD_DAMPEN,
            threshold_offset=threshold_offset,
            v_noise_std=v_noise_std,
        )
    return lif_step(
        v,
        coba_current(g_e, v, g_i),
        ref,
        C_m,
        g_L,
        ref_steps,
        spike_biophysical,
        v_grad_dampen=V_GRAD_DAMPEN,
    )


def i_step_coba(
    v,
    ref,
    g_e,
    g_i=None,
    threshold_offset=None,
    v_noise_std=0.0,
    C_m=None,
    g_L=None,
    ref_steps=None,
):
    """One I-neuron LIF step with COBA driving force.

    ``g_i`` is the I→I inhibitory conductance on the I cell, used for
    Brunel/Vreeswijk-style balanced-network experiments where I-cells have
    recurrent self-inhibition. Default ``None`` preserves the canonical PING
    architecture (no I→I)."""
    if ref_steps is None:
        ref_steps = refractory_steps(ref_ms_I, dt, policy="nearest")
    C_m = C_m_I if C_m is None else C_m
    g_L = g_L_I if g_L is None else g_L
    if COBA_INTEGRATOR == "expeuler":
        return lif_step_expeuler(
            v,
            ref,
            g_e,
            g_i,
            C_m,
            g_L,
            ref_steps,
            spike_biophysical,
            v_grad_dampen=V_GRAD_DAMPEN,
            threshold_offset=threshold_offset,
            v_noise_std=v_noise_std,
        )
    return lif_step(
        v,
        coba_current(g_e, v, g_i),
        ref,
        C_m,
        g_L,
        ref_steps,
        spike_biophysical,
        v_grad_dampen=V_GRAD_DAMPEN,
    )
