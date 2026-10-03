"""SNN model definitions and layer primitives.

All constants are hardcoded defaults. Override via module-level assignment
or by passing arguments to model constructors.

Models: COBANet.
Layer primitives: exp_synapse, lif_step.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snnlab.sim.timing import duration_steps, refractory_metadata, refractory_steps

# ── Simulation ────────────────────────────────────────────────────────────
dt: float = 0.25  # ms — integration timestep
T_ms: float = 1000.0  # ms — total simulation time per sample
T_steps: int = duration_steps(T_ms, dt)

# ── Biophysics ────────────────────────────────────────────────────────────
tau_m_E = 20.0  # ms — excitatory membrane time constant
tau_m_ratio = 4.0  # tau_m_E / tau_m_I (Börgers: 20ms / 5ms)
C_m_E = 1.0  # nF — excitatory capacitance (fixed reference)
_CM_RATIO = 2.0  # C_m_E / C_m_I (fixed)
g_L_E = C_m_E / tau_m_E  # 0.05 uS
tau_m_I = tau_m_E / tau_m_ratio  # 5.0 ms
C_m_I = C_m_E / _CM_RATIO  # 0.5 nF
g_L_I = C_m_I / tau_m_I  # 0.1 uS
E_L = -65.0  # mV — leak / resting potential
E_e = 0.0  # mV — excitatory (AMPA) reversal
E_i = -80.0  # mV — inhibitory (GABA) reversal
V_th = -50.0  # mV — spike threshold
V_reset = -65.0  # mV — post-spike reset voltage
V_floor = -200.0  # mV — hard lower clamp
# Upper magnitude bound on conductances under the forward-pass state clamp
# (self.state_clamp). Well above any physiological operating value (init E→I is
# ~1 µS), so it only bites on a genuine runaway. The load-bearing part of the
# clamp is the LOWER bound at 0: with signed weights a conductance can go
# negative and drive g_tot = g_L + g_e + g_i to ≤ 0, which makes the exp-Euler
# v_inf = (…)/g_tot diverge to NaN. Flooring conductances at 0 (physical — a
# conductance cannot be negative) keeps g_tot ≥ g_L > 0 and bounds v_inf between
# the reversal potentials, while the weights stay signed.
G_CLAMP_MAX = 100.0  # µS
ref_ms_E = 3.0  # ms — excitatory refractory period
_REF_RATIO = 2.0  # ref_ms_E / ref_ms_I (Börgers)
ref_ms_I = ref_ms_E / _REF_RATIO  # 1.5 ms
tau_ampa = 2.0  # ms — AMPA decay
# GABA decay. Library default is the literature biophysical value. The collection
# TRAINS at 6 ms (the canonical operating point that centres the loop in gamma,
# f_γ ≈ 44 Hz) by passing --tau-gaba explicitly; cells carry tau_gaba_ms in their
# config and inherit it on --load-config, so this default only sets fresh runs
# that pass neither flag.
tau_gaba = 9.0  # ms — GABA decay (Börgers: 9 ms; Buzsaki & Wang: 8-12 ms)
TRAINABLE_TAU_M_E_BOUNDS_MS = (5.0, 50.0)
TRAINABLE_TAU_M_I_BOUNDS_MS = (2.0, 20.0)
ADAPT_TAU_BOUNDS_MS = (50.0, 500.0)
ADAPT_STRENGTH_MAX_MV = 20.0

# ── Input encoding ────────────────────────────────────────────────────────
max_rate_hz = (
    25.0  # Hz — max Poisson rate for fully-on pixel (sensory-input scale, LGN-ish)
)

# ── snnTorch ──────────────────────────────────────────────────────────────
tau_snn = 10.0  # ms — membrane time constant
# Output-LIF time constant for the spiking readouts. Smaller than
# tau_snn means faster leak: the output membrane decays before saturating
# under high-rate hidden drive, which is what was breaking the spike-count
# readout for snnTorch-family models at coarse dt where hidden rates run
# 60–1000 Hz. Override per-run via --readout-tau-out.
tau_out_ms = 2.0  # ms
thr_snn = 1.0  # spike threshold
CUMULATIVE_READOUT_TAU_BOUNDS_MS = (5.0, 25.0)
CUMULATIVE_READOUT_REDUCTION = "sum_softmax_potential_per_timestep"
CUMULATIVE_READOUT_REFERENCE = (
    "https://github.com/idiap/sparch/blob/"
    "f8756254ab8d0e3337eb69542c684b922d6b6cbd/"
    "sparch/models/snns.py#L730-L825"
)

# ── Architecture ──────────────────────────────────────────────────────────
N_IN: int = (
    64  # input neurons (module default; overridden per-run by the dataset loader)
)
N_HID: int = 64  # hidden excitatory neurons (last layer size for compat)
N_INH: int = 16  # inhibitory neurons (PING only, per E-I layer)
N_OUT: int = 10  # output neurons (one per digit class)
HIDDEN_SIZES: list[int] = [64]  # hidden layer sizes (N_HID is always last entry)

# ── Weight init ───────────────────────────────────────────────────────────
# Parent Gaussian parameters for the nominal summed coupling. init_weight
# lower-clamps the draw, applies optional initialization zeroing, then divides
# by fan-in. These are not per-edge moments.
W_FF_MEAN = 5.1
W_FF_STD = 3.8
W_IN_MEAN = W_FF_MEAN  # alias
W_IN_STD = W_FF_STD  # alias
W_HID_MEAN = W_FF_MEAN  # alias
W_HID_STD = W_FF_STD  # alias
W_EE_MEAN = 0.0  # uS — E→E recurrent init mean (0 per Börgers: PING needs no E→E)
W_EE_STD = 0.0  # uS — E→E recurrent init std (0 per Börgers: PING needs no E→E)
W_EI_MEAN = 1.0  # uS — E→I init mean (just suprathreshold, Börgers)
W_EI_STD = 0.5  # uS — E→I init std (pre-fan-in)
W_IE_MEAN = 3.0  # uS — I→E init mean (2-3× E→I, Viriyopase et al.)
W_IE_STD = 1.5  # uS — I→E init std (pre-fan-in)
W_II_MEAN = 0.0  # uS — I→I recurrent init mean (0 by default: PING needs no I→I,
W_II_STD = 0.0  #  enable explicitly for Brunel/Vreeswijk balanced-network experiments)

# ── Training ──────────────────────────────────────────────────────────────
BATCH_SIZE = 64
# Surrogate-gradient steepness (fast-sigmoid slope). 5 is the stable
# end-to-end value at pinglab's current BPTT depth / clip / optimizer
# config — slope=10 (Neftci canonical) and slope=40 (Cramer) both blow
# up to gn 1e10+ even with the global-norm gradient clip at 100.
SURROGATE_SLOPE = 5.0
V_GRAD_DAMPEN = 80.0


# Legacy constants for callers that explicitly request the default-grid counters.
# Production updates derive counters from each model's physical durations and
# actual timestep; the utility functions also resolve their defaults at runtime.
_dt_default = 0.25
ref_steps_E = refractory_steps(ref_ms_E, _dt_default)
ref_steps_I = refractory_steps(ref_ms_I, _dt_default)


def _env_no_compile() -> bool:
    """PINGLAB_NO_COMPILE=1 disables torch.compile in the model forward path.
    Set this for ablation runs that want the eager baseline (e.g. comparing
    compile vs eager wall time on the same hardware)."""
    import os

    return os.environ.get("PINGLAB_NO_COMPILE", "") == "1"


# Dynamo compile-cache size. Default 8 is exhausted by:
#   - train vs eval (requires_grad differs on state tensors per forward call)
#   - last partial batch in an epoch (smaller batch dim than 256)
#   - any incidental shape variation in compiled functions
# Bump to 32 so the cache holds a generous superset of distinct trace
# variants without falling back to eager. Unlike force_parameter_static_shapes
# (which we tried in 85cd304 and reverted: dynamic-shape codegen produced
# slower kernels on CUDA), this knob only affects how many specialised
# variants we keep around — each one is still a fully-static, shape-
# specialised compiled graph.
import torch._dynamo  # noqa: E402

torch._dynamo.config.recompile_limit = 32  # ty: ignore[invalid-assignment]


# ── Surrogate gradient ───────────────────────────────────────────────────


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


def _bounded_logit(value: float, lo: float, hi: float) -> float:
    """Logit whose sigmoid maps to ``value`` inside the open interval [lo, hi]."""
    if not lo < value < hi:
        raise ValueError(f"initial value {value} must lie inside ({lo}, {hi})")
    y = (value - lo) / (hi - lo)
    y = min(max(y, 1e-6), 1.0 - 1e-6)
    return math.log(y / (1.0 - y))


def _bounded_from_logit(logit, lo: float, hi: float):
    """Map an unconstrained trainable tensor to a bounded positive parameter."""
    return lo + (hi - lo) * torch.sigmoid(logit)


# ── Layer primitives ─────────────────────────────────────────────────────


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
    The graph-break was forcing a per-call recompile cascade that defeated
    torch.compile on COBANet's CUDA path (recompile_limit hit silently).
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
    streams that cli/infer build (was copy-pasted per drive channel).
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


def _parse_weight_spec(w, default_dist, default_initial_zero_fraction):
    """Parse a weight spec tuple: (p1, p2), (p1, p2, dist), or (p1, p2, dist, initial_zero_fraction)."""
    if len(w) >= 4:
        return w[0], w[1], w[2], w[3]
    elif len(w) == 3:
        return w[0], w[1], w[2], default_initial_zero_fraction
    return w[0], w[1], default_dist, default_initial_zero_fraction


# Connectivity mode for the sparsifier in init_weight:
#   False (default) — per-entry Bernoulli: each entry zeroed independently
#     with probability `initial_zero_fraction`. Fan-in per post cell is binomial, so it
#     varies cell to cell.
#   True            — fixed fan-in (exact-K): every post cell (column) keeps
#     exactly K = round((1-initial_zero_fraction)·N_pre) random presynaptic inputs.
#     Removes the binomial fan-in variance — the Brunel/Vreeswijk convention.
# Set via the --exact-k-initialization CLI flag (M.EXACT_K_INITIALIZATION = True). Annotated bool
# (not the inferred Literal[False]) so entry points can flip it to True.
EXACT_K_INITIALIZATION: bool = False


def _weight_statistics(weight, *, clamp_zero_mask, explicit_zero_mask):
    """Return JSON-safe initialization statistics with zero sources separated."""
    flat = weight.detach().reshape(-1)
    active = ~explicit_zero_mask.reshape(-1)
    initially_nonzero = flat != 0

    def moments(values):
        if values.numel() == 0:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    column_sums = weight.detach().sum(dim=0) if weight.ndim == 2 else flat.sum()[None]
    return {
        "n_parameters": flat.numel(),
        "initialization_zero_count": int((flat == 0).sum()),
        "initialization_zero_fraction": float((flat == 0).float().mean()),
        "lower_clamp_zero_count": int((clamp_zero_mask.reshape(-1) & active).sum()),
        "lower_clamp_zero_fraction_of_unzeroed": float(
            (clamp_zero_mask.reshape(-1)[active]).float().mean()
        )
        if active.any()
        else 0.0,
        "explicit_zero_count": int(explicit_zero_mask.sum()),
        "explicit_zero_fraction": float(explicit_zero_mask.float().mean()),
        "all_entries": moments(flat),
        "initially_nonzero_entries": moments(flat[initially_nonzero]),
        "realized_column_sum": moments(column_sums),
    }


def _lower_clamped_normal_mean(mean, std):
    """E[max(0, X)] for X ~ Normal(mean, std**2)."""
    if std == 0:
        return max(0.0, float(mean))
    z = float(mean) / float(std)
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return float(std) * phi + float(mean) * Phi


def _lower_clamp_zero_probability(mean, std):
    if std == 0:
        return 1.0 if mean <= 0 else 0.0
    z = -float(mean) / float(std)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def init_weight(
    shape,
    dist="lower_clamped_normal",
    p1=0.0,
    p2=0.1,
    initial_zero_fraction=0.0,
    *,
    return_provenance=False,
):
    """Initialize a fan-in-normalized trainable matrix.

    ``initial_zero_fraction`` controls epoch-zero values only. Its zeros are
    ordinary trainable parameters, not a persistent connectivity mask.
    """
    if not 0.0 <= initial_zero_fraction < 1.0:
        raise ValueError("initial_zero_fraction must satisfy 0 <= fraction < 1")
    n_pre = shape[0]
    if dist == "signed_normal":
        w = torch.randn(*shape).mul_(p2).add_(p1)
    elif dist in ("normal", "lower_clamped_normal"):
        w = torch.randn(*shape).mul_(p2).add_(p1).clamp_(min=0)
    elif dist == "uniform":
        w = torch.rand(*shape).mul_(p2 - p1).add_(p1)
    elif dist == "constant":
        w = torch.full(shape, p1)
    elif dist == "zeros":
        w = torch.zeros(*shape)
    else:
        raise ValueError(f"Unknown dist: {dist!r}")
    clamp_zero_mask = w == 0
    explicit_zero_mask = torch.zeros(shape, dtype=torch.bool, device=w.device)
    if initial_zero_fraction > 0:
        if EXACT_K_INITIALIZATION and len(shape) == 2:
            # Fixed fan-in: each column keeps exactly K random rows.
            n_post = shape[1]
            k = max(1, int(round((1.0 - initial_zero_fraction) * n_pre)))
            mask = torch.zeros(n_pre, n_post)
            for j in range(n_post):
                idx = torch.randperm(n_pre)[:k]
                mask[idx, j] = 1.0
            explicit_zero_mask = mask == 0
            w = w * mask
            # Rescale by exact fan-in so per-column expected drive is
            # preserved (matches the Bernoulli path's 1/(1-initial_zero_fraction)).
            w = w * (n_pre / k)
        else:
            mask = torch.rand(*shape) > initial_zero_fraction
            explicit_zero_mask = ~mask
            w = w * mask.float()
            w = w / (1.0 - initial_zero_fraction)
    w = w / n_pre
    if not return_provenance:
        return w
    law = "lower_clamped_normal" if dist == "normal" else dist
    expected_unscaled = (
        _lower_clamped_normal_mean(p1, p2)
        if law == "lower_clamped_normal"
        else float(p1)
    )
    provenance = {
        "distribution": law,
        "parent_parameters": {"mean": float(p1), "std": float(p2)},
        "fan_in": int(n_pre),
        "scaling_convention": "fan_in_normalized_expected_summed_coupling",
        "requested_initial_zero_fraction": float(initial_zero_fraction),
        "initial_zeroing": (
            "exact_k"
            if EXACT_K_INITIALIZATION and initial_zero_fraction > 0
            else "bernoulli"
            if initial_zero_fraction > 0
            else "none"
        ),
        "expected_summed_coupling_after_clamp": expected_unscaled,
        "theoretical_lower_clamp_zero_fraction": (
            _lower_clamp_zero_probability(p1, p2)
            if law == "lower_clamped_normal"
            else None
        ),
        "zeros_remain_trainable": True,
        "statistics": _weight_statistics(
            w,
            clamp_zero_mask=clamp_zero_mask,
            explicit_zero_mask=explicit_zero_mask,
        ),
    }
    return w, provenance


def init_readout_weight(shape, mean, std, *, return_provenance=False):
    """Initialize directly stored readout weights from a lower-clamped normal."""
    weight = torch.randn(*shape).mul_(std).add_(mean).clamp_(min=0)
    if not return_provenance:
        return weight
    zeros = weight == 0
    return weight, {
        "distribution": "lower_clamped_normal",
        "parent_parameters": {"mean": float(mean), "std": float(std)},
        "fan_in": int(shape[0]),
        "scaling_convention": "direct_stored_weight",
        "requested_initial_zero_fraction": 0.0,
        "initial_zeroing": "none",
        "expected_summed_coupling_after_clamp": None,
        "theoretical_lower_clamp_zero_fraction": (
            _lower_clamp_zero_probability(mean, std)
        ),
        "zeros_remain_trainable": True,
        "statistics": _weight_statistics(
            weight,
            clamp_zero_mask=zeros,
            explicit_zero_mask=torch.zeros_like(zeros),
        ),
    }


# ── E-step and I-step composites ─────────────────────────────────────────

COBA_INTEGRATOR = "expeuler"  # "expeuler" | "fwd"  — parity toggle for COBA integration


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


# ── Model class ──────────────────────────────────────────────────────────


class COBANet(nn.Module):
    recording = False
    recording_mode = "full"
    signed_weights = False

    def _set_meta(self, B, n_spk, rec, sizes):
        t_sec = T_steps * dt / 1000.0
        self.rates = {k: v / (B * sizes[k] * t_sec) for k, v in n_spk.items()}
        if rec is not None:
            # Accept either pre-stacked tensors or lists of per-timestep tensors
            self.spike_record = {
                k: (v if isinstance(v, torch.Tensor) else torch.stack(v))
                for k, v in rec.items()
            }

    def __init__(
        self,
        w_in=(W_IN_MEAN, W_IN_STD),
        w_in_i=None,
        w_hid=(W_HID_MEAN, W_HID_STD),
        w_ee=(W_EE_MEAN, W_EE_STD),
        w_ei=(W_EI_MEAN, W_EI_STD),
        w_ie=(W_IE_MEAN, W_IE_STD),
        w_ii=(W_II_MEAN, W_II_STD),
        dist="lower_clamped_normal",
        initial_zero_fraction=0.0,
        dales_law=True,
        hidden_sizes=None,
        readout_mode="rate",
        signed_readout=False,
        readout_bias=False,
        readout_w_init=None,
        trainable_w_ee=False,
        trainable_w_ei=False,
        trainable_w_ie=False,
        trainable_w_ii=False,
        n_inh_per_layer=None,
        state_clamp=False,
        train_leak=False,
        tau_m_e_bounds_ms=TRAINABLE_TAU_M_E_BOUNDS_MS,
        tau_m_i_bounds_ms=TRAINABLE_TAU_M_I_BOUNDS_MS,
        adaptive_threshold=False,
        adapt_tau_bounds_ms=ADAPT_TAU_BOUNDS_MS,
        adapt_strength_init_mv=1.0,
        adapt_strength_max_mv=ADAPT_STRENGTH_MAX_MV,
        refractory_e_ms=None,
        refractory_i_ms=None,
        refractory_policy="nearest",
    ):
        super().__init__()
        self.refractory_e_ms = float(
            ref_ms_E if refractory_e_ms is None else refractory_e_ms
        )
        self.refractory_i_ms = float(
            ref_ms_I if refractory_i_ms is None else refractory_i_ms
        )
        for duration in (self.refractory_e_ms, self.refractory_i_ms):
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("refractory durations must be finite and positive")
        if refractory_policy not in ("exact", "nearest"):
            raise ValueError("refractory policy must be 'exact' or 'nearest'")
        self.refractory_policy = refractory_policy
        if readout_mode not in (
            "rate",
            "mem-mean",
            "spike-count",
            "spike-rate",
            "cumulative-potential",
        ):
            raise ValueError(
                "readout_mode must be 'rate', 'mem-mean', 'spike-count', "
                f"'spike-rate', or 'cumulative-potential', got {readout_mode!r}"
            )
        self.readout_mode = readout_mode
        self.signed_readout = bool(signed_readout)
        self.readout_bias_enabled = bool(readout_bias)
        self.readout_w_init = readout_w_init
        if self.readout_bias_enabled and self.readout_mode != "cumulative-potential":
            raise ValueError(
                "readout_bias is supported only by the cumulative-potential readout"
            )
        self.signed_weights = not dales_law
        # Forward-pass state clamp: floor conductances at 0 (and cap magnitude)
        # each timestep. Off by default, so every existing run is unchanged.
        self.state_clamp = state_clamp
        self.train_leak = bool(train_leak)
        self.tau_m_e_bounds_ms = tuple(float(x) for x in tau_m_e_bounds_ms)
        self.tau_m_i_bounds_ms = tuple(float(x) for x in tau_m_i_bounds_ms)
        self.adaptive_threshold = bool(adaptive_threshold)
        self.adapt_tau_bounds_ms = tuple(float(x) for x in adapt_tau_bounds_ms)
        self.adapt_strength_init_mv = float(adapt_strength_init_mv)
        self.adapt_strength_max_mv = float(adapt_strength_max_mv)
        # Lazy-compile cache for the per-timestep body (compiled on first call).
        self._compiled_cache: dict = {}
        # Optional per-step hook fired right after every layer's spikes are
        # emitted and before they propagate (readout + next-step recurrence +
        # I-loop). Signature: (s_e, s_i_or_None, layer_idx) -> (s_e', s_i').
        # Set on the live instance to do hidden-spike perturbation experiments
        # without modifying the forward graph. Forces the eager step body.
        self._hidden_perturb_fn = None

        sizes = hidden_sizes if hidden_sizes is not None else HIDDEN_SIZES
        self.hidden_sizes = list(sizes)
        self.n_layers = len(sizes)
        all_sizes = [N_IN] + list(sizes) + [N_OUT]
        self.all_sizes = all_sizes
        self.weight_initialization = {}

        # Every hidden layer gets E-I structure (1-indexed).
        self.ei_layers = set(range(1, self.n_layers + 1))

        # Per-layer N_I override (1-indexed). When None, falls back to
        # n_e // 4. Used by the I-pool sweep in nb047 to vary the E:I
        # ratio without retraining the rest of the architecture.
        self.n_inh_per_layer = dict(n_inh_per_layer or {})

        # Feedforward weights: input→H1, H1→H2, ..., HN→output
        self.W_ff = nn.ParameterList()
        for idx, (n_pre, n_post) in enumerate(zip(all_sizes[:-1], all_sizes[1:])):
            spec = w_in if idx == 0 else w_hid
            p1, p2, d, s = _parse_weight_spec(spec, dist, initial_zero_fraction)
            is_readout = idx == len(all_sizes) - 2
            if is_readout and self.readout_w_init is not None:
                mean, std = self.readout_w_init
                weight, init_meta = init_readout_weight(
                    (n_pre, n_post), mean, std, return_provenance=True
                )
            else:
                weight, init_meta = init_weight(
                    (n_pre, n_post), d, p1, p2, s, return_provenance=True
                )
            self.W_ff.append(nn.Parameter(weight))
            role = "W_in" if idx == 0 else "W_out" if is_readout else f"W_ff_{idx}"
            init_meta.update({"role": role, "shape": [n_pre, n_post]})
            self.weight_initialization[role] = init_meta

        if w_in_i is None:
            self.register_parameter("W_in_i", None)
        else:
            p1, p2, d, s = _parse_weight_spec(w_in_i, dist, initial_zero_fraction)
            n_i = self.n_inh_per_layer.get(1, sizes[0] // 4)
            weight, init_meta = init_weight(
                (N_IN, n_i), d, p1, p2, s, return_provenance=True
            )
            self.W_in_i = nn.Parameter(weight)
            init_meta.update({"role": "W_in_i", "shape": [N_IN, n_i]})
            self.weight_initialization["W_in_i"] = init_meta

        # The classifier may be an abstract signed decoder while the simulated
        # feed-forward and recurrent synapses remain Dale-constrained.  Keep it
        # in W_ff[-1] so legacy checkpoint shape inference continues to work,
        # but initialise it like nn.Linear rather than as a positive
        # conductance when signed decoding is requested.
        if self.signed_readout:
            if self.readout_w_init is not None:
                raise ValueError(
                    "readout_w_init cannot be combined with signed_readout"
                )
            nn.init.kaiming_uniform_(self.W_ff[-1], a=math.sqrt(5))
            weight = self.W_ff[-1].detach()
            zeros = torch.zeros_like(weight, dtype=torch.bool)
            self.weight_initialization["W_out"] = {
                "role": "W_out",
                "shape": list(weight.shape),
                "distribution": "kaiming_uniform_signed",
                "parent_parameters": None,
                "fan_in": int(weight.shape[0]),
                "scaling_convention": "torch_kaiming_uniform",
                "requested_initial_zero_fraction": 0.0,
                "initial_zeroing": "none",
                "expected_summed_coupling_after_clamp": None,
                "zeros_remain_trainable": True,
                "statistics": _weight_statistics(
                    weight, clamp_zero_mask=zeros, explicit_zero_mask=zeros
                ),
            }

        if self.readout_bias_enabled:
            self.b_out = nn.Parameter(torch.empty(N_OUT))
            bound = 1.0 / math.sqrt(self.hidden_sizes[-1])
            nn.init.uniform_(self.b_out, -bound, bound)
        else:
            self.register_parameter("b_out", None)

        # Bittar & Garner's cumulative-potential decoder uses one trainable
        # membrane decay per class, bounded to 5--25 ms.  Store alpha directly
        # to match the reference implementation; clamp it in the forward pass.
        if self.readout_mode == "cumulative-potential":
            tau_lo, tau_hi = CUMULATIVE_READOUT_TAU_BOUNDS_MS
            alpha_lo = math.exp(-dt / tau_lo)
            alpha_hi = math.exp(-dt / tau_hi)
            self.readout_alpha = nn.Parameter(torch.empty(N_OUT))
            nn.init.uniform_(self.readout_alpha, alpha_lo, alpha_hi)
        else:
            self.register_parameter("readout_alpha", None)

        # E-I weights per E-I layer. By default the recurrent circuit is fixed
        # anatomical connectivity — a substrate the readout learns to read — but
        # each of the four blocks (W_ee / W_ei / W_ie / W_ii) can be made a
        # trainable parameter via its trainable_w_* flag. Training all four with
        # --no-dales-law turns the hidden layer into a free signed recurrent
        # matrix (a generic RSNN, no longer E/I-constrained).
        self.W_ee = nn.ParameterDict()
        self.W_ei = nn.ParameterDict()
        self.W_ie = nn.ParameterDict()
        # W_ii is the I→I recurrent matrix (added for Brunel/Vreeswijk balanced-
        # network experiments). Zero-mean / zero-std by default so the canonical
        # PING architecture is unchanged; set w_ii to non-zero to enable.
        self.W_ii = nn.ParameterDict()
        for i in self.ei_layers:
            n_e = sizes[i - 1]
            n_i = self.n_inh_per_layer.get(i, n_e // 4)
            k = str(i)
            p1, p2, d, s = _parse_weight_spec(w_ee, dist, initial_zero_fraction)
            w_ee_init, w_ee_meta = init_weight(
                (n_e, n_e), d, p1, p2, s, return_provenance=True
            )
            w_ee_t = nn.Parameter(
                w_ee_init,
                requires_grad=trainable_w_ee,
            )
            p1, p2, d, s = _parse_weight_spec(w_ei, dist, initial_zero_fraction)
            w_ei_init, w_ei_meta = init_weight(
                (n_e, n_i), d, p1, p2, s, return_provenance=True
            )
            w_ei_t = nn.Parameter(
                w_ei_init,
                requires_grad=trainable_w_ei,
            )
            p1, p2, d, s = _parse_weight_spec(w_ie, dist, initial_zero_fraction)
            w_ie_init, w_ie_meta = init_weight(
                (n_i, n_e), d, p1, p2, s, return_provenance=True
            )
            w_ie_t = nn.Parameter(
                w_ie_init,
                requires_grad=trainable_w_ie,
            )
            p1, p2, d, s = _parse_weight_spec(w_ii, dist, initial_zero_fraction)
            w_ii_init, w_ii_meta = init_weight(
                (n_i, n_i), d, p1, p2, s, return_provenance=True
            )
            w_ii_t = nn.Parameter(
                w_ii_init,
                requires_grad=trainable_w_ii,
            )
            self.W_ee[k] = w_ee_t
            self.W_ei[k] = w_ei_t
            self.W_ie[k] = w_ie_t
            self.W_ii[k] = w_ii_t
            for role, meta, trainable in (
                (f"W_EE_{k}", w_ee_meta, trainable_w_ee),
                (f"W_EI_{k}", w_ei_meta, trainable_w_ei),
                (f"W_IE_{k}", w_ie_meta, trainable_w_ie),
                (f"W_II_{k}", w_ii_meta, trainable_w_ii),
            ):
                tensor = {
                    f"W_EE_{k}": w_ee_t,
                    f"W_EI_{k}": w_ei_t,
                    f"W_IE_{k}": w_ie_t,
                    f"W_II_{k}": w_ii_t,
                }[role]
                meta.update(
                    {
                        "role": role,
                        "shape": list(tensor.shape),
                        "trainable": bool(trainable),
                    }
                )
                self.weight_initialization[role] = meta

        for idx, parameter in enumerate(self.W_ff):
            role = (
                "W_in"
                if idx == 0
                else "W_out"
                if idx == len(self.W_ff) - 1
                else f"W_ff_{idx}"
            )
            self.weight_initialization[role]["trainable"] = bool(
                parameter.requires_grad
            )

        self.tau_m_e_logit = nn.ParameterDict()
        self.tau_m_i_logit = nn.ParameterDict()
        if self.train_leak:
            e_lo, e_hi = self.tau_m_e_bounds_ms
            i_lo, i_hi = self.tau_m_i_bounds_ms
            e_init = _bounded_logit(tau_m_E, e_lo, e_hi)
            i_init = _bounded_logit(tau_m_I, i_lo, i_hi)
            for i in self.ei_layers:
                n_e = sizes[i - 1]
                n_i = self.n_inh_per_layer.get(i, n_e // 4)
                k = str(i)
                self.tau_m_e_logit[k] = nn.Parameter(torch.full((n_e,), e_init))
                self.tau_m_i_logit[k] = nn.Parameter(torch.full((n_i,), i_init))

        self.adapt_tau_logit = nn.ParameterDict()
        self.adapt_strength_logit = nn.ParameterDict()
        if self.adaptive_threshold:
            tau_lo, tau_hi = self.adapt_tau_bounds_ms
            tau_init = _bounded_logit((tau_lo * tau_hi) ** 0.5, tau_lo, tau_hi)
            strength_lo, strength_hi = 0.0, self.adapt_strength_max_mv
            strength_init = min(
                max(self.adapt_strength_init_mv, 1e-6), strength_hi - 1e-6
            )
            strength_logit = _bounded_logit(strength_init, strength_lo, strength_hi)
            for i in self.ei_layers:
                n_e = sizes[i - 1]
                k = str(i)
                self.adapt_tau_logit[k] = nn.Parameter(torch.full((n_e,), tau_init))
                self.adapt_strength_logit[k] = nn.Parameter(
                    torch.full((n_e,), strength_logit)
                )

    def leak_params(self, layer_key: str):
        """Return (C_m_E, g_L_E, C_m_I, g_L_I) for one layer.

        With ``train_leak`` off this is the historical scalar COBA leak.  With it
        on, per-neuron bounded membrane time constants are converted back to
        leak conductances via g_L = C_m / τ_m while giving cells heterogeneous
        integration horizons.
        """
        if not self.train_leak:
            return C_m_E, g_L_E, C_m_I, g_L_I
        e_lo, e_hi = self.tau_m_e_bounds_ms
        i_lo, i_hi = self.tau_m_i_bounds_ms
        tau_e = _bounded_from_logit(self.tau_m_e_logit[layer_key], e_lo, e_hi)
        tau_i = _bounded_from_logit(self.tau_m_i_logit[layer_key], i_lo, i_hi)
        return C_m_E, C_m_E / tau_e, C_m_I, C_m_I / tau_i

    def adapt_params(self, layer_key: str):
        """Return bounded adaptive-threshold decay and strength for one E layer."""
        if not self.adaptive_threshold:
            return None, None
        tau_lo, tau_hi = self.adapt_tau_bounds_ms
        tau = _bounded_from_logit(self.adapt_tau_logit[layer_key], tau_lo, tau_hi)
        strength = _bounded_from_logit(
            self.adapt_strength_logit[layer_key],
            0.0,
            self.adapt_strength_max_mv,
        )
        return torch.exp(-dt / tau), strength

    def _hid_key(self, layer_idx):
        if self.n_layers == 1:
            return "hid"
        return f"hid_{layer_idx}"

    def _inh_key(self, layer_idx):
        if self.n_layers == 1:
            return "inh"
        return f"inh_{layer_idx}"

    def forward(
        self,
        noise_std=0.0,
        randomize_init=False,
        ref_mean=0.0,
        ref_std=0.0,
        ext_g=None,
        ext_g_i=None,
        ext_g_inhib_e=None,
        ext_g_inhib_i=None,
        drive_sigma=0.0,
        input_spikes=None,
        input_spikes_i=None,
        readout_reset_mask=None,
        v_perturb_eps=0.0,
        v_perturb_seed=0,
        noise_on_inh=True,
        recurrent_weight_scales=None,
    ):
        has_ext_g = ext_g is not None
        has_ext_g_i = ext_g_i is not None
        has_ext_g_inhib_e = ext_g_inhib_e is not None
        has_ext_g_inhib_i = ext_g_inhib_i is not None
        has_input_spikes = input_spikes is not None
        has_input_spikes_i = input_spikes_i is not None
        has_readout_reset = readout_reset_mask is not None

        if has_ext_g and ext_g.dim() == 3:
            B, device = ext_g.shape[1], ext_g.device
        elif has_input_spikes and input_spikes.dim() == 3:
            B, device = input_spikes.shape[1], input_spikes.device
        else:
            B, device = (
                1,
                (
                    ext_g.device
                    if has_ext_g
                    else input_spikes.device
                    if has_input_spikes
                    else torch.device("cpu")
                ),
            )

        # A schedule scales an initialized recurrent matrix without replacing
        # it. This preserves topology and continuous neuron/synapse state.
        recurrent_weight_scales = recurrent_weight_scales or {}
        scale_series = {}
        for name in ("w_ee", "w_ei", "w_ie", "w_ii"):
            series = recurrent_weight_scales.get(name)
            if series is None:
                series = torch.ones(T_steps, device=device)
            else:
                series = torch.as_tensor(series, dtype=torch.float32, device=device)
                if series.ndim != 1 or len(series) < T_steps:
                    raise ValueError(f"{name} weight scale must be a length-T vector")
            scale_series[name] = series

        if self.signed_weights:
            W_ff = list(self.W_ff)
        else:
            W_ff = [W.clamp(min=0) for W in self.W_ff]
            if self.signed_readout:
                W_ff[-1] = self.W_ff[-1]
        # `noise_std` is the diffusive membrane-noise amplitude (mV, on the
        # voltage), applied per-cell-independent to every E and I cell inside
        # the LIF step (see lif_step_expeuler). Zero-mean and dt-invariant —
        # replaces the old rectified, first-layer-E-only g_E conductance noise.
        v_noise_std = float(noise_std) if noise_std > 0 else 0.0

        # Per-step matmul on every device — single code path (no CUDA-only
        # fast-path).

        # Per-layer state
        v_e, ref_e, ge_e, gi_e, s_e = {}, {}, {}, {}, {}
        v_i, ref_i, ge_i, gi_i, s_i = {}, {}, {}, {}, {}
        a_e = {}
        drive_gains = {}
        for i in range(1, self.n_layers + 1):
            n_e = self.hidden_sizes[i - 1]
            k = str(i)
            v_e[k], ref_e[k] = init_lif_state(
                B,
                n_e,
                device,
                randomize=randomize_init,
                ref_mean=ref_mean,
                ref_std=ref_std,
            )
            ge_e[k] = init_conductance(B, n_e, device)
            s_e[k] = torch.zeros(B, n_e, device=device)
            if self.adaptive_threshold:
                a_e[k] = torch.zeros(B, n_e, device=device)
            if i in self.ei_layers:
                n_i = self.n_inh_per_layer.get(i, n_e // 4)
                gi_e[k] = init_conductance(B, n_e, device)
                v_i[k], ref_i[k] = init_lif_state(
                    B, n_i, device, randomize=randomize_init
                )
                ge_i[k] = init_conductance(B, n_i, device)
                gi_i[k] = init_conductance(B, n_i, device)
                s_i[k] = torch.zeros(B, n_i, device=device)
            if drive_sigma > 0 and i == 1:
                drive_gains[k] = (
                    1.0 + drive_sigma * torch.randn(B, n_e, device=device)
                ).clamp(min=0)

        # Lyapunov perturbation: add a fixed-norm random offset to every
        # membrane voltage at t=0 so a second forward pass on identical
        # input diverges only by the chaos of the dynamics. Seeded so the
        # perturbation is reproducible across the clean/perturbed pair.
        if v_perturb_eps > 0:
            pgen = torch.Generator(device="cpu").manual_seed(int(v_perturb_seed))
            for i in range(1, self.n_layers + 1):
                k = str(i)
                dv = (
                    torch.randn(v_e[k].shape, generator=pgen).to(device) * v_perturb_eps
                )
                v_e[k] = v_e[k] + dv
                if k in v_i:
                    dvi = (
                        torch.randn(v_i[k].shape, generator=pgen).to(device)
                        * v_perturb_eps
                    )
                    v_i[k] = v_i[k] + dvi

        # Output state. ``rate`` uses hidden_accum directly; the spiking
        # readouts share v_out and differ in how the output-LIF trajectory is
        # reduced to class logits.
        hidden_accum = init_conductance(B, self.hidden_sizes[-1], device)
        v_out = torch.zeros(B, N_OUT, device=device)
        mem_sum = torch.zeros(B, N_OUT, device=device)
        out_spike_count = torch.zeros(B, N_OUT, device=device)
        s_out = torch.zeros(B, N_OUT, device=device)
        evidence_sum = torch.zeros(B, N_OUT, device=device)

        # Pre-allocate recording buffers on GPU
        rec_buf = None
        if self.recording:
            if self.recording_mode not in ("full", "spikes", "inhibitory"):
                raise ValueError(f"unknown recording mode: {self.recording_mode}")
            full_recording = self.recording_mode == "full"
            rec_buf = {}
            if full_recording:
                rec_buf["out"] = torch.zeros(T_steps, B, N_OUT, device=device)
            if full_recording and self.readout_mode in ("spike-count", "spike-rate"):
                rec_buf["out_spikes"] = torch.zeros(T_steps, B, N_OUT, device=device)
                rec_buf["v_out"] = torch.zeros(T_steps, B, N_OUT, device=device)
            if full_recording and has_input_spikes:
                rec_buf["input"] = torch.zeros(T_steps, B, N_IN, device=device)
            for i in range(1, self.n_layers + 1):
                n_e = self.hidden_sizes[i - 1]
                if self.recording_mode != "inhibitory":
                    rec_buf[self._hid_key(i)] = torch.zeros(T_steps, B, n_e, device=device)
                # Extra trace buffers: membrane voltage and conductances for
                # E (and I, where present). Lets downstream image-mode dump
                # per-neuron v/g traces alongside spikes.
                if full_recording:
                    rec_buf[f"v_e_{i}"] = torch.zeros(T_steps, B, n_e, device=device)
                    rec_buf[f"ge_e_{i}"] = torch.zeros(T_steps, B, n_e, device=device)
                if i in self.ei_layers:
                    n_inh = self.n_inh_per_layer.get(i, n_e // 4)
                    rec_buf[self._inh_key(i)] = torch.zeros(
                        T_steps, B, n_inh, device=device
                    )
                    if full_recording:
                        rec_buf[f"gi_e_{i}"] = torch.zeros(T_steps, B, n_e, device=device)
                        rec_buf[f"v_i_{i}"] = torch.zeros(T_steps, B, n_inh, device=device)
                        rec_buf[f"ge_i_{i}"] = torch.zeros(T_steps, B, n_inh, device=device)
                        rec_buf[f"gi_i_{i}"] = torch.zeros(T_steps, B, n_inh, device=device)
        # GPU-side spike accumulators
        n_spk_tensors = {}
        for i in range(1, self.n_layers + 1):
            n_spk_tensors[self._hid_key(i)] = torch.zeros(1, device=device)
            if i in self.ei_layers:
                n_spk_tensors[self._inh_key(i)] = torch.zeros(1, device=device)
        n_spk_tensors["out"] = torch.zeros(1, device=device)
        # Per-layer (B, n_e) spike-count accumulator for the firing-rate
        # regulariser — must keep gradient attached, so it sums state["s_e"]
        # post-step.
        rate_counts = [torch.zeros(B, n, device=device) for n in self.hidden_sizes]

        # Bundle mutating state and per-call config so _step_body can be
        # compiled per-timestep. The Python
        # int `t` and rec_buf writes stay in _forward_loop so the compiled
        # graph never has to re-trace per-t.
        state = {
            "v_e": v_e,
            "ref_e": ref_e,
            "ge_e": ge_e,
            "gi_e": gi_e,
            "s_e": s_e,
            "a_e": a_e,
            "v_i": v_i,
            "ref_i": ref_i,
            "ge_i": ge_i,
            "gi_i": gi_i,
            "s_i": s_i,
            "hidden_accum": hidden_accum,
            "v_out": v_out,
            "mem_sum": mem_sum,
            "out_spike_count": out_spike_count,
            "s_out": s_out,
            "evidence_sum": evidence_sum,
        }
        # Compute dt-dependent constants locally so torch.compile specializes on dt
        decay_ampa = np.exp(-dt / tau_ampa)
        decay_gaba = np.exp(-dt / tau_gaba)
        timing = refractory_metadata(
            self.refractory_e_ms,
            self.refractory_i_ms,
            dt,
            policy=self.refractory_policy,
        )
        self.timing_metadata = {
            **timing,
            "nominal_duration_ms": float(T_ms),
            "duration_steps": int(T_steps),
            "realized_duration_ms": T_steps * dt,
        }
        beta_snn = np.exp(-dt / tau_snn)
        beta_out = np.exp(-dt / tau_out_ms)
        leak_params = {
            str(i): self.leak_params(str(i)) for i in range(1, self.n_layers + 1)
        }
        adapt_params = {
            str(i): self.adapt_params(str(i)) for i in range(1, self.n_layers + 1)
        }

        cfg = {
            "B": B,
            "device": device,
            "W_ff": W_ff,
            "drive_gains": drive_gains,
            "ei_layers": self.ei_layers,
            "has_input_spikes": has_input_spikes,
            "has_input_spikes_i": has_input_spikes_i,
            "has_ext_g": has_ext_g,
            "has_ext_g_i": has_ext_g_i,
            "has_ext_g_inhib_e": has_ext_g_inhib_e,
            "has_ext_g_inhib_i": has_ext_g_inhib_i,
            "has_input_i": self.W_in_i is not None,
            "W_in_i": self.W_in_i.clamp(min=0)
            if self.W_in_i is not None and not self.signed_weights
            else self.W_in_i,
            "has_readout_reset": has_readout_reset,
            "readout_mode": self.readout_mode,
            "readout_bias": self.b_out,
            "readout_alpha": self.readout_alpha,
            "v_noise_std": v_noise_std,
            "noise_on_inh": bool(noise_on_inh),
            "n_e0": self.hidden_sizes[0],
            "n_spk_tensors": n_spk_tensors,
            "decay_ampa": decay_ampa,
            "decay_gaba": decay_gaba,
            "ref_steps_E": timing["refractory_e_steps"],
            "ref_steps_I": timing["refractory_i_steps"],
            "beta_snn": beta_snn,
            "beta_out": beta_out,
            "leak_params": leak_params,
            "adaptive_threshold": self.adaptive_threshold,
            "adapt_params": adapt_params,
        }

        # Lazy-init torch.compile on the per-timestep body, with a CPU-skip
        # and a PINGLAB_NO_COMPILE escape hatch. CPU is skipped because
        # Inductor's cpp build fails on
        # some hosts and the error escapes the try/except (surfaces only
        # at first compiled call, not at torch.compile() construction).
        if (
            "step" not in self._compiled_cache
            and not _env_no_compile()
            and device.type != "cpu"
            and self._hidden_perturb_fn is None
        ):
            try:
                self._compiled_cache["step"] = torch.compile(
                    self._step_body, dynamic=False
                )
            except Exception as exc:  # noqa: BLE001
                self._compiled_cache["step"] = self._step_body
                self._compiled_cache["compile_error"] = str(exc)
        # When the perturb hook is set, always use the eager body so the
        # Python callable doesn't break (or trigger recompile on) the graph.
        step = (
            self._step_body
            if self._hidden_perturb_fn is not None
            else self._compiled_cache.get("step", self._step_body)
        )

        for t in range(T_steps):
            slc = {
                "in_t": (
                    input_spikes[t].unsqueeze(0)
                    if has_input_spikes and input_spikes.dim() == 2
                    else (input_spikes[t] if has_input_spikes else None)
                ),
                "in_i_t": (
                    input_spikes_i[t].unsqueeze(0)
                    if has_input_spikes_i and input_spikes_i.dim() == 2
                    else (input_spikes_i[t] if has_input_spikes_i else None)
                ),
                "ext_t": (
                    ext_g[t].unsqueeze(0)
                    if has_ext_g and ext_g.dim() == 2
                    else (ext_g[t] if has_ext_g else None)
                ),
                "ext_t_i": (
                    ext_g_i[t].unsqueeze(0)
                    if has_ext_g_i and ext_g_i.dim() == 2
                    else (ext_g_i[t] if has_ext_g_i else None)
                ),
                "ext_inhib_e_t": (
                    ext_g_inhib_e[t].unsqueeze(0)
                    if has_ext_g_inhib_e and ext_g_inhib_e.dim() == 2
                    else (ext_g_inhib_e[t] if has_ext_g_inhib_e else None)
                ),
                "ext_inhib_i_t": (
                    ext_g_inhib_i[t].unsqueeze(0)
                    if has_ext_g_inhib_i and ext_g_inhib_i.dim() == 2
                    else (ext_g_inhib_i[t] if has_ext_g_inhib_i else None)
                ),
                "readout_reset_t": (
                    readout_reset_mask[t] if has_readout_reset else None
                ),
                "recurrent_scale_t": (
                    scale_series["w_ee"][t],
                    scale_series["w_ei"][t],
                    scale_series["w_ie"][t],
                    scale_series["w_ii"][t],
                ),
            }
            logits_t = step(slc, cfg, state)
            # Accumulate per-neuron E spike counts for fr-reg (grad-attached).
            for i in range(1, self.n_layers + 1):
                rate_counts[i - 1] = rate_counts[i - 1] + state["s_e"][str(i)]
            if rec_buf is not None:
                if slc["in_t"] is not None and "input" in rec_buf:
                    rec_buf["input"][t] = slc["in_t"]
                for i in range(1, self.n_layers + 1):
                    k = str(i)
                    if self._hid_key(i) in rec_buf:
                        rec_buf[self._hid_key(i)][t] = state["s_e"][k]
                    if full_recording:
                        rec_buf[f"v_e_{i}"][t] = state["v_e"][k]
                        rec_buf[f"ge_e_{i}"][t] = state["ge_e"][k]
                    if i in self.ei_layers:
                        rec_buf[self._inh_key(i)][t] = state["s_i"][k]
                        if full_recording:
                            rec_buf[f"gi_e_{i}"][t] = state["gi_e"][k]
                            rec_buf[f"v_i_{i}"][t] = state["v_i"][k]
                            rec_buf[f"ge_i_{i}"][t] = state["ge_i"][k]
                            rec_buf[f"gi_i_{i}"][t] = state["gi_i"][k]
                if full_recording:
                    rec_buf["out"][t] = logits_t
                if "out_spikes" in rec_buf:
                    rec_buf["out_spikes"][t] = state["s_out"]
                    rec_buf["v_out"][t] = state["v_out"]

        sizes = {}
        for i in range(1, self.n_layers + 1):
            sizes[self._hid_key(i)] = self.hidden_sizes[i - 1]
            if i in self.ei_layers:
                sizes[self._inh_key(i)] = self.hidden_sizes[i - 1] // 4
        sizes["out"] = N_OUT
        n_spk = {k: v.item() for k, v in n_spk_tensors.items()}
        rec = None
        if rec_buf is not None:
            rec = {
                k: (v.squeeze(1).cpu() if B == 1 else v.cpu())
                for k, v in rec_buf.items()
            }
        self._set_meta(B, n_spk, rec, sizes)
        # Expose grad-attached per-neuron spike counts so the trainer's
        # firing-rate regulariser (train.py) can build its loss.
        self.last_spike_counts = rate_counts
        # Spike-count readouts need their own activity diagnostics.  Preserve
        # the per-sample, per-class counts from the completed presentation so
        # training can detect silent and saturated decoders without enabling
        # the much larger timestep recording buffers.
        self.last_output_spike_counts = state["out_spike_count"]
        if self.readout_mode == "mem-mean":
            return state["mem_sum"] / float(T_steps)
        return logits_t

    def _step_body(self, slc, cfg, state):
        """One timestep: all PING layers + readout. The compile target.

        slc:    per-t inputs (in_t, ext_t) sliced in _forward_loop.
        cfg:    per-call constants (W_ff, drive_gains, ei_layers, flags,
                spike accumulators).
        state:  per-call mutable dicts (membrane voltages, conductances,
                spike outputs, readout accumulators). Mutated in place.

        Returns logits_t for the rec_buf write in _forward_loop. Per-t
        rec_buf writes and Python int `t` are kept out so the compiled
        graph reuses across all T_steps invocations.
        """
        W_ff = cfg["W_ff"]
        ei_layers = cfg["ei_layers"]
        drive_gains = cfg["drive_gains"]
        has_input_spikes = cfg["has_input_spikes"]
        has_ext_g = cfg["has_ext_g"]
        n_spk_tensors = cfg["n_spk_tensors"]

        prev_spk = None
        for i in range(1, self.n_layers + 1):
            k = str(i)
            W = W_ff[i - 1]
            is_ei = i in ei_layers

            if is_ei:
                # exp_synapse inlined: when called as a separate function
                # Dynamo compiles it as its own trace boundary and hits the
                # recompile limit (3 different W shapes × inner-loop call
                # sites > recompile_limit=8), silently falling back to
                # eager. Inlining keeps everything inside _step_body's
                # single compiled graph where the three weight shapes are
                # specialized once per (W_ee, W_ei, W_ie) tuple.
                scale_ee, scale_ei, scale_ie, scale_ii = slc["recurrent_scale_t"]
                ee_drive = (state["s_e"][k] @ self.W_ee[k]) * scale_ee
                state["ge_e"][k] = state["ge_e"][k] * cfg["decay_ampa"] + ee_drive
                ei_drive = (state["s_e"][k] @ self.W_ei[k]) * scale_ei
                if k == "1" and cfg["has_ext_g_i"]:
                    ei_drive = ei_drive + slc["ext_t_i"]
                if k == "1" and cfg["has_input_i"] and has_input_spikes:
                    input_i = (
                        slc["in_i_t"] if cfg["has_input_spikes_i"] else slc["in_t"]
                    )
                    ei_drive = ei_drive + input_i @ cfg["W_in_i"]
                state["ge_i"][k] = state["ge_i"][k] * cfg["decay_ampa"] + ei_drive
                state["gi_e"][k] = (
                    state["gi_e"][k] * cfg["decay_gaba"]
                    + (state["s_i"][k] @ self.W_ie[k]) * scale_ie
                )
                state["gi_i"][k] = (
                    state["gi_i"][k] * cfg["decay_gaba"]
                    + (state["s_i"][k] @ self.W_ii[k]) * scale_ii
                )
                if k == "1" and cfg["has_ext_g_inhib_e"]:
                    state["gi_e"][k] = state["gi_e"][k] + slc["ext_inhib_e_t"]
                if k == "1" and cfg["has_ext_g_inhib_i"]:
                    state["gi_i"][k] = state["gi_i"][k] + slc["ext_inhib_i_t"]
            else:
                state["ge_e"][k] = state["ge_e"][k] * cfg["decay_ampa"]

            if i == 1:
                if has_input_spikes:
                    g_ext = slc["in_t"] @ W
                    if k in drive_gains:
                        g_ext = g_ext * drive_gains[k]
                    state["ge_e"][k] = state["ge_e"][k] + g_ext
                if has_ext_g:
                    state["ge_e"][k] = state["ge_e"][k] + slc["ext_t"]
            else:
                ff_drive = prev_spk @ W
                state["ge_e"][k] = state["ge_e"][k] + ff_drive

            # Forward-pass state clamp (opt-in): floor conductances at 0 and cap
            # their magnitude so a signed-weight net cannot drive g_tot ≤ 0 (the
            # exp-Euler v_inf = …/g_tot divergence) or accumulate unboundedly. The
            # weights stay signed — only the state is bounded. The `if` is a
            # static per-net constant, so torch.compile specialises one branch.
            if self.state_clamp:
                state["ge_e"][k] = state["ge_e"][k].clamp(0.0, G_CLAMP_MAX)
                if is_ei:
                    state["ge_i"][k] = state["ge_i"][k].clamp(0.0, G_CLAMP_MAX)
                    state["gi_e"][k] = state["gi_e"][k].clamp(0.0, G_CLAMP_MAX)
                    state["gi_i"][k] = state["gi_i"][k].clamp(0.0, G_CLAMP_MAX)

            g_e_for_step = state["ge_e"][k]
            g_i_for_e = state["gi_e"][k] if is_ei else None
            g_e_for_i = state["ge_i"][k] if is_ei else None
            g_i_for_i = state["gi_i"][k] if is_ei else None
            v_noise = cfg["v_noise_std"]
            v_noise_i = v_noise if cfg["noise_on_inh"] else 0.0
            c_m_e, g_l_e, c_m_i, g_l_i = cfg["leak_params"][k]
            threshold_e = state["a_e"][k] if cfg["adaptive_threshold"] else None
            if is_ei:
                state["v_e"][k], state["s_e"][k], state["ref_e"][k] = e_step_coba(
                    state["v_e"][k],
                    state["ref_e"][k],
                    g_e_for_step,
                    g_i_for_e,
                    ref_steps=cfg["ref_steps_E"],
                    threshold_offset=threshold_e,
                    v_noise_std=v_noise,
                    C_m=c_m_e,
                    g_L=g_l_e,
                )
                state["v_i"][k], state["s_i"][k], state["ref_i"][k] = i_step_coba(
                    state["v_i"][k],
                    state["ref_i"][k],
                    g_e_for_i,
                    g_i_for_i,
                    ref_steps=cfg["ref_steps_I"],
                    v_noise_std=v_noise_i,
                    C_m=c_m_i,
                    g_L=g_l_i,
                )
            else:
                state["v_e"][k], state["s_e"][k], state["ref_e"][k] = e_step_coba(
                    state["v_e"][k],
                    state["ref_e"][k],
                    g_e_for_step,
                    ref_steps=cfg["ref_steps_E"],
                    threshold_offset=threshold_e,
                    v_noise_std=v_noise,
                    C_m=c_m_e,
                    g_L=g_l_e,
                )
            if cfg["adaptive_threshold"]:
                adapt_decay, adapt_strength = cfg["adapt_params"][k]
                state["a_e"][k] = (
                    state["a_e"][k] * adapt_decay + state["s_e"][k] * adapt_strength
                )

            if self._hidden_perturb_fn is not None:
                new_s_e, new_s_i = self._hidden_perturb_fn(
                    state["s_e"][k],
                    state["s_i"].get(k) if is_ei else None,
                    i,
                )
                state["s_e"][k] = new_s_e
                if is_ei and new_s_i is not None:
                    state["s_i"][k] = new_s_i

            prev_spk = state["s_e"][k]

            hk = self._hid_key(i)
            n_spk_tensors[hk] += state["s_e"][k].detach().sum()
            if is_ei:
                ik = self._inh_key(i)
                n_spk_tensors[ik] += state["s_i"][k].detach().sum()

        if cfg["readout_mode"] in ("mem-mean", "spike-count", "spike-rate"):
            if cfg["has_readout_reset"]:
                reset = slc["readout_reset_t"].to(
                    device=state["v_out"].device, dtype=torch.bool
                )
                if reset.ndim == 0:
                    reset = reset.expand(state["v_out"].shape[0])
                reset = reset.reshape(-1, 1)
                state["v_out"] = torch.where(
                    reset, torch.zeros_like(state["v_out"]), state["v_out"]
                )
                state["mem_sum"] = torch.where(
                    reset, torch.zeros_like(state["mem_sum"]), state["mem_sum"]
                )
                state["out_spike_count"] = torch.where(
                    reset,
                    torch.zeros_like(state["out_spike_count"]),
                    state["out_spike_count"],
                )
            # Exp-Euler ZOH on output LIF + subtract reset. COBANet's W_ff has
            # no bias term — bias scaling is moot here.
            one_minus_beta = 1.0 - cfg["beta_out"]
            spike_scale = one_minus_beta / dt
            I_out = spike_scale * (prev_spk @ W_ff[-1])
            state["v_out"] = cfg["beta_out"] * state["v_out"] + I_out
            s_out = fast_sigmoid_spike(state["v_out"] - thr_snn, SURROGATE_SLOPE)
            state["s_out"] = s_out
            n_spk_tensors["out"] += s_out.detach().sum()
            if cfg["readout_mode"] in ("spike-count", "spike-rate"):
                state["out_spike_count"] = state["out_spike_count"] + s_out
            else:
                state["mem_sum"] = state["mem_sum"] + state["v_out"]
            state["v_out"] = state["v_out"] - s_out * thr_snn
            if cfg["readout_mode"] == "spike-count":
                return state["out_spike_count"]
            if cfg["readout_mode"] == "spike-rate":
                duration_s = float(T_steps) * dt / 1000.0
                return state["out_spike_count"] / duration_s
            return state["mem_sum"] / float(T_steps)
        if cfg["readout_mode"] == "cumulative-potential":
            drive = prev_spk @ W_ff[-1]
            if cfg["readout_bias"] is not None:
                drive = drive + cfg["readout_bias"]
            tau_lo, tau_hi = CUMULATIVE_READOUT_TAU_BOUNDS_MS
            alpha_lo = math.exp(-dt / tau_lo)
            alpha_hi = math.exp(-dt / tau_hi)
            alpha = cfg["readout_alpha"].clamp(min=alpha_lo, max=alpha_hi)
            state["v_out"] = alpha * state["v_out"] + (1.0 - alpha) * drive
            state["evidence_sum"] = state["evidence_sum"] + F.softmax(
                state["v_out"], dim=1
            )
            return state["evidence_sum"]
        state["hidden_accum"] = state["hidden_accum"] + prev_spk
        return state["hidden_accum"] @ W_ff[-1]

    def _dale_params(self):
        """Trainable weights subject to Dale's law: the feedforward stack plus
        any recurrent E↔I matrices flipped to trainable (--trainable-w-*).
        Non-trainable recurrent buffers are skipped — they are init'd
        non-negative and never updated."""
        dicts = (self.W_ee, self.W_ei, self.W_ie, self.W_ii)
        feedforward = list(self.W_ff)
        if self.signed_readout:
            feedforward = feedforward[:-1]
        params = feedforward + [p for d in dicts for p in d.values()]
        return [p for p in params if p.requires_grad]

    def weight_final_statistics(self):
        """Summarize trained tensors so initialization-zero regrowth is visible."""
        tensors = {}
        for idx, parameter in enumerate(self.W_ff):
            role = (
                "W_in"
                if idx == 0
                else "W_out"
                if idx == len(self.W_ff) - 1
                else f"W_ff_{idx}"
            )
            tensors[role] = parameter.detach()
        for prefix, parameter_dict in (
            ("W_EE", self.W_ee),
            ("W_EI", self.W_ei),
            ("W_IE", self.W_ie),
            ("W_II", self.W_ii),
        ):
            for layer, parameter in parameter_dict.items():
                tensors[f"{prefix}_{layer}"] = parameter.detach()

        result = {}
        for role, weight in tensors.items():
            flat = weight.reshape(-1)
            column_sums = weight.sum(dim=0)
            result[role] = {
                "zero_count": int((flat == 0).sum()),
                "zero_fraction": float((flat == 0).float().mean()),
                "effective_nonzero_fan_in": {
                    "mean": float((weight != 0).sum(dim=0).float().mean()),
                    "min": int((weight != 0).sum(dim=0).min()),
                    "max": int((weight != 0).sum(dim=0).max()),
                },
                "all_entries": {
                    "mean": float(flat.mean()),
                    "std": float(flat.std(unbiased=False)),
                    "min": float(flat.min()),
                    "max": float(flat.max()),
                },
                "realized_column_sum": {
                    "mean": float(column_sums.mean()),
                    "std": float(column_sums.std(unbiased=False)),
                    "min": float(column_sums.min()),
                    "max": float(column_sums.max()),
                },
            }
        return result

    @torch.no_grad()
    def project_dales(self) -> None:
        """Projected-gradient step for Dale's law: clamp every trainable
        constrained weight back onto the non-negative orthant. Registered as an
        optimiser step-post-hook (train.py), so it runs automatically after each
        opt.step(); no-op when signed weights are allowed. Eager and outside the
        compiled per-timestep graph, so it does not affect torch.compile."""
        if self.signed_weights:
            return
        for p in self._dale_params():
            p.clamp_(min=0)
