"""Acceptance tests for stdp_nonlinear: pair-based STDP with power-law weight
dependence and soft bounds (Gütig et al. 2003; NEST stdp_synapse).

Layers:
- N05-N26 drive the update step alone, on fixed spike trains.
- N27-N28 compare it with analytic equilibria (slow, statistical).
- N29-N30 compare it with Brian2 and NEST (slow; skipped without them).
- N31-N35 run the rule inside snnlab, with the update step as their reference.
  Post spikes are imposed with ReplaySpikes, so neuron dynamics play no part.
The lang-side tests N01-N04 live in tests/lang/test_snnlang.py.

The rule. With w_hat = (w - w_min) / (w_max - w_min), a presynaptic spike
depresses by a_minus * w_hat**mu_minus * y (clipped at w_min), then a
postsynaptic spike potentiates by a_plus * (1 - w_hat)**mu_plus * x, with the
weight left after depression (clipped at w_max). Traces, step order and the
same-step convention are those of stdp_simple: spikes in the same step do not
pair. With mu_plus = mu_minus = 0 the rule is stdp_simple.

Conventions: weights use the runtime layout (batch, source, target). Sizes are
non-square (3 sources, 2 targets), so a transposed layout fails. Defaults use
distinct amplitudes, time constants and exponents, so a swapped constant fails.

The tests are skipped until snnlab.sim.plasticity.stdp_nonlinear exists.
"""

from __future__ import annotations

import importlib
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from snnlab import lang as snn
from snnlab import sim
from snnlab.sim.plasticity import stdp_simple as simple_kernel

KERNEL_MODULE = "snnlab.sim.plasticity.stdp_nonlinear"  # PROVISIONAL: location
try:
    kernel = importlib.import_module(KERNEL_MODULE)
except ModuleNotFoundError as error:
    # Skip only while the module itself is absent; any other import failure fails.
    if error.name not in {"snnlab.sim.plasticity", KERNEL_MODULE}:
        raise
    kernel = None

pytestmark = pytest.mark.skipif(
    kernel is None, reason="stdp_nonlinear is not implemented yet"
)

PAIRINGS = ("all_to_all", "nearest")
EXPONENTS = (0.0, 0.02, 0.5, 1.0)
DT_MS = 0.5
F64, F32 = torch.float64, torch.float32
TOL = {F64: 1e-12, F32: 1e-5}
# Distinct values, so a swapped amplitude, time constant or exponent fails.
DEFAULTS = {
    "a_plus": 0.01,
    "a_minus": 0.0105,
    "mu_plus": 0.3,
    "mu_minus": 0.7,
    "tau_plus_ms": 16.8,
    "tau_minus_ms": 33.7,
    "w_max": 1.0,
}
N_SOURCE, N_TARGET = 3, 2


# Helpers: the update step ----------------------------------------------------


def make_rule(**overrides):
    return kernel.STDPNonlinear(**{**DEFAULTS, **overrides})


def simple_rule(rule):
    """The stdp_simple rule with the same amplitudes, bounds and pairing."""
    return simple_kernel.STDPSimple(
        a_plus=rule.a_plus,
        a_minus=rule.a_minus,
        tau_plus_ms=rule.tau_plus_ms,
        tau_minus_ms=rule.tau_minus_ms,
        w_max=rule.w_max,
        w_min=rule.w_min,
        pairing=rule.pairing,
    )


def w_hat(rule, w):
    return (w - rule.w_min) / (rule.w_max - rule.w_min)


def ltp(rule, lag, w):
    """Potentiation of one pre-post pair at `lag` steps, from weight w."""
    factor = (1.0 - w_hat(rule, w)) ** rule.mu_plus
    return rule.a_plus * factor * math.exp(-lag * DT_MS / rule.tau_plus_ms)


def ltd(rule, lag, w):
    """Depression (a positive number) of one post-pre pair, from weight w."""
    factor = w_hat(rule, w) ** rule.mu_minus
    return rule.a_minus * factor * math.exp(-lag * DT_MS / rule.tau_minus_ms)


def full(shape, value, dtype=F64):
    return torch.full(shape, value, dtype=dtype)


def spike_tensor(steps_count, trains, *, dtype=F64):
    """(steps, batch, neurons) 0/1 tensor; trains[b][n] lists spike steps."""
    out = torch.zeros(steps_count, len(trains), len(trains[0]), dtype=dtype)
    for b, neurons in enumerate(trains):
        for n, steps in enumerate(neurons):
            for s in steps:
                out[s, b, n] = 1
    return out


def random_trains(generator, steps_count, *, batch, n, rate, dtype=F64):
    draws = torch.rand(steps_count, batch, n, generator=generator, dtype=F64)
    return (draws < rate).to(dtype)


def run(rule, w0, pre, post, *, mask=None, state=None, dt_ms=DT_MS, module=None):
    """Apply an update step over every time step.

    Returns (final weights, final state, end-of-step weight trajectory).
    `module` defaults to the stdp_nonlinear kernel; N12 and N17 pass stdp_simple.
    """
    module = kernel if module is None else module
    steps_count, batch, n_source = pre.shape
    if state is None:
        state = module.init_state(
            rule,
            batch=batch,
            n_source=n_source,
            n_target=post.shape[2],
            dtype=w0.dtype,
            device=w0.device,
        )
    weights, trajectory = w0, []
    for t in range(steps_count):
        weights, state = module.step(
            rule, state, weights, pre[t], post[t], dt_ms=dt_ms, mask=mask
        )
        trajectory.append(weights)
    return weights, state, torch.stack(trajectory)


def pair_dw(rule, pre_steps, post_steps, *, w0, dtype=F64):
    """Weight change of a single synapse for the given spike steps.

    The change is taken in float64 from the weights the step computed, so a
    float32 run is judged by its own rounding only.
    """
    steps_count = max([*pre_steps, *post_steps]) + 2
    pre = spike_tensor(steps_count, [[pre_steps]], dtype=dtype)
    post = spike_tensor(steps_count, [[post_steps]], dtype=dtype)
    start = full((1, 1, 1), w0, dtype=dtype)
    final, _, _ = run(rule, start, pre, post)
    return float(final[0, 0, 0]) - float(start[0, 0, 0])


def approx(expected, dtype=F64, scale=1.0):
    """Relative tolerance of the dtype, plus the rounding of a weight of `scale`."""
    return pytest.approx(
        expected, rel=TOL[dtype], abs=4 * torch.finfo(dtype).eps * scale
    )


def state_with_traces(x, y, dtype=F64):
    """A one-synapse state whose traces, after one step of decay, equal x and y."""
    return kernel.STDPState(
        pre_trace=torch.tensor(
            [[x * math.exp(DT_MS / DEFAULTS["tau_plus_ms"])]], dtype=dtype
        ),
        post_trace=torch.tensor(
            [[y * math.exp(DT_MS / DEFAULTS["tau_minus_ms"])]], dtype=dtype
        ),
    )


def event_reference(rule, pre_steps, post_steps, *, w0, steps_count):
    """Weight of one synapse computed from spike times, not traces (N23).

    Each spike applies one update with the current weight. Its pairing sum is
    built from the spike times: all earlier partner spikes (all-to-all) or the
    latest one (nearest). Spikes in the same step do not pair, and within a step
    depression comes before potentiation.
    """
    pre, post = set(pre_steps), set(post_steps)

    def window(earlier, t, tau_ms):
        if not earlier:
            return 0.0
        if rule.pairing == "nearest":
            earlier = [max(earlier)]
        return sum(math.exp(-(t - s) * DT_MS / tau_ms) for s in earlier)

    w = w0
    for t in range(steps_count):
        if t in pre:
            y = window([s for s in post if s < t], t, rule.tau_minus_ms)
            step = rule.a_minus * w_hat(rule, w) ** rule.mu_minus * y
            w = max(rule.w_min, w - step)
        if t in post:
            x = window([s for s in pre if s < t], t, rule.tau_plus_ms)
            step = rule.a_plus * (1.0 - w_hat(rule, w)) ** rule.mu_plus * x
            w = min(rule.w_max, w + step)
    return w


def window_sum(dt_ms, tau_ms):
    """Discrete window integral, sum over k >= 1 of dt exp(-k dt / tau).

    Same-step pairs are excluded, so this replaces tau in the equilibria.
    """
    return dt_ms / math.expm1(dt_ms / tau_ms)


def bisect(function, low=1e-12, high=1 - 1e-12):
    """Root of a function that changes sign once on [low, high]."""
    f_low = function(low)
    for _ in range(200):
        middle = 0.5 * (low + high)
        if (function(middle) > 0) == (f_low > 0):
            low, f_low = middle, function(middle)
        else:
            high = middle
    return 0.5 * (low + high)


# N05: the runtime rule and a functional step ----------------------------------


def test_n05_rule_mirrors_the_spec_and_step_is_functional():
    rule = kernel.STDPNonlinear(
        a_plus=0.01,
        a_minus=0.0105,
        mu_plus=0.3,
        mu_minus=0.7,
        tau_plus_ms=16.8,
        tau_minus_ms=33.7,
        w_max=1.0,
    )
    assert (rule.kind, rule.update, rule.bounds) == (
        "stdp_nonlinear",
        "nonlinear",
        "soft",
    )
    assert (rule.pairing, rule.w_min) == ("all_to_all", 0.0)
    assert set(kernel.PAIRINGS) == set(PAIRINGS)
    with pytest.raises(TypeError):
        kernel.STDPNonlinear(0.01, 0.0105, 0.3, 0.7, 16.8, 33.7, 1.0)  # keyword-only
    with pytest.raises(TypeError):
        make_rule(update="additive")  # the update is not a parameter
    with pytest.raises((AttributeError, TypeError)):
        rule.mu_plus = 0.5  # frozen
    for bad in ({"mu_plus": -0.1}, {"mu_minus": math.nan}, {"w_min": 1.0}):
        with pytest.raises(ValueError):
            make_rule(**bad)

    state = kernel.init_state(rule, batch=1, n_source=2, n_target=2, dtype=F64)
    weights = full((1, 2, 2), 0.5)
    pre = torch.tensor([[1.0, 0.0]], dtype=F64)
    post = torch.tensor([[0.0, 1.0]], dtype=F64)
    weights, state = kernel.step(rule, state, weights, pre, 0 * post, dt_ms=DT_MS)
    before = (
        weights.clone(),
        pre.clone(),
        post.clone(),
        state.pre_trace.clone(),
        state.post_trace.clone(),
    )
    new_weights, new_state = kernel.step(
        rule, state, weights, 0 * pre, post, dt_ms=DT_MS
    )
    assert torch.equal(weights, before[0])
    assert torch.equal(pre, before[1]) and torch.equal(post, before[2])
    assert torch.equal(state.pre_trace, before[3])
    assert torch.equal(state.post_trace, before[4])
    assert new_weights is not weights and not torch.equal(new_weights, weights)
    assert new_state is not state


# N06-N08: single pairs and the two exponents ---------------------------------


@pytest.mark.parametrize("pairing", PAIRINGS)
@pytest.mark.parametrize("mu_plus", EXPONENTS)
@pytest.mark.parametrize("w0", [0.0, 0.3, 0.9])
@pytest.mark.parametrize("lag", [1, 5, 80])
@pytest.mark.parametrize("dtype", [F64, F32], ids=["f64", "f32"])
def test_n06_single_pair_potentiation_is_exact(dtype, lag, w0, mu_plus, pairing):
    rule = make_rule(mu_plus=mu_plus, pairing=pairing)
    start = float(torch.tensor(w0, dtype=dtype))  # w0 as the dtype holds it
    dw = pair_dw(rule, pre_steps=[3], post_steps=[3 + lag], w0=w0, dtype=dtype)
    assert dw == approx(ltp(rule, lag, start), dtype, scale=1.0)


@pytest.mark.parametrize("pairing", PAIRINGS)
@pytest.mark.parametrize("mu_minus", EXPONENTS)
@pytest.mark.parametrize("w0", [1.0, 0.6, 0.1])
@pytest.mark.parametrize("lag", [1, 5, 80])
@pytest.mark.parametrize("dtype", [F64, F32], ids=["f64", "f32"])
def test_n07_single_pair_depression_is_exact(dtype, lag, w0, mu_minus, pairing):
    rule = make_rule(mu_minus=mu_minus, pairing=pairing)
    start = float(torch.tensor(w0, dtype=dtype))
    dw = pair_dw(rule, pre_steps=[3 + lag], post_steps=[3], w0=w0, dtype=dtype)
    assert dw == approx(-ltd(rule, lag, start), dtype, scale=1.0)


def test_n08_each_exponent_acts_on_its_own_side():
    rule = make_rule(mu_plus=0.3, mu_minus=0.7)
    w0, lag = 0.4, 10
    potentiation = pair_dw(rule, [3], [3 + lag], w0=w0)
    depression = pair_dw(rule, [3 + lag], [3], w0=w0)
    # Changing the other side's exponent changes nothing.
    assert pair_dw(make_rule(mu_minus=0.1), [3], [3 + lag], w0=w0) == potentiation
    assert pair_dw(make_rule(mu_plus=0.9), [3 + lag], [3], w0=w0) == depression
    # Each side uses its own exponent, not the other one.
    assert potentiation == approx(ltp(rule, lag, w0))
    assert depression == approx(-ltd(rule, lag, w0))
    swapped = make_rule(mu_plus=0.7, mu_minus=0.3)
    assert abs(ltp(swapped, lag, w0) - potentiation) > 1e-4  # a swap is visible
    assert abs(-ltd(swapped, lag, w0) - depression) > 1e-4


# N09-N12: same-step spikes, step order, traces --------------------------------


@pytest.mark.parametrize("mu", [0.0, 0.5])
@pytest.mark.parametrize("pairing", PAIRINGS)
def test_n09_spikes_in_the_same_step_do_not_pair(pairing, mu):
    rule = make_rule(mu_plus=mu, mu_minus=mu, pairing=pairing)
    assert pair_dw(rule, [5], [5], w0=0.4) == 0.0
    one_spike = spike_tensor(1, [[[0]]])
    _, state, _ = run(rule, full((1, 1, 1), 0.4), one_spike, one_spike)
    assert float(state.pre_trace[0, 0]) == 1.0  # both spikes entered the traces
    assert float(state.post_trace[0, 0]) == 1.0


@pytest.mark.parametrize("pairing", PAIRINGS)
def test_n10_same_step_spikes_still_pair_with_earlier_spikes(pairing):
    # NEST stdp_nn_symm_synapse example: pre {10, 20} and post {20} give one
    # pair, 10 -> 20, with the weight factor of the current weight.
    rule = make_rule(pairing=pairing)
    potentiation = pair_dw(rule, pre_steps=[10, 20], post_steps=[20], w0=0.4)
    depression = pair_dw(rule, pre_steps=[20], post_steps=[10, 20], w0=0.4)
    assert potentiation == approx(ltp(rule, 10, 0.4))
    assert depression == approx(-ltd(rule, 10, 0.4))


def test_n11_potentiation_uses_the_weight_after_depression():
    # One synapse whose pre and post neurons both spike in this step, with both
    # traces from earlier spikes. Depression comes first, with f_minus of the
    # current weight; potentiation then uses f_plus of the depressed weight.
    rule = make_rule(a_plus=0.2, a_minus=0.2)
    x, y, w0 = 0.9, 0.8, 0.5
    ones = torch.ones(1, 1, dtype=F64)
    weights, _ = kernel.step(
        rule, state_with_traces(x, y), full((1, 1, 1), w0), ones, ones, dt_ms=DT_MS
    )
    depressed = w0 - rule.a_minus * w_hat(rule, w0) ** rule.mu_minus * y
    sequential = (
        depressed + rule.a_plus * (1 - w_hat(rule, depressed)) ** rule.mu_plus * x
    )
    simultaneous = (
        w0
        - rule.a_minus * w_hat(rule, w0) ** rule.mu_minus * y
        + rule.a_plus * (1 - w_hat(rule, w0)) ** rule.mu_plus * x
    )
    assert float(weights) == approx(sequential)
    assert abs(sequential - simultaneous) > 1e-3  # the two orders differ visibly


@pytest.mark.parametrize("pairing", PAIRINGS)
def test_n12_traces_decay_exactly_and_equal_stdp_simple_traces(pairing):
    rule = make_rule(pairing=pairing)
    lag = 40
    one_spike = spike_tensor(lag + 1, [[[0]]])
    _, state, _ = run(rule, full((1, 1, 1), 0.5), one_spike, one_spike)
    assert float(state.pre_trace[0, 0]) == pytest.approx(
        math.exp(-lag * DT_MS / rule.tau_plus_ms), rel=1e-12
    )
    assert float(state.post_trace[0, 0]) == pytest.approx(
        math.exp(-lag * DT_MS / rule.tau_minus_ms), rel=1e-12
    )
    g = torch.Generator().manual_seed(12)
    pre = random_trains(g, 300, batch=2, n=3, rate=0.1)
    post = random_trains(g, 300, batch=2, n=2, rate=0.1)
    w0 = full((2, 3, 2), 0.5)
    _, mine, _ = run(rule, w0, pre, post)
    _, simple, _ = run(simple_rule(rule), w0, pre, post, module=simple_kernel)
    torch.testing.assert_close(mine.pre_trace, simple.pre_trace, rtol=1e-14, atol=0)
    torch.testing.assert_close(mine.post_trace, simple.post_trace, rtol=1e-14, atol=0)


# N13-N16: soft bounds, clipping, the factors at the bounds, w_min -------------


def test_n13_soft_bounds_follow_the_closed_form():
    # With mu = 1 and the same trace at every update, the distance to the bound
    # shrinks geometrically: 1 - w_hat_n = (1 - w_hat_0)(1 - a_hat x)^n for
    # potentiation, w_hat_n = w_hat_0 (1 - a_hat y)^n for depression. The bound
    # is approached but never reached.
    rule = make_rule(a_plus=0.05, a_minus=0.04, mu_plus=1.0, mu_minus=1.0, w_min=0.2)
    span = rule.w_max - rule.w_min
    x = y = 0.9
    n_updates = 150
    zero, one = torch.zeros(1, 1, dtype=F64), torch.ones(1, 1, dtype=F64)

    w, expected = full((1, 1, 1), 0.5), []
    for n in range(1, n_updates + 1):
        w, _ = kernel.step(rule, state_with_traces(x, 0.0), w, zero, one, dt_ms=DT_MS)
        gap = (1 - w_hat(rule, 0.5)) * (1 - rule.a_plus / span * x) ** n
        expected.append((float(w), rule.w_max - span * gap))
        assert float(w) < rule.w_max
    actual, closed = np.array(expected).T
    np.testing.assert_allclose(actual, closed, rtol=0, atol=1e-14)

    w, expected = full((1, 1, 1), 0.5), []
    for n in range(1, n_updates + 1):
        w, _ = kernel.step(rule, state_with_traces(0.0, y), w, one, zero, dt_ms=DT_MS)
        level = w_hat(rule, 0.5) * (1 - rule.a_minus / span * y) ** n
        expected.append((float(w), rule.w_min + span * level))
        assert float(w) > rule.w_min
    actual, closed = np.array(expected).T
    np.testing.assert_allclose(actual, closed, rtol=0, atol=1e-14)


def test_n14_clipping_catches_overshoot_without_nan():
    # With a small exponent the factor stays large close to a bound, so one
    # step can overshoot. The bound clips it exactly.
    rule = make_rule(a_plus=0.01, a_minus=0.01, mu_plus=0.02, mu_minus=0.02)
    ones, zero = torch.ones(1, 1, dtype=F64), torch.zeros(1, 1, dtype=F64)
    near_top = full((1, 1, 1), rule.w_max - 1e-4)
    w, _ = kernel.step(
        rule, state_with_traces(1.0, 0.0), near_top, zero, ones, dt_ms=DT_MS
    )
    assert float(w) == rule.w_max
    near_bottom = full((1, 1, 1), rule.w_min + 1e-4)
    w, _ = kernel.step(
        rule, state_with_traces(0.0, 1.0), near_bottom, ones, zero, dt_ms=DT_MS
    )
    assert float(w) == rule.w_min

    g = torch.Generator().manual_seed(14)
    pre = random_trains(g, 400, batch=2, n=3, rate=0.1)
    post = random_trains(g, 400, batch=2, n=4, rate=0.1)
    w0 = 0.1 + 0.5 * torch.rand(2, 3, 4, generator=g, dtype=F64)
    for mu in EXPONENTS:
        for pairing in PAIRINGS:
            loud = make_rule(
                a_plus=0.2,
                a_minus=0.2,
                mu_plus=mu,
                mu_minus=mu,
                w_min=0.1,
                w_max=0.6,
                pairing=pairing,
            )
            _, _, trajectory = run(loud, w0, pre, post)
            assert torch.isfinite(trajectory).all()
            assert float(trajectory.min()) >= 0.1
            assert float(trajectory.max()) <= 0.6


@pytest.mark.parametrize("mu", EXPONENTS)
def test_n15_factors_at_the_bounds(mu):
    # Every exponent, including 0, gives exact and finite results at the bounds.
    rule = make_rule(mu_plus=mu, mu_minus=mu, w_min=0.2)
    lag = 7
    # A causal pair at w_max and an acausal pair at w_min leave the weight there.
    assert pair_dw(rule, [3], [3 + lag], w0=rule.w_max) == 0.0
    assert pair_dw(rule, [3 + lag], [3], w0=rule.w_min) == 0.0
    # From the opposite bound the factor is 1: f_plus(0) = f_minus(1) = 1.
    assert pair_dw(rule, [3], [3 + lag], w0=rule.w_min) == approx(
        rule.a_plus * math.exp(-lag * DT_MS / rule.tau_plus_ms)
    )
    assert pair_dw(rule, [3 + lag], [3], w0=rule.w_max) == approx(
        -rule.a_minus * math.exp(-lag * DT_MS / rule.tau_minus_ms)
    )


def test_n16_factors_are_anchored_at_w_min():
    rule = make_rule(w_min=0.2, w_max=1.0)
    # Potentiation from w = 0.6, halfway between the bounds: w_hat = 0.5.
    dw = pair_dw(rule, [3], [13], w0=0.6)
    assert dw == approx(ltp(rule, 10, 0.6))
    assert dw == approx(
        rule.a_plus * 0.5**rule.mu_plus * math.exp(-10 * DT_MS / rule.tau_plus_ms)
    )
    # Just above w_min, depression is small and needs no clipping. A factor
    # computed from w / w_max would be large here and clip to w_min.
    start = rule.w_min + 1e-3
    dw = pair_dw(rule, [13], [3], w0=start)
    assert dw == approx(-ltd(rule, 10, start))
    assert start + dw > rule.w_min
    unanchored = rule.a_minus * (start / rule.w_max) ** rule.mu_minus
    assert unanchored * math.exp(-10 * DT_MS / rule.tau_minus_ms) > 1e-3


# N17-N18: equivalence with stdp_simple, pairing patterns ----------------------


@pytest.mark.parametrize("w_min", [0.0, 0.2])
@pytest.mark.parametrize("pairing", PAIRINGS)
def test_n17_zero_exponents_equal_stdp_simple(pairing, w_min):
    rule = make_rule(
        a_plus=0.05,
        a_minus=0.05,
        mu_plus=0.0,
        mu_minus=0.0,
        w_min=w_min,
        pairing=pairing,
    )
    g = torch.Generator().manual_seed(17)
    # A high rate, so many pre and post spikes share a step.
    pre = random_trains(g, 400, batch=2, n=3, rate=0.15)
    post = random_trains(g, 400, batch=2, n=2, rate=0.15)
    assert bool(((pre.amax(dim=2) > 0) & (post.amax(dim=2) > 0)).any())
    w0 = w_min + (1 - w_min) * torch.rand(2, 3, 2, generator=g, dtype=F64)
    mine, mine_state, mine_path = run(rule, w0, pre, post)
    simple, simple_state, simple_path = run(
        simple_rule(rule), w0, pre, post, module=simple_kernel
    )
    torch.testing.assert_close(mine_path, simple_path, rtol=1e-12, atol=1e-15)
    torch.testing.assert_close(mine_state.pre_trace, simple_state.pre_trace)
    torch.testing.assert_close(mine_state.post_trace, simple_state.post_trace)
    assert bool((simple_path == w_min).any()) and bool((simple_path == 1.0).any())


def test_n18_spike_patterns_separate_the_pairing_schemes():
    nearest = make_rule(pairing="nearest")
    all_to_all = make_rule(pairing="all_to_all")
    w0 = 0.4

    def k(lag, tau_ms):
        return math.exp(-lag * DT_MS / tau_ms)

    f_plus = (1 - w0) ** DEFAULTS["mu_plus"]
    f_minus = w0 ** DEFAULTS["mu_minus"]
    a_plus, a_minus = DEFAULTS["a_plus"], DEFAULTS["a_minus"]
    tau_p, tau_m = DEFAULTS["tau_plus_ms"], DEFAULTS["tau_minus_ms"]
    # Pre-pre-post: nearest pairs the post spike with the second pre spike only;
    # all-to-all with both, in one update with one weight factor.
    assert pair_dw(nearest, [0, 20], [30], w0=w0) == approx(
        a_plus * f_plus * k(10, tau_p)
    )
    assert pair_dw(all_to_all, [0, 20], [30], w0=w0) == approx(
        a_plus * f_plus * (k(10, tau_p) + k(30, tau_p))
    )
    # Post-post-pre: the same distinction for depression.
    assert pair_dw(nearest, [30], [0, 20], w0=w0) == approx(
        -a_minus * f_minus * k(10, tau_m)
    )
    assert pair_dw(all_to_all, [30], [0, 20], w0=w0) == approx(
        -a_minus * f_minus * (k(10, tau_m) + k(30, tau_m))
    )
    # Pre-post-post: each post spike pairs with the one pre spike, in two
    # updates; the second sees the weight after the first. Both schemes agree.
    first = w0 + a_plus * f_plus * k(10, tau_p)
    second = first + a_plus * (1 - first) ** DEFAULTS["mu_plus"] * k(30, tau_p)
    for rule in (nearest, all_to_all):
        assert pair_dw(rule, [0], [10, 30], w0=w0) == approx(second - w0)


# N19-N22: masks, locality, batches, continuation ------------------------------


def test_n19_masked_synapses_never_change():
    rule = make_rule(a_plus=0.05, a_minus=0.05)
    g = torch.Generator().manual_seed(19)
    pre = random_trains(g, 300, batch=1, n=3, rate=0.08)
    post = random_trains(g, 300, batch=1, n=3, rate=0.08)
    w0 = full((1, 3, 3), 0.5)
    w0[0, 0, 2] = 0.0  # a zero entry the user chose to freeze
    mask = ~torch.eye(3, dtype=torch.bool)  # no autapses
    mask[0, 2] = False
    masked, _, _ = run(rule, w0, pre, post, mask=mask)
    free, _, _ = run(rule, w0, pre, post)
    assert torch.equal(masked[0][~mask], w0[0][~mask])
    torch.testing.assert_close(masked[0][mask], free[0][mask], rtol=0, atol=0)
    assert not torch.equal(free[0][~mask], w0[0][~mask])
    # Unmasked, a zero weight grows at its first causal pair: f_plus(0) = 1.
    assert pair_dw(rule, [3], [8], w0=0.0) == approx(
        rule.a_plus * math.exp(-5 * DT_MS / rule.tau_plus_ms)
    )


@pytest.mark.parametrize("pairing", PAIRINGS)
def test_n20_each_synapse_follows_only_its_own_weight_and_neurons(pairing):
    rule = make_rule(a_plus=0.05, a_minus=0.05, pairing=pairing)
    g = torch.Generator().manual_seed(20)
    pre = random_trains(g, 200, batch=1, n=3, rate=0.06)
    post = random_trains(g, 200, batch=1, n=2, rate=0.06)
    w0 = 0.2 + 0.6 * torch.rand(1, 3, 2, generator=g, dtype=F64)
    final, _, _ = run(rule, w0, pre, post)
    assert final.shape == (1, 3, 2)
    for j in range(3):
        for i in range(2):
            alone, _, _ = run(
                rule,
                w0[:, j : j + 1, i : i + 1],
                pre[:, :, j : j + 1],
                post[:, :, i : i + 1],
            )
            torch.testing.assert_close(
                final[:, j, i], alone[:, 0, 0], rtol=1e-12, atol=0
            )
    transposed, _, _ = run(rule, w0, post[:, :, [0, 1, 1]], pre[:, :, :2])
    assert not torch.allclose(transposed, final)


def test_n21_batch_trials_are_independent():
    rule = make_rule(a_plus=0.05, a_minus=0.05)
    g = torch.Generator().manual_seed(21)
    pre = random_trains(g, 200, batch=4, n=3, rate=0.06)
    post = random_trains(g, 200, batch=4, n=2, rate=0.06)
    # Different initial weights per trial, so the weight factors differ too.
    w0 = 0.1 + 0.8 * torch.rand(4, 3, 2, generator=g, dtype=F64)
    batched, _, _ = run(rule, w0, pre, post)
    for b in range(4):
        alone, _, _ = run(rule, w0[b : b + 1], pre[:, b : b + 1], post[:, b : b + 1])
        torch.testing.assert_close(batched[b : b + 1], alone, rtol=0, atol=0)


@pytest.mark.parametrize("pairing", PAIRINGS)
def test_n22_a_run_split_in_two_chunks_equals_one_run(pairing):
    rule = make_rule(a_plus=0.05, a_minus=0.05, pairing=pairing)
    g = torch.Generator().manual_seed(22)
    pre = random_trains(g, 300, batch=2, n=3, rate=0.05)
    post = random_trains(g, 300, batch=2, n=2, rate=0.05)
    w0 = full((2, 3, 2), 0.5)
    whole, whole_state, _ = run(rule, w0, pre, post)
    first, first_state, _ = run(rule, w0, pre[:137], post[:137])
    second, second_state, _ = run(rule, first, pre[137:], post[137:], state=first_state)
    torch.testing.assert_close(second, whole, rtol=0, atol=0)
    torch.testing.assert_close(
        second_state.pre_trace, whole_state.pre_trace, rtol=0, atol=0
    )
    torch.testing.assert_close(
        second_state.post_trace, whole_state.post_trace, rtol=0, atol=0
    )


# N23-N24: independent reference, one-to-one ------------------------------------


@pytest.mark.parametrize(
    ("mu_plus", "mu_minus"),
    [(0.0, 0.0), (0.02, 0.02), (0.5, 0.5), (1.0, 1.0), (0.0, 1.0), (0.3, 0.7)],
)
@pytest.mark.parametrize("pairing", PAIRINGS)
def test_n23_matches_an_independent_event_reference(pairing, mu_plus, mu_minus):
    rule = make_rule(
        a_plus=0.03, a_minus=0.03, mu_plus=mu_plus, mu_minus=mu_minus, pairing=pairing
    )
    g = torch.Generator().manual_seed(23)
    steps_count = 400
    pre = random_trains(g, steps_count, batch=1, n=3, rate=0.06)
    post = random_trains(g, steps_count, batch=1, n=2, rate=0.06)
    w0 = 0.2 + 0.6 * torch.rand(1, 3, 2, generator=g, dtype=F64)
    final, _, _ = run(rule, w0, pre, post)
    for j in range(3):
        for i in range(2):
            expected = event_reference(
                rule,
                torch.nonzero(pre[:, 0, j]).flatten().tolist(),
                torch.nonzero(post[:, 0, i]).flatten().tolist(),
                w0=float(w0[0, j, i]),
                steps_count=steps_count,
            )
            assert float(final[0, j, i]) == pytest.approx(
                expected, rel=1e-12, abs=1e-14
            )


@pytest.mark.parametrize("pairing", PAIRINGS)
def test_n24_one_to_one_update_is_the_diagonal_of_the_dense_update(pairing):
    g = torch.Generator().manual_seed(24)
    n = 4
    pre = random_trains(g, 300, batch=2, n=n, rate=0.08)
    post = random_trains(g, 300, batch=2, n=n, rate=0.08)
    rule = make_rule(a_plus=0.05, a_minus=0.05, w_min=0.1, pairing=pairing)
    frozen = torch.tensor([True, True, False, True])
    w0 = 0.2 + 0.6 * torch.rand(2, n, generator=g, dtype=F64)
    dense, _, _ = run(rule, torch.diag_embed(w0), pre, post, mask=torch.diag(frozen))
    state = kernel.init_state(rule, batch=2, n_source=n, n_target=n)
    channel = w0
    for t in range(pre.shape[0]):
        channel, state = kernel.step_one_to_one(
            rule, state, channel, pre[t], post[t], dt_ms=DT_MS, mask=frozen
        )
    torch.testing.assert_close(
        channel, torch.diagonal(dense, dim1=-2, dim2=-1), rtol=0, atol=1e-15
    )
    assert torch.all(channel[:, 2] == w0[:, 2])  # the masked channel never moved


# N25-N26: dtypes and devices, convergence in dt -------------------------------


def _device(name, dtype):
    available = {
        "cpu": True,
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }[name]
    marks = [] if name == "cpu" else [pytest.mark.accelerator]
    marks.append(pytest.mark.skipif(not available, reason=f"{name} unavailable"))
    return pytest.param(name, dtype, id=f"{name}-{str(dtype)[6:]}", marks=marks)


# MPS has no float64, so it runs in float32 only.
DEVICE_CASES = [
    _device("cpu", F32),
    _device("cpu", F64),
    _device("mps", F32),
    _device("cuda", F32),
    _device("cuda", F64),
]


@pytest.mark.parametrize(("device", "dtype"), DEVICE_CASES)
def test_n25_dtypes_and_devices_match_the_reference(device, dtype):
    rule = make_rule(a_plus=0.05, a_minus=0.05, w_min=0.1)
    g = torch.Generator().manual_seed(25)
    pre = random_trains(g, 300, batch=2, n=3, rate=0.08)
    post = random_trains(g, 300, batch=2, n=2, rate=0.08)
    w0 = 0.1 + 0.9 * torch.rand(2, 3, 2, generator=g, dtype=F64)
    w0[0, 0, 0], w0[1, 2, 1] = 1.0 - 1e-6, 0.1 + 1e-6  # close to both bounds
    expected, _, _ = run(rule, w0, pre, post)
    actual, _, trajectory = run(
        rule,
        w0.to(device=device, dtype=dtype),
        pre.to(device=device, dtype=dtype),
        post.to(device=device, dtype=dtype),
    )
    assert actual.device.type == device and actual.dtype == dtype
    # Compared in the dtype the update step computes in.
    rtol, atol = {F32: (1e-5, 1e-6), F64: (1e-12, 1e-13)}[dtype]
    torch.testing.assert_close(actual.cpu(), expected.to(dtype), rtol=rtol, atol=atol)
    assert float(trajectory.max()) <= 1.0 and float(trajectory.min()) >= float(
        torch.tensor(0.1, dtype=dtype)
    )


def _binned_run(rule, times_pre, times_post, *, dt_ms, duration_ms, w0):
    steps_count = round(duration_ms / dt_ms)
    pre = torch.zeros(steps_count, 1, len(times_pre), dtype=F64)
    post = torch.zeros(steps_count, 1, len(times_post), dtype=F64)
    for out, times in ((pre, times_pre), (post, times_post)):
        for n, row in enumerate(times):
            steps = np.floor(row / dt_ms).astype(int)
            out[steps[steps < steps_count], 0, n] = 1  # merges same-step spikes
    final, _, _ = run(rule, w0, pre, post, dt_ms=dt_ms)
    return final


@pytest.mark.slow
def test_n26_halving_dt_converges_to_the_continuous_rule():
    # The same continuous-time spike trains, binned at dt, dt/2 and dt/4. The
    # binning moves each spike by less than one step and merges coincidences
    # into same-step spikes, which do not pair. Both effects are O(dt), so the
    # error against a fine-grid run should shrink about linearly with dt.
    rule = make_rule(a_plus=0.005, a_minus=0.00525)
    rng = np.random.default_rng(26)
    duration_ms, rate_hz, n = 4000.0, 20.0, 16

    def poisson(count):
        spikes = rng.random((count, int(duration_ms))) < rate_hz / 1000.0
        offsets = rng.random(spikes.shape)
        return [np.flatnonzero(row) + offsets[k][row] for k, row in enumerate(spikes)]

    times_pre, times_post = poisson(n), poisson(n)
    w0 = full((1, n, n), 0.5)
    reference = _binned_run(
        rule, times_pre, times_post, dt_ms=0.025, duration_ms=duration_ms, w0=w0
    )
    errors = [
        float(
            (
                _binned_run(
                    rule,
                    times_pre,
                    times_post,
                    dt_ms=dt,
                    duration_ms=duration_ms,
                    w0=w0,
                )
                - reference
            )
            .abs()
            .mean()
        )
        for dt in (0.4, 0.2, 0.1)
    ]
    # Observed ratios are about 0.5 per halving; the bounds leave a margin for
    # the seed and for the reference's own error.
    assert errors[1] <= 0.7 * errors[0]
    assert errors[2] <= 0.4 * errors[0]
    assert errors[2] < 1e-3  # mean error at dt = 0.1 ms, as a fraction of w_max


# N27-N28: analytic equilibria (slow, statistical) ------------------------------


EQUILIBRIUM_DT_MS = 1.0
EQUILIBRIUM_EXPONENTS = [(1.0, 1.0), (0.4, 0.4), (0.0, 1.0), (0.3, 0.7)]


def _drift_root(rule, *, dt_ms, causal=0.0):
    """Root of a_+ T_+ (1 - w)^mu_+ (1 + causal) = a_- T_- w^mu_-, T the window sums."""
    t_plus = window_sum(dt_ms, rule.tau_plus_ms)
    t_minus = window_sum(dt_ms, rule.tau_minus_ms)
    return bisect(
        lambda w: (
            rule.a_plus * t_plus * (1 - w) ** rule.mu_plus * (1 + causal)
            - rule.a_minus * t_minus * w**rule.mu_minus
        )
    )


@pytest.mark.slow
@pytest.mark.parametrize(("mu_plus", "mu_minus"), EQUILIBRIUM_EXPONENTS)
def test_n27_equilibrium_with_independent_poisson_trains(mu_plus, mu_minus):
    # Independent pre and post Poisson trains: no causal correlation, so the
    # mean weight settles where the mean drift vanishes,
    # a_+ T_+ (1 - w)^mu_+ = a_- T_- w^mu_-, with the discrete window sums T.
    rule = make_rule(a_plus=0.004, a_minus=0.0042, mu_plus=mu_plus, mu_minus=mu_minus)
    expected = _drift_root(rule, dt_ms=EQUILIBRIUM_DT_MS)
    g = torch.Generator().manual_seed(27)
    rate = 50.0 * EQUILIBRIUM_DT_MS / 1000.0
    n_source, n_target = 40, 50
    burn_in, measured = 30_000, 50_000
    state = kernel.init_state(rule, batch=1, n_source=n_source, n_target=n_target)
    w = full((1, n_source, n_target), 0.5)
    total, samples = 0.0, 0
    for t in range(burn_in + measured):
        pre = (torch.rand(1, n_source, generator=g, dtype=F64) < rate).to(F64)
        post = (torch.rand(1, n_target, generator=g, dtype=F64) < rate).to(F64)
        w, state = kernel.step(rule, state, w, pre, post, dt_ms=EQUILIBRIUM_DT_MS)
        if t >= burn_in and t % 100 == 0:
            total += float(w.mean())
            samples += 1
    # Observed errors are below 0.002 (finite amplitude, finite run).
    assert total / samples == pytest.approx(expected, abs=0.005)


@pytest.mark.slow
@pytest.mark.parametrize(("mu_plus", "mu_minus"), EQUILIBRIUM_EXPONENTS)
def test_n28_equilibrium_of_the_linear_poisson_neuron(mu_plus, mu_minus):
    # The linear Poisson neuron of Gütig et al. (2003, Eq. 5): each input spike
    # triggers an output spike one step later with probability w_hat_i / N. Its
    # own spike adds a causal pair, so the equilibrium moves up by
    # C_0 = exp(-dt / tau_+) / (T_+ r N), which tends to 1 / (tau_+ r N).
    rule = make_rule(a_plus=0.004, a_minus=0.0042, mu_plus=mu_plus, mu_minus=mu_minus)
    dt_ms, rate_hz, n_inputs, batch = EQUILIBRIUM_DT_MS, 50.0, 10, 200
    t_plus = window_sum(dt_ms, rule.tau_plus_ms)
    c0 = math.exp(-dt_ms / rule.tau_plus_ms) / (t_plus * rate_hz / 1000.0 * n_inputs)
    expected = _drift_root(rule, dt_ms=dt_ms, causal=c0)
    independent = _drift_root(rule, dt_ms=dt_ms)
    assert expected - independent > 0.02  # the causal term is visible
    if mu_plus == mu_minus:  # Eq. 19 of the paper, with the discrete window sums
        alpha = (
            rule.a_minus * window_sum(dt_ms, rule.tau_minus_ms) / (rule.a_plus * t_plus)
        )
        eq19 = 1 / (1 + (alpha / (1 + c0)) ** (1 / mu_plus))
        assert eq19 == pytest.approx(expected, abs=1e-9)
    g = torch.Generator().manual_seed(28)
    p = rate_hz * dt_ms / 1000.0
    burn_in, measured = 60_000, 60_000
    state = kernel.init_state(rule, batch=batch, n_source=n_inputs, n_target=1)
    w = full((batch, n_inputs, 1), 0.5)
    previous = torch.zeros(batch, n_inputs, dtype=F64)
    total, samples = 0.0, 0
    for t in range(burn_in + measured):
        triggered = torch.rand(batch, n_inputs, generator=g, dtype=F64) < (
            previous * w_hat(rule, w[:, :, 0]) / n_inputs
        )
        post = triggered.any(dim=1, keepdim=True).to(F64)
        pre = (torch.rand(batch, n_inputs, generator=g, dtype=F64) < p).to(F64)
        w, state = kernel.step(rule, state, w, pre, post, dt_ms=dt_ms)
        previous = pre
        if t >= burn_in and t % 100 == 0:
            total += float(w.mean())
            samples += 1
    # Observed errors are below 0.006; the causal shift is above 0.02, so the
    # tolerance separates this equilibrium from the independent one.
    assert total / samples == pytest.approx(expected, abs=0.01)


# N29-N30: independent simulators ------------------------------------------------


def _brian2_final_weights(b2, rule, pre, post, w0):
    """Final weights from Brian2 event-driven traces, as (source, target).

    Each pathway is split in two and the four parts are ordered as snnlab orders
    a step: depression, potentiation, then both trace jumps. Spikes in the same
    step then do not pair.
    """
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    dt = DT_MS * b2.ms
    steps_count, _, n_source = pre.shape
    n_target = post.shape[2]

    def generator(spikes, size):
        steps, cells = np.nonzero(spikes[:, 0].numpy())
        return b2.SpikeGeneratorGroup(size, cells, steps * dt, dt=dt)

    jump = "= 1" if rule.pairing == "nearest" else "+= 1"
    hat = "((w - w_min) / (w_max - w_min))"
    pre_group, post_group = generator(pre, n_source), generator(post, n_target)
    synapses = b2.Synapses(
        pre_group,
        post_group,
        model="""w : 1
                 dapre/dt = -apre / tau_plus : 1 (event-driven)
                 dapost/dt = -apost / tau_minus : 1 (event-driven)""",
        on_pre={
            "depress": f"w = clip(w - a_minus * {hat}**mu_minus * apost, w_min, w_max)",
            "pre_jump": f"apre {jump}",
        },
        on_post={
            "potentiate": f"w = clip(w + a_plus * (1 - {hat})**mu_plus * apre, w_min, w_max)",
            "post_jump": f"apost {jump}",
        },
        namespace={
            "tau_plus": rule.tau_plus_ms * b2.ms,
            "tau_minus": rule.tau_minus_ms * b2.ms,
            "a_plus": rule.a_plus,
            "a_minus": rule.a_minus,
            "mu_plus": rule.mu_plus,
            "mu_minus": rule.mu_minus,
            "w_min": rule.w_min,
            "w_max": rule.w_max,
        },
        dt=dt,
    )
    for order, name in enumerate(("depress", "potentiate", "pre_jump", "post_jump")):
        getattr(synapses, name).order = order
    synapses.connect()
    sources, targets = synapses.i[:], synapses.j[:]
    synapses.w[:] = w0[0].numpy()[sources, targets]
    network = b2.Network(pre_group, post_group, synapses)
    network.run((steps_count + 1) * dt)
    out = np.zeros((n_source, n_target))
    out[sources, targets] = synapses.w[:]
    return out


@pytest.mark.slow
@pytest.mark.brian2
@pytest.mark.parametrize(
    ("mu_plus", "mu_minus"), [(0.02, 0.02), (1.0, 1.0), (0.3, 0.7)]
)
@pytest.mark.parametrize("pairing", PAIRINGS)
def test_n29_matches_brian2(pairing, mu_plus, mu_minus):
    b2 = pytest.importorskip("brian2")
    rule = make_rule(
        a_plus=0.02, a_minus=0.021, mu_plus=mu_plus, mu_minus=mu_minus, pairing=pairing
    )
    g = torch.Generator().manual_seed(29)
    # A high rate, so same-step spikes occur and the ordered pathways matter.
    pre = random_trains(g, 2000, batch=1, n=3, rate=0.08)
    post = random_trains(g, 2000, batch=1, n=2, rate=0.08)
    w0 = 0.2 + 0.6 * torch.rand(1, 3, 2, generator=g, dtype=F64)
    expected = _brian2_final_weights(b2, rule, pre, post, w0)
    final, _, _ = run(rule, w0, pre, post)
    np.testing.assert_allclose(
        final[0].numpy(), expected, rtol=0, atol=1e-12 * rule.w_max
    )


def _nest_final_weights(nest, rule, pre, post, w0):
    """Final weights from NEST stdp_synapse or stdp_nn_symm_synapse.

    NEST has no lower bound parameter, so w_min = 0. NEST treats the delay as
    dendritic: with a delay of one step d, a post spike at t pairs with a pre
    spike at t - d. Pre spikes are therefore sent one step later than they
    arrive in snnlab. NEST updates weights only at pre spikes, so a last pre
    spike, 60 tau_minus after the end, applies the final potentiation; its
    depression is below 1e-26 of a_minus.
    """
    steps_count, _, n_source = pre.shape
    n_target = post.shape[2]
    end_ms = (steps_count + 1) * DT_MS + 60 * rule.tau_minus_ms
    nest.ResetKernel()
    nest.set_verbosity("M_ERROR")
    nest.resolution = DT_MS
    model = "stdp_synapse" if rule.pairing == "all_to_all" else "stdp_nn_symm_synapse"
    out = np.zeros((n_source, n_target))
    for j in range(n_source):
        pre_times = [
            (s + 2) * DT_MS for s in torch.nonzero(pre[:, 0, j]).flatten().tolist()
        ]
        pre_times.append(end_ms)
        for i in range(n_target):
            post_times = [
                (s + 1) * DT_MS for s in torch.nonzero(post[:, 0, i]).flatten().tolist()
            ]
            pre_source = nest.Create(
                "spike_generator", params={"spike_times": pre_times}
            )
            post_source = nest.Create(
                "spike_generator", params={"spike_times": post_times}
            )
            pre_parrot = nest.Create("parrot_neuron")
            post_parrot = nest.Create(
                "parrot_neuron", params={"tau_minus": rule.tau_minus_ms}
            )
            nest.Connect(pre_source, pre_parrot, syn_spec={"delay": DT_MS})
            nest.Connect(post_source, post_parrot, syn_spec={"delay": DT_MS})
            nest.Connect(
                pre_parrot,
                post_parrot,
                syn_spec={
                    "synapse_model": model,
                    "receptor_type": 1,  # parrot neurons ignore spikes at port 1
                    "delay": DT_MS,
                    "weight": float(w0[0, j, i]),
                    "Wmax": rule.w_max,
                    "lambda": rule.a_plus / rule.w_max,
                    "alpha": rule.a_minus / rule.a_plus,
                    "mu_plus": rule.mu_plus,
                    "mu_minus": rule.mu_minus,
                    "tau_plus": rule.tau_plus_ms,
                },
            )
            nest.Simulate(end_ms + 5 * DT_MS)
            connection = nest.GetConnections(pre_parrot, post_parrot)
            out[j, i] = connection.get("weight")
            nest.ResetKernel()
            nest.set_verbosity("M_ERROR")
            nest.resolution = DT_MS
    return out


@pytest.mark.slow
@pytest.mark.parametrize(
    ("mu_plus", "mu_minus"), [(0.02, 0.02), (1.0, 1.0), (0.3, 0.7)]
)
@pytest.mark.parametrize("pairing", PAIRINGS)
def test_n30_matches_nest_stdp_synapse(pairing, mu_plus, mu_minus):
    # Run once outside CI, where NEST is installed. NEST's own same-time
    # convention differs, so no pre and post spike share a step here.
    nest = pytest.importorskip("nest")
    rule = make_rule(
        a_plus=0.02, a_minus=0.021, mu_plus=mu_plus, mu_minus=mu_minus, pairing=pairing
    )
    g = torch.Generator().manual_seed(30)
    pre = random_trains(g, 1000, batch=1, n=3, rate=0.03)
    post = random_trains(g, 1000, batch=1, n=2, rate=0.03)
    post[pre.amax(dim=2, keepdim=True).expand_as(post) > 0] = 0
    w0 = 0.2 + 0.6 * torch.rand(1, 3, 2, generator=g, dtype=F64)
    expected = _nest_final_weights(nest, rule, pre, post, w0)
    final, _, _ = run(rule, w0, pre, post)
    np.testing.assert_allclose(final[0].numpy(), expected, rtol=1e-10, atol=1e-12)


# Helpers: the rule inside snnlab ------------------------------------------------


def lang_rule(**overrides):
    p = {**DEFAULTS, **overrides}
    return snn.STDPNonlinear(  # PROVISIONAL: name until review
        a_plus=p["a_plus"],
        a_minus=p["a_minus"],
        mu_plus=p["mu_plus"],
        mu_minus=p["mu_minus"],
        tau_plus=p["tau_plus_ms"] * snn.ms,
        tau_minus=p["tau_minus_ms"] * snn.ms,
        w_max=p["w_max"],
        **{
            key: p[key]
            for key in ("w_min", "pairing", "freeze_zero_weights")
            if key in p
        },
    )


def network(
    rule=None, *, n_source=N_SOURCE, n_target=N_TARGET, w0=0.5, delay_steps=0, **connect
):
    net = snn.Network("stdp_nonlinear", dt=DT_MS * snn.ms)
    pre = net.input(
        "pre", shape=("time", "batch", n_source), signal_type="spikes", unit="spike"
    )
    post = net.population("post", size=n_target, neuron=snn.CUBA_LIF())
    if rule is not None:
        connect["plasticity"] = rule
    net.connect(
        pre,
        post.excitatory,
        name="syn",
        synapse=snn.ExponentialCurrent(tau=2 * snn.ms),
        weight=snn.Constant(w0),
        initialization_scaling="direct",
        delay=delay_steps * DT_MS * snn.ms if delay_steps else None,
        **connect,
    )
    net.output("post_spikes", post.spikes)
    net.output("post_voltage", post.voltage)
    return net


def model_for(net, **executor):
    bundle = snn.compile(net, target=None)
    return sim.GraphExecutor(sim.plan_graph(bundle.graph), seed=0, **executor)


def replay(post, population="post"):
    replayed = sim.DenseSpikeReplay.from_tensor(post, dt_ms=DT_MS)
    return (sim.ReplaySpikes(population, replayed),)


def weights(result, name="syn", n_source=N_SOURCE, n_target=N_TARGET):
    """Final weights per trial as (batch, source, target)."""
    w = result.parameters[f"{name}.weight"]
    if w.ndim == 2:
        w = w.unsqueeze(0)
    if n_source != n_target and w.shape[-2:] == (n_target, n_source):
        w = w.transpose(-1, -2)
    return w


def trains(seed, steps, batch=1, n_source=N_SOURCE, n_target=N_TARGET, rate=0.05):
    g = torch.Generator().manual_seed(seed)
    pre = random_trains(g, steps, batch=batch, n=n_source, rate=rate)
    post = random_trains(g, steps, batch=batch, n=n_target, rate=rate)
    return pre, post


# N31-N35: the rule inside snnlab ------------------------------------------------

# The rule-agnostic integration tests of stdp_simple (T23-T49), which the rule
# interface makes runnable for any rule. PROVISIONAL: PR 2 may move the shared
# suite to another file; then update SHARED_SUITE.
SHARED_SUITE = "tests/sim/test_plasticity_stdp_simple.py"
RULE_AGNOSTIC = (
    "t23", "t24", "t26", "t27", "t28", "t29", "t30", "t31", "t32", "t35", "t37",
    "t38", "t39", "t40", "t41", "t42", "t43", "t44", "t45", "t46", "t47", "t48",
    "t49",
)  # fmt: skip


def test_n31_rule_agnostic_integration_suite_runs_for_both_rules():
    root = Path(__file__).resolve().parents[2]
    collected = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            SHARED_SUITE,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    for test_id in RULE_AGNOSTIC:
        ids = [
            match.group(1)
            for line in collected
            if f"::test_{test_id}_" in line and (match := re.search(r"\[(.*)\]$", line))
        ]
        assert any("stdp_simple" in i for i in ids), (
            f"{test_id} does not run for stdp_simple"
        )
        assert any("stdp_nonlinear" in i for i in ids), (
            f"{test_id} does not run for stdp_nonlinear"
        )


@pytest.mark.parametrize("pairing", PAIRINGS)
@pytest.mark.parametrize("connectivity", ["dense", "one_to_one"])
def test_n32_simulation_matches_the_update_step(connectivity, pairing):
    dense = connectivity == "dense"
    n_source, n_target = (N_SOURCE, N_TARGET) if dense else (2, 2)
    sizes = {"n_source": n_source, "n_target": n_target}
    extra = {} if dense else {"connectivity": snn.OneToOne()}
    delay = 2
    pre, post = trains(32, 300, batch=2, **sizes, rate=0.08)
    params = {"a_plus": 0.05, "a_minus": 0.05, "pairing": pairing}
    model = model_for(network(lang_rule(**params), **sizes, delay_steps=delay, **extra))
    whole = model({"pre": pre.float()}, interventions=replay(post.float()))

    # Pre spikes count at arrival, after the delay.
    arrivals = torch.cat([torch.zeros(delay, 2, n_source, dtype=F64), pre[:-delay]])
    mask = None if dense else torch.eye(2, dtype=torch.bool)
    w0 = torch.full((2, n_source, n_target), 0.5, dtype=F64)
    expected, _, _ = run(make_rule(**params), w0, arrivals, post, mask=mask)
    actual = weights(whole, **sizes).double()
    if not dense:  # OneToOne stores one weight per channel
        expected = torch.diagonal(expected, dim1=-2, dim2=-1)
        actual = actual.reshape(expected.shape)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    first = model({"pre": pre[:150].float()}, interventions=replay(post.float()))
    second = model(
        {"pre": pre[150:].float()},
        runtime_state=first.runtime_state,
        interventions=replay(post.float()),
    )
    torch.testing.assert_close(
        weights(second, **sizes), weights(whole, **sizes), rtol=0, atol=0
    )


def test_n33_freeze_zero_weights_keeps_structural_zeros():
    pre, post = torch.zeros(30, 1, N_SOURCE), torch.zeros(30, 1, N_TARGET)
    pre[5, 0, 0] = 1
    post[10, 0, 1] = 1
    # Without the freeze, a zero weight grows at its first causal pair.
    grown = weights(
        model_for(network(lang_rule(), w0=0.0))(
            {"pre": pre}, interventions=replay(post)
        )
    )
    expected = DEFAULTS["a_plus"] * math.exp(-5 * DT_MS / DEFAULTS["tau_plus_ms"])
    assert grown[0, 0, 1].item() == pytest.approx(expected, rel=1e-5)
    # With it, the weight stays 0, even below w_min = 0.2: frozen entries are
    # exempt from the bounds check.
    frozen = lang_rule(w_min=0.2, freeze_zero_weights=True)
    kept = weights(
        model_for(network(frozen, w0=0.0))({"pre": pre}, interventions=replay(post))
    )
    assert torch.all(kept == 0.0)


@pytest.mark.parametrize("connectivity", ["dense", "one_to_one"])
def test_n34_transmission_uses_the_learned_weights_from_the_previous_step(connectivity):
    # A learning synapse and a control synapse share the synapse model, so the
    # ratio of their current peaks after a probe pre spike equals the ratio of
    # their weights. Probes are far apart (97.5 ms; synapse tau 2 ms).
    dense = connectivity == "dense"
    n_source, n_target = (N_SOURCE, N_TARGET) if dense else (2, 2)
    sizes = {"n_source": n_source, "n_target": n_target}
    extra = {} if dense else {"connectivity": snn.OneToOne()}
    learner, control = (1, 0) if dense else (0, 1)
    pairing_step, post_step, probes = 5, 15, (200, 400)
    pre = torch.zeros(450, 1, n_source)
    post = torch.zeros(450, 1, n_target)
    for j in (0,) if dense else (0, 1):
        pre[[pairing_step, *probes], 0, j] = 1
    post[post_step, 0, learner] = 1
    params = {"a_plus": 0.2, "a_minus": 0.2}
    rule = make_rule(**params)
    model = model_for(network(lang_rule(**params), **sizes, **extra))
    recording = sim.RecordingSpec(signals=(sim.SignalRecording("syn.current"),))
    result = model({"pre": pre}, interventions=replay(post), recording=recording)
    current = result.recorded_signals["syn.current"][:, 0].abs()  # (time, target)

    w0 = 0.5
    # Learned at step 15 from the pair 5 -> 15, with f_plus of w0.
    potentiated = w0 + ltp(rule, post_step - pairing_step, w0)
    # The probe at step 200 also depresses, with f_minus of the potentiated
    # weight, but only after it has transmitted.
    depressed = potentiated - ltd(rule, probes[0] - post_step, potentiated)
    for probe, weight in zip(probes, (potentiated, depressed), strict=True):
        window = current[probe : probe + 40]
        ratio = float(window[:, learner].max() / window[:, control].max())
        assert ratio == pytest.approx(weight / w0, rel=1e-5)


def test_n35_saved_state_does_not_cross_rules(tmp_path):
    pre, post = trains(35, 60)
    pre, post = pre.float(), post.float()
    simple = snn.STDPSimple(
        a_plus=DEFAULTS["a_plus"],
        a_minus=DEFAULTS["a_minus"],
        tau_plus=DEFAULTS["tau_plus_ms"] * snn.ms,
        tau_minus=DEFAULTS["tau_minus_ms"] * snn.ms,
        w_max=DEFAULTS["w_max"],
    )

    def saved(rule, name):
        first = model_for(network(rule))({"pre": pre[:30]}, interventions=replay(post))
        return sim.load_runtime_state(
            sim.save_runtime_state(tmp_path / name, first.runtime_state)
        )

    from_simple, from_nonlinear = (
        saved(simple, "simple"),
        saved(lang_rule(), "nonlinear"),
    )
    cases = (
        (lang_rule(), from_simple),
        (simple, from_nonlinear),
        (lang_rule(mu_plus=0.0), from_nonlinear),
        (lang_rule(mu_minus=0.5), from_nonlinear),
        # Even with zero exponents, which update like stdp_simple, the rules differ.
        (lang_rule(mu_plus=0.0, mu_minus=0.0), from_simple),
    )
    for rule, state in cases:
        with pytest.raises(ValueError, match="incompatible.*plasticity"):
            model_for(network(rule))(
                {"pre": pre[30:]}, runtime_state=state, interventions=replay(post)
            )
    # The identical rule continues.
    model_for(network(lang_rule()))(
        {"pre": pre[30:]}, runtime_state=from_nonlinear, interventions=replay(post)
    )
