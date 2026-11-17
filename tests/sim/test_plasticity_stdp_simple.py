"""Acceptance tests for stdp_simple: pair-based, additive STDP with hard bounds.

The tests cover the update step and its conventions (T01-T21, T33, T36); the rule
inside the simulator (T23-T28, T31, T32, T35, T37); training and transmission
controls (T29, T38-T41); recording, persistence and learning frozen (T30,
T42-T46); devices, dtypes and the precision of the bounds (T47, T48); and
refusals with their cause (T49). The lang-side tests (T22, T25, T34) live in
tests/lang/test_snnlang.py.

Layers:
- T01-T20 and T33 drive the update step alone, on fixed spike trains.
- T21 compares the update step with an independent Brian2 model.
- T23-T32 and T35-T49 run the rule inside snnlab, with the update step as their
  reference. Post spikes are imposed with ReplaySpikes, so neuron dynamics play
  no part. T32 checks that transmission uses the learned weights.

Conventions: weights use the runtime layout (batch, source, target). Sizes are
non-square (3 sources, 2 targets), so a transposed layout fails. Defaults use
distinct amplitudes and time constants, so a swapped constant fails.
"""

from __future__ import annotations

import copy
import dataclasses
import math

import numpy as np
import pytest
import torch

from snnlab import lang as snn
from snnlab import sim
from snnlab._shared.plasticity import stdp_simple_from_json, unsupported_reasons
from snnlab.sim.plasticity import stdp_simple as kernel

PAIRINGS = ("nearest", "all_to_all")
DT_MS = 0.5
TOL = {torch.float64: 1e-12, torch.float32: 1e-5}
F64 = torch.float64
# Bi & Poo (1998) time constants; amplitudes differ so swaps fail.
DEFAULTS = {
    "a_plus": 0.01,
    "a_minus": 0.0105,
    "tau_plus_ms": 16.8,
    "tau_minus_ms": 33.7,
    "w_max": 1.0,
}
N_SOURCE, N_TARGET = 3, 2


# Helpers: the update step ----------------------------------------------------


def make_rule(**overrides):
    return kernel.STDPSimple(**{**DEFAULTS, **overrides})


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


def run(rule, w0, pre, post, *, mask=None, state=None):
    """Apply the update step over every time step.

    Returns (final weights, final state, end-of-step weight trajectory).
    """
    steps_count, batch, n_source = pre.shape
    if state is None:
        state = kernel.init_state(
            rule, batch=batch, n_source=n_source, n_target=post.shape[2], dtype=w0.dtype
        )
    weights, trajectory = w0, []
    for t in range(steps_count):
        weights, state = kernel.step(
            rule, state, weights, pre[t], post[t], dt_ms=DT_MS, mask=mask
        )
        trajectory.append(weights)
    return weights, state, torch.stack(trajectory)


def run_with_traces(rule, w0, pre, post):
    """Like run, but also returns the end-of-step traces."""
    state = kernel.init_state(
        rule, batch=pre.shape[1], n_source=pre.shape[2], n_target=post.shape[2]
    )
    weights, pre_traces, post_traces = w0, [], []
    for t in range(pre.shape[0]):
        weights, state = kernel.step(rule, state, weights, pre[t], post[t], dt_ms=DT_MS)
        pre_traces.append(state.pre_trace)
        post_traces.append(state.post_trace)
    return torch.stack(pre_traces), torch.stack(post_traces)


def pair_dw(rule, pre_steps, post_steps, *, w0, dtype=F64):
    """Weight change of a single synapse for the given spike steps."""
    steps_count = max([*pre_steps, *post_steps]) + 2
    pre = spike_tensor(steps_count, [[pre_steps]], dtype=dtype)
    post = spike_tensor(steps_count, [[post_steps]], dtype=dtype)
    start = torch.full((1, 1, 1), w0, dtype=dtype)
    final, _, _ = run(rule, start, pre, post)
    return float(final[0, 0, 0]) - float(start[0, 0, 0])


def ltp(rule, lag):
    return rule.a_plus * math.exp(-lag * DT_MS / rule.tau_plus_ms)


def ltd(rule, lag):
    return rule.a_minus * math.exp(-lag * DT_MS / rule.tau_minus_ms)


def full(shape, value):
    return torch.full(shape, value, dtype=F64)


def pairwise_delta_w(pre_steps, post_steps, *, rule):
    """Independent reference from spike times, ignoring bounds (T20).

    Spikes in the same step never pair. Nearest-spike pairing is symmetric.
    """
    pre, post = sorted(set(pre_steps)), sorted(set(post_steps))
    total = 0.0
    if rule.pairing == "all_to_all":
        total += sum(ltp(rule, tp - tq) for tp in post for tq in pre if tq < tp)
        total -= sum(ltd(rule, tq - tp) for tq in pre for tp in post if tp < tq)
        return total
    for tp in post:
        earlier = [tq for tq in pre if tq < tp]
        if earlier:
            total += ltp(rule, tp - max(earlier))
    for tq in pre:
        earlier = [tp for tp in post if tp < tq]
        if earlier:
            total -= ltd(rule, tq - max(earlier))
    return total


# T01-T03: spec, validation, functional step -----------------------------------


def test_t01_spec_states_additive_hard_bounds_and_defaults():
    rule = kernel.STDPSimple(
        a_plus=0.01, a_minus=0.0105, tau_plus_ms=16.8, tau_minus_ms=33.7, w_max=1.0
    )
    assert (rule.kind, rule.update, rule.bounds) == ("stdp_simple", "additive", "hard")
    assert rule.pairing == "all_to_all"
    assert rule.w_min == 0.0
    assert set(kernel.PAIRINGS) == set(PAIRINGS)
    assert rule.json() == {
        "kind": "stdp_simple",
        "update": "additive",
        "bounds": "hard",
        "pairing": "all_to_all",
        "a_plus": 0.01,
        "a_minus": 0.0105,
        "tau_plus_ms": 16.8,
        "tau_minus_ms": 33.7,
        "w_min": 0.0,
        "w_max": 1.0,
    }
    assert make_rule(pairing="nearest").pairing == "nearest"
    # One default in every layer: lang, a serialized spec without the field,
    # and the update step.
    lang_spec = lang_rule().json()
    assert lang_spec["pairing"] == "all_to_all"
    without_pairing = {k: v for k, v in lang_spec.items() if k != "pairing"}
    assert stdp_simple_from_json(without_pairing)["pairing"] == "all_to_all"
    assert make_rule(a_plus=0.0, a_minus=0.0).a_plus == 0.0
    with pytest.raises(TypeError):
        kernel.STDPSimple(0.01, 0.0105, 16.8, 33.7, 1.0)  # keyword-only
    with pytest.raises(TypeError):
        make_rule(update="multiplicative")  # additivity is not a parameter
    with pytest.raises(dataclasses.FrozenInstanceError):
        rule.a_plus = 0.02


@pytest.mark.parametrize(
    "override",
    [
        {"a_plus": -0.01},
        {"a_minus": -0.01},
        {"tau_plus_ms": 0.0},
        {"tau_minus_ms": -1.0},
        {"w_min": -0.1},
        {"w_min": 1.0},
        {"w_min": 0.6, "w_max": 0.5},
        {"pairing": "presynaptic_centred"},
        {"a_plus": math.nan},
        {"a_minus": math.inf},
        {"tau_plus_ms": math.inf},
        {"tau_minus_ms": math.nan},
        {"w_max": math.inf},
        {"w_min": math.nan},
    ],
    ids=[
        "a_plus<0",
        "a_minus<0",
        "tau_plus=0",
        "tau_minus<0",
        "w_min<0",
        "w_min=w_max",
        "w_min>w_max",
        "unknown_pairing",
        "a_plus=nan",
        "a_minus=inf",
        "tau_plus=inf",
        "tau_minus=nan",
        "w_max=inf",
        "w_min=nan",
    ],
)
def test_t02_invalid_parameters_raise(override):
    with pytest.raises(ValueError):
        make_rule(**override)


def test_t03_step_does_not_modify_its_inputs():
    rule = make_rule()
    state = kernel.init_state(rule, batch=1, n_source=2, n_target=2, dtype=F64)
    weights = full((1, 2, 2), 0.5)
    pre = torch.tensor([[1.0, 0.0]], dtype=F64)
    post = torch.tensor([[0.0, 1.0]], dtype=F64)
    weights, state = kernel.step(rule, state, weights, pre, 0 * post, dt_ms=DT_MS)
    before = (weights.clone(), state.pre_trace.clone(), state.post_trace.clone())
    new_weights, _ = kernel.step(rule, state, weights, 0 * pre, post, dt_ms=DT_MS)
    assert torch.equal(weights, before[0])
    assert torch.equal(state.pre_trace, before[1])
    assert torch.equal(state.post_trace, before[2])
    assert not torch.equal(new_weights, weights)


# T04-T08: single pairs, step order, traces ------------------------------------


@pytest.mark.parametrize("pairing", PAIRINGS)
@pytest.mark.parametrize("lag", [1, 2, 5, 20, 80])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
def test_t04_single_pair_potentiation_is_exact(dtype, lag, pairing):
    rule = make_rule(pairing=pairing)
    dw = pair_dw(rule, pre_steps=[3], post_steps=[3 + lag], w0=0.0, dtype=dtype)
    assert dw == pytest.approx(ltp(rule, lag), rel=TOL[dtype])


@pytest.mark.parametrize("pairing", PAIRINGS)
@pytest.mark.parametrize("lag", [1, 2, 5, 20, 80])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
def test_t05_single_pair_depression_is_exact(dtype, lag, pairing):
    rule = make_rule(pairing=pairing)
    # Start at a_minus, the largest single depression, so w_min is never hit.
    dw = pair_dw(
        rule, pre_steps=[3 + lag], post_steps=[3], w0=rule.a_minus, dtype=dtype
    )
    assert dw == pytest.approx(-ltd(rule, lag), rel=TOL[dtype])


@pytest.mark.parametrize("pairing", PAIRINGS)
def test_t06_spikes_in_the_same_step_do_not_pair_with_each_other(pairing):
    assert pair_dw(make_rule(pairing=pairing), [5], [5], w0=0.5) == 0.0


@pytest.mark.parametrize("pairing", PAIRINGS)
def test_t07_same_step_spikes_still_pair_with_earlier_spikes(pairing):
    # NEST stdp_nn_symm_synapse example: pre {10, 20} and post {20} give one
    # pair, 10 -> 20. The mirror case gives one depression pair.
    rule = make_rule(pairing=pairing)
    potentiation = pair_dw(rule, pre_steps=[10, 20], post_steps=[20], w0=0.5)
    depression = pair_dw(rule, pre_steps=[20], post_steps=[10, 20], w0=0.5)
    assert potentiation == pytest.approx(ltp(rule, 10), rel=1e-12)
    assert depression == pytest.approx(-ltd(rule, 10), rel=1e-12)


@pytest.mark.parametrize("pairing", PAIRINGS)
def test_t08_traces_decay_exactly_and_keep_same_step_spikes_for_later(pairing):
    rule = make_rule(pairing=pairing)
    lag = 40
    one_spike = spike_tensor(lag + 1, [[[0]]])
    _, state, _ = run(rule, full((1, 1, 1), 0.5), one_spike, one_spike)
    # The same-step pair at step 0 did not pair, but both spikes entered the traces.
    assert float(state.pre_trace[0, 0]) == pytest.approx(
        math.exp(-lag * DT_MS / rule.tau_plus_ms), rel=1e-12
    )
    assert float(state.post_trace[0, 0]) == pytest.approx(
        math.exp(-lag * DT_MS / rule.tau_minus_ms), rel=1e-12
    )
    two_spikes = spike_tensor(2, [[[0, 1]]])
    _, state, _ = run(rule, full((1, 1, 1), 0.5), two_spikes, two_spikes)
    carry = pairing == "all_to_all"
    expected_pre = 1.0 + carry * math.exp(-DT_MS / rule.tau_plus_ms)
    expected_post = 1.0 + carry * math.exp(-DT_MS / rule.tau_minus_ms)
    assert float(state.pre_trace[0, 0]) == pytest.approx(expected_pre, rel=1e-12)
    assert float(state.post_trace[0, 0]) == pytest.approx(expected_post, rel=1e-12)


# T09-T12: bounds and additivity -------------------------------------------------


def test_t09_potentiation_saturates_exactly_at_w_max():
    rule = make_rule(a_plus=0.05, a_minus=0.0)
    pre_steps = list(range(0, 400, 10))
    post_steps = [s + 1 for s in pre_steps]
    final, _, trajectory = run(
        rule,
        full((1, 1, 1), 0.9),
        spike_tensor(402, [[pre_steps]]),
        spike_tensor(402, [[post_steps]]),
    )
    assert float(final) == rule.w_max
    assert float(trajectory.max()) == rule.w_max


@pytest.mark.parametrize("w_min", [0.0, 0.2])
def test_t10_depression_stops_exactly_at_w_min(w_min):
    rule = make_rule(a_plus=0.0, a_minus=0.05, w_min=w_min)
    post_steps = list(range(0, 400, 10))
    pre_steps = [s + 1 for s in post_steps]
    final, _, trajectory = run(
        rule,
        full((1, 1, 1), 0.35),
        spike_tensor(402, [[pre_steps]]),
        spike_tensor(402, [[post_steps]]),
    )
    assert float(final) == w_min
    assert float(trajectory.min()) == w_min


@pytest.mark.parametrize("pairing", PAIRINGS)
def test_t11_weights_never_leave_the_bounds(pairing):
    rule = make_rule(a_plus=0.2, a_minus=0.2, w_min=0.1, w_max=0.6, pairing=pairing)
    g = torch.Generator().manual_seed(11)
    pre = random_trains(g, 400, batch=2, n=3, rate=0.1)
    post = random_trains(g, 400, batch=2, n=4, rate=0.1)
    w0 = 0.1 + 0.5 * torch.rand(2, 3, 4, generator=g, dtype=F64)
    _, _, trajectory = run(rule, w0, pre, post)
    assert float(trajectory.min()) >= 0.1
    assert float(trajectory.max()) <= 0.6
    assert bool((trajectory == 0.1).any())  # both bounds are reached
    assert bool((trajectory == 0.6).any())


@pytest.mark.parametrize("pairing", PAIRINGS)
def test_t12_weight_change_does_not_depend_on_the_weight(pairing):
    rule = make_rule(pairing=pairing)
    pre_steps, post_steps = [5, 31, 60], [12, 40, 52]
    expected = pairwise_delta_w(pre_steps, post_steps, rule=rule)
    for w0 in (0.1, 0.3, 0.5, 0.7):
        dw = pair_dw(rule, pre_steps, post_steps, w0=w0)
        assert dw == pytest.approx(expected, rel=1e-12, abs=1e-14)


# T13-T15: pairing schemes ------------------------------------------------------


def test_t13_pre_pre_post_separates_the_pairing_schemes():
    nearest = make_rule(pairing="nearest")
    all_to_all = make_rule(pairing="all_to_all")
    dw_nearest = pair_dw(nearest, pre_steps=[0, 20], post_steps=[30], w0=0.5)
    dw_all = pair_dw(all_to_all, pre_steps=[0, 20], post_steps=[30], w0=0.5)
    assert dw_nearest == pytest.approx(ltp(nearest, 10), rel=1e-12)
    expected_all = ltp(all_to_all, 10) + ltp(all_to_all, 30)
    assert dw_all == pytest.approx(expected_all, rel=1e-12)


def test_t14_post_post_pre_separates_the_pairing_schemes():
    nearest = make_rule(pairing="nearest")
    all_to_all = make_rule(pairing="all_to_all")
    dw_nearest = pair_dw(nearest, pre_steps=[30], post_steps=[0, 20], w0=0.5)
    dw_all = pair_dw(all_to_all, pre_steps=[30], post_steps=[0, 20], w0=0.5)
    assert dw_nearest == pytest.approx(-ltd(nearest, 10), rel=1e-12)
    expected_all = -ltd(all_to_all, 10) - ltd(all_to_all, 30)
    assert dw_all == pytest.approx(expected_all, rel=1e-12)


def test_t15_symmetric_nearest_pairs_every_post_spike_with_the_latest_pre_spike():
    # Pre at 0, posts at 10 and 30: each post pairs with the same pre spike, so
    # both schemes agree. A presynaptic-centred scheme would pair it only once.
    expected = ltp(make_rule(), 10) + ltp(make_rule(), 30)
    for pairing in PAIRINGS:
        dw = pair_dw(make_rule(pairing=pairing), [0], [10, 30], w0=0.5)
        assert dw == pytest.approx(expected, rel=1e-12)


# T16-T17: masks and locality ---------------------------------------------------


def test_t16_masked_synapses_never_change():
    rule = make_rule(a_plus=0.05, a_minus=0.05)
    g = torch.Generator().manual_seed(16)
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


@pytest.mark.parametrize("pairing", PAIRINGS)
def test_t17_each_synapse_follows_only_its_own_pre_and_post_neuron(pairing):
    rule = make_rule(pairing=pairing)
    g = torch.Generator().manual_seed(17)
    pre = random_trains(g, 200, batch=1, n=3, rate=0.05)
    post = random_trains(g, 200, batch=1, n=2, rate=0.05)
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


# T18-T20: batch, continuation, independent reference ---------------------------


def test_t18_batch_trials_are_independent():
    rule = make_rule()
    g = torch.Generator().manual_seed(18)
    pre = random_trains(g, 200, batch=4, n=3, rate=0.05)
    post = random_trains(g, 200, batch=4, n=2, rate=0.05)
    w0 = 0.2 + 0.6 * torch.rand(1, 3, 2, generator=g, dtype=F64)
    w0 = w0.expand(4, 3, 2).clone()
    batched, _, _ = run(rule, w0, pre, post)
    for b in range(4):
        alone, _, _ = run(rule, w0[b : b + 1], pre[:, b : b + 1], post[:, b : b + 1])
        torch.testing.assert_close(batched[b : b + 1], alone, rtol=0, atol=0)


@pytest.mark.parametrize("pairing", PAIRINGS)
def test_t19_a_run_split_in_two_chunks_equals_one_run(pairing):
    rule = make_rule(a_plus=0.05, a_minus=0.05, pairing=pairing)
    g = torch.Generator().manual_seed(19)
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


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("pairing", PAIRINGS)
def test_t20_matches_an_independent_pairwise_reference(pairing, seed):
    rule = make_rule(a_plus=1e-3, a_minus=1.05e-3, pairing=pairing)
    g = torch.Generator().manual_seed(seed)
    pre = random_trains(g, 300, batch=1, n=3, rate=0.04)
    post = random_trains(g, 300, batch=1, n=2, rate=0.04)
    final, _, trajectory = run(rule, full((1, 3, 2), 0.5), pre, post)
    assert float(trajectory.min()) > 0.0 and float(trajectory.max()) < 1.0
    for j in range(3):
        for i in range(2):
            expected = pairwise_delta_w(
                torch.nonzero(pre[:, 0, j]).flatten().tolist(),
                torch.nonzero(post[:, 0, i]).flatten().tolist(),
                rule=rule,
            )
            dw = float(final[0, j, i]) - 0.5
            assert dw == pytest.approx(expected, rel=1e-10, abs=1e-13)


# T33: where clipping happens ---------------------------------------------------


@pytest.mark.parametrize(
    ("w0", "depression", "potentiation", "expected"),
    [
        # Lower bound: clipping in sequence gives 0.08; clipping the combined
        # update would give 0.05 - 0.10 + 0.08 = 0.03.
        (0.05, 0.10, 0.08, 0.08),
        # Upper bound: depression first gives 0.95 - 0.08 + 0.10 = 0.97;
        # potentiating first would clip at 1.0 and end at 0.92.
        (0.95, 0.08, 0.10, 0.97),
    ],
    ids=["lower_bound_sequential", "upper_bound_depression_first"],
)
def test_t33_clipping_is_sequential_with_depression_first(
    w0, depression, potentiation, expected
):
    # One synapse whose pre and post neurons both spike in this step, with both
    # traces at 1 from earlier spikes. Amplitudes are chosen so that, after the
    # one-step decay, depression and potentiation have the sizes above.
    x = math.exp(-DT_MS / DEFAULTS["tau_plus_ms"])
    y = math.exp(-DT_MS / DEFAULTS["tau_minus_ms"])
    rule = make_rule(a_plus=potentiation / x, a_minus=depression / y)
    ones = torch.ones(1, 1, dtype=F64)
    state = kernel.STDPState(pre_trace=ones, post_trace=ones)
    weights, _ = kernel.step(rule, state, full((1, 1, 1), w0), ones, ones, dt_ms=DT_MS)
    assert float(weights) == pytest.approx(expected, rel=1e-12)


# T36: the channel-wise update of one-to-one projections -----------------------


@pytest.mark.parametrize("pairing", PAIRINGS)
def test_t36_one_to_one_update_is_the_diagonal_of_the_dense_update(pairing):
    # Synapse i joins source i to target i only. The channel-wise step must give
    # exactly the diagonal of the dense step under an identity mask, including
    # a frozen channel.
    g = torch.Generator().manual_seed(36)
    n = 4
    pre = random_trains(g, 300, batch=2, n=n, rate=0.08)
    post = random_trains(g, 300, batch=2, n=n, rate=0.08)
    rule = make_rule(a_plus=0.05, a_minus=0.05, w_min=0.1, pairing=pairing)
    frozen = torch.tensor([True, True, False, True])
    w0 = 0.2 + 0.6 * torch.rand(2, n, generator=g, dtype=F64)
    dense, _, _ = run(
        rule,
        torch.diag_embed(w0),
        pre,
        post,
        mask=torch.diag(frozen),
    )
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


# T21: independent Brian2 model --------------------------------------------------


def _brian2_final_weights(b2, rule, pre, post, w0):
    """Final weights from Brian2 event-driven traces, as (source, target)."""
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    dt = DT_MS * b2.ms
    steps_count, _, n_source = pre.shape
    n_target = post.shape[2]

    def generator(spikes, size):
        steps, cells = np.nonzero(spikes[:, 0].numpy())
        return b2.SpikeGeneratorGroup(size, cells, steps * dt, dt=dt)

    jump = "= 1" if rule.pairing == "nearest" else "+= 1"
    pre_group, post_group = generator(pre, n_source), generator(post, n_target)
    synapses = b2.Synapses(
        pre_group,
        post_group,
        model="""w : 1
                 dapre/dt = -apre / tau_plus : 1 (event-driven)
                 dapost/dt = -apost / tau_minus : 1 (event-driven)""",
        on_pre=f"""w = clip(w - a_minus * apost, w_min, w_max)
                   apre {jump}""",
        on_post=f"""w = clip(w + a_plus * apre, w_min, w_max)
                    apost {jump}""",
        namespace={
            "tau_plus": rule.tau_plus_ms * b2.ms,
            "tau_minus": rule.tau_minus_ms * b2.ms,
            "a_plus": rule.a_plus,
            "a_minus": rule.a_minus,
            "w_min": rule.w_min,
            "w_max": rule.w_max,
        },
        dt=dt,
    )
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
@pytest.mark.parametrize("pairing", PAIRINGS)
def test_t21_matches_brian2(pairing):
    b2 = pytest.importorskip("brian2")
    rule = make_rule(a_plus=0.02, a_minus=0.021, pairing=pairing)
    g = torch.Generator().manual_seed(21)
    pre = random_trains(g, 2000, batch=1, n=3, rate=0.03)
    post = random_trains(g, 2000, batch=1, n=2, rate=0.03)
    # Same-step order is a convention (Q6) that Brian2 cannot arbitrate.
    post[pre.amax(dim=2, keepdim=True).expand_as(post) > 0] = 0
    w0 = 0.2 + 0.6 * torch.rand(1, 3, 2, generator=g, dtype=F64)
    expected = _brian2_final_weights(b2, rule, pre, post, w0)
    final, _, _ = run(rule, w0, pre, post)
    np.testing.assert_allclose(final[0].numpy(), expected, rtol=1e-9, atol=1e-12)


# Helpers: the rule inside snnlab ------------------------------------------------


def lang_rule(**overrides):
    p = {**DEFAULTS, **overrides}
    return snn.STDPSimple(
        a_plus=p["a_plus"],
        a_minus=p["a_minus"],
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
    rule=None,
    *,
    n_source=N_SOURCE,
    n_target=N_TARGET,
    w0=0.5,
    delay_steps=0,
    **connect,
):
    net = snn.Network("stdp_simple", dt=DT_MS * snn.ms)
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


def as_batch_source_target(w, n_source=N_SOURCE, n_target=N_TARGET):
    """Final weights per trial as (batch, source, target)."""
    if w.ndim == 2:
        w = w.unsqueeze(0)
    if n_source != n_target and w.shape[-2:] == (n_target, n_source):
        w = w.transpose(-1, -2)
    return w


def weights(result, name="syn", **sizes):
    return as_batch_source_target(result.parameters[f"{name}.weight"], **sizes)


def trains(seed, steps, batch=1, n_source=N_SOURCE, n_target=N_TARGET, rate=0.05):
    g = torch.Generator().manual_seed(seed)
    pre = random_trains(g, steps, batch=batch, n=n_source, rate=rate)
    post = random_trains(g, steps, batch=batch, n=n_target, rate=rate)
    return pre, post


# T23-T24: compatibility and bounds ------------------------------------------------


def test_t23_zero_amplitudes_reproduce_the_static_network_and_runs_never_write_back():
    pre, post = trains(23, 200, batch=2)
    pre, post = pre.float(), post.float()
    static = model_for(network())({"pre": pre}, interventions=replay(post))
    frozen_rule = lang_rule(a_plus=0.0, a_minus=0.0)
    frozen = model_for(network(frozen_rule))({"pre": pre}, interventions=replay(post))
    for name, value in static.outputs.items():
        torch.testing.assert_close(frozen.outputs[name], value, rtol=0, atol=0)
    bundle = snn.compile(network(lang_rule(a_plus=0.05, a_minus=0.05)), target=None)
    graph = copy.deepcopy(bundle.graph)
    model = sim.GraphExecutor(sim.plan_graph(bundle.graph), seed=0)
    first = model({"pre": pre}, interventions=replay(post))
    second = model({"pre": pre}, interventions=replay(post))
    torch.testing.assert_close(weights(first), weights(second), rtol=0, atol=0)
    assert not torch.all(weights(first) == 0.5)  # the rule did act
    assert torch.all(model.parameter_map()["syn.weight"] == 0.5)  # model untouched
    assert bundle.graph == graph  # bundle untouched


def _recurrent_e_to_e(rule):
    """E -> E with autapses excluded: the diagonal is a structural zero."""
    net = snn.Network("stdp_recurrent", dt=DT_MS * snn.ms)
    pre = net.input(
        "pre", shape=("time", "batch", 3), signal_type="spikes", unit="spike"
    )
    e = net.population("E", size=3, neuron=snn.CUBA_LIF())
    net.connect(
        pre,
        e.excitatory,
        name="drive",
        synapse=snn.ExponentialCurrent(tau=2 * snn.ms),
        weight=snn.Constant(0.5),
        initialization_scaling="direct",
    )
    net.connect(
        e.spikes,
        e.excitatory,
        name="syn",
        synapse=snn.ExponentialCurrent(tau=2 * snn.ms),
        weight=snn.Constant(0.5),
        initialization_scaling="direct",
        connection="recurrent",
        delay=DT_MS * snn.ms,
        plasticity=rule,
    )
    net.output("E_spikes", e.spikes)
    return net


def test_t24_bounds_are_checked_on_plastic_entries_only():
    pre, post = torch.zeros(20, 1, N_SOURCE), torch.zeros(20, 1, N_TARGET)
    # A plastic entry outside the bounds raises before the run.
    with pytest.raises(ValueError, match="w_max|w_min|bound"):
        model_for(network(lang_rule(w_max=1.0), w0=1.5))(
            {"pre": pre}, interventions=replay(post)
        )
    # Excluded autapses are structural zeros: below w_min = 0.2, yet allowed,
    # and they stay exactly zero.
    rule = lang_rule(w_min=0.2, a_plus=0.05, a_minus=0.05)
    pre3, post3 = trains(24, 200, n_source=3, n_target=3, rate=0.08)
    result = model_for(_recurrent_e_to_e(rule))(
        {"pre": pre3.float()}, interventions=replay(post3.float(), population="E")
    )
    recurrent = weights(result, n_source=3, n_target=3)[0]
    assert torch.all(torch.diagonal(recurrent) == 0.0)
    assert torch.all(recurrent[~torch.eye(3, dtype=torch.bool)] >= 0.2)
    # Opt-in frozen zeros: below w_min, allowed, unchanged.
    frozen = lang_rule(w_min=0.2, freeze_zero_weights=True)
    result = model_for(network(frozen, w0=0.0))(
        {"pre": pre}, interventions=replay(post)
    )
    assert torch.all(weights(result) == 0.0)


# T26-T31: delays, execution, training, recording, disabled projections ------------


@pytest.mark.parametrize("delay_steps", [0, 4])
def test_t26_pre_spikes_count_at_arrival_after_the_delay(delay_steps):
    lag = 10  # the post spike comes 10 steps after the pre spike is emitted
    pre, post = torch.zeros(30, 1, N_SOURCE), torch.zeros(30, 1, N_TARGET)
    pre[5, 0, 0] = 1
    post[5 + lag, 0, 1] = 1
    net = network(lang_rule(), w0=0.0, delay_steps=delay_steps)
    final = weights(model_for(net)({"pre": pre}, interventions=replay(post)))
    expected = DEFAULTS["a_plus"] * math.exp(
        -(lag - delay_steps) * DT_MS / DEFAULTS["tau_plus_ms"]
    )
    assert final[0, 0, 1].item() == pytest.approx(expected, rel=1e-5)
    final[0, 0, 1] = 0.0
    assert torch.all(final == 0.0)  # no other synapse moved


@pytest.mark.parametrize("pairing", PAIRINGS)
@pytest.mark.parametrize("connectivity", ["dense", "one_to_one"])
def test_t27_simulation_matches_the_update_step_and_resumes_from_runtime_state(
    connectivity, pairing
):
    dense = connectivity == "dense"
    n_source, n_target = (N_SOURCE, N_TARGET) if dense else (2, 2)
    sizes = {"n_source": n_source, "n_target": n_target}
    extra = {} if dense else {"connectivity": snn.OneToOne()}
    pre, post = trains(27, 300, batch=2, **sizes)
    params = {"a_plus": 0.05, "a_minus": 0.05, "pairing": pairing}
    model = model_for(network(lang_rule(**params), **sizes, **extra))
    whole = model({"pre": pre.float()}, interventions=replay(post.float()))

    mask = None if dense else torch.eye(2, dtype=torch.bool)
    w0 = torch.full((2, n_source, n_target), 0.5, dtype=F64)
    expected, _, _ = run(make_rule(**params), w0, pre, post, mask=mask)
    actual = weights(whole, **sizes).double()
    if not dense:  # OneToOne stores one weight per channel: (batch, 1, channels)
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


def test_t28_plasticity_selects_the_step_by_step_path_and_says_why():
    pre, post = torch.zeros(16, 1, N_SOURCE), torch.zeros(16, 1, N_TARGET)
    model = model_for(network(lang_rule()), compile_block_steps=8)
    path = model({"pre": pre}, interventions=replay(post)).metrics["execution_path"]
    assert path["selected"] == "kernel_loop"
    assert any("plastic" in reason for reason in path["fallback_reasons"])


def test_t29_training_keeps_plasticity_off_and_runtime_weights_win_over_checkpoint(
    tmp_path,
):
    steps, samples = 20, 4

    def trainable(rule):
        net = snn.Network("stdp_train", dt=DT_MS * snn.ms)
        source = net.input(
            "pre",
            shape=("time", "batch", N_SOURCE),
            signal_type="spikes",
            unit="spike",
        )
        cells = net.population("post", size=N_TARGET, neuron=snn.CUBA_LIF())
        extra = {} if rule is None else {"plasticity": rule}
        net.connect(
            source,
            cells.excitatory,
            name="syn",
            synapse=snn.ExponentialCurrent(tau=3 * snn.ms),
            weight=snn.Constant(0.5),
            initialization_scaling="direct",
            constraint=snn.NonNegative(),  # allowed: constraints act only in training
            **extra,
        )
        scores = snn.MeanVoltage(source=cells.spikes, classes=2, name="scores")
        net.output("prediction", scores)
        recipe = snn.TrainSpec(
            objectives=(snn.CrossEntropy(prediction=scores, target="label"),),
            parameter_groups=(
                snn.ParameterGroup(
                    [row["id"] for row in net.parameters], name="all", lr=0.01
                ),
            ),
            optimizer=snn.AdamW(weight_decay=0.0),
            presentation_duration=steps * DT_MS * snn.ms,
        )
        return snn.compile(net, training=recipe)

    g = torch.Generator().manual_seed(29)
    values = (torch.rand(steps, samples, N_SOURCE, generator=g) < 0.3).float()
    common = {
        "input_bindings": (sim.DenseArrayBinding("pre", values),),
        "targets": {"label": torch.arange(samples) % 2},
        "device": "cpu",
        "seed": 7,
        "epochs": 2,
        "batch_size": 2,
        "diagnostics": False,
        "performance": sim.Performance(precision="highest"),
    }
    rule_params = {"a_plus": 0.05, "a_minus": 0.05}
    static_bundle = trainable(None)
    plastic_bundle = trainable(lang_rule(**rule_params))
    static = sim.train(
        graph=static_bundle.graph, training=static_bundle.training, **common
    )
    checkpoint = tmp_path / "trained"
    plastic = sim.train(
        graph=plastic_bundle.graph,
        training=plastic_bundle.training,
        save_final_checkpoint=checkpoint,
        **common,
    )
    trained = plastic.parameters["syn.weight"]
    # Training is identical with and without plasticity: STDP is off in train.
    torch.testing.assert_close(trained, static.parameters["syn.weight"], rtol=0, atol=0)

    # The checkpoint loads into the same plastic graph; STDP starts from it.
    post = (torch.rand(steps, 1, N_TARGET, generator=g) < 0.2).float()

    def simulate(chunk, **extra):
        return sim.simulate(
            graph=plastic_bundle.graph,
            checkpoint=checkpoint,
            input_bindings=(sim.DenseArrayBinding("pre", values[chunk, :1]),),
            interventions=replay(post),
            device="cpu",
            **extra,
        )

    whole = simulate(slice(None))
    start = as_batch_source_target(trained).double()
    expected, _, _ = run(
        make_rule(**rule_params), start, values[:, :1].double(), post.double()
    )
    torch.testing.assert_close(weights(whole).double(), expected, rtol=1e-5, atol=1e-6)

    # Checkpoint and RuntimeState together: the runtime weights win for plastic
    # projections, so continuation never restarts from the checkpoint.
    first = simulate(slice(0, steps // 2))
    second = simulate(slice(steps // 2, None), runtime_state=first.runtime_state)
    torch.testing.assert_close(weights(second), weights(whole), rtol=0, atol=0)


def test_t30_recordings_follow_the_update_and_one_schedule_across_chunks():
    pre, post = trains(30, 100)
    pre, post = pre.float(), post.float()
    params = {"a_plus": 0.05, "a_minus": 0.05}
    model = model_for(network(lang_rule(**params)))
    recording = sim.RecordingSpec(
        signals=(
            sim.SignalRecording("syn.weight", every_steps=10),
            sim.SignalRecording("syn.pre_trace"),
            sim.SignalRecording("syn.post_trace"),
        )
    )
    whole = model({"pre": pre}, interventions=replay(post), recording=recording)
    w0 = torch.full((1, N_SOURCE, N_TARGET), 0.5, dtype=F64)
    _, _, trajectory = run(make_rule(**params), w0, pre.double(), post.double())
    pre_traces, post_traces = run_with_traces(
        make_rule(**params), w0, pre.double(), post.double()
    )

    # Weights: (snapshot, batch, source, target), taken after each step's update,
    # at global steps 9, 19, ..., 99. The last equals the final weights. The
    # steps are the recording's time coordinates, as for every recording.
    snapshots = whole.recorded_signals["syn.weight"]
    steps = whole.numpy().recording.info["syn.weight"].step_indices
    assert snapshots.shape == (10, 1, N_SOURCE, N_TARGET)
    assert list(steps) == list(range(9, 100, 10))
    torch.testing.assert_close(
        snapshots.double(), trajectory[9::10], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(snapshots[-1], weights(whole), rtol=0, atol=0)

    # Traces: (time, batch, neurons), every step, after the step's spikes.
    pre_recorded = whole.recorded_signals["syn.pre_trace"]
    post_recorded = whole.recorded_signals["syn.post_trace"]
    assert pre_recorded.shape == (100, 1, N_SOURCE)
    assert post_recorded.shape == (100, 1, N_TARGET)
    torch.testing.assert_close(pre_recorded.double(), pre_traces, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        post_recorded.double(), post_traces, rtol=1e-5, atol=1e-6
    )

    # Chunks of 37 and 63 steps follow the same global schedule.
    first = model({"pre": pre[:37]}, interventions=replay(post), recording=recording)
    second = model(
        {"pre": pre[37:]},
        runtime_state=first.runtime_state,
        interventions=replay(post),
        recording=recording,
    )
    joined = torch.cat(
        [first.recorded_signals["syn.weight"], second.recorded_signals["syn.weight"]]
    )
    joined_steps = (
        first.numpy().recording.info["syn.weight"].step_indices
        + second.numpy().recording.info["syn.weight"].step_indices
    )
    torch.testing.assert_close(joined, snapshots, rtol=0, atol=0)
    assert list(joined_steps) == list(steps)

    # Recording off: nothing recorded, weights unchanged.
    off = model({"pre": pre}, interventions=replay(post))
    assert not any(name.startswith("syn.") for name in off.recorded_signals)
    torch.testing.assert_close(weights(off), weights(whole), rtol=0, atol=0)


def test_t31_a_disabled_projection_keeps_its_weights_and_its_traces_track_spikes():
    pre, post = trains(31, 100)
    pre, post = pre.float(), post.float()
    params = {"a_plus": 0.05, "a_minus": 0.05}
    model = model_for(network(lang_rule(**params), enabled=False))
    recording = sim.RecordingSpec(
        signals=(
            sim.SignalRecording("syn.pre_trace"),
            sim.SignalRecording("syn.post_trace"),
        )
    )
    result = model({"pre": pre}, interventions=replay(post), recording=recording)
    assert torch.all(weights(result) == 0.5)  # weights frozen
    w0 = torch.full((1, N_SOURCE, N_TARGET), 0.5, dtype=F64)
    pre_traces, post_traces = run_with_traces(
        make_rule(**params), w0, pre.double(), post.double()
    )
    torch.testing.assert_close(
        result.recorded_signals["syn.pre_trace"].double(),
        pre_traces,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        result.recorded_signals["syn.post_trace"].double(),
        post_traces,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("connectivity", ["dense", "one_to_one"])
def test_t32_transmission_uses_the_learned_weights_from_the_previous_step(
    connectivity,
):
    # A learning synapse and a control synapse share the synapse model, so the
    # ratio of their current peaks after a probe pre spike equals the ratio of
    # their weights, whatever the synapse kernel. Probes are far apart (97.5 ms,
    # synapse tau 2 ms), so earlier currents have decayed away.
    dense = connectivity == "dense"
    n_source, n_target = (N_SOURCE, N_TARGET) if dense else (2, 2)
    sizes = {"n_source": n_source, "n_target": n_target}
    extra = {} if dense else {"connectivity": snn.OneToOne()}
    # Dense: pre 0 drives target 1 (learns) and target 0 (control).
    # OneToOne: pre 0 -> target 0 learns; pre 1 -> target 1 is the control.
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
    # Learned at step 15 from the pair 5 -> 15.
    potentiated = w0 + ltp(rule, post_step - pairing_step)
    # The probe at step 200 also depresses, but only after it has transmitted.
    depressed = potentiated - ltd(rule, probes[0] - post_step)
    for probe, weight in zip(probes, (potentiated, depressed), strict=True):
        window = current[probe : probe + 40]
        ratio = float(window[:, learner].max() / window[:, control].max())
        assert ratio == pytest.approx(weight / w0, rel=1e-5)


# T35: several plastic projections -------------------------------------------------


def test_t35_plastic_projections_learn_independently():
    # Two plastic projections onto one population, with different rules, sizes
    # and delays, beside a static projection. Each keeps its own traces and
    # weights and follows its own update step; they share only the post spikes.
    delay = 2
    g = torch.Generator().manual_seed(35)
    pre_a = random_trains(g, 300, batch=2, n=3, rate=0.05)
    pre_b = random_trains(g, 300, batch=2, n=2, rate=0.05)
    post = random_trains(g, 300, batch=2, n=2, rate=0.05)
    rule_a = {"a_plus": 0.05, "a_minus": 0.04, "pairing": "nearest"}
    rule_b = {
        "a_plus": 0.03,
        "a_minus": 0.06,
        "tau_plus_ms": 10.0,
        "tau_minus_ms": 20.0,
        "w_min": 0.1,
        "w_max": 0.9,
        "pairing": "all_to_all",
    }
    net = snn.Network("stdp_two_projections", dt=DT_MS * snn.ms)
    a = net.input(
        "pre_a", shape=("time", "batch", 3), signal_type="spikes", unit="spike"
    )
    b = net.input(
        "pre_b", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cells = net.population("post", size=2, neuron=snn.CUBA_LIF())
    common = {
        "synapse": snn.ExponentialCurrent(tau=2 * snn.ms),
        "initialization_scaling": "direct",
    }
    net.connect(
        a,
        cells.excitatory,
        name="syn_a",
        weight=snn.Constant(0.5),
        plasticity=lang_rule(**rule_a),
        **common,
    )
    net.connect(
        b,
        cells.excitatory,
        name="syn_b",
        weight=snn.Constant(0.4),
        delay=delay * DT_MS * snn.ms,
        plasticity=lang_rule(**rule_b),
        **common,
    )
    net.connect(b, cells.excitatory, name="static", weight=snn.Constant(0.3), **common)
    net.output("post_spikes", cells.spikes)
    result = model_for(net)(
        {"pre_a": pre_a.float(), "pre_b": pre_b.float()},
        interventions=replay(post.float()),
    )

    expected_a, _, _ = run(make_rule(**rule_a), full((2, 3, 2), 0.5), pre_a, post)
    arrived_b = torch.cat([torch.zeros(delay, 2, 2, dtype=F64), pre_b[:-delay]])
    expected_b, _, _ = run(make_rule(**rule_b), full((2, 2, 2), 0.4), arrived_b, post)
    torch.testing.assert_close(
        weights(result, "syn_a").double(), expected_a, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        weights(result, "syn_b", n_source=2).double(),
        expected_b,
        rtol=1e-5,
        atol=1e-6,
    )
    assert not torch.allclose(expected_a, full((2, 3, 2), 0.5))  # the rules acted
    assert torch.all(result.parameters["static.weight"] == 0.3)


# T37: unsupported uses ------------------------------------------------------------


def test_t37_unsupported_uses_of_plastic_projections_are_rejected(tmp_path):
    bundle = snn.compile(network(lang_rule()), target=None)
    # Prepared streaming keeps privately owned weights; continuation goes
    # through sim.simulate with runtime_state instead.
    with pytest.raises(ValueError, match="streaming.*plastic"):
        sim.prepare_stream(bundle.graph, window_steps=4, device="cpu")
    # Dataset inputs would need one weight set per sample (decision of
    # 16 Nov 2026): rejected before anything runs.
    snapshot = tmp_path / "samples.npz"
    np.savez(
        snapshot,
        features=np.ones((3, N_SOURCE), dtype=np.float32),
        labels=np.zeros(3, dtype=np.int64),
    )
    binding = sim.DatasetSnapshotBinding(
        path=snapshot,
        input_id="pre",
        dataset_id="stdp-fixture",
        split="test",
        encoder=sim.DatasetEncoder(
            "rate_poisson", duration_ms=5.0, max_rate_hz=10.0, seed=1
        ),
    )
    with pytest.raises(NotImplementedError, match="dataset inputs.*plastic"):
        sim.simulate(graph=bundle.graph, input_bindings=(binding,), device="cpu")
    # CUDA graph replay is never used: weights change every step.
    from snnlab.sim.graph.replay import _replay_fallback_reason

    model = model_for(network(lang_rule()))
    reason = _replay_fallback_reason(model, {"pre": torch.zeros(4, 1, N_SOURCE)}, {})
    assert "plastic" in reason


# T38-T41: training and transmission controls ---------------------------------------


def trainable_bundle(rule, steps):
    """A tiny classifier whose only weights are the projection "syn"."""
    net = snn.Network("stdp_train", dt=DT_MS * snn.ms)
    source = net.input(
        "pre", shape=("time", "batch", N_SOURCE), signal_type="spikes", unit="spike"
    )
    cells = net.population("post", size=N_TARGET, neuron=snn.CUBA_LIF())
    extra = {} if rule is None else {"plasticity": rule}
    net.connect(
        source,
        cells.excitatory,
        name="syn",
        synapse=snn.ExponentialCurrent(tau=3 * snn.ms),
        weight=snn.Constant(0.5),
        initialization_scaling="direct",
        constraint=snn.NonNegative(),
        **extra,
    )
    scores = snn.MeanVoltage(source=cells.spikes, classes=2, name="scores")
    net.output("prediction", scores)
    recipe = snn.TrainSpec(
        objectives=(snn.CrossEntropy(prediction=scores, target="label"),),
        parameter_groups=(
            snn.ParameterGroup(
                [row["id"] for row in net.parameters], name="all", lr=0.01
            ),
        ),
        optimizer=snn.AdamW(weight_decay=0.0),
        presentation_duration=steps * DT_MS * snn.ms,
    )
    return snn.compile(net, training=recipe)


def test_t38_validation_and_final_evaluation_run_with_plasticity_off():
    # Training, the initial and per-epoch evaluations, validation and the final
    # test all measure the trained weights. With strong amplitudes, any of these
    # passes applying the rule would change a loss; every number must equal the
    # static network's exactly.
    steps = 20
    g = torch.Generator().manual_seed(38)
    values = (torch.rand(steps, 6, N_SOURCE, generator=g) < 0.3).float()
    labels = torch.arange(6) % 2

    def held_out(spec):
        return spec(
            input_bindings=(sim.DenseArrayBinding("pre", values[:, 4:]),),
            targets={"label": labels[4:]},
        )

    def train(rule):
        bundle = trainable_bundle(rule, steps)
        return sim.train(
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(sim.DenseArrayBinding("pre", values[:, :4]),),
            targets={"label": labels[:4]},
            validation=held_out(sim.ValidationSpec),
            test=held_out(sim.TestSpec),
            metrics=sim.TrainingMetrics(
                train="evaluate", initial=("train", "validation")
            ),
            device="cpu",
            seed=7,
            epochs=2,
            batch_size=2,
            diagnostics=False,
            performance=sim.Performance(precision="highest"),
        )

    static = train(None)
    plastic = train(lang_rule(a_plus=0.2, a_minus=0.2))
    assert plastic.metrics["epochs"] == static.metrics["epochs"]
    assert "validation_loss" in plastic.metrics["epochs"][0]
    assert plastic.metrics["test"]["loss"] == static.metrics["test"]["loss"]
    assert plastic.metrics["execution_paths"] == static.metrics["execution_paths"]
    torch.testing.assert_close(
        plastic.parameters["syn.weight"],
        static.parameters["syn.weight"],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("scale", [0.0, 0.5])
def test_t39_projection_scales_act_on_transmission_only(scale):
    # A plastic projection is scaled where it transmits: its current is scaled,
    # while its rule learns exactly as at scale 1, even at scale 0. A static
    # projection is scaled as before, through its weights.
    pre, post = trains(39, 200)
    pre, post = pre.float(), post.float()
    params = {"a_plus": 0.05, "a_minus": 0.05}
    recording = sim.RecordingSpec(signals=(sim.SignalRecording("syn.current"),))

    def simulate(rule, scales):
        bundle = snn.compile(network(rule), target=None)
        return sim.simulate(
            graph=bundle.graph,
            input_bindings=(sim.DenseArrayBinding("pre", pre),),
            interventions=replay(post),
            recording=recording,
            inference_overrides={"projection_scales": scales},
            device="cpu",
        )

    plastic_full = simulate(lang_rule(**params), {})
    plastic = simulate(lang_rule(**params), {"syn": scale})
    torch.testing.assert_close(
        plastic.recorded_signals["syn.current"],
        scale * plastic_full.recorded_signals["syn.current"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(weights(plastic), weights(plastic_full), rtol=0, atol=0)
    w0 = torch.full((1, N_SOURCE, N_TARGET), 0.5, dtype=F64)
    expected, _, _ = run(make_rule(**params), w0, pre.double(), post.double())
    torch.testing.assert_close(
        weights(plastic).double(), expected, rtol=1e-5, atol=1e-6
    )
    assert not torch.allclose(expected, w0)  # it learned, even without transmitting

    static_full = simulate(None, {})
    static = simulate(None, {"syn": scale})
    torch.testing.assert_close(
        static.recorded_signals["syn.current"],
        scale * static_full.recorded_signals["syn.current"],
        rtol=0,
        atol=0,
    )
    assert torch.all(static.parameters["syn.weight"] == 0.5 * scale)


def test_t40_plasticity_off_runs_a_plastic_network_as_the_static_one():
    # The explicit switch training uses: with plasticity=False the plastic
    # network runs exactly like the static one, from the model's weights, on
    # the same execution path, and leaves no plasticity state.
    pre, _ = trains(40, 200, rate=0.3)
    pre = pre.float()
    recording = sim.RecordingSpec(signals=(sim.SignalRecording("syn.current"),))
    plastic = model_for(network(lang_rule(a_plus=0.2, a_minus=0.2)))
    static = model_for(network())
    off = plastic({"pre": pre}, recording=recording, plasticity=False)
    reference = static({"pre": pre}, recording=recording)
    for name, value in reference.outputs.items():
        torch.testing.assert_close(off.outputs[name], value, rtol=0, atol=0)
    torch.testing.assert_close(
        off.recorded_signals["syn.current"],
        reference.recorded_signals["syn.current"],
        rtol=0,
        atol=0,
    )
    assert torch.equal(off.parameters["syn.weight"], reference.parameters["syn.weight"])
    custom = off.runtime_state.custom_state
    assert not any(key.startswith("plasticity/") for key in custom)
    for key in ("selected", "fallback_reasons"):
        assert (
            off.metrics["execution_path"][key]
            == reference.metrics["execution_path"][key]
        )
    # CUDA graph replay stays available, as for the static network.
    from snnlab.sim.graph.replay import _replay_fallback_reason

    reason = _replay_fallback_reason(plastic, {"pre": pre}, {"plasticity": False})
    assert "plastic" not in (reason or "")
    # The rule's updates are not differentiable: with gradients tracked, a
    # plastic run must say so instead of training through the updates.
    weight = plastic.parameter_map()["syn.weight"]
    weight.requires_grad_(True)
    try:
        with pytest.raises(ValueError, match="gradients"):
            plastic({"pre": pre})
        assert plastic({"pre": pre}, plasticity=False).outputs  # training's path
    finally:
        weight.requires_grad_(False)
    with pytest.raises(ValueError, match="plasticity is on"):
        plastic({"pre": pre}, plasticity=False, transmission_scales={"syn": 0.5})


def test_t41_a_disabled_projection_stays_frozen_at_any_scale():
    # A disabled plastic projection transmits nothing and keeps its weights, at
    # every projection scale; its traces still follow the arrivals and the
    # target spikes.
    pre, post = trains(41, 200)
    pre, post = pre.float(), post.float()
    rule = make_rule(a_plus=0.05, a_minus=0.05)
    net = network(lang_rule(a_plus=0.05, a_minus=0.05), enabled=False)
    bundle = snn.compile(net, target=None)
    recording = sim.RecordingSpec(
        signals=tuple(
            sim.SignalRecording(f"syn.{name}")
            for name in ("current", "pre_trace", "post_trace")
        )
    )
    w0 = torch.full((1, N_SOURCE, N_TARGET), 0.5, dtype=F64)
    xs, ys = run_with_traces(rule, w0, pre.double(), post.double())
    for scale in (0.0, 0.5, 2.0):
        result = sim.simulate(
            graph=bundle.graph,
            input_bindings=(sim.DenseArrayBinding("pre", pre),),
            interventions=replay(post),
            recording=recording,
            inference_overrides={"projection_scales": {"syn": scale}},
            device="cpu",
        )
        assert torch.all(weights(result) == 0.5)
        assert torch.all(result.recorded_signals["syn.current"] == 0)
        # All-to-all traces accumulate above 1: compare at float32 precision.
        torch.testing.assert_close(
            result.recorded_signals["syn.pre_trace"].double(),
            xs,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            result.recorded_signals["syn.post_trace"].double(),
            ys,
            rtol=1e-5,
            atol=1e-6,
        )


# T42-T46: persistence, recording and learning frozen -------------------------------


def masked_recurrent_model(rule, start):
    """E -> E with autapses excluded and frozen zeros; start is (source, target)."""
    net = snn.Network("stdp_masks", dt=DT_MS * snn.ms)
    kick = net.input(
        "kick", shape=("time", "batch", 4), signal_type="spikes", unit="spike"
    )
    cells = net.population("E", size=4, neuron=snn.CUBA_LIF())
    common = {
        "synapse": snn.ExponentialCurrent(tau=2 * snn.ms),
        "initialization_scaling": "direct",
    }
    net.connect(
        kick, cells.excitatory, name="kick_E", weight=snn.Constant(0.0), **common
    )
    net.connect(
        cells.spikes,
        cells.excitatory,
        name="rec",
        weight=snn.Constant(0.5),
        connection="recurrent",
        delay=DT_MS * snn.ms,
        plasticity=rule,
        **common,
    )
    net.output("E_spikes", cells.spikes)
    model = model_for(net)
    with torch.no_grad():
        model.parameter_map()["rec.weight"].copy_(start)
    return model


def test_t42_saved_state_round_trips_weights_traces_and_masks(tmp_path):
    # A run saved to disk mid-way and reloaded continues exactly like one long
    # run. The saved state holds the weights, both traces and the mask of
    # entries that can change; the saved mask wins over the current model's,
    # as the saved weights do.
    rule = lang_rule(a_plus=0.05, a_minus=0.05, w_min=0.2, freeze_zero_weights=True)
    start = torch.full((4, 4), 0.5)
    start[0, 1] = start[2, 3] = 0.0  # frozen zeros
    g = torch.Generator().manual_seed(42)
    spikes = (torch.rand(200, 1, 4, generator=g) < 0.05).float()
    kick = torch.zeros(200, 1, 4)
    model = masked_recurrent_model(rule, start)

    def run_chunk(model, chunk, state=None):
        return model(
            {"kick": kick[chunk]},
            runtime_state=state,
            interventions=replay(spikes, population="E"),
        )

    whole = run_chunk(model, slice(None))
    first = run_chunk(model, slice(0, 90))
    keys = {
        key for key in first.runtime_state.custom_state if key.startswith("plasticity/")
    }
    assert keys == {
        f"plasticity/rec/{name}"
        for name in ("weight", "pre_trace", "post_trace", "mask")
    }
    saved = sim.save_runtime_state(tmp_path / "state", first.runtime_state)
    loaded = sim.load_runtime_state(saved)
    second = run_chunk(model, slice(90, None), loaded)
    torch.testing.assert_close(
        second.parameters["rec.weight"], whole.parameters["rec.weight"], rtol=0, atol=0
    )
    for name in ("pre_trace", "post_trace"):
        key = f"plasticity/rec/{name}"
        torch.testing.assert_close(
            second.runtime_state.custom_state[key],
            whole.runtime_state.custom_state[key],
            rtol=0,
            atol=0,
        )
    final = second.parameters["rec.weight"][0]
    assert torch.all(final.diagonal() == 0) and final[0, 1] == 0 and final[2, 3] == 0

    # Continue with a model whose zeros sit elsewhere: the saved mask still
    # decides which entries can change, and the saved weights are used.
    other_start = torch.full((4, 4), 0.5)
    other_start[1, 0] = 0.0
    other = masked_recurrent_model(rule, other_start)
    resumed = run_chunk(other, slice(90, None), loaded)
    torch.testing.assert_close(
        resumed.parameters["rec.weight"], whole.parameters["rec.weight"], rtol=0, atol=0
    )


def test_t43_a_changed_rule_or_a_missing_plastic_state_is_rejected(tmp_path):
    # Continuation requires the identical rule, amplitudes included, and a
    # state that carries the plastic weights.
    pre, post = trains(43, 60)
    pre, post = pre.float(), post.float()
    base = {"a_plus": 0.05, "a_minus": 0.05}
    first = model_for(network(lang_rule(**base)))(
        {"pre": pre[:30]}, interventions=replay(post)
    )
    state = sim.load_runtime_state(
        sim.save_runtime_state(tmp_path / "state", first.runtime_state)
    )
    for changed in (
        {"a_plus": 0.06},
        {"tau_minus_ms": 30.0},
        {"pairing": "nearest"},
        {"w_max": 0.9},
        {"freeze_zero_weights": True},
    ):
        model = model_for(network(lang_rule(**{**base, **changed})))
        with pytest.raises(ValueError, match="incompatible.*plasticity"):
            model({"pre": pre[30:]}, runtime_state=state, interventions=replay(post))
    # The identical rule continues.
    model_for(network(lang_rule(**base)))(
        {"pre": pre[30:]}, runtime_state=state, interventions=replay(post)
    )
    # A state from the static network cannot start a plastic run, and the
    # reverse is refused too.
    static_state = model_for(network())({"pre": pre[:30]}).runtime_state
    with pytest.raises(ValueError, match="incompatible"):
        model_for(network(lang_rule(**base)))(
            {"pre": pre[30:]}, runtime_state=static_state
        )
    with pytest.raises(ValueError, match="incompatible"):
        model_for(network())({"pre": pre[30:]}, runtime_state=state)


@pytest.mark.parametrize("split", [33, 35, 37])
def test_t44_delayed_spikes_cross_chunk_boundaries(split, tmp_path):
    # With a 4-step delay, spikes emitted just before the split arrive after it.
    # They must pair at arrival in the second chunk, also through a saved state.
    delay = 4
    pre = torch.zeros(80, 1, N_SOURCE)
    post = torch.zeros(80, 1, N_TARGET)
    pre[[31, 32, 33, 34, 50], 0, 0] = 1
    post[[37, 45, 60], 0, 1] = 1
    params = {"a_plus": 0.05, "a_minus": 0.05}
    model = model_for(network(lang_rule(**params), delay_steps=delay))
    whole = model({"pre": pre}, interventions=replay(post))
    first = model({"pre": pre[:split]}, interventions=replay(post))
    state = sim.load_runtime_state(
        sim.save_runtime_state(tmp_path / "state", first.runtime_state)
    )
    second = model(
        {"pre": pre[split:]}, runtime_state=state, interventions=replay(post)
    )
    torch.testing.assert_close(weights(second), weights(whole), rtol=0, atol=0)
    arrivals = torch.cat([torch.zeros(delay, 1, N_SOURCE), pre[:-delay]])
    w0 = torch.full((1, N_SOURCE, N_TARGET), 0.5, dtype=F64)
    expected, _, _ = run(make_rule(**params), w0, arrivals.double(), post.double())
    torch.testing.assert_close(weights(whole).double(), expected, rtol=1e-6, atol=1e-7)


def test_t45_weight_recordings_export_to_numpy_and_disk_and_reload(tmp_path):
    # Weight snapshots carry their step coordinates into NumPy, into exported
    # files and into a disk sink written during the run; every route reads
    # back the same arrays and coordinates. Thinning works for any dense signal.
    pre, post = trains(45, 100, batch=2)
    pre, post = pre.float(), post.float()
    model = model_for(network(lang_rule(a_plus=0.05, a_minus=0.05)))
    signals = (
        sim.SignalRecording("syn.weight", every_steps=25),
        sim.SignalRecording("syn.pre_trace"),
        sim.SignalRecording("post.voltage", every_steps=10),
    )
    result = model(
        {"pre": pre}, interventions=replay(post), recording=sim.RecordingSpec(signals)
    )
    exported = result.numpy()
    info = exported.recording.info["syn.weight"]
    assert info.axes == ("time", "batch", "source", "target")
    assert info.step_indices == (24, 49, 74, 99)
    assert exported.recording.signals["syn.weight"].shape == (4, 2, N_SOURCE, N_TARGET)
    np.testing.assert_array_equal(
        exported.recording.signals["syn.weight"][-1], weights(result).numpy()
    )
    assert exported.recording.info["post.voltage"].step_indices == tuple(
        range(9, 100, 10)
    )
    assert exported.recording.info["syn.pre_trace"].step_indices == tuple(range(100))
    one = result.numpy(batch=1).recording
    assert one.info["syn.weight"].axes == ("time", "source", "target")
    np.testing.assert_array_equal(
        one.signals["syn.weight"], exported.recording.signals["syn.weight"][:, 1]
    )

    # Export to files and read back.
    path = result.write_recording(tmp_path / "exported")
    reread = sim.RecordingData.read(path)
    for name in ("syn.weight", "syn.pre_trace", "post.voltage"):
        np.testing.assert_array_equal(
            reread.signals[name], exported.recording.signals[name]
        )
        assert (
            reread.info[name].step_indices == exported.recording.info[name].step_indices
        )
        assert reread.info[name].axes == exported.recording.info[name].axes

    # A disk sink during a run in two chunks writes the same snapshots.
    sink = sim.NPZRecordingSink(tmp_path / "sink")
    streamed = sim.RecordingSpec(
        signals, sink=sink, sink_id=sink.identity, block_steps=2
    )
    first = model({"pre": pre[:60]}, interventions=replay(post), recording=streamed)
    model(
        {"pre": pre[60:]},
        runtime_state=first.runtime_state,
        interventions=replay(post),
        recording=streamed,
    )
    from_disk = sim.RecordingData.read(tmp_path / "sink")
    for name in ("syn.weight", "syn.pre_trace", "post.voltage"):
        np.testing.assert_array_equal(
            from_disk.signals[name], exported.recording.signals[name]
        )
        assert (
            from_disk.info[name].step_indices
            == exported.recording.info[name].step_indices
        )
    assert from_disk.info["syn.weight"].unit == "nA"

    # Thinning is for dense signals; weight matrices are recorded whole.
    for bad in (
        sim.SignalRecording("syn.weight", every_steps=0),
        sim.SignalRecording("syn.weight", cells=(0,)),
        sim.SignalRecording("post.spikes", kind="spike_events", every_steps=5),
    ):
        with pytest.raises(ValueError):
            model({"pre": pre}, recording=sim.RecordingSpec((bad,)))


def test_t46_learning_frozen_transmits_the_current_weights_without_changing_them():
    # A probe between learning phases: learning=False runs with the weights
    # from runtime_state, does not change them, and transmits exactly like a
    # static network that holds those weights. The traces keep tracking spikes.
    pre, post = trains(46, 200, rate=0.1)
    pre, post = pre.float(), post.float()
    # Inputs fall silent for 40 ms before the probe, so the synaptic current
    # left from the learning phase (tau 2 ms) has decayed when the probe starts.
    pre[20:100] = 0
    params = {"a_plus": 0.05, "a_minus": 0.05}
    bundle = snn.compile(network(lang_rule(**params)), target=None)
    recording = sim.RecordingSpec(
        signals=(
            sim.SignalRecording("syn.current"),
            sim.SignalRecording("syn.pre_trace"),
        )
    )

    def simulate(chunk, **extra):
        return sim.simulate(
            graph=bundle.graph,
            input_bindings=(sim.DenseArrayBinding("pre", pre[chunk]),),
            interventions=replay(post),
            recording=recording,
            device="cpu",
            **extra,
        )

    learned = simulate(slice(0, 100))
    probe = simulate(
        slice(100, None), runtime_state=learned.runtime_state, learning=False
    )
    torch.testing.assert_close(weights(probe), weights(learned), rtol=0, atol=0)
    assert not torch.allclose(weights(learned), torch.tensor(0.5))  # it had learned

    # The probe transmits with the learned weights: a static network holding
    # them produces the same current.
    static = model_for(network())
    with torch.no_grad():
        static.parameter_map()["syn.weight"].copy_(weights(learned)[0])
    current = sim.RecordingSpec(signals=(sim.SignalRecording("syn.current"),))
    reference = static(
        {"pre": pre[100:]}, interventions=replay(post), recording=current
    )
    torch.testing.assert_close(
        probe.recorded_signals["syn.current"],
        reference.recorded_signals["syn.current"],
        rtol=1e-6,
        atol=1e-7,
    )
    # The traces keep tracking, so learning can resume from the probe's state.
    w0 = torch.full((1, N_SOURCE, N_TARGET), 0.5, dtype=F64)
    xs, _ = run_with_traces(make_rule(**params), w0, pre.double(), post.double())
    torch.testing.assert_close(
        probe.recorded_signals["syn.pre_trace"].double(), xs[100:], rtol=1e-6, atol=1e-6
    )
    path = probe.metrics["execution_path"]
    assert any("per-trial weights" in reason for reason in path["fallback_reasons"])
    with pytest.raises(TypeError, match="learning"):
        simulate(slice(0, 10), learning="no")


# T47-T48: devices, dtypes and precision of the bounds ------------------------------


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
    _device("cpu", torch.float32),
    _device("cpu", torch.float64),
    _device("mps", torch.float32),
    _device("cuda", torch.float32),
    _device("cuda", torch.float64),
]
TOLERANCES = {torch.float32: (1e-5, 1e-6), torch.float64: (1e-12, 1e-13)}


@pytest.mark.parametrize(("device", "dtype"), DEVICE_CASES)
def test_t47_plastic_runs_match_the_update_step_on_every_device_and_dtype(
    device, dtype
):
    # The same plastic simulation, with a delay, two trials, weight recording
    # and a continuation, on each available device in float32 and float64.
    delay = 2
    pre, post = trains(47, 120, batch=2, rate=0.08)
    params = {"a_plus": 0.05, "a_minus": 0.05, "pairing": "all_to_all"}
    model = model_for(network(lang_rule(**params), delay_steps=delay))
    model = model.to(device=device, dtype=dtype)
    recording = sim.RecordingSpec(
        signals=(sim.SignalRecording("syn.weight", every_steps=30),)
    )
    inputs = pre.to(device=device, dtype=dtype)
    first = model(
        {"pre": inputs[:70]}, interventions=replay(post.to(dtype)), recording=recording
    )
    second = model(
        {"pre": inputs[70:]},
        runtime_state=first.runtime_state,
        interventions=replay(post.to(dtype)),
        recording=recording,
    )
    final = second.parameters["syn.weight"]
    assert final.device.type == device and final.dtype == dtype
    arrivals = torch.cat([torch.zeros(delay, 2, N_SOURCE, dtype=F64), pre[:-delay]])
    w0 = torch.full((2, N_SOURCE, N_TARGET), 0.5, dtype=F64)
    expected, _, trajectory = run(make_rule(**params), w0, arrivals, post)
    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(final.cpu().double(), expected, rtol=rtol, atol=atol)
    snapshots = torch.cat(
        [
            first.recorded_signals["syn.weight"],
            second.recorded_signals["syn.weight"],
        ]
    )
    torch.testing.assert_close(
        snapshots.cpu().double(),
        trajectory[[29, 59, 89, 119]],
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("w_max", [0.0015, 0.7, 1.3])
def test_t48_float32_weights_at_the_bounds_are_accepted(w_max):
    # 0.0015 has no exact float32 form: a weight drawn at w_max rounds to
    # float32(w_max), which lies just above w_max. The bounds check compares in
    # the weights' precision, as the update step clips there, so such weights
    # start; values truly outside the bounds are still refused.
    w_min = w_max / 3
    rule = lang_rule(a_plus=0.1 * w_max, a_minus=0.1 * w_max, w_max=w_max, w_min=w_min)
    pre, post = trains(48, 60)
    for value in (w_max, w_min):
        model = model_for(network(rule, w0=value))
        assert model.parameter_map()["syn.weight"].dtype == torch.float32
        result = model({"pre": pre.float()}, interventions=replay(post.float()))
        final = weights(result)
        assert final.max() <= torch.tensor(w_max, dtype=torch.float32)
        assert final.min() >= torch.tensor(w_min, dtype=torch.float32)
    with pytest.raises(ValueError, match="w_max"):
        model_for(network(rule, w0=w_max * 1.001))({"pre": pre.float()})


# T49: invalid values, graphs and states are refused with their cause ---------------


def test_t49_invalid_values_graphs_and_states_are_refused_with_their_cause():
    with pytest.raises(ValueError, match="tau_plus must be a time"):
        snn.STDPSimple(
            a_plus=0.01,
            a_minus=0.01,
            tau_plus=20.0,
            tau_minus=20 * snn.ms,
            w_max=1.0,
        )
    with pytest.raises(ValueError, match="freeze_zero_weights must be boolean"):
        lang_rule(freeze_zero_weights="yes")
    rule = make_rule()
    state = kernel.init_state(rule, batch=1, n_source=N_SOURCE, n_target=N_TARGET)
    w0 = torch.full((1, N_SOURCE, N_TARGET), 0.5, dtype=F64)
    silent_pre = torch.zeros(1, N_SOURCE, dtype=F64)
    silent_post = torch.zeros(1, N_TARGET, dtype=F64)
    with pytest.raises(ValueError, match="dt_ms must be positive"):
        kernel.step(rule, state, w0, silent_pre, silent_post, dt_ms=0.0)

    # A graph edited after compilation still meets the simulator's own check.
    graph = snn.compile(network(lang_rule()), target=None).graph
    edits = (
        ("a_plus", -1.0, "a_plus"),
        ("a_plus", "large", "a_plus must be a finite number"),
        ("tau_plus", {"value": 0.02, "unit": "s"}, "tau_plus must be a time in ms"),
        ("freeze_zero_weights", "yes", "freeze_zero_weights must be boolean"),
        (None, "stdp_simple", "must be a serialized spec mapping"),
    )
    for field, value, message in edits:
        edited = copy.deepcopy(graph)
        if field is None:
            edited["projections"][0]["plasticity"] = value
        else:
            edited["projections"][0]["plasticity"][field] = value
        with pytest.raises(ValueError, match=f"projection 'syn'.*{message}"):
            sim.GraphExecutor(sim.plan_graph(edited), seed=0)
    unsupported = copy.deepcopy(graph)
    unsupported["projections"][0]["polarity"] = "inhibitory"
    with pytest.raises(ValueError, match="projection 'syn'.*cannot apply"):
        sim.GraphExecutor(sim.plan_graph(unsupported), seed=0)
    # A source that is neither an input nor a population emits no spikes.
    row = {**graph["projections"][0], "source": "elsewhere.spikes"}
    assert "its source does not emit spikes" in unsupported_reasons(graph, row)

    pre, post = trains(49, 40)
    model = model_for(network(lang_rule()))
    with pytest.raises(ValueError, match="finite and non-negative"):
        model({"pre": pre.float()}, transmission_scales={"syn": -1.0})
    with pytest.raises(ValueError, match="not plastic: \\['other'\\]"):
        model({"pre": pre.float()}, transmission_scales={"other": 1.0})

    # A saved state that lacks, reshapes or recasts a plastic tensor.
    saved = model({"pre": pre.float()}, interventions=replay(post.float()))
    custom = dict(saved.runtime_state.custom_state)
    key = "plasticity/syn/weight"
    edits = (
        ({k: v for k, v in custom.items() if k != key}, "missing"),
        ({**custom, key: custom[key][..., :1]}, "shape expected"),
        ({**custom, key: custom[key].double()}, "dtype expected"),
    )
    for edited, message in edits:
        state = dataclasses.replace(saved.runtime_state, custom_state=edited)
        with pytest.raises(ValueError, match=f"{key} {message}|{message} {key}"):
            model({"pre": pre.float()}, runtime_state=state)
