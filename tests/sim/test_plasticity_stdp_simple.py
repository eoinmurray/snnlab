"""Acceptance tests for stdp_simple: pair-based, additive STDP with hard bounds.

The tests are skipped until the feature exists. Lines marked PROVISIONAL follow
the issue draft and may change in review. The lang-side tests (T22, T25) live in
tests/lang/test_snnlang.py.

Layers:
- T01-T20 drive the update step alone, on fixed spike trains.
- T21 compares the update step with an independent Brian2 model.
- T23-T32 run the rule inside snnlab, with the update step as their reference.
  Post spikes are imposed with ReplaySpikes, so neuron dynamics play no part.
  T32 checks that transmission uses the learned weights.

Conventions: weights use the runtime layout (batch, source, target). Sizes are
non-square (3 sources, 2 targets), so a transposed layout fails. Defaults use
distinct amplitudes and time constants, so a swapped constant fails.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import math

import numpy as np
import pytest
import torch

from snnlab import lang as snn
from snnlab import sim

KERNEL_MODULE = "snnlab.sim.plasticity.stdp_simple"  # PROVISIONAL: location
try:
    kernel = importlib.import_module(KERNEL_MODULE)
except ModuleNotFoundError as error:
    # Skip only while the module itself is absent. A missing dependency or a
    # broken import inside an existing module must fail collection.
    if error.name not in {"snnlab.sim.plasticity", KERNEL_MODULE}:
        raise
    kernel = None

pytestmark = pytest.mark.skipif(
    kernel is None, reason=f"{KERNEL_MODULE} is not implemented yet"
)
requires_lang = pytest.mark.skipif(
    not hasattr(snn, "STDPSimple"), reason="lang.STDPSimple is not implemented yet"
)

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
    assert rule.pairing == "nearest"
    assert rule.w_min == 0.0
    assert set(kernel.PAIRINGS) == set(PAIRINGS)
    assert rule.json() == {
        "kind": "stdp_simple",
        "update": "additive",
        "bounds": "hard",
        "pairing": "nearest",
        "a_plus": 0.01,
        "a_minus": 0.0105,
        "tau_plus_ms": 16.8,
        "tau_minus_ms": 33.7,
        "w_min": 0.0,
        "w_max": 1.0,
    }
    assert make_rule(pairing="all_to_all").pairing == "all_to_all"
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
    return snn.STDPSimple(  # PROVISIONAL: name and arguments
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
        connect["plasticity"] = rule  # PROVISIONAL: keyword on Network.connect
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
    """Weights as (batch, source, target). PROVISIONAL: stored layout."""
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


@requires_lang
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
        plasticity=rule,  # PROVISIONAL
    )
    net.output("E_spikes", e.spikes)
    return net


@requires_lang
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
    # Opt-in frozen zeros (PROVISIONAL flag): below w_min, allowed, unchanged.
    frozen = lang_rule(w_min=0.2, freeze_zero_weights=True)
    result = model_for(network(frozen, w0=0.0))(
        {"pre": pre}, interventions=replay(post)
    )
    assert torch.all(weights(result) == 0.0)


# T26-T31: delays, execution, training, recording, disabled projections ------------


@requires_lang
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


@requires_lang
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
    if not dense:  # PROVISIONAL: OneToOne stores one weight per channel pair
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


@requires_lang
def test_t28_plasticity_selects_the_step_by_step_path_and_says_why():
    pre, post = torch.zeros(16, 1, N_SOURCE), torch.zeros(16, 1, N_TARGET)
    model = model_for(network(lang_rule()), compile_block_steps=8)
    path = model({"pre": pre}, interventions=replay(post)).metrics["execution_path"]
    assert path["selected"] == "kernel_loop"
    assert any("plastic" in reason for reason in path["fallback_reasons"])


@requires_lang
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
        "precision": "highest",
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


@requires_lang
def test_t30_recordings_follow_the_update_and_one_schedule_across_chunks():
    pre, post = trains(30, 100)
    pre, post = pre.float(), post.float()
    params = {"a_plus": 0.05, "a_minus": 0.05}
    model = model_for(network(lang_rule(**params)))
    recording = sim.RecordingSpec(
        signals=(
            sim.SignalRecording("syn.weight", every_steps=10),  # PROVISIONAL
            sim.SignalRecording("syn.pre_trace"),  # PROVISIONAL signal names
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
    # at global steps 9, 19, ..., 99. The last equals the final weights.
    snapshots = whole.recorded_signals["syn.weight"]
    steps = whole.recorded_signals["syn.weight.steps"]  # PROVISIONAL
    assert snapshots.shape == (10, 1, N_SOURCE, N_TARGET)
    assert steps.tolist() == list(range(9, 100, 10))
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
    joined_steps = torch.cat(
        [
            first.recorded_signals["syn.weight.steps"],
            second.recorded_signals["syn.weight.steps"],
        ]
    )
    torch.testing.assert_close(joined, snapshots, rtol=0, atol=0)
    assert joined_steps.tolist() == steps.tolist()

    # Recording off: nothing recorded, weights unchanged.
    off = model({"pre": pre}, interventions=replay(post))
    assert not any(name.startswith("syn.") for name in off.recorded_signals)
    torch.testing.assert_close(weights(off), weights(whole), rtol=0, atol=0)


@requires_lang
def test_t31_a_disabled_projection_keeps_its_weights_and_its_traces_track_spikes():
    pre, post = trains(31, 100)
    pre, post = pre.float(), post.float()
    params = {"a_plus": 0.05, "a_minus": 0.05}
    model = model_for(network(lang_rule(**params), enabled=False))
    recording = sim.RecordingSpec(
        signals=(
            sim.SignalRecording("syn.pre_trace"),  # PROVISIONAL
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


@requires_lang
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
