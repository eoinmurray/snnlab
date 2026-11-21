"""Independent numerical protocols for the built-in local learning rules."""

import json
import math
from dataclasses import replace

import pytest
import torch

from snnlab import lang, sim
from snnlab.plasticity import (
    InhibitorySTDP,
    PairSTDP,
    Plasticity,
    RewardModulatedSTDP,
    TripletSTDP,
)


def pair(**kwargs):
    return PairSTDP(
        **dict(
            a_plus=0.1 * lang.nA,
            a_minus=0.2 * lang.nA,
            tau_pre=10 * lang.ms,
            tau_post=20 * lang.ms,
            w_min=0 * lang.nA,
            w_max=1 * lang.nA,
        )
        | kwargs
    )


def triplet(**kwargs):
    return TripletSTDP(
        **dict(
            a2_plus=0.1 * lang.nA,
            a2_minus=0.02 * lang.nA,
            a3_plus=0.3 * lang.nA,
            a3_minus=0.04 * lang.nA,
            tau_pre=10 * lang.ms,
            tau_post=20 * lang.ms,
            tau_x=40 * lang.ms,
            tau_y=30 * lang.ms,
            w_min=0 * lang.nA,
            w_max=2 * lang.nA,
        )
        | kwargs
    )


def inhibitory(**kwargs):
    return InhibitorySTDP(
        **dict(
            eta=0.1 * lang.nA,
            target_rate=5 * lang.Hz,
            tau=20 * lang.ms,
            w_min=0 * lang.nA,
            w_max=1 * lang.nA,
        )
        | kwargs
    )


def reward_rule(**kwargs):
    return RewardModulatedSTDP(
        **(
            pair().__dict__
            | dict(
                reward_input="reward",
                tau_eligibility=100 * lang.ms,
                tau_weight=10 * lang.ms,
            )
            | kwargs
        )
    )


def make_graph(rule, *, initial=0.5, delay=0):
    net = lang.Network("rules", dt=1 * lang.ms)
    pre = net.input(
        "pre", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    post = net.population("post", size=1, neuron=lang.LIF())
    if isinstance(rule, RewardModulatedSTDP):
        net.input("reward", shape=("time", "batch", 1), signal_type="analog", unit="1")
    net.connect(
        pre,
        post.inhibitory if isinstance(rule, InhibitorySTDP) else post.excitatory,
        name="p",
        synapse=lang.ExponentialCurrent(tau=1 * lang.ms),
        weight=lang.Constant(initial),
        initialization_scaling="direct",
        delay=delay * lang.ms,
    )
    net.output("spikes", post.spikes)
    return lang.compile(net, target=None).graph


def run(
    rule,
    pre,
    post,
    *,
    reward=None,
    state=None,
    graph=None,
    initial=0.5,
    delay=0,
    **kwargs,
):
    graph = make_graph(rule, initial=initial, delay=delay) if graph is None else graph
    bindings = [
        sim.DenseArrayBinding(
            "pre", torch.tensor(pre, dtype=torch.float32).reshape(-1, 1, 1)
        )
    ]
    if isinstance(rule, RewardModulatedSTDP):
        bindings.append(
            sim.DenseArrayBinding(
                "reward",
                torch.tensor(
                    reward if reward is not None else [0.0] * len(pre),
                    dtype=torch.float32,
                ).reshape(-1, 1, 1),
            )
        )
    replay = sim.DenseSpikeReplay.from_tensor(
        torch.tensor(post, dtype=torch.float32).reshape(-1, 1, 1),
        dt_ms=graph["timebase"]["dt"]["value"],
        start_step=state.completed_steps if state is not None else 0,
    )
    return sim.simulate(
        graph=graph,
        device="cpu",
        input_bindings=bindings,
        interventions=[sim.ReplaySpikes("post", replay)],
        plasticity=Plasticity(rules={"p": rule}),
        runtime_state=state,
        **kwargs,
    )


def weight(result):
    return result.parameters["p.weight"].item()


@pytest.mark.parametrize(
    "pre,post,expected",
    [
        ([1, 0], [0, 1], 0.5 + 0.1 * math.exp(-0.1) * 0.5),
        ([0, 1], [1, 0], 0.5 - 0.2 * math.exp(-0.05) * 0.5),
    ],
)
def test_weight_dependent_pair(pre, post, expected):
    assert weight(
        run(pair(weight_dependence="multiplicative"), pre, post)
    ) == pytest.approx(expected)


def test_weight_dependence_normalized_distance_and_boundaries():
    rule = pair(
        weight_dependence="multiplicative", w_min=0.2 * lang.nA, w_max=1.2 * lang.nA
    )
    assert weight(run(rule, [1, 0], [0, 1], initial=0.7)) == pytest.approx(
        0.7 + 0.05 * math.exp(-0.1)
    )
    assert weight(run(rule, [1, 0], [0, 1], initial=1.2)) == pytest.approx(1.2)
    assert weight(run(rule, [0, 1], [1, 0], initial=0.2)) == pytest.approx(0.2)


def test_triplet_potentiation_and_depression():
    expected = (
        0.5 + 0.1 * (math.exp(-0.1) + math.exp(-0.2)) + 0.3 * math.exp(-0.2 - 1 / 30)
    )
    assert weight(run(triplet(), [1, 0, 0], [0, 1, 1])) == pytest.approx(expected)
    expected = (
        0.5 - 0.02 * (math.exp(-0.05) + math.exp(-0.1)) - 0.04 * math.exp(-0.1 - 1 / 40)
    )
    assert weight(run(triplet(), [0, 1, 1], [1, 0, 0])) == pytest.approx(expected)


def test_triplet_reduces_to_pair_and_excludes_simultaneous_pairs():
    pre, post = [1, 0, 1, 1, 0], [0, 1, 1, 0, 1]
    rule = triplet(
        a3_plus=0 * lang.nA,
        a3_minus=0 * lang.nA,
        a2_minus=0.2 * lang.nA,
        w_max=1 * lang.nA,
    )
    assert weight(run(rule, pre, post)) == pytest.approx(weight(run(pair(), pre, post)))
    assert weight(run(triplet(), [1], [1])) == 0.5


def test_inhibitory_symmetric_window_and_target_depression():
    alpha = 2 * 20 * 5 / 1000
    expected = 0.5 + 0.1 * (math.exp(-0.05) - alpha)
    assert weight(run(inhibitory(), [1, 0], [0, 1])) == pytest.approx(expected)
    assert weight(run(inhibitory(), [0, 1], [1, 0])) == pytest.approx(expected)
    assert weight(run(inhibitory(), [1, 1], [0, 0])) == pytest.approx(
        0.5 - 2 * 0.1 * alpha
    )
    assert weight(run(inhibitory(target_rate=0 * lang.Hz), [1], [0])) == 0.5


def test_inhibitory_rule_rejects_excitatory_projection():
    with pytest.raises(ValueError, match="requires an inhibitory projection"):
        run(inhibitory(), [1], [0], graph=make_graph(pair()))


@pytest.mark.parametrize(
    "reward,sign", [([0, 0, 0], 0), ([0, 0, 1], 1), ([0, 0, -1], -1)]
)
def test_delayed_reward_and_signed_update(reward, sign):
    expected = 0.5 + sign * 0.1 * math.exp(-0.1) * math.exp(-0.01) / 10
    result = run(reward_rule(), [1, 0, 0], [0, 1, 0], reward=reward)
    assert weight(result) == pytest.approx(expected)
    assert result.runtime_state.custom_state[
        "plasticity/p/trace/eligibility"
    ].item() == pytest.approx(0.1 * math.exp(-0.1 - 0.01))


def test_reward_integrates_during_silence():
    expected = 0.5 + 0.1 * math.exp(-0.1) * (1 + math.exp(-0.01)) / 10
    assert weight(
        run(reward_rule(), [1, 0, 0], [0, 1, 0], reward=[0, 1, 1])
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: pair(weight_dependence="multiplicative"),
        triplet,
        inhibitory,
        reward_rule,
    ],
)
def test_json_and_saved_chunk_continuation(factory, tmp_path):
    request = Plasticity(rules={"p": factory()})
    restored = Plasticity.from_json(json.loads(json.dumps(request.json())))
    assert restored.json() == request.json()
    rule = restored.rules["p"]
    pre, post, reward = [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, -1, 1, 1]
    full = run(rule, pre, post, reward=reward, delay=1)
    first = run(rule, pre[:2], post[:2], reward=reward[:2], delay=1)
    sim.save_runtime_state(tmp_path / "state", first.runtime_state)
    second = run(
        rule,
        pre[2:],
        post[2:],
        reward=reward[2:],
        state=sim.load_runtime_state(tmp_path / "state"),
        delay=1,
    )
    torch.testing.assert_close(
        full.parameters["p.weight"], second.parameters["p.weight"], rtol=0, atol=0
    )
    for key, tensor in full.runtime_state.custom_state.items():
        torch.testing.assert_close(
            tensor, second.runtime_state.custom_state[key], rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"reward_input": ""},
        {"tau_weight": 0 * lang.ms},
        {"tau_eligibility": -1 * lang.ms},
    ],
)
def test_reward_parameters_are_validated(changes):
    with pytest.raises(ValueError):
        reward_rule(**changes)


def test_invalid_reward_input_contracts():
    rule = reward_rule()
    for field, value in [
        ("unit", "nA"),
        ("signal_type", "spikes"),
        ("shape", ["time", "batch", 2]),
    ]:
        g = make_graph(rule)
        g["inputs"][1][field] = value
        with pytest.raises(ValueError):
            run(rule, [1], [0], graph=g)
    with pytest.raises(ValueError):
        run(rule, [1], [0], reward=[float("nan")])


def test_rule_change_rejects_continuation():
    rule = pair()
    first = run(rule, [1], [0])
    with pytest.raises(ValueError, match="incompatible"):
        run(
            replace(rule, weight_dependence="multiplicative"),
            [0],
            [1],
            state=first.runtime_state,
        )


@pytest.mark.parametrize("dt_ms", [0.5, 2.0])
def test_reward_uses_physical_decay_and_timestep_integration(dt_ms):
    rule = reward_rule()
    g = make_graph(rule)
    g["timebase"]["dt"]["value"] = dt_ms
    expected = 0.5 + 0.1 * math.exp(-dt_ms / 10 - dt_ms / 100) * dt_ms / 10
    assert weight(
        run(rule, [1, 0, 0], [0, 1, 0], reward=[0, 0, 1], graph=g)
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (pair, {"weight_dependence": "unknown"}),
        (triplet, {"tau_x": 0 * lang.ms}),
        (triplet, {"a3_plus": -0.1 * lang.nA}),
        (inhibitory, {"target_rate": -1 * lang.Hz}),
        (inhibitory, {"target_rate": 1 * lang.ms}),
        (inhibitory, {"eta": 0.1 * lang.uS}),
        (reward_rule, {"tau_weight": 0 * lang.ms}),
        (reward_rule, {"tau_eligibility": float("inf") * lang.ms}),
    ],
)
def test_new_rule_parameter_validation(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)
