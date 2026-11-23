"""Independent numerical protocols for the built-in local learning rules."""

import json
import math
from dataclasses import replace

import pytest
import torch

from snnlab import lang, sim
from snnlab.plasticity import (
    CalciumPlasticity,
    InhibitorySTDP,
    NonlinearSTDP,
    PairSTDP,
    Plasticity,
    RewardModulatedSTDP,
    SynapticScaling,
    TripletSTDP,
    VoltageSTDP,
)
from snnlab.sim.runtime import plasticity_rules as kernels


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
    learning=None,
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
        plasticity=learning if learning is not None else Plasticity(rules={"p": rule}),
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


def nonlinear(**changes):
    return NonlinearSTDP(**(pair().__dict__ | changes))


def voltage_rule(**changes):
    return VoltageSTDP(
        **dict(
            a_plus=0.001 * lang.nA,
            a_minus=0.002 * lang.nA,
            theta_minus=-65 * lang.mV,
            theta_plus=-45 * lang.mV,
            voltage_rest=-70 * lang.mV,
            spike_voltage=20 * lang.mV,
            tau_pre=10 * lang.ms,
            tau_minus=20 * lang.ms,
            tau_plus=30 * lang.ms,
            w_min=0 * lang.nA,
            w_max=1 * lang.nA,
        )
        | changes
    )


def calcium(**changes):
    return CalciumPlasticity(**dict(w_min=0 * lang.nA, w_max=1 * lang.nA) | changes)


def scaling(**changes):
    return SynapticScaling(
        **dict(
            target_rate=5 * lang.Hz,
            tau_rate=100 * lang.ms,
            beta=0.1,
            gamma=0.01,
            w_min=0 * lang.nA,
            w_max=1 * lang.nA,
        )
        | changes
    )


@pytest.mark.parametrize(
    "mu_plus,mu_minus", [(0, 0), (0, 1), (0.2, 0.7), (1, 1), (2, 3)]
)
def test_nonlinear_independent_power_law_windows(mu_plus, mu_minus):
    rule = nonlinear(
        mu_plus=mu_plus, mu_minus=mu_minus, w_min=0.2 * lang.nA, w_max=1.2 * lang.nA
    )
    initial = 0.6
    assert weight(run(rule, [1, 0], [0, 1], initial=initial)) == pytest.approx(
        initial + 0.1 * math.exp(-0.1) * 0.6**mu_plus
    )
    assert weight(run(rule, [0, 1], [1, 0], initial=initial)) == pytest.approx(
        initial - 0.2 * math.exp(-0.05) * 0.4**mu_minus
    )


@pytest.mark.parametrize("mode,exponent", [("additive", 0), ("multiplicative", 1)])
def test_nonlinear_special_cases_preserve_existing_pair_convention(mode, exponent):
    pre, post = [1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 1]
    assert weight(
        run(nonlinear(mu_plus=exponent, mu_minus=exponent), pre, post)
    ) == weight(run(pair(weight_dependence=mode), pre, post))


def test_calcium_thresholds_bistability_and_decay():
    rule = calcium()
    w = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    pre, post = torch.zeros(1, 1, dtype=w.dtype), torch.zeros(1, 2, dtype=w.dtype)
    delta, state = kernels.step(rule, w, pre, post, kernels.initial_state(rule, w), 1)
    expected = w * (1 - w) * (w - 0.5) / 150000
    torch.testing.assert_close(delta, expected)
    # Calcium between thresholds depresses; above both thresholds both terms act.
    state["calcium"] = torch.tensor([[1.2, 2.0]], dtype=w.dtype)
    delta, state = kernels.step(rule, w, pre, post, state, 1)
    expected[0, 0] -= 200 * 0.25 / 150000
    expected[0, 1] += (321.808 * 0.25 - 200 * 0.75) / 150000
    torch.testing.assert_close(delta, expected)
    torch.testing.assert_close(
        state["calcium"], torch.tensor([[1.2, 2.0]], dtype=w.dtype) * math.exp(-1 / 20)
    )
    # Same-step events sum in calcium, unlike the pair timing convention.
    delta, state = kernels.step(
        rule, w, pre + 1, post + 1, kernels.initial_state(rule, w), 1
    )
    torch.testing.assert_close(state["calcium"], torch.full_like(w, 3))
    expected = (w * (1 - w) * (w - 0.5) + 321.808 * (1 - w) - 200 * w) / 150000
    torch.testing.assert_close(delta, expected)


def test_voltage_uses_voltage_gates_and_old_presynaptic_trace():
    rule = voltage_rule()
    w = torch.full((2, 2), 0.5, dtype=torch.float64)
    pre = torch.tensor([[1.0, 0.0]], dtype=w.dtype)
    zero = torch.zeros_like(pre)
    initial = kernels.initial_state(rule, w)
    # Steady subthreshold depolarisation drives LTD, without postsynaptic spikes.
    initial["voltage_minus"].fill_(-60)
    initial["voltage_plus"].fill_(-60)
    delta, _ = kernels.step(rule, w, pre, zero, initial, 1, voltage=zero - 60)
    torch.testing.assert_close(
        delta, torch.tensor([[-0.01, -0.01], [0, 0]], dtype=w.dtype)
    )
    # Prior presynaptic activity gates LTP; only the depolarised target changes.
    initial["pre"] = pre
    sample = torch.tensor([[20.0, -70.0]], dtype=w.dtype)
    delta, _ = kernels.step(rule, w, zero, zero, initial, 1, voltage=sample)
    filtered = 20 + (-60 - 20) * math.exp(-1 / 30)
    expected = 0.001 * math.exp(-0.1) * 65 * (filtered + 65)
    torch.testing.assert_close(
        delta, torch.tensor([[expected, 0], [0, 0]], dtype=w.dtype)
    )
    # New pre activity cannot gate LTP immediately.
    initial["pre"].zero_()
    delta, _ = kernels.step(
        replace(rule, a_minus=0 * lang.nA), w, pre, pre, initial, 1, voltage=sample
    )
    assert torch.count_nonzero(delta) == 0


def test_voltage_spike_excursion_and_missing_sample():
    rule = voltage_rule()
    w = torch.tensor([[0.5]])
    initial = kernels.initial_state(rule, w)
    initial["pre"].fill_(1)
    initial["voltage_plus"].fill_(-60)
    spike = torch.ones(1, 1)
    delta, _ = kernels.step(rule, w, spike * 0, spike, initial, 1, voltage=spike * -70)
    assert delta.item() > 0
    with pytest.raises(ValueError, match="voltage samples"):
        kernels.step(rule, w, spike, spike, initial, 1)


def test_scaling_proportional_integral_controller_and_composition():
    rule = pair(a_plus=0 * lang.nA, a_minus=0 * lang.nA)
    homeo = scaling()
    learning = Plasticity(rules={"p": rule}, scaling={"p": homeo})
    result = run(rule, [0, 0, 0], [0, 0, 0], learning=learning)
    expected, rate, integral = 0.5, 5.0, 0.0
    for _ in range(3):
        rate *= math.exp(-0.01)
        error = 5 - rate
        integral += 0.001 * error
        expected += expected * 0.001 * (0.1 * error + 0.01 * integral)
    assert weight(result) == pytest.approx(expected)
    standalone = run(
        rule, [0, 0, 0], [0, 0, 0], learning=Plasticity(rules={}, scaling={"p": homeo})
    )
    assert weight(standalone) == weight(result)
    active = run(rule, [0] * 3, [1] * 3, learning=learning)
    assert weight(active) < 0.5 < weight(result)


@pytest.mark.parametrize("factory", [nonlinear, voltage_rule, calcium])
@pytest.mark.parametrize("with_scaling", [False, True])
def test_new_rules_json_portable_continuation_and_bounds(
    factory, with_scaling, tmp_path
):
    rule = factory()
    learning = Plasticity(
        rules={"p": rule}, scaling={"p": scaling()} if with_scaling else {}
    )
    restored = Plasticity.from_json(json.loads(json.dumps(learning.json())))
    assert learning.json() == restored.json()
    pre, post = [1, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0]
    full = run(rule, pre, post, learning=restored, delay=1)
    first = run(rule, pre[:3], post[:3], learning=restored, delay=1)
    sim.save_runtime_state(tmp_path / "state", first.runtime_state)
    second = run(
        rule,
        pre[3:],
        post[3:],
        state=sim.load_runtime_state(tmp_path / "state"),
        learning=restored,
        delay=1,
    )
    assert 0 <= weight(full) <= 1
    torch.testing.assert_close(
        full.parameters["p.weight"], second.parameters["p.weight"], rtol=0, atol=0
    )
    for key, tensor in full.runtime_state.custom_state.items():
        torch.testing.assert_close(
            tensor, second.runtime_state.custom_state[key], rtol=0, atol=0
        )


def test_scaling_only_snapshot_continuation(tmp_path):
    rule = pair()
    learning = Plasticity(rules={}, scaling={"p": scaling()})
    first = run(rule, [0], [1], learning=learning)
    second = run(
        rule,
        [0],
        [0],
        learning=learning,
        state=first.runtime_state,
        snapshot=sim.SnapshotSpec(tmp_path / "snapshot", stage="both"),
    )
    torch.testing.assert_close(
        second.snapshots["start"].parameters["p.weight"], first.parameters["p.weight"]
    )
    torch.testing.assert_close(
        second.snapshots["end"].parameters["p.weight"], second.parameters["p.weight"]
    )


@pytest.mark.parametrize(
    "factory,changes",
    [
        (nonlinear, {"mu_plus": -1}),
        (nonlinear, {"mu_minus": float("nan")}),
        (nonlinear, {"weight_dependence": "multiplicative"}),
        (voltage_rule, {"theta_plus": -80 * lang.mV}),
        (voltage_rule, {"spike_voltage": -50 * lang.mV}),
        (voltage_rule, {"theta_minus": 0 * lang.nA}),
        (voltage_rule, {"tau_minus": 0 * lang.ms}),
        (calcium, {"rho_star": 1}),
        (calcium, {"c_pre": -1}),
        (calcium, {"theta_depression": float("nan")}),
        (calcium, {"gamma_potentiation": float("inf")}),
        (scaling, {"target_rate": 0 * lang.Hz}),
        (scaling, {"beta": -1}),
        (scaling, {"gamma": float("nan")}),
    ],
)
def test_new_spec_validation(factory, changes):
    with pytest.raises(ValueError):
        factory(**changes)


def test_composed_bounds_and_scaling_types():
    with pytest.raises(ValueError, match="bounds must agree"):
        Plasticity(rules={"p": pair()}, scaling={"p": scaling(w_max=2 * lang.nA)})
    with pytest.raises(TypeError, match="SynapticScaling"):
        Plasticity(rules={}, scaling={"p": pair()})
    with pytest.raises(TypeError, match="built-in"):
        Plasticity(rules={"p": scaling()})


def test_calcium_converges_to_continuous_deterministic_reference():
    from scipy.integrate import solve_ivp

    rule = calcium(tau_weight=10000 * lang.ms)

    # Integrate through both calcium threshold crossings with an independent ODE.
    def ode(t, y):
        c = 2 * math.exp(-t / 20)
        rho = y[0]
        drift = rho * (1 - rho) * (rho - 0.5)
        if c > 1.3:
            drift += 321.808 * (1 - rho)
        if c > 1:
            drift -= 200 * rho
        return [drift / 10000]

    reference = solve_ivp(ode, (0, 30), [0.4], rtol=1e-11, atol=1e-13, max_step=0.01).y[
        0, -1
    ]
    errors = []
    for dt in (1.0, 0.01):
        w = torch.tensor([[0.4]], dtype=torch.float64)
        state = {"calcium": torch.full_like(w, 2)}
        zero = torch.zeros(1, 1, dtype=w.dtype)
        for _ in range(round(30 / dt)):
            delta, state = kernels.step(rule, w, zero, zero, state, dt)
            w = w + delta
        errors.append(abs(w.item() - reference))
    assert errors[1] < errors[0] / 10
    assert errors[1] < 0.0001


def test_scaling_preserves_ratios_per_target_and_units():
    rule = scaling()
    w = torch.tensor([[0.2, 0.4], [0.4, 0.6]], dtype=torch.float64)
    rate = torch.tensor([[5.0, 5.0]], dtype=w.dtype)
    post = torch.tensor([[0.0, 1.0]], dtype=w.dtype)
    delta, rate, integral = kernels.scaling_step(
        rule, w, post, rate, torch.zeros_like(rate), 1
    )
    candidate = w + delta
    torch.testing.assert_close(candidate[0] / candidate[1], w[0] / w[1])
    assert candidate[0, 0] > w[0, 0]
    assert candidate[0, 1] < w[0, 1]
    assert integral[0, 0] > 0 > integral[0, 1]


def test_voltage_session_reads_membrane_samples_and_spike_interventions():
    rule = voltage_rule(tau_plus=1 * lang.ms)
    first = run(rule, [1], [0])
    sample = first.runtime_state.voltages["post"].item()
    filters = first.runtime_state.custom_state
    expected_filter = sample + (-70 - sample) * math.exp(-1)
    assert filters["plasticity/p/trace/voltage_plus"].item() == pytest.approx(
        expected_filter
    )
    second = run(rule, [0], [1], state=first.runtime_state)
    next_filter = 20 + (expected_filter - 20) * math.exp(-1)
    expected_increment = 0.001 * math.exp(-0.1) * 65 * max(next_filter + 65, 0)
    assert weight(second) == pytest.approx(min(1, weight(first) + expected_increment))
