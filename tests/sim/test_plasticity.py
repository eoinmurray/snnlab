"""Local-learning timing, ownership, persistence and training isolation."""

import json
import math
from dataclasses import replace

import pytest
import torch

from snnlab import extensions as E
from snnlab import lang, sim
from snnlab.plasticity import CustomPlasticity, PairSTDP, Plasticity
from snnlab.sim.requests.spec import ExecutionSpec


def graph(*, delay=0, recurrent=False, constraint=None):
    net = lang.Network("plastic", dt=1 * lang.ms)
    drive = net.input(
        "drive", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    pop = net.population("post", size=2, neuron=lang.LIF(), spiking=True)
    net.connect(
        pop.spikes if recurrent else drive,
        pop.excitatory,
        name="plastic",
        synapse=lang.ExponentialCurrent(tau=1 * lang.ms),
        weight=lang.Constant(0.5),
        initialization_scaling="direct",
        constraint=constraint,
        delay=delay * lang.ms,
        connection="recurrent" if recurrent else "feedforward",
    )
    net.output("spikes", pop.spikes)
    return lang.compile(net, target=None).graph


def rule(**changes):
    return PairSTDP(
        **dict(
            a_plus=0.1 * lang.nA,
            a_minus=0.2 * lang.nA,
            w_min=0 * lang.nA,
            w_max=1 * lang.nA,
            tau_pre=10 * lang.ms,
            tau_post=20 * lang.ms,
        )
        | changes
    )


def run(pre, post, *, g=None, request=None, runtime_state=None, retain=True, **kwargs):
    post = torch.as_tensor(post, dtype=torch.float32).reshape(-1, 1, 2)
    pre = torch.as_tensor(pre, dtype=torch.float32).reshape(-1, 1, 2)
    start = runtime_state.completed_steps if runtime_state is not None else 0
    return sim.simulate(
        graph=graph() if g is None else g,
        device="cpu",
        input_bindings=[sim.DenseArrayBinding("drive", pre)],
        interventions=[
            sim.ReplaySpikes(
                "post",
                sim.DenseSpikeReplay.from_tensor(post, dt_ms=1, start_step=start),
            )
        ],
        plasticity=request or Plasticity(rules={"plastic": rule()}),
        runtime_state=runtime_state,
        retain_parameters=retain,
        **kwargs,
    )


@pytest.mark.parametrize(
    "pre,post,expected",
    [
        ([[1, 0], [0, 0]], [[0, 0], [0, 1]], 0.5 + 0.1 * math.exp(-0.1)),
        ([[0, 0], [1, 0]], [[0, 1], [0, 0]], 0.5 - 0.2 * math.exp(-0.05)),
        ([[1, 0]], [[0, 1]], 0.5),
    ],
)
def test_pair_timing_and_matrix_orientation(pre, post, expected):
    result = run(pre, post)
    assert result.parameters["plastic.weight"][0, 1].item() == pytest.approx(expected)
    assert result.parameters["plastic.weight"][1, 0].item() == 0.5
    assert result.metrics["plasticity"]["execution"] == "eager"
    assert "online plasticity" in result.metrics["execution_path"]["fallback_reasons"]
    assert result.metrics["performance"]["resolved"]["compile_mode"] is None
    assert not result.parameters["plastic.weight"].requires_grad


def test_all_pairs_and_delayed_arrivals():
    pre = [[1, 0], [1, 0], [0, 0], [0, 0]]
    post = [[0, 0], [0, 0], [0, 0], [0, 1]]
    result = run(pre, post, g=graph(delay=1))
    expected = 0.5 + 0.1 * (math.exp(-0.2) + math.exp(-0.1))
    assert result.parameters["plastic.weight"][0, 1].item() == pytest.approx(expected)


def test_saved_chunk_continuation_matches_full_and_owns_copies(tmp_path):
    pre = [[1, 0], [0, 0], [1, 1], [0, 0], [0, 1]]
    post = [[0, 0], [0, 1], [0, 0], [1, 0], [0, 0]]
    g = graph(delay=2)
    full = run(pre, post, g=g)
    first = run(pre[:2], post[:2], g=g, retain=False)
    before = first.runtime_state.detached()
    sim.save_runtime_state(tmp_path / "state", first.runtime_state)
    state = sim.load_runtime_state(tmp_path / "state")
    second = run(pre[2:], post[2:], g=g, runtime_state=state)
    torch.testing.assert_close(
        full.parameters["plastic.weight"],
        second.parameters["plastic.weight"],
        rtol=0,
        atol=0,
    )
    for key, value in full.runtime_state.custom_state.items():
        torch.testing.assert_close(
            value, second.runtime_state.custom_state[key], rtol=0, atol=0
        )
    for key, value in before.custom_state.items():
        torch.testing.assert_close(
            value, first.runtime_state.custom_state[key], rtol=0, atol=0
        )
    with pytest.raises(ValueError, match="incompatible"):
        run(
            pre[2:],
            post[2:],
            g=g,
            runtime_state=state,
            request=Plasticity(rules={"plastic": rule(tau_pre=11 * lang.ms)}),
        )
    with pytest.raises(ValueError, match="incompatible"):
        sim.GraphExecutor(sim.plan_graph(g), compile_mode=None)(
            {"drive": torch.zeros(1, 1, 2)}, runtime_state=state
        )


def init_custom(ctx):
    assert not torch.is_grad_enabled()
    return {"count": ctx.weights.new_zeros(())}


def custom_step(ctx):
    ctx.pre_spikes.zero_()
    return torch.full_like(ctx.weights, ctx.config["increment"]), {
        "count": ctx.state["count"] + 1
    }


E.register_plasticity(
    "tests.increment/v1",
    custom_step,
    initialize=init_custom,
    state_units={"count": "1"},
)


def custom_request(definition="tests.increment/v1", **config):
    return Plasticity(
        rules={
            "plastic": CustomPlasticity(
                definition=definition,
                config=config or {"increment": 0.1},
                w_min=0 * lang.nA,
                w_max=1 * lang.nA,
            )
        }
    )


def test_custom_json_state_bounds_and_no_input_mutation():
    request = custom_request()
    restored = Plasticity.from_json(json.loads(json.dumps(request.json())))
    assert restored.json() == request.json()
    result = run([[1, 0]] * 8, [[0, 0]] * 8, request=restored)
    torch.testing.assert_close(result.parameters["plastic.weight"], torch.ones(2, 2))
    assert (
        result.runtime_state.custom_state["plasticity/plastic/trace/count"].item() == 8
    )
    torch.testing.assert_close(
        result.model.parameter_map()["plastic.weight"],
        result.parameters["plastic.weight"],
    )


def test_recurrent_autapses_remain_excluded():
    result = run(
        [[0, 0]] * 3, [[1, 1]] * 3, g=graph(recurrent=True), request=custom_request()
    )
    torch.testing.assert_close(
        result.parameters["plastic.weight"].diagonal(), torch.zeros(2)
    )
    assert result.parameters["plastic.weight"][0, 1].item() == pytest.approx(0.8)


@pytest.mark.parametrize("failure", ["shape", "nan", "keys", "dtype"])
def test_custom_validation_and_failed_call_does_not_change_model(failure):
    def bad(ctx):
        delta = torch.ones_like(ctx.weights)
        state = ctx.state
        if failure == "shape":
            delta = delta[:1]
        elif failure == "nan":
            delta.fill_(float("nan"))
        elif failure == "keys":
            state = {}
        else:
            state = {"count": ctx.state["count"].double()}
        return delta, state

    name = f"tests.bad_{failure}/v1"
    E.register_plasticity(name, bad, initialize=init_custom, state_units={"count": "1"})
    model = sim.GraphExecutor(sim.plan_graph(graph()), compile_mode=None)
    before = model.parameter_map()["plastic.weight"].detach().clone()
    with pytest.raises((ValueError, TypeError)):
        model({"drive": torch.ones(2, 1, 2)}, plasticity=custom_request(name))
    torch.testing.assert_close(
        model.parameter_map()["plastic.weight"], before, rtol=0, atol=0
    )


def test_training_and_batch_boundaries():
    request = Plasticity(rules={"plastic": rule()})
    with pytest.raises(ValueError, match="requires simulate"):
        ExecutionSpec(kind="train", plasticity=request)
    with pytest.raises(TypeError, match="plasticity"):
        sim.train(plasticity=request)
    model = sim.GraphExecutor(sim.plan_graph(graph()), compile_mode=None)
    with pytest.raises(ValueError, match="batch_size=1"):
        model({"drive": torch.zeros(1, 2, 2)}, plasticity=request)
    trainable = sim.GraphExecutor(
        sim.plan_graph(graph()),
        compile_mode=None,
        trainable_parameters=["plastic.weight"],
    )
    with pytest.raises(ValueError, match="without gradients"):
        trainable({"drive": torch.zeros(1, 1, 2)}, plasticity=request)


def test_invalid_unit_unknown_projection_and_corrupt_state():
    with pytest.raises(ValueError, match="require nA"):
        run(
            [[0, 0]],
            [[0, 0]],
            request=Plasticity(
                rules={
                    "plastic": PairSTDP(
                        a_plus=0.1 * lang.uS,
                        a_minus=0.1 * lang.uS,
                        w_min=0 * lang.uS,
                        w_max=1 * lang.uS,
                    )
                }
            ),
        )
    with pytest.raises(ValueError, match="unknown plasticity projection"):
        run([[0, 0]], [[0, 0]], request=Plasticity(rules={"missing": rule()}))
    first = run([[1, 0]], [[0, 0]])
    broken = replace(first.runtime_state, custom_state={})
    with pytest.raises(ValueError, match="tensor keys"):
        run([[0, 0]], [[0, 1]], runtime_state=broken)


def test_all_pairs_and_arrival_delay():
    result = run([[1, 0], [1, 0], [0, 0]], [[0, 0], [0, 0], [0, 1]])
    expected = 0.5 + 0.1 * (math.exp(-0.2) + math.exp(-0.1))
    assert result.parameters["plastic.weight"][0, 1].item() == pytest.approx(expected)
    delayed = run([[1, 0], [0, 0], [0, 0]], [[0, 0], [0, 1], [0, 0]], g=graph(delay=2))
    # Arrival at step 2 follows the post spike at step 1: depression, not potentiation.
    assert delayed.parameters["plastic.weight"][0, 1].item() == pytest.approx(
        0.5 - 0.2 * math.exp(-0.05)
    )


def test_updated_weights_drive_the_next_step():
    g = graph()
    neuron = g["populations"][0]["neuron"]
    neuron.update(
        resting_mv=0,
        reset_mv=0,
        initial_voltage_mv=0,
        threshold_mv=1.4,
        tau_mem={"value": 100, "unit": "ms"},
    )
    model = sim.GraphExecutor(sim.plan_graph(g), compile_mode=None)
    baseline = model({"drive": torch.tensor([[[1.0, 0]], [[1.0, 0]]])})
    adapted = model(
        {"drive": torch.tensor([[[1.0, 0]], [[1.0, 0]]])},
        plasticity=custom_request(increment=0.5),
    )
    assert baseline.outputs["spikes"].sum() == 0
    assert adapted.outputs["spikes"][0].sum() == 0
    assert adapted.outputs["spikes"][1].sum() == 2


def test_resumed_snapshots_capture_dynamic_weights(tmp_path):
    first = run([[0, 0]], [[0, 0]], request=custom_request())
    second = run(
        [[0, 0]],
        [[0, 0]],
        request=custom_request(),
        runtime_state=first.runtime_state,
        snapshot=sim.SnapshotSpec(tmp_path / "snapshot", stage="both", parameters=True),
    )
    torch.testing.assert_close(
        second.snapshots["start"].parameters["plastic.weight"],
        first.parameters["plastic.weight"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        second.snapshots["end"].parameters["plastic.weight"],
        second.parameters["plastic.weight"],
        rtol=0,
        atol=0,
    )


def test_invalid_graph_shapes_and_sparse_initialization():
    g = graph()
    g["parameters"][0]["initializer"]["initial_zero_fraction"] = 0.5
    with pytest.raises(ValueError, match="sparse initialization"):
        run([[0, 0]], [[0, 0]], g=g)
    g = graph()
    g["projections"].append({**g["projections"][0], "id": "shared"})
    with pytest.raises(ValueError, match="unshared weights"):
        run([[0, 0]], [[0, 0]], g=g)


@pytest.mark.parametrize(
    "field,value",
    [
        ("a_plus", -0.1 * lang.nA),
        ("tau_pre", 0 * lang.ms),
        ("tau_post", float("nan") * lang.ms),
        ("w_min", -1 * lang.nA),
        ("w_max", 0 * lang.nA),
    ],
)
def test_rule_quantities_are_validated(field, value):
    with pytest.raises(ValueError):
        rule(**{field: value})


def cap_weight(value, config):
    return value.clamp(max=config["maximum"])


extensions_name = "tests.plasticity_cap/v1"
E.register("constraint", extensions_name, cap_weight)


def test_constraints_and_private_callback_inputs():
    g = graph(constraint=lang.CustomConstraint(extensions_name, maximum=0.55))
    model = sim.GraphExecutor(sim.plan_graph(g), compile_mode=None)
    inputs = torch.ones(2, 1, 2)
    original = inputs.clone()
    result = model({"drive": inputs}, plasticity=custom_request())
    torch.testing.assert_close(inputs, original, rtol=0, atol=0)
    torch.testing.assert_close(
        result.parameters["plastic.weight"], torch.full((2, 2), 0.55)
    )
    assert not result.outputs["spikes"].requires_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_runtime_dtype_and_custom_continuation(dtype):
    request = custom_request()
    g = graph()
    inputs = {"drive": torch.ones(2, 1, 2, dtype=dtype)}
    model = sim.GraphExecutor(sim.plan_graph(g), compile_mode=None).to(dtype=dtype)
    full = model(inputs, plasticity=request)
    first = sim.GraphExecutor(sim.plan_graph(g), compile_mode=None).to(dtype=dtype)(
        {"drive": inputs["drive"][:1]}, plasticity=request
    )
    second = sim.GraphExecutor(sim.plan_graph(g), compile_mode=None).to(dtype=dtype)(
        {"drive": inputs["drive"][1:]},
        plasticity=request,
        runtime_state=first.runtime_state,
    )
    torch.testing.assert_close(
        full.parameters["plastic.weight"],
        second.parameters["plastic.weight"],
        rtol=0,
        atol=0,
    )
    assert (
        second.runtime_state.custom_state["plasticity/plastic/trace/count"].item() == 2
    )
    assert second.parameters["plastic.weight"].dtype == dtype


def test_simulation_freezes_supplied_training_membership():
    result = run(
        [[1, 0]],
        [[0, 1]],
        training={
            "resolved_parameters": {"trainable": ["plastic.weight"]},
        },
    )
    assert not result.model.parameter_map()["plastic.weight"].requires_grad


def test_scaling_composes_with_custom_rule_and_preserves_callback_state():
    from snnlab.plasticity import SynapticScaling

    request = replace(
        custom_request(),
        scaling={
            "plastic": SynapticScaling(
                target_rate=5 * lang.Hz,
                tau_rate=100 * lang.ms,
                beta=0.1,
                gamma=0.01,
                w_min=0 * lang.nA,
                w_max=1 * lang.nA,
            ),
        },
    )
    request = Plasticity.from_json(json.loads(json.dumps(request.json())))
    full = run([[0, 0]] * 3, [[0, 1]] * 3, request=request)
    first = run([[0, 0]], [[0, 1]], request=request)
    second = run(
        [[0, 0]] * 2, [[0, 1]] * 2, request=request, runtime_state=first.runtime_state
    )
    torch.testing.assert_close(
        full.parameters["plastic.weight"],
        second.parameters["plastic.weight"],
        rtol=0,
        atol=0,
    )
    assert full.runtime_state.custom_state["plasticity/plastic/trace/count"].item() == 3
    assert (
        full.parameters["plastic.weight"][0, 0]
        > full.parameters["plastic.weight"][0, 1]
    )
