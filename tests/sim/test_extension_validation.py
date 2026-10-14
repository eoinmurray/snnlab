"""Strict diagnostics and synchronization-free structural extension contracts."""

import pytest
import torch

from snnlab import extensions as E
from snnlab import lang
from snnlab.lang import training as recipe
from snnlab.sim import extensions as X
from snnlab.sim.execution import ExecutionSpec, GraphExecutor, build, plan_graph
from tests.sim.test_execution_extensions import custom_network


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(E, "_REGISTRY", {})


def test_state_scans_each_tensor_once():
    state = {
        "voltage": torch.zeros(2, 3),
        "refractory": torch.zeros(2, 3, dtype=torch.long),
    }
    with torch.profiler.profile() as profile:
        X.checked_state(state, state, "probe")
    assert (
        next(row.count for row in profile.key_averages() if row.key == "aten::isfinite")
        == 2
    )


@pytest.mark.parametrize("policy", ["strict", "structural"])
@pytest.mark.parametrize("bad", ["shape", "dtype", "keys", "type", "device"])
def test_structural_contracts_remain_enforced(policy, bad):
    template = {"voltage": torch.zeros(2, 3)}
    value = dict(template)
    if bad == "shape":
        value["voltage"] = torch.zeros(3)
    elif bad == "dtype":
        value["voltage"] = torch.zeros(2, 3, dtype=torch.float64)
    elif bad == "keys":
        value["extra"] = torch.zeros(2, 3)
    elif bad == "device":
        value["voltage"] = torch.empty(2, 3, device="meta")
    else:
        value["voltage"] = None
    with pytest.raises((TypeError, ValueError)):
        X.checked_state(value, template, "probe", check_values=policy == "strict")


@pytest.mark.parametrize("bad", ["state", "spikes", "binary", "derivative"])
def test_strict_rejects_invalid_callback_values(bad):
    net, _, _, projection = custom_network()
    original = E.get("neuron", "test.adaptive/v1")

    def step(context):
        state, spikes = original.function(context)
        if bad == "state":
            state["voltage"] = state["voltage"] * float("nan")
        if bad == "spikes":
            spikes = spikes * float("inf")
        if bad == "binary":
            spikes = spikes + 0.5
        return state, spikes

    E.register_neuron(
        "test.invalid/v1",
        step,
        initialize=original.initialize,
        state_units={"adaptation": "nA"},
    )
    net.populations[0]["neuron"] = lang.CustomNeuron(
        "test.invalid/v1", threshold=0.9
    ).json()
    E.register_surrogate(
        "test.invalid/v1", lambda x, config: torch.full_like(x, float("nan"))
    )
    model = GraphExecutor(
        plan_graph(lang.compile(net).graph),
        trainable_parameters=(projection.weight.id,),
        surrogate=recipe.CustomSurrogate("test.invalid/v1").json()
        if bad == "derivative"
        else None,
    )
    with pytest.raises(ValueError, match="finite|binary"):
        result = model({"inputs": torch.ones(4, 2, 1)})
        result.outputs["spikes"].sum().backward()


def test_cached_surrogate_and_structural_derivative_contract(monkeypatch):
    E.register_surrogate("test.derivative/v1", lambda x, config: torch.ones_like(x))
    dispatch = X.make_spike(
        slope=25,
        custom=recipe.CustomSurrogate("test.derivative/v1").json(),
        check_values=False,
    )
    monkeypatch.setattr(
        E, "resolve", lambda *a: pytest.fail("surrogate resolved again")
    )
    x = torch.tensor([-1.0, 1.0], requires_grad=True)
    with torch.profiler.profile() as profile:
        (dispatch(x) + dispatch(x)).sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 2))
    assert not {"aten::isfinite", "aten::_local_scalar_dense"} & {
        r.key for r in profile.key_averages()
    }


@pytest.mark.parametrize("check_values", [True, False])
def test_malformed_derivative_fails_in_both_policies(check_values):
    E.register_surrogate("test.bad/v1", lambda x, config: torch.ones(9))
    dispatch = X.make_spike(
        slope=25,
        custom=recipe.CustomSurrogate("test.bad/v1").json(),
        check_values=check_values,
    )
    with pytest.raises(ValueError, match="shape"):
        dispatch(torch.ones(2, requires_grad=True)).sum().backward()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_policy_agreement_state_outputs_gradients_and_continuation(device):
    net, _, _, projection = custom_network()
    E.register_surrogate(
        "test.triangle/v1", lambda x, config: (1 - x.abs()).clamp(min=0)
    )
    plan = plan_graph(lang.compile(net).graph)
    values = torch.ones(14, 2, 1, device=device)
    results = []
    for policy in ("strict", "structural"):
        model = GraphExecutor(
            plan,
            trainable_parameters=(projection.weight.id,),
            surrogate=recipe.CustomSurrogate("test.triangle/v1").json(),
            extension_validation=policy,
        ).to(device)
        full = model({"inputs": values})
        full.outputs["spikes"].sum().backward()
        gradient = model.parameter_map()[projection.weight.id].grad.clone()
        first = model({"inputs": values[:6]})
        second = model({"inputs": values[6:]}, runtime_state=first.runtime_state)
        for name in full.outputs:
            torch.testing.assert_close(
                torch.cat([first.outputs[name], second.outputs[name]]),
                full.outputs[name],
                rtol=0,
                atol=0,
            )
        results.append((full, gradient))
    for name in results[0][0].outputs:
        torch.testing.assert_close(
            results[0][0].outputs[name], results[1][0].outputs[name], rtol=0, atol=0
        )
    for name in results[0][0].runtime_state.custom_state:
        torch.testing.assert_close(
            results[0][0].runtime_state.custom_state[name],
            results[1][0].runtime_state.custom_state[name],
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)


def test_execution_spec_policy_propagates_and_is_recorded():
    net, *_ = custom_network()
    result = build(
        ExecutionSpec(
            kind="build",
            graph=lang.compile(net).graph,
            extension_validation="structural",
        )
    )
    assert (
        result.model.extension_validation
        == result.metrics["extension_validation"]
        == "structural"
    )
    with pytest.raises(ValueError, match="extension_validation"):
        ExecutionSpec(kind="build", extension_validation="off")
    with pytest.raises(ValueError, match="extension_validation"):
        GraphExecutor(result.model.plan, extension_validation="off")


@pytest.mark.parametrize("policy", ["strict", "structural"])
def test_step_value_scan_counts(policy):
    net, *_ = custom_network()
    model = GraphExecutor(
        plan_graph(lang.compile(net).graph), extension_validation=policy
    )
    counts = []
    for steps in (1, 10):
        with torch.profiler.profile() as profile:
            model({"inputs": torch.ones(steps, 2, 1)})
        counts.append(
            sum(
                row.count
                for row in profile.key_averages()
                if row.key == "aten::isfinite"
            )
        )
    # Initialization stays strict; only strict mode adds scans per timestep.
    assert counts[1] > counts[0] if policy == "strict" else counts[1] == counts[0]


@pytest.mark.parametrize("policy", ["strict", "structural"])
def test_initialization_remains_strict(policy):
    net, *_ = custom_network()
    original = E.get("neuron", "test.adaptive/v1")

    def initialize(context):
        state = original.initialize(context)
        state["voltage"].fill_(float("nan"))
        return state

    E.register_neuron(
        "test.nonfinite/v1",
        original.function,
        initialize=initialize,
        state_units={"adaptation": "nA"},
    )
    net.populations[0]["neuron"] = lang.CustomNeuron(
        "test.nonfinite/v1", threshold=0.9
    ).json()
    model = GraphExecutor(
        plan_graph(lang.compile(net).graph), extension_validation=policy
    )
    with pytest.raises(ValueError, match="finite"):
        model({"inputs": torch.ones(2, 1, 1)})


@pytest.mark.parametrize("mode", ["sim", "train"])
def test_cli_policy_maps_to_execution_spec(mode):
    from snnlab.sim.execution import execution_spec_from_args
    from snnlab.sim.tool import parse_args

    assert (
        execution_spec_from_args(
            parse_args([mode, "--device", "cpu"])
        ).extension_validation
        == "strict"
    )
    spec = execution_spec_from_args(
        parse_args([mode, "--device", "cpu", "--extension-validation", "structural"])
    )
    assert spec.extension_validation == "structural"
    assert "extension_validation" not in spec.options
