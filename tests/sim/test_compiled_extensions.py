"""Compiled extension numerics, contracts and explicit eager escape hatch."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from snnlab import extensions as E
from snnlab import lang
from snnlab.sim import GraphExecutor, plan_graph
from snnlab.sim.cli import execution_spec_from_args, parse_args
from snnlab.sim.requests.handlers import build
from snnlab.sim.requests.spec import ExecutionSpec
from tests.sim.test_execution_extensions import custom_network


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(E, "_REGISTRY", {})


@pytest.mark.parametrize("policy", ["strict", "structural"])
@pytest.mark.parametrize("custom_surrogate", [False, True])
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
def test_compiled_custom_state_gradients_and_continuation(
    policy, custom_surrogate, device
):
    # Use actual Inductor, including the custom autograd backward.
    net, _, _, projection = custom_network()
    E.register_surrogate(
        "compiled.triangle/v1", lambda x, config: (1 - x.abs()).clamp(min=0)
    )
    surrogate = (
        lang.CustomSurrogate("compiled.triangle/v1").json()
        if custom_surrogate
        else None
    )
    plan = plan_graph(lang.compile(net).graph)
    values = torch.ones(14, 2, 1, device=device)
    runs = []
    for mode in (None, "default"):
        model = GraphExecutor(
            plan,
            trainable_parameters=(projection.weight.id,),
            extension_validation=policy,
            surrogate=surrogate,
            compile_mode=mode,
        ).to(device)
        result = model({"inputs": values})
        result.outputs["spikes"].sum().backward()
        gradient = model.parameter_map()[projection.weight.id].grad.clone()
        with torch.no_grad():
            first = model({"inputs": values[:6]})
            second = model({"inputs": values[6:]}, runtime_state=first.runtime_state)
        for name in result.outputs:
            torch.testing.assert_close(
                torch.cat([first.outputs[name], second.outputs[name]]),
                result.outputs[name],
                rtol=2e-5,
                atol=2e-5,
            )
        runs.append((result, gradient))
    for group in ("outputs", "diagnostics"):
        for name in getattr(runs[0][0], group):
            torch.testing.assert_close(
                getattr(runs[0][0], group)[name],
                getattr(runs[1][0], group)[name],
                rtol=2e-5,
                atol=2e-5,
            )
    for name, value in runs[0][0].runtime_state.custom_state.items():
        torch.testing.assert_close(
            value, runs[1][0].runtime_state.custom_state[name], rtol=2e-5, atol=2e-5
        )
    torch.testing.assert_close(runs[0][1], runs[1][1], rtol=2e-5, atol=2e-5)


def test_compilation_defaults_metrics_and_cli_opt_out():
    net, *_ = custom_network()
    spec = ExecutionSpec(kind="build", graph=lang.compile(net).graph)
    result = build(spec)
    assert result.model.compile_mode == spec.compile_mode == "default"
    assert result.metrics["compiled_populations"] == ["cells"]
    assert result.metrics["compiled_projections"] == ["input"]
    eager = build(replace(spec, compile_mode=None))
    assert eager.metrics["compiled_populations"] == []
    assert eager.metrics["compiled_projections"] == []
    assert (
        execution_spec_from_args(parse_args(["sim", "--device", "cpu"])).compile_mode
        == "default"
    )
    assert (
        execution_spec_from_args(SimpleNamespace(mode="sim", device="cpu")).compile_mode
        == "default"
    )
    assert (
        execution_spec_from_args(
            parse_args(["sim", "--device", "cpu", "--compile-mode", "eager"])
        ).compile_mode
        is None
    )


def test_dynamics_registry_validation_is_not_repeated_per_step(monkeypatch):
    net, *_ = custom_network()
    model = GraphExecutor(plan_graph(lang.compile(net).graph), compile_mode=None)
    original = E.resolve
    calls = []

    def resolve(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(E, "resolve", resolve)
    model({"inputs": torch.ones(1, 2, 1)})
    once = list(calls)
    calls.clear()
    model({"inputs": torch.ones(20, 2, 1)})
    assert calls == once


def test_untraceable_update_requires_explicit_eager():
    net, *_ = custom_network()
    original = E.get("neuron", "test.adaptive/v1")

    def step(context):
        # Host decisions on tensor data cannot be captured as one tensor graph.
        if context.excitatory.sum().item() > 0:
            return original.function(context)
        return original.function(context)

    E.register_neuron(
        "compiled.host/v1",
        step,
        initialize=original.initialize,
        state_units={"adaptation": "nA"},
    )
    net.populations[0]["neuron"] = lang.CustomNeuron(
        "compiled.host/v1", threshold=0.9
    ).json()
    plan = plan_graph(lang.compile(net).graph)
    with pytest.raises((torch._dynamo.exc.Unsupported, torch._dynamo.exc.UserError)):
        GraphExecutor(plan)({"inputs": torch.ones(2, 1, 1)})
    assert GraphExecutor(plan, compile_mode=None)(
        {"inputs": torch.ones(2, 1, 1)}
    ).outputs


def test_compiled_weight_constraint_matches_eager():
    E.register_constraint(
        "compiled.bounds/v1",
        lambda value, config: value.clamp(-config["limit"], config["limit"]),
    )
    net, _, _, projection = custom_network()
    net.parameters[0]["constraint"] = lang.CustomConstraint(
        "compiled.bounds/v1", limit=0.1
    ).json()
    plan = plan_graph(lang.compile(net).graph)
    for mode in (None, "default"):
        model = GraphExecutor(plan, compile_mode=mode)
        weight = model.parameter_map()[projection.weight.id]
        with torch.no_grad():
            weight.fill_(-2)
        model.enforce_constraints()
        torch.testing.assert_close(weight, torch.full_like(weight, -0.1))
