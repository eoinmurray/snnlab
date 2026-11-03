"""Forward extraction compatibility and numerical regression coverage."""

import dataclasses
import inspect
from pathlib import Path

import pytest
import torch

from snnlab import lang
from snnlab.sim import GraphExecutor, plan_graph
from snnlab.sim.graph import executor as executor_module
from snnlab.sim.graph import forward_pass
from tests.sim.test_gpu_efficiency import circuit


def _pack(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if dataclasses.is_dataclass(value):
        return {
            field.name: _pack(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {key: _pack(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_pack(item) for item in value)
    return value


def _assert_equal(expected, actual):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            _assert_equal(expected[name], actual[name])
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            _assert_equal(left, right)
    else:
        assert actual == expected


def test_forward_impl_forwards_every_keyword_and_returns_same_tuple(monkeypatch):
    signature = inspect.signature(GraphExecutor._forward_impl)
    sentinel = object()
    inputs = {"events": object()}
    options = {
        name: object()
        for name, parameter in signature.parameters.items()
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY
    }
    expected = (object(), object())

    def run(executor, supplied_inputs, **kwargs):
        assert executor is sentinel
        assert supplied_inputs is inputs
        assert kwargs == options
        return expected

    monkeypatch.setattr(executor_module, "run_forward_pass", run)
    assert GraphExecutor._forward_impl(sentinel, inputs, **options) is expected
    extracted = inspect.signature(forward_pass.run_forward_pass)
    for name in options:
        assert signature.parameters[name] == extracted.parameters[name]


def test_delay_buffer_reexport_is_the_same_class():
    assert executor_module.DelayBuffer is forward_pass.DelayBuffer


def test_recurrent_forward_matches_pre_extraction_cpu_reference():
    with torch.random.fork_rng():
        torch.manual_seed(103)
        graph, parameter = circuit(lang.CUBA_LIF)
        model = GraphExecutor(plan_graph(graph), trainable_parameters=(parameter,))
        inputs = torch.ones(41, 2, 2, requires_grad=True)
        result = model({"events": inputs})
        assert result.model is model
        result.outputs["voltage"].sum().backward()
        assert inputs.grad.count_nonzero() > 0
        assert model.parameter_map()[parameter].grad.count_nonzero() > 0
        actual = _pack(
            {
                "outputs": result.outputs,
                "diagnostics": result.diagnostics,
                "state": result.runtime_state,
                "metrics": result.metrics,
                "parameters": result.parameters,
                "final_state": result.final_state,
                "input_gradient": inputs.grad,
                "parameter_gradient": model.parameter_map()[parameter].grad,
            }
        )
    # Captured on 0728656 before extraction: CPU CUBA_LIF, seed 103, 41 steps.
    expected = torch.load(
        Path(__file__).parent / "fixtures" / "forward-pass-reference.pt",
        weights_only=True,
    )
    _assert_equal(expected, actual)


@pytest.mark.parametrize("split", [19, 20])
def test_resumed_recurrent_forward_preserves_outputs_and_state(split):
    graph, _ = circuit(lang.CUBA_LIF)
    model = GraphExecutor(plan_graph(graph))
    inputs = torch.ones(41, 2, 2)
    with torch.no_grad():
        whole = model({"events": inputs})
        first = model({"events": inputs[:split]})
        tail = model({"events": inputs[split:]}, runtime_state=first.runtime_state)
    for name in ("voltage", "spikes"):
        torch.testing.assert_close(
            torch.cat((first.outputs[name], tail.outputs[name])),
            whole.outputs[name],
            rtol=0,
            atol=0,
        )
    for name in whole.diagnostics:
        torch.testing.assert_close(
            torch.cat((first.diagnostics[name], tail.diagnostics[name])),
            whole.diagnostics[name],
            rtol=0,
            atol=0,
        )
    _assert_equal(_pack(whole.runtime_state), _pack(tail.runtime_state))
