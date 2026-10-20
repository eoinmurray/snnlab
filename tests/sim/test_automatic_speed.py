"""Automatic policies retain opt-outs, result ownership and numerical validity."""

import pytest
import torch

from snnlab import extensions as E
from snnlab import sim
from snnlab.sim.graph.precision import automatic_fused_optimizer
from snnlab.sim.requests.spec import ExecutionSpec
from tests.sim.test_training_replay import compare, request

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def test_defaults_and_cpu_fallback():
    for kind in ("build", "simulate", "infer", "train"):
        spec = ExecutionSpec(kind=kind)
        assert spec.compile_block_steps == "auto"
        assert spec.dataset_prefetch == 1
        assert spec.precision == "auto"
        assert spec.cuda_graphs == "auto"
    kwargs = request(recurrent=True)
    kwargs.pop("compile_block_steps")
    kwargs.pop("precision")
    result = sim.train(**kwargs)
    assert result.model.compile_block_steps == 1
    assert result.metrics["precision"] == {"policy": "auto", "autocast": "off"}
    assert not result.metrics["fused_optimizer"]


def test_extension_capture_contract_roundtrip(monkeypatch):
    monkeypatch.setattr(E, "_REGISTRY", {})
    E.register_objective(
        "test.capture/v1", "tests.sim._extension_callbacks:mse", capture_safe=True
    )
    descriptor = E.required(
        {"kind": "custom_objective", "definition": "test.capture/v1"}
    )[0]
    assert descriptor["capture_safe"] is True
    E.restore({"extensions": [descriptor]})
    assert E.get("objective", "test.capture/v1").capture_safe
    legacy = {key: value for key, value in descriptor.items() if key != "capture_safe"}
    E.restore({"extensions": [legacy]})
    assert E.get("objective", "test.capture/v1").capture_safe
    with pytest.raises(TypeError, match="boolean"):
        E.register_operation("test.bad/v1", lambda *args: None, capture_safe="yes")


@CUDA
def test_default_amp_replay_matches_same_precision_without_replay():
    kwargs = request(device="cuda", recurrent=True, block_steps=4)
    kwargs["precision"] = "auto"
    previous = torch.get_float32_matmul_precision()
    expected = sim.train(**kwargs, cuda_graphs="off")
    actual = sim.train(**kwargs)
    compare(actual, expected)
    assert torch.get_float32_matmul_precision() == previous
    assert actual.metrics["precision"]["autocast"] == "bfloat16"
    assert actual.metrics["fused_optimizer"]
    assert actual.metrics["cuda_graphs"]["replayed_updates"] == 6
    assert all(torch.isfinite(value).all() for value in actual.parameters.values())


@CUDA
def test_forward_replay_owns_results_and_reads_current_inputs():
    kwargs = request(device="cuda", recurrent=True, block_steps=4)
    built = sim.build(
        graph=kwargs["graph"],
        training=kwargs["training"],
        device="cuda",
        compile_block_steps=4,
    )
    model = built.model
    values = kwargs["input_bindings"][0].value.to("cuda")
    with torch.no_grad():
        for _ in range(3):
            first = model({"events": values}, diagnostics=False)
        assert first.metrics["cuda_graphs"]["replayed_calls"] == 1, first.metrics[
            "cuda_graphs"
        ]
        saved = {name: value.clone() for name, value in first.outputs.items()}
        second = model({"events": values * 0}, diagnostics=False)
        model.cuda_graphs = "off"
        reference = model({"events": values * 0}, diagnostics=False)
        for name, value in reference.outputs.items():
            torch.testing.assert_close(second.outputs[name], value)
            torch.testing.assert_close(first.outputs[name], saved[name])
        continued = model(
            {"events": values}, diagnostics=False, runtime_state=second.runtime_state
        )
        assert continued.runtime_state.completed_steps == 18


@CUDA
def test_fused_optimizer_respects_explicit_configuration():
    parameter = torch.nn.Parameter(torch.ones(2, device="cuda"))
    optimizer = torch.optim.Adam([parameter], fused=False)
    assert not automatic_fused_optimizer(optimizer)
    automatic = torch.optim.Adam([parameter])
    assert automatic_fused_optimizer(automatic)
    (parameter.square().sum()).backward()
    automatic.step()
    assert automatic.param_groups[0]["fused"]


def captured_ce(prediction, target, config):
    return torch.nn.functional.cross_entropy(prediction, target.long())


@CUDA
def test_capture_safe_objective_keeps_value_validation(monkeypatch):
    monkeypatch.setattr(E, "_REGISTRY", {})
    E.register_objective(
        "test.cuda_ce/v1", captured_ce, capture_safe=True, classification=True
    )
    kwargs = request(device="cuda", recurrent=True, block_steps=4)
    kwargs["training"]["objectives"][0].update(
        kind="custom_objective", definition="test.cuda_ce/v1", config={}
    )
    expected = sim.train(**kwargs, cuda_graphs="off")
    actual = sim.train(**kwargs)
    compare(actual, expected)
    assert actual.metrics["cuda_graphs"]["replayed_updates"] == 6


def test_deferred_finite_validation_still_names_invalid_callback():
    from snnlab.sim.graph.validation import (
        check_finite,
        deferred_validation,
        validate_pending,
    )

    with deferred_validation() as checks:
        check_finite(torch.tensor(float("nan")), "my_callback")
    with pytest.raises(ValueError, match="my_callback tensor must be finite"):
        validate_pending(checks)


def test_inference_selects_no_grad_without_hiding_input_gradients():
    kwargs = request(recurrent=True)
    model = sim.build(graph=kwargs["graph"], device="cpu", compile_mode=None).model
    values = kwargs["input_bindings"][0].value
    assert torch.is_grad_enabled()
    result = model({"events": values}, diagnostics=False)
    assert torch.is_grad_enabled()
    assert all(not value.requires_grad for value in result.outputs.values())
    differentiable = model(
        {"events": values.clone().requires_grad_()}, diagnostics=False
    )
    assert any(value.requires_grad for value in differentiable.outputs.values())
    sum(value.sum() for value in differentiable.outputs.values()).backward()


@CUDA
def test_default_forward_replay_reuses_shape_across_dataset_offsets():
    kwargs = request(recurrent=True)
    model = sim.build(graph=kwargs["graph"], device="cuda", compile_block_steps=4).model
    values = kwargs["input_bindings"][0].value.cuda()
    for index in range(4):
        result = model(
            {"events": values * index}, diagnostics=False, batch_offset=5 * index
        )
        assert torch.is_grad_enabled()
        assert (
            result.metrics["inference_retention"]["window"]["batch_offset"] == 5 * index
        )
    assert result.metrics["cuda_graphs"]["captures"] == 1
    assert result.metrics["cuda_graphs"]["replayed_calls"] == 2
    model.cuda_graphs = "off"
    expected = model({"events": values * 3}, diagnostics=False, batch_offset=15)
    for name, value in expected.outputs.items():
        torch.testing.assert_close(result.outputs[name], value)


def test_highest_precision_keeps_legacy_cuda_protocol_identity(monkeypatch):
    from snnlab.sim.requests import handlers

    monkeypatch.setattr(handlers, "resolve_device", lambda _: "cuda")
    highest = ExecutionSpec(kind="train", precision="highest", compile_block_steps=1)
    assert handlers._compilation_protocol({}, highest) == {}
    automatic = ExecutionSpec(kind="train", compile_block_steps=1)
    protocol = handlers._compilation_protocol({}, automatic)
    assert protocol["execution_precision"]["policy"] == "auto"
    with pytest.raises(ValueError, match="precision does not match"):
        handlers._compilation_protocol(protocol, highest)
