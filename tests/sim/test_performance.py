"""Public performance policies preserve execution, compatibility and reporting."""

import json
from dataclasses import FrozenInstanceError, asdict, fields, replace

import numpy as np
import pytest
import torch

from snnlab import sim
from snnlab.sim.requests import handlers
from snnlab.sim.requests.spec import ExecutionSpec
from snnlab.sim.training import trainer
from tests.sim.test_execution_api import inputs
from tests.sim.test_training_replay import compare, request


def test_policy_is_immutable_keyword_only_and_has_unchanged_defaults():
    policy = sim.Performance()
    assert asdict(policy) == {
        "compile_mode": "default",
        "compile_block_steps": "auto",
        "cuda_graphs": "auto",
        "precision": "auto",
        "fused_optimizer": "auto",
        "dataset_prefetch": 1,
        "projection_mode": "fast",
    }
    with pytest.raises(FrozenInstanceError):
        policy.precision = "highest"
    with pytest.raises(TypeError):
        sim.Performance(None)
    for kind in ("build", "simulate", "train"):
        spec = ExecutionSpec(kind=kind)
        assert asdict(policy) == {f.name: getattr(spec, f.name) for f in fields(policy)}


@pytest.mark.parametrize(
    "name,value",
    [
        ("compile_mode", "eager"),
        ("compile_block_steps", 0),
        ("compile_block_steps", 65),
        ("compile_block_steps", True),
        ("compile_block_steps", 2.5),
        ("cuda_graphs", "on"),
        ("precision", "float16"),
        ("fused_optimizer", True),
        ("dataset_prefetch", -1),
        ("dataset_prefetch", 9),
        ("dataset_prefetch", True),
        ("projection_mode", "unknown"),
    ],
)
def test_invalid_settings_fail_at_construction_and_internal_boundary(name, value):
    with pytest.raises(ValueError, match=name):
        sim.Performance(**{name: value})
    with pytest.raises(ValueError, match=name):
        ExecutionSpec(kind="train", **{name: value})


@pytest.mark.parametrize("name", ["build", "simulate", "train"])
def test_all_public_operations_resolve_policy_before_dispatch(name, monkeypatch):
    observed = []
    sentinel = object()

    def execute(spec):
        observed.append(spec)
        return sentinel

    monkeypatch.setattr(trainer if name == "train" else handlers, name, execute)
    function = getattr(sim, name)
    policy = sim.Performance(
        compile_mode=None,
        compile_block_steps=3,
        cuda_graphs="off",
        precision="highest",
        dataset_prefetch=0,
        fused_optimizer="off",
        projection_mode="strict",
    )
    for supplied in (None, sim.Performance(), policy):
        assert function(performance=supplied) is sentinel
        expected = supplied or sim.Performance()
        assert asdict(expected) == {
            f.name: getattr(observed[-1], f.name) for f in fields(expected)
        }
    with pytest.warns(DeprecationWarning, match="Performance"):
        assert function(compile_mode=None, projection_mode="strict") is sentinel
    assert observed[-1].compile_mode is None
    assert observed[-1].projection_mode == "strict"
    assert observed[-1].dataset_prefetch == 1


@pytest.mark.parametrize("name", ["build", "simulate", "train"])
def test_ambiguous_policies_and_wrong_types_fail_before_execution(name):
    function = getattr(sim, name)
    with pytest.raises(TypeError, match="performance must be a Performance"):
        function(performance={"compile_mode": None})
    # Even an explicitly supplied default conflicts; omission needs a sentinel.
    with pytest.raises(TypeError, match="cannot be combined.*compile_mode"):
        function(performance=sim.Performance(), compile_mode="default")
    with pytest.raises(TypeError, match="cannot be combined.*compile_mode"):
        function(performance=sim.Performance(), compile_mode=None)


def test_build_reports_preparation_without_claiming_work_was_executed():
    kwargs = inputs()
    kwargs.pop("input_bindings")
    kwargs.pop("compile_mode")
    policy = sim.Performance(compile_mode=None)
    result = sim.build(**kwargs, performance=policy)
    report = result.metrics["performance"]
    assert report["requested"] == asdict(policy)
    assert report["resolved"]["precision"]["autocast"] == "not_run"
    assert report["resolved"]["cuda_graphs"]["selected"] == "not_run"
    assert report["resolved"]["execution_paths"] == []
    assert report["resolved"]["fused_optimizer"] is None
    assert report["resolved"]["dataset_prefetch"] is None


def test_simulation_policy_preserves_outputs_state_and_independent_metadata():
    kwargs = inputs()
    kwargs.pop("compile_mode")
    # Keep the default no-gradient inference path eligible.
    kwargs.pop("training")
    policy = sim.Performance(compile_mode=None, precision="highest")
    actual = sim.simulate(**kwargs, performance=policy, diagnostics=False)
    with pytest.warns(DeprecationWarning):
        expected = sim.simulate(
            **kwargs, compile_mode=None, precision="highest", diagnostics=False
        )
    for key, value in expected.outputs.items():
        torch.testing.assert_close(actual.outputs[key], value, rtol=0, atol=0)
    continued = sim.simulate(
        **kwargs,
        performance=policy,
        diagnostics=False,
        runtime_state=actual.runtime_state,
    )
    assert continued.runtime_state.completed_steps == 6
    report = actual.metrics["performance"]
    assert report["requested"] == asdict(policy)
    assert report["resolved"]["device"] == "cpu"
    assert report["resolved"]["precision"] == {"autocast": "off", "matmul": "caller"}
    assert report["resolved"]["cuda_graphs"]["replayed_calls"] == 0
    assert report["resolved"]["execution_paths"] == actual.model._execution_paths
    json.dumps(report, allow_nan=False)
    report["resolved"]["execution_paths"].clear()
    assert actual.model._execution_paths
    assert continued.metrics["performance"]["resolved"]["execution_paths"]


def test_training_policy_preserves_updates_optimizer_checkpoint_and_resume(tmp_path):
    kwargs = request(recurrent=True)
    fields_to_move = {f.name for f in fields(sim.Performance)}
    settings = {k: kwargs.pop(k) for k in list(kwargs) if k in fields_to_move}
    policy = sim.Performance(**settings, compile_mode=None)
    actual = sim.train(
        **kwargs, performance=policy, save_final_checkpoint=tmp_path / "final"
    )
    with pytest.warns(DeprecationWarning):
        expected = sim.train(**kwargs, **asdict(policy))
    compare(actual, expected)
    report = actual.metrics["performance"]
    assert report["requested"] == asdict(policy)
    assert report["resolved"]["fused_optimizer"] is False
    assert report["resolved"]["cuda_graphs"]["normal_updates"] == 6
    assert "CUDA device required" in report["fallbacks"]["cuda_graphs"]
    assert report["resolved"]["dataset_prefetch"] is None
    json.dumps(report, allow_nan=False)
    # The object normalizes to the existing protocol, so either call style resumes.
    sim.train(
        **kwargs,
        performance=policy,
        updates=2,
        save_final_checkpoint=tmp_path / "partial",
    )
    resumed_kwargs = {**kwargs, "checkpoint": tmp_path / "partial"}
    resumed = sim.train(**resumed_kwargs, performance=policy)
    with pytest.warns(DeprecationWarning):
        legacy_resumed = sim.train(**resumed_kwargs, **asdict(policy))
    compare(resumed, legacy_resumed)
    compare(resumed, actual)
    assert replace(policy, cuda_graphs="off").compile_mode is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_policy_reports_actual_precision_fusion_and_replay():
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("native BF16 unavailable")
    kwargs = request(device="cuda", recurrent=True, block_steps=4)
    settings = {
        f.name: kwargs.pop(f.name) for f in fields(sim.Performance) if f.name in kwargs
    }
    policy = replace(sim.Performance(**settings), precision="auto")
    actual = sim.train(**kwargs, performance=policy)
    with pytest.warns(DeprecationWarning):
        expected = sim.train(**kwargs, **asdict(policy))
    compare(actual, expected)
    resolved = actual.metrics["performance"]["resolved"]
    assert resolved["precision"] == {"autocast": "bfloat16", "matmul": "high"}
    assert resolved["fused_optimizer"] is True
    assert resolved["cuda_graphs"]["replayed_updates"] == 6
    assert resolved["cuda_graphs"]["captures"] == 2
    assert resolved["execution_paths"] == actual.metrics["execution_paths"]


@pytest.mark.parametrize("name", ["train", "simulate"])
def test_dataset_policy_preserves_samples_and_reports_prefetch(name, tmp_path):
    kwargs = inputs()
    kwargs.pop("compile_mode")
    spikes = kwargs.pop("input_bindings")[0].value
    snapshot = tmp_path / "samples.npz"
    np.savez(snapshot, features=spikes.numpy(), labels=np.asarray([0, 1]))
    kwargs["input_bindings"] = (
        sim.DatasetSnapshotBinding(
            path=snapshot,
            input_id="events",
            target_id="label",
            dataset_id="performance-test",
            split="train",
            encoder=sim.DatasetEncoder("prebinned_spikes"),
        ),
    )
    kwargs.update(batch_size=1, diagnostics=False)
    if name == "train":
        kwargs["epochs"] = 1
    else:
        kwargs.pop("training")
    policy = sim.Performance(compile_mode=None, dataset_prefetch=1)
    actual = getattr(sim, name)(**kwargs, performance=policy)
    synchronous = getattr(sim, name)(
        **kwargs, performance=replace(policy, dataset_prefetch=0)
    )
    for key, value in synchronous.outputs.items():
        torch.testing.assert_close(actual.outputs[key], value, rtol=0, atol=0)
    for key, value in synchronous.parameters.items():
        torch.testing.assert_close(actual.parameters[key], value, rtol=0, atol=0)
    assert actual.metrics["performance"]["resolved"]["dataset_prefetch"] == 1
    assert synchronous.metrics["performance"]["resolved"]["dataset_prefetch"] == 0
