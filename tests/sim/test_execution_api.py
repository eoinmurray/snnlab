"""Public keyword execution preserves the internal handlers' numerical contract."""

import inspect
from dataclasses import fields

import pytest
import torch

from snnlab import sim
from snnlab.sim.requests import handlers
from snnlab.sim.requests.performance import _UNSET
from snnlab.sim.requests.spec import ExecutionSpec
from snnlab.sim.training.trainer import train as train_request
from tests.sim._execution_builders import direct_train_bundle


@pytest.mark.parametrize("name", ["build", "simulate", "infer", "train"])
def test_execution_functions_have_explicit_operation_specific_keywords(name):
    function = getattr(sim, name)
    signature = inspect.signature(function)
    assert all(
        p.kind is inspect.Parameter.KEYWORD_ONLY for p in signature.parameters.values()
    )
    assert not {"kind", "executor", "spec", "options"} & signature.parameters.keys()
    with pytest.raises(TypeError):
        function(ExecutionSpec(kind=name))
    with pytest.raises(TypeError, match="kind"):
        function(kind=name)
    if name != "train":
        with pytest.raises(TypeError, match="epochs"):
            function(epochs=2)
    else:
        with pytest.raises(TypeError, match="interventions"):
            function(interventions=())
    # Shared defaults remain consistent with CLI/internal execution.
    default = ExecutionSpec(kind=name)
    performance_fields = {field.name for field in fields(sim.Performance)}
    for field in fields(default):
        if field.name in signature.parameters:
            public_default = signature.parameters[field.name].default
            internal_default = getattr(default, field.name)
            if field.name in performance_fields:
                assert public_default is _UNSET
                assert getattr(sim.Performance(), field.name) == internal_default
            elif isinstance(internal_default, dict):
                assert public_default is None
            else:
                assert public_default == internal_default


def test_request_and_dispatcher_are_internal():
    for name in ("ExecutionSpec", "execute_request"):
        assert name not in sim.__all__
        assert name not in dir(sim)
        with pytest.raises(AttributeError):
            getattr(sim, name)


def inputs():
    bundle = direct_train_bundle()
    spikes = torch.zeros(3, 2, 2)
    spikes[:, 0, 0] = 1
    spikes[:, 1, 1] = 1
    return dict(
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(sim.DenseArrayBinding("events", spikes),),
        seed=23,
        device="cpu",
        compile_mode=None,
    )


@pytest.mark.parametrize("name", ["simulate", "infer"])
def test_keyword_runs_match_internal_execution_and_continue(name, tmp_path):
    kwargs = inputs()
    kwargs.update(
        diagnostics=False,
        protocol={"source": "public-api-test"},
        input_bindings=(
            sim.PoissonInputBinding(
                "events", steps_count=3, batch_size=2, rates_hz=(50.0,), seed=7
            ),
        ),
        inference_overrides={"duration_ms": 0.6, "input_rate_hz": 100.0},
        snapshot=sim.SnapshotSpec(tmp_path / name, stage="both"),
    )
    result = getattr(sim, name)(**kwargs)
    options = {"inference_overrides": kwargs.pop("inference_overrides")}
    kwargs.pop("snapshot")
    expected = getattr(handlers, name)(
        ExecutionSpec(kind=name, **kwargs, options=options)
    )
    for key, value in expected.outputs.items():
        torch.testing.assert_close(result.outputs[key], value, rtol=0, atol=0)
    assert not result.diagnostics
    assert result.runtime_state.completed_steps == 6
    assert result.snapshots.keys() == {"start", "end"}
    assert result.snapshots["start"].metadata["kind"] == name
    continued = getattr(sim, name)(
        **kwargs,
        inference_overrides=options["inference_overrides"],
        runtime_state=result.runtime_state,
    )
    assert continued.runtime_state.completed_steps == 12


def test_keyword_build_initializes_without_running():
    kwargs = inputs()
    kwargs.pop("input_bindings")
    built = sim.build(**kwargs)
    expected = handlers.build(ExecutionSpec(kind="build", **kwargs))
    assert isinstance(built.model, sim.GraphExecutor)
    assert not built.outputs
    for key, value in expected.parameters.items():
        torch.testing.assert_close(built.parameters[key], value, rtol=0, atol=0)


def test_keyword_training_validation_test_and_resume_match_internal_run(tmp_path):
    kwargs = inputs()
    targets = {"label": torch.tensor([0, 1])}
    kwargs.update(
        targets=targets,
        epochs=2,
        batch_size=1,
        shuffle=True,
        validation=sim.ValidationSpec(
            input_bindings=kwargs["input_bindings"], targets=targets
        ),
        test=sim.TestSpec(input_bindings=kwargs["input_bindings"], targets=targets),
        training_timing=True,
        diagnostics=False,
    )
    expected = train_request(ExecutionSpec(kind="train", **kwargs))
    partial = sim.train(**kwargs, updates=1, save_final_checkpoint=tmp_path / "partial")
    assert partial.metrics["test"] is None
    resumed = sim.train(**kwargs, checkpoint=tmp_path / "partial")
    assert (
        resumed.training_checkpoint.completed_updates
        == expected.training_checkpoint.completed_updates
    )
    assert [
        {k: v for k, v in row.items() if k != "timing"}
        for row in resumed.metrics["epochs"]
    ] == [
        {k: v for k, v in row.items() if k != "timing"}
        for row in expected.metrics["epochs"]
    ]
    assert resumed.metrics["test"] == expected.metrics["test"]
    for key, value in expected.parameters.items():
        torch.testing.assert_close(resumed.parameters[key], value, rtol=0, atol=0)
