"""Epoch evaluation measures fixed weights without changing training."""

from dataclasses import replace

import pytest
import torch

from snnlab.sim.execution import (
    DenseArrayBinding,
    ExecutionSpec,
    TargetArrayBinding,
    ValidationSpec,
    save_training_checkpoint,
    train,
)
from tests.sim._execution_builders import direct_train_bundle


def request():
    bundle = direct_train_bundle()
    spikes = torch.zeros(3, 5, 2)
    labels = torch.tensor([0, 1, 0, 1, 0])
    spikes[:, torch.arange(5), labels] = 1
    validation = ValidationSpec(
        input_bindings=(DenseArrayBinding("events", spikes[:, :3].clone()),),
        target_bindings=(TargetArrayBinding("label", 1 - labels[:3]),),
        protocol={"split": "validation"},
    )
    return ExecutionSpec(
        kind="train",
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(DenseArrayBinding("events", spikes),),
        targets={"label": labels},
        validation=validation,
        device="cpu",
        seed=23,
        diagnostics=False,
        epochs=2,
        batch_size=2,
        shuffle=True,
    )


def test_epoch_evaluation_is_sample_weighted_and_validation_never_trains():
    spec = request()
    result = train(spec)
    without_validation = train(replace(spec, validation=None))
    assert result.metrics["updates"] == without_validation.metrics["updates"]
    for name, parameter in result.parameters.items():
        torch.testing.assert_close(
            parameter, without_validation.parameters[name], rtol=0, atol=0
        )
    for name, state in result.optimizer_state.items():
        for key, value in state.items():
            torch.testing.assert_close(
                value, without_validation.optimizer_state[name][key], rtol=0, atol=0
            )
    torch.testing.assert_close(
        result.training_checkpoint.rng_state,
        without_validation.training_checkpoint.rng_state,
    )
    history = result.metrics["epochs"]
    assert [row["epoch"] for row in history] == [0, 1, 2]
    assert history[0]["train_loss"] == pytest.approx(
        torch.log(torch.tensor(2.0)).item()
    )
    assert history[0]["train_accuracy"] == pytest.approx(3 / 5)
    assert history[0]["validation_accuracy"] == pytest.approx(1 / 3)
    assert result.metrics["validation_protocol"]["split"] == "validation"
    assert "validation_loss" not in without_validation.metrics["epochs"][-1]
    assert result.diagnostics == {}
    with torch.no_grad():
        for split, inputs, labels in (
            ("train", spec.input_bindings[0].value, spec.targets["label"]),
            (
                "validation",
                spec.validation.input_bindings[0].value,
                spec.validation.target_bindings[0].value,
            ),
        ):
            scores = next(
                iter(
                    result.model({"events": inputs}, diagnostics=False).outputs.values()
                )
            )
            assert history[-1][f"{split}_loss"] == pytest.approx(
                torch.nn.functional.cross_entropy(scores, labels).item()
            )
            assert history[-1][f"{split}_accuracy"] == pytest.approx(
                float((scores.argmax(-1) == labels).float().mean())
            )
            assert (
                history[-1][f"{split}_accuracies"]["objective[0]"]
                == history[-1][f"{split}_accuracy"]
            )


def test_epoch_history_resume_reports_only_completed_epochs(tmp_path):
    spec = request()
    full = train(spec)
    partial = train(replace(spec, updates=1))
    assert [row["epoch"] for row in partial.metrics["epochs"]] == [0]
    checkpoint = save_training_checkpoint(
        tmp_path / "partial", partial.training_checkpoint
    )
    resumed = train(replace(spec, checkpoint=checkpoint))
    assert resumed.metrics["epochs"] == full.metrics["epochs"][1:]
    boundary = train(replace(spec, updates=3))
    checkpoint = save_training_checkpoint(
        tmp_path / "boundary", boundary.training_checkpoint
    )
    resumed_boundary = train(replace(spec, checkpoint=checkpoint))
    assert resumed_boundary.metrics["epochs"] == full.metrics["epochs"][1:]


def test_validation_requires_epochs_and_matching_target_ids():
    spec = request()
    with pytest.raises(ValueError, match="positive epochs"):
        train(replace(spec, epochs=0, updates=2))
    with pytest.raises(ValueError, match="target ids do not match recipe"):
        train(
            replace(
                spec,
                validation=ValidationSpec(
                    input_bindings=spec.validation.input_bindings,
                    targets={"wrong": torch.tensor([0, 1, 0])},
                ),
            )
        )


@pytest.mark.parametrize(
    "name",
    [
        "epochs",
        "batch_size",
        "shuffle",
        "updates",
        "save_final_checkpoint",
        "save_selected_checkpoint",
    ],
)
def test_training_options_require_direct_fields(name):
    with pytest.raises(ValueError, match="must be ExecutionSpec fields"):
        train(replace(request(), options={name: None}))


@pytest.mark.parametrize(
    "fields",
    [
        {"epochs": -1},
        {"epochs": 1.5},
        {"epochs": True},
        {"batch_size": 0},
        {"updates": 0},
        {"updates": False},
    ],
)
def test_training_counts_reject_invalid_values(fields):
    with pytest.raises(ValueError, match="must be an integer"):
        train(replace(request(), **fields))


def test_training_shuffle_requires_boolean():
    with pytest.raises(TypeError, match="shuffle must be boolean"):
        train(replace(request(), shuffle="yes"))
