"""Test data stays held out until training and checkpoint selection finish."""

import copy
from dataclasses import replace

import pytest
import torch

from snnlab import extensions
from snnlab.sim import (
    CheckpointSelection,
    DenseArrayBinding,
    GraphExecutor,
    SelectionMetric,
    TrainingMetrics,
    build,
    train,
)
from snnlab.sim import (
    TestSpec as HeldOutTestSpec,
)
from tests.sim.test_execution_epochs import request


def test_request(**changes):
    spec = request()
    # Seven examples exercise sample weighting of the final one-example batch.
    indices = torch.tensor([0, 1, 2, 3, 4, 0, 1])
    changes.setdefault(
        "test",
        HeldOutTestSpec(
            input_bindings=(
                DenseArrayBinding("events", spec.input_bindings[0].value[:, indices]),
            ),
            targets={"label": spec.targets["label"][indices]},
            protocol={"split": "test"},
        ),
    )
    changes.setdefault("compile_mode", None)
    return replace(spec, **changes)


# This is a request builder, not a test case.
test_request.__test__ = False


@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_default_evaluation_order_and_best_validation_weights(
    monkeypatch, compile_mode
):
    events = []
    original = GraphExecutor._forward

    def tracked(self, *args, **kwargs):
        events.append("train" if torch.is_grad_enabled() else "evaluate")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GraphExecutor, "_forward", tracked)
    spec = test_request(
        compile_mode=compile_mode,
        on_epoch_end=lambda report: events.append(report.epoch),
    )
    result = train(spec)
    assert events == [
        "train",
        "train",
        "train",
        "evaluate",
        "evaluate",
        1,
        "train",
        "train",
        "train",
        "evaluate",
        "evaluate",
        2,
        "evaluate",
        "evaluate",
        "evaluate",
        "evaluate",
    ]
    assert "validation_loss" not in result.metrics["epochs"][0]
    rows = result.metrics["epochs"][1:]
    assert all("validation_loss" in row for row in rows)
    selected = result.selected_checkpoint
    assert selected.selection_record["epoch"] == 1
    assert selected.selection_record["values"] == [rows[0]["validation_loss"]]
    measured = result.metrics["test"]
    assert measured["checkpoint"] == "selected"
    assert measured["epoch"] == 1
    assert measured["update"] == 3
    assert measured["examples"] == 7
    assert measured["protocol"]["split"] == "test"
    assert all(not any(key.startswith("test_") for key in row) for row in rows)
    model = build(spec).model
    with torch.no_grad():
        for name, parameter in model.parameter_map().items():
            parameter.copy_(selected.parameters[name])
        scores = model(
            {"events": spec.test.input_bindings[0].value}, diagnostics=False
        ).outputs["class_scores"]
        expected_loss = torch.nn.functional.cross_entropy(
            scores, spec.test.targets["label"]
        )
        expected_accuracy = (
            (scores.argmax(-1) == spec.test.targets["label"]).float().mean()
        )
    assert measured["loss"] == pytest.approx(float(expected_loss))
    assert measured["accuracy"] == pytest.approx(float(expected_accuracy))


def test_test_evaluation_preserves_final_weights_optimizer_rng_and_selection():
    spec = test_request(training_timing=True)
    measured = train(spec)
    baseline = train(replace(spec, test=None))
    assert measured.metrics["training_timing"]["test_s"] > 0
    assert measured.metrics["selection_record"] == baseline.metrics["selection_record"]
    assert measured.model.training
    for name, parameter in measured.parameters.items():
        torch.testing.assert_close(parameter, baseline.parameters[name], rtol=0, atol=0)
        torch.testing.assert_close(
            measured.model.parameter_map()[name], parameter, rtol=0, atol=0
        )
        for key, value in measured.optimizer_state.get(name, {}).items():
            torch.testing.assert_close(
                value, baseline.optimizer_state[name][key], rtol=0, atol=0
            )
    assert torch.equal(
        measured.training_checkpoint.rng_state, baseline.training_checkpoint.rng_state
    )
    assert torch.equal(torch.get_rng_state(), measured.training_checkpoint.rng_state)
    assert baseline.metrics["test"] is None


@pytest.mark.parametrize("updates", [1, 3, 4])
def test_partial_run_never_resolves_test_data_and_resume_evaluates_once(
    tmp_path, monkeypatch, updates
):
    from snnlab.sim.training import trainer

    original = trainer.resolve_input_bindings
    accesses = []

    def tracked(*args, **kwargs):
        if kwargs.get("protocol", {}).get("split") == "test":
            accesses.append("test")
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer, "resolve_input_bindings", tracked)
    spec = test_request()
    path = tmp_path / "partial"
    partial = train(replace(spec, updates=updates, save_final_checkpoint=path))
    assert partial.metrics["test"] is None
    assert accesses == []
    resumed = train(replace(spec, checkpoint=path))
    assert accesses == ["test"]
    full = train(spec)
    assert accesses == ["test", "test"]
    assert resumed.metrics["test"] == full.metrics["test"]


@pytest.mark.parametrize("selection_disabled", [False, True])
def test_final_weights_used_without_validation_or_with_selection_disabled(
    selection_disabled,
):
    spec = test_request()
    spec = (
        replace(spec, checkpoint_selection=CheckpointSelection.disabled())
        if selection_disabled
        else replace(spec, validation=None)
    )
    result = train(spec)
    assert result.selected_checkpoint is None
    assert result.metrics["test"]["checkpoint"] == "final"
    assert result.metrics["test"]["epoch"] == 2
    assert result.metrics["test"]["update"] == 6
    with torch.no_grad():
        scores = result.model(
            {"events": spec.test.input_bindings[0].value}, diagnostics=False
        ).outputs["class_scores"]
        expected = torch.nn.functional.cross_entropy(scores, spec.test.targets["label"])
    assert result.metrics["test"]["loss"] == pytest.approx(float(expected))


def test_disabled_test_evaluation_does_not_resolve_invalid_binding():
    spec = test_request(metrics=TrainingMetrics(test="none"))
    result = train(replace(spec, test=HeldOutTestSpec()))
    assert result.metrics["test"] is None


def test_test_data_and_policy_can_change_on_resume(tmp_path):
    spec = test_request()
    path = tmp_path / "partial"
    train(
        replace(
            spec,
            test=None,
            metrics=TrainingMetrics(test="none"),
            updates=1,
            save_final_checkpoint=path,
        )
    )
    resumed = train(replace(spec, checkpoint=path))
    assert resumed.metrics["test"] == train(spec).metrics["test"]


def test_dataset_test_draws_are_batched_and_sample_weighted(tmp_path):
    from tests.sim.test_batch_encoding import dataset, training_request

    spec = training_request(tmp_path, validation=True)
    binding = dataset(tmp_path, count=7, name="test")
    spec = replace(
        spec,
        compile_mode=None,
        test=HeldOutTestSpec(
            input_bindings=(replace(binding, split="test"),),
            encoding_seeds=(107, 109),
        ),
    )
    multiple = train(spec)
    singles = [
        train(replace(spec, test=replace(spec.test, encoding_seeds=(seed,))))
        for seed in (107, 109)
    ]
    measured = multiple.metrics["test"]
    assert measured["examples"] == 14
    assert measured["encoding_draws"] == [107, 109]
    assert measured["loss"] == pytest.approx(
        sum(result.metrics["test"]["loss"] for result in singles) / 2
    )
    assert measured["accuracy"] == pytest.approx(
        sum(result.metrics["test"]["accuracy"] for result in singles) / 2
    )


def test_test_labels_cannot_influence_checkpoint_selection():
    spec = test_request()
    left = train(spec)
    right = train(
        replace(
            spec,
            test=replace(spec.test, targets={"label": 1 - spec.test.targets["label"]}),
        )
    )
    assert (
        left.selected_checkpoint.selection_record
        == right.selected_checkpoint.selection_record
    )
    assert left.metrics["test"]["loss"] != right.metrics["test"]["loss"]


def test_invalid_test_spec_and_zero_epochs_are_rejected():
    with pytest.raises(TypeError, match="test must be a TestSpec"):
        train(test_request(test="invalid"))
    with pytest.raises(ValueError, match="test requires training with positive epochs"):
        train(test_request(epochs=0, validation=None))
    with pytest.raises(ValueError, match="target ids do not match recipe"):
        train(
            test_request(
                test=HeldOutTestSpec(
                    input_bindings=request().input_bindings,
                    targets={"wrong": torch.zeros(5, dtype=torch.long)},
                )
            )
        )
    with pytest.raises(ValueError, match="metrics test must be final or none"):
        TrainingMetrics(test="epoch")


def test_total_loss_selection_supports_nonclassification_objectives():
    extensions.register_objective(
        "tests.final-evaluation-regression/v1",
        lambda predictions, targets, config: (
            (predictions - targets[:, None]).square().mean()
        ),
    )
    spec = test_request()
    recipe = copy.deepcopy(spec.training)
    recipe["objectives"][0] = {
        **recipe["objectives"][0],
        "kind": "custom_objective",
        "definition": "tests.final-evaluation-regression/v1",
        "config": {},
    }
    result = train(replace(spec, training=recipe))
    assert (
        result.selected_checkpoint.selection_record["policy"]["metric"]["kind"]
        == "loss"
    )
    assert "accuracy" not in result.metrics["test"]
    assert result.metrics["test"]["loss"] >= 0
    with pytest.raises(ValueError, match="duplicate checkpoint selection metric"):
        CheckpointSelection(
            metric=SelectionMetric("loss"), tie_break=(SelectionMetric("loss"),)
        ).validate(recipe, epochs=2, has_validation=True)
