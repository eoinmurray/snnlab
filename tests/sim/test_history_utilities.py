"""History readers validate native evidence without inventing observations."""

import copy
import io
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from snnlab import sim
from snnlab.sim.training.trainer import train
from tests.sim.test_final_test_evaluation import test_request


def test_native_history_export_baseline_counts_and_provenance(tmp_path):
    result = train(test_request())
    path = result.write_metrics(tmp_path / "metrics.json")
    history = sim.TrainingHistory.read(path)
    history.validate(
        expected_epochs=2,
        require_complete=True,
        selection=sim.CheckpointSelection(metric=sim.SelectionMetric("loss")),
        expected_examples={"test": 7, "validation": 3},
    )
    assert history.splits == ("train", "validation")
    assert history.coverage.complete is True
    assert history.test.epoch == history.selection.epoch == 1
    assert (
        history.epochs[0].splits["validation"].measurement == "fixed_weight_evaluation"
    )
    assert "train" not in history.epochs[0].splits
    curve = history.curve(metric="accuracy", split="train")
    np.testing.assert_array_equal(curve.available, [False, True, True])
    assert np.isnan(curve.values[0])
    assert curve.measurement == (None, "online_pre_update", "online_pre_update")
    with pytest.raises(FileExistsError):
        result.write_metrics(path)
    result.write_metrics(path, overwrite=True)
    assert sim.read_metrics(path) == result.metrics
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(result.metrics))
    with pytest.raises(ValueError, match="legacy"):
        sim.TrainingHistory.read(legacy)
    assert sim.TrainingHistory.read(legacy, legacy=True).test.examples == 7


@pytest.mark.parametrize("updates", [1, 3, 4])
def test_interrupted_and_resumed_histories_remain_distinct(tmp_path, updates):
    spec = test_request(updates=updates, save_final_checkpoint=tmp_path / "checkpoint")
    partial = train(spec)
    history = sim.TrainingHistory.from_metrics(partial.metrics)
    assert history.coverage.complete is False
    assert history.test is None
    with pytest.raises(ValueError, match="incomplete"):
        history.validate(expected_epochs=2, require_complete=True)
    resumed = train(
        replace(
            spec,
            updates=None,
            checkpoint=tmp_path / "checkpoint",
            save_final_checkpoint=None,
        )
    )
    sim.TrainingHistory.from_metrics(resumed.metrics).validate(require_complete=True)
    full = train(replace(spec, updates=None, save_final_checkpoint=None))
    assert resumed.metrics["epochs"] == full.metrics["epochs"]
    assert resumed.metrics["test"] == full.metrics["test"]


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m["epochs"].reverse(),
        lambda m: m["epochs"][1].update(train_accuracy=1.1),
        lambda m: m["epochs"][1].update(train_loss=float("inf")),
        lambda m: m["epochs"][0].update(
            train_loss=0.0, train_measurement="online_pre_update"
        ),
        lambda m: m["epochs"][1].update(validation_examples=-1),
        lambda m: m["selection_record"].update(epoch=2),
        lambda m: m["test"].update(epoch=2),
        lambda m: m["test"].update(checkpoint="final"),
    ],
)
def test_inconsistent_native_evidence_is_rejected(change):
    metrics = train(test_request()).metrics
    change(metrics)
    with pytest.raises(ValueError):
        sim.TrainingHistory.from_metrics(metrics)


def test_multiple_objectives_unknown_semantics_and_negative_custom_loss():
    history = sim.TrainingHistory.from_metrics(
        {
            "epochs": [
                {
                    "epoch": 1,
                    "train_loss": -2.0,
                    "train_components": {"objective[0]": -3.0},
                    "train_accuracies": {"objective[0]": 0.2, "objective[1]": 0.8},
                },
                {"epoch": 3, "train_loss": -1.0},
            ]
        }
    )
    assert history.coverage.complete is None
    assert history.coverage.contiguous is False
    with pytest.raises(ValueError, match="objective"):
        history.curve(metric="accuracy", split="train")
    curve = history.curve(metric="accuracy", split="train", objective=1)
    np.testing.assert_array_equal(curve.available, [True, False])
    assert curve.measurement == (None, None)
    assert history.curve(metric="loss", split="train", objective=0).values[0] == -3.0
    with pytest.raises(ValueError, match="missing epochs"):
        history.validate(expected_epochs=3, require_complete=True)
    with pytest.raises(ValueError, match="count"):
        history.validate(expected_examples={"test": 7})


def test_ties_initial_eligibility_and_encoding_draws():
    policy = sim.CheckpointSelection(
        metric=sim.SelectionMetric("loss"), include_initial=True
    )
    metrics = {
        "epochs": [
            {
                "epoch": epoch,
                "validation_loss": 1.0,
                "validation_examples": 6,
                "validation_encoding_draws": [7, 11],
                "validation_measurement": "fixed_weight_evaluation",
            }
            for epoch in range(3)
        ],
        "checkpoint_selection": {"policy": policy.to_dict()},
        "selection_record": {
            "epoch": 0,
            "update": 0,
            "values": [1.0],
            "policy": policy.to_dict(),
        },
    }
    history = sim.TrainingHistory.from_metrics(metrics)
    history.validate(selection=policy, expected_encoding_draws={"validation": (7, 11)})
    with pytest.raises(ValueError, match="draws"):
        history.validate(expected_encoding_draws={"validation": (7,)})
    changed = copy.deepcopy(metrics)
    changed["selection_record"]["epoch"] = 1
    with pytest.raises(ValueError, match="tie"):
        sim.TrainingHistory.from_metrics(changed)
    changed = copy.deepcopy(metrics)
    changed["epochs"] = changed["epochs"][1:]
    partial = sim.TrainingHistory.from_metrics(changed)
    assert partial.coverage.selection_verified is False
    with pytest.raises(ValueError, match="eligible candidates"):
        partial.validate(expected_epochs=2, require_complete=True)


def test_metrics_reject_opaque_and_nonfinite_values_before_publication(tmp_path):
    for value in (torch.ones(2), float("nan")):
        result = sim.ExecutionResult("graph", metrics={"value": value})
        with pytest.raises((TypeError, ValueError)):
            result.write_metrics(tmp_path / "bad.json")
        assert not (tmp_path / "bad.json").exists()
    for contents in (
        '{"schema":"unknown","metrics":{}}',
        '{"schema":"snnlab.metrics/v1","metrics":{"x":1e999}}',
    ):
        path = tmp_path / "invalid.json"
        path.write_text(contents)
        with pytest.raises(ValueError):
            sim.read_metrics(path)


def test_epoch_formatter_and_logger_preserve_trajectory_and_resume(tmp_path):
    stream = io.StringIO()
    logger = sim.EpochLogger(
        stream=stream, timing=True, precision=2, labels={"validation": "val"}
    )
    spec = test_request(
        on_epoch_end=logger, updates=3, save_final_checkpoint=tmp_path / "checkpoint"
    )
    train(spec)
    assert "epoch 1/2" in stream.getvalue()
    assert "train[online]" in stream.getvalue()
    assert "val[fixed]" in stream.getvalue()
    assert "time=" not in stream.getvalue()
    stream.seek(0)
    stream.truncate()
    result = train(
        replace(
            spec,
            updates=None,
            checkpoint=tmp_path / "checkpoint",
            save_final_checkpoint=None,
        )
    )
    assert "epoch 2/2" in stream.getvalue() and "epoch 1/2" not in stream.getvalue()
    baseline = train(test_request())
    assert result.metrics["epochs"] == baseline.metrics["epochs"]
    for name, value in result.parameters.items():
        torch.testing.assert_close(value, baseline.parameters[name], rtol=0, atol=0)
    torch.testing.assert_close(
        result.training_checkpoint.rng_state, baseline.training_checkpoint.rng_state
    )


def test_max_direction_secondary_score_and_earliest_exact_tie():
    policy = sim.CheckpointSelection(
        metric=sim.SelectionMetric("accuracy", direction="max"),
        tie_break=(sim.SelectionMetric("loss"),),
    )
    metrics = {
        "epochs": [
            {
                "epoch": epoch,
                "validation_loss": loss,
                "validation_accuracies": {"objective[0]": accuracy},
                "validation_measurement": "fixed_weight_evaluation",
            }
            for epoch, loss, accuracy in (
                (0, 0.1, 1.0),
                (1, 1.0, 0.9),
                (2, 0.5, 0.9),
                (3, 0.5, 0.9),
            )
        ],
        "checkpoint_selection": {"policy": policy.to_dict()},
        "selection_record": {
            "epoch": 2,
            "update": 4,
            "values": [0.9, 0.5],
            "policy": policy.to_dict(),
        },
    }
    history = sim.TrainingHistory.from_metrics(metrics)
    history.validate(expected_epochs=3, require_complete=True)
    assert history.selection.epoch == 2
    metrics["selection_record"]["epoch"] = 3
    with pytest.raises(ValueError, match="tie"):
        sim.TrainingHistory.from_metrics(metrics)


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m["selection_record"].update(update=1234),
        lambda m: m["selection_record"].update(values=[float("nan")]),
        lambda m: m["selection_record"].update(phase="initial"),
        lambda m: m["history_coverage"].update(complete=1),
        lambda m: m["epochs"][1].update(validation_accuracy=0.1234),
    ],
)
def test_selection_and_coverage_metadata_corruption_is_rejected(change):
    metrics = train(test_request()).metrics
    change(metrics)
    with pytest.raises(ValueError):
        sim.TrainingHistory.from_metrics(metrics)


def test_legacy_update_selection_retains_its_distinct_epoch_semantics():
    spec = test_request(
        checkpoint_selection=sim.CheckpointSelection.legacy_training_batch_loss(),
        metrics=sim.TrainingMetrics(update_history=True),
    )
    result = train(spec)
    history = sim.TrainingHistory.from_metrics(result.metrics)
    assert history.selection.phase == "legacy_training_batch_loss"
    assert history.test.update == history.selection.update
    history.validate(expected_epochs=2, require_complete=True)
    # Full-batch updates have no epoch coordinate in the native selection record.
    result = train(
        replace(
            spec,
            epochs=0,
            validation=None,
            test=None,
            batch_size=None,
            metrics=None,
            updates=2,
        )
    )
    assert sim.TrainingHistory.from_metrics(result.metrics).selection.epoch is None
