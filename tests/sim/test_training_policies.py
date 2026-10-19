"""Measurement, scheduling and reporting share exact optimizer boundaries."""

import copy
import inspect
import json
import random
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch

from snnlab import extensions
from snnlab.sim import (
    Batching,
    CheckpointSelection,
    EpochObservations,
    GraphExecutor,
    TrainingMetrics,
    load_training_checkpoint,
    save_training_checkpoint,
    train,
)
from tests.sim.test_batch_encoding import training_request
from tests.sim.test_epoch_observations import observations
from tests.sim.test_epoch_observations import request as observation_request
from tests.sim.test_execution_epochs import request


def online_spec(**changes):
    return replace(
        request(),
        metrics=TrainingMetrics(
            train="online", initial=("validation",), update_history=False
        ),
        **changes,
    )


def weighted_objective(scores, targets, config):
    weights = scores.new_tensor([1.0, 3.0])
    return (
        torch.nn.functional.cross_entropy(scores, targets.long(), reduction="none")
        * weights[targets.long()]
    ).mean()


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_online_reuses_forwards_and_matches_manual_unequal_batches(
    monkeypatch, custom, device
):
    spec = online_spec(device=device, shuffle=False)
    if custom:
        extensions.register_objective(
            "tests.weighted-online/v1", weighted_objective, classification=True
        )
        recipe = copy.deepcopy(spec.training)
        recipe["objectives"][0] = {
            **recipe["objectives"][0],
            "kind": "custom_objective",
            "definition": "tests.weighted-online/v1",
            "config": {},
        }
        spec = replace(spec, training=recipe)
    original = GraphExecutor._forward
    batches, evaluations = [], []

    def tracked(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        scores = next(iter(result[0].outputs.values())).detach()
        (batches if torch.is_grad_enabled() else evaluations).append(scores)
        return result

    monkeypatch.setattr(GraphExecutor, "_forward", tracked)
    result = train(spec)
    assert len(batches) == 6
    assert len(evaluations) == 6  # validation: two batches, initial + two epochs
    assert result.metrics["updates"] == []
    assert "train_loss" not in result.metrics["epochs"][0]
    for epoch, row in enumerate(result.metrics["epochs"][1:]):
        losses, ces, correct = [], [], 0
        for batch, scores in enumerate(batches[epoch * 3 : (epoch + 1) * 3]):
            labels = spec.targets["label"][batch * 2 : batch * 2 + len(scores)].to(
                device
            )
            ce = torch.nn.functional.cross_entropy(scores, labels)
            raw = weighted_objective(scores, labels, {}) if custom else ce
            losses.append(float(raw) * len(scores))
            ces.append(float(ce) * len(scores))
            correct += int((scores.argmax(-1) == labels).sum())
        assert row["train_loss"] == pytest.approx(sum(losses) / 5)
        assert row["train_cross_entropies"]["objective[0]"] == pytest.approx(
            sum(ces) / 5
        )
        assert row["train_accuracy"] == correct / 5
        assert row["train_examples"] == 5
        assert row["train_measurement"] == "online_pre_update"
    assert (
        result.selected_checkpoint.selection_record["policy"]["metric"]["kind"]
        == "loss"
    )


@pytest.mark.parametrize("order", ["sequential", "shuffle", "stratified"])
@pytest.mark.parametrize("counts", [(10, 10), (17, 2, 1), (1,), (2, 1)])
def test_batching_unique_coverage_and_repeatable_class_scarcity(order, counts):
    labels = torch.cat(
        [
            torch.full((count,), i * 2, dtype=torch.long)
            for i, count in enumerate(counts)
        ]
    )
    policy = Batching(
        order=order, stratify_by="label" if order == "stratified" else None
    )
    batches = policy.indices(len(labels), 7, seed=31, labels=labels)
    actual = torch.cat(batches)
    assert sorted(actual.tolist()) == list(range(len(labels)))
    assert torch.equal(
        actual, torch.cat(policy.indices(len(labels), 7, seed=31, labels=labels))
    )
    if order == "shuffle":
        assert torch.equal(
            actual,
            torch.randperm(len(labels), generator=torch.Generator().manual_seed(31)),
        )
    elif order == "sequential":
        assert torch.equal(actual, torch.arange(len(labels)))
    else:
        positions = []
        class_seen = {}
        for index in actual:
            label = int(labels[index])
            j = class_seen.get(label, 0)
            positions.append((j + 0.5) / counts[label // 2])
            class_seen[label] = j + 1
        assert positions == sorted(positions)


@pytest.mark.parametrize(
    "policy,sizes",
    [
        (Batching(), [64, 64, 64, 8]),
        (Batching(remainder="drop"), [64, 64, 64]),
        (Batching(remainder="merge_small", min_batch_size=20), [64, 64, 72]),
    ],
)
def test_tail_policies(policy, sizes):
    batches = policy.indices(200, 64, seed=0)
    assert [len(batch) for batch in batches] == sizes
    assert torch.equal(torch.cat(batches), torch.arange(sum(sizes)))


def test_small_dataset_and_threshold_equality():
    policy = Batching(remainder="merge_small", min_batch_size=20)
    assert policy.sizes(8, 64) == [8]
    assert policy.sizes(148, 64) == [64, 64, 20]
    with pytest.raises(ValueError, match="drops every"):
        Batching(remainder="drop").sizes(8, 64)


@pytest.mark.parametrize(
    "fields",
    [
        {"order": "invalid"},
        {"remainder": "invalid"},
        {"order": "stratified"},
        {"stratify_by": "label"},
        {"min_batch_size": 1},
        {"remainder": "merge_small"},
        {"remainder": "merge_small", "min_batch_size": True},
    ],
)
def test_invalid_batching(fields):
    with pytest.raises(ValueError):
        Batching(**fields)


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([-1, 0]),
        torch.tensor([0.5, 1.0]),
        torch.ones(2, 1, dtype=torch.long),
        torch.tensor([0]),
    ],
)
def test_invalid_stratification_target(labels):
    with pytest.raises(ValueError):
        Batching(order="stratified", stratify_by="label").indices(
            2, 2, seed=0, labels=labels
        )


@pytest.mark.parametrize(
    "remainder,threshold", [("keep", None), ("merge_small", 2), ("drop", None)]
)
@pytest.mark.parametrize("updates", [1, 2])
@pytest.mark.parametrize("prefetch", [0, 2])
def test_partial_resume_keeps_encoding_online_sums_parameters_and_callbacks(
    tmp_path, remainder, threshold, updates, prefetch
):
    spec = training_request(tmp_path, validation=True)
    reports = []
    spec = replace(
        spec,
        batch_size=4,
        shuffle=False,
        dataset_prefetch=prefetch,
        batching=Batching(
            order="stratified",
            stratify_by="label",
            remainder=remainder,
            min_batch_size=threshold,
        ),
        metrics=TrainingMetrics(train="online", initial=("validation",)),
        checkpoint_selection=CheckpointSelection(),
        on_epoch_end=reports.append,
    )
    full = train(spec)
    reports.clear()
    partial = train(replace(spec, updates=updates))
    completed = [r.epoch for r in reports]
    saved = save_training_checkpoint(tmp_path / "partial", partial.training_checkpoint)
    reports.clear()
    resumed = train(replace(spec, checkpoint=saved))
    assert completed + [r.epoch for r in reports] == [1, 2, 3]
    assert resumed.metrics["epochs"] == full.metrics["epochs"]
    assert (
        partial.metrics["updates"] + resumed.metrics["updates"]
        == full.metrics["updates"]
    )
    for name in full.parameters:
        torch.testing.assert_close(
            full.parameters[name], resumed.parameters[name], rtol=0, atol=0
        )
        for key in full.optimizer_state[name]:
            torch.testing.assert_close(
                full.optimizer_state[name][key],
                resumed.optimizer_state[name][key],
                rtol=0,
                atol=0,
            )
    assert torch.equal(
        full.training_checkpoint.rng_state, resumed.training_checkpoint.rng_state
    )
    assert (
        full.selected_checkpoint.selection_record
        == resumed.selected_checkpoint.selection_record
    )
    sizes = spec.batching.sizes(9, 4)
    assert [row["epoch_updates"] for row in full.metrics["epochs"][1:]] == [
        len(sizes)
    ] * 3
    assert [row["epoch_examples"] for row in full.metrics["epochs"][1:]] == [
        sum(sizes)
    ] * 3


def test_optout_reads_no_scalars_per_update_and_explicit_update_selection_stays(
    monkeypatch,
):
    calls = []
    original = torch.Tensor.__float__

    def tracked(tensor):
        if inspect.currentframe().f_back.f_code is train.__code__:
            calls.append(tensor)
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "__float__", tracked)
    spec = online_spec(validation=None)
    result = train(spec)
    assert len(calls) == 1  # final checkpoint loss, after all updates
    assert result.training_checkpoint.completed_updates == 6
    assert result.selected_checkpoint is None
    calls.clear()
    selected = train(
        replace(
            spec, checkpoint_selection=CheckpointSelection.legacy_training_batch_loss()
        )
    )
    assert len(calls) == 7
    assert selected.selected_checkpoint is not None
    assert selected.metrics["updates"] == []


@pytest.mark.parametrize("timed", [False, True])
def test_callback_snapshot_observations_rng_and_no_extra_forwards(monkeypatch, timed):
    spec = observation_request()
    reports, calls = [], []
    original = GraphExecutor._forward

    def tracked(self, *args, **kwargs):
        calls.append(torch.is_grad_enabled())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GraphExecutor, "_forward", tracked)
    spec = replace(
        spec,
        observations=observations(spec),
        training_timing=timed,
        metrics=TrainingMetrics(
            train="evaluate", initial=("train", "validation"), update_history=True
        ),
    )
    baseline = train(spec)
    count = len(calls)
    calls.clear()

    def noisy(report):
        random.random()
        np.random.random()
        torch.rand(5)
        reports.append(report)
        with pytest.raises(FrozenInstanceError):
            report.epoch = 99
        with pytest.raises(TypeError):
            report.train["loss"] = 0
        with pytest.raises(AttributeError):
            report.train.loss = 0

    measured = train(replace(spec, on_epoch_end=noisy))
    assert len(calls) == count
    assert [r.epoch for r in reports] == [1, 2]
    assert [r.updates for r in reports] == [3, 6]
    assert [r.examples for r in reports] == [5, 10]
    for report, row in zip(reports, measured.metrics["epochs"][1:]):
        assert report.train.loss == row["train_loss"]
        assert report.validation.accuracy == row["validation_accuracy"]
        assert (
            report.observations["train"]["population_rates_hz"]
            == row["observations"]["train"]["population_rates_hz"]
        )
        assert (report.timing is None) == (not timed)
        if timed:
            assert report.timing == row["timing"]
    for name in measured.parameters:
        torch.testing.assert_close(
            measured.parameters[name], baseline.parameters[name], rtol=0, atol=0
        )
    assert torch.equal(
        measured.training_checkpoint.rng_state, baseline.training_checkpoint.rng_state
    )


def test_callback_missing_splits_and_exception_rng_restoration():
    reports = []
    spec = replace(
        request(),
        validation=None,
        metrics=TrainingMetrics(train="none", validation="none", initial=()),
        on_epoch_end=reports.append,
    )
    train(spec)
    assert all(
        report.train is None and report.validation is None and report.timing is None
        for report in reports
    )
    states = []

    def explode(report):
        states.append(torch.get_rng_state().clone())
        torch.rand(9)
        raise RuntimeError("logging failed")

    with pytest.raises(RuntimeError, match="logging failed"):
        train(replace(spec, on_epoch_end=explode))
    assert torch.equal(torch.get_rng_state(), states[0])


def test_incompatible_policies_and_checkpoint_authentication(tmp_path):
    with pytest.raises(ValueError, match="conflicts"):
        train(replace(request(), batching=Batching(order="sequential")))
    with pytest.raises(ValueError, match="training target"):
        train(
            replace(
                request(),
                shuffle=False,
                batching=Batching(order="stratified", stratify_by="missing"),
            )
        )
    with pytest.raises(ValueError, match="fixed-weight"):
        train(online_spec(checkpoint_selection=CheckpointSelection(split="train")))
    with pytest.raises(ValueError, match="initial split"):
        train(
            replace(
                request(),
                metrics=TrainingMetrics(initial=()),
                checkpoint_selection=CheckpointSelection(include_initial=True),
            )
        )
    with pytest.raises(ValueError, match="split activity"):
        train(
            replace(
                observation_request(),
                metrics=TrainingMetrics(train="online", initial=("validation",)),
                observations=EpochObservations(population_rates=("out",)),
            )
        )
    spec = online_spec()
    saved = save_training_checkpoint(
        tmp_path / "partial", train(replace(spec, updates=1)).training_checkpoint
    )
    with pytest.raises(ValueError, match="protocol"):
        train(
            replace(
                spec,
                checkpoint=saved,
                metrics=TrainingMetrics(train="evaluate", initial=("validation",)),
            )
        )
    manifest_path = saved / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["reporting_state"]["online_count"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reporting state digest"):
        load_training_checkpoint(saved)


def test_callback_enabled_on_default_boundary_resume_does_not_replay(tmp_path):
    spec = request()
    saved = save_training_checkpoint(
        tmp_path / "legacy", train(replace(spec, updates=3)).training_checkpoint
    )
    reports = []
    result = train(replace(spec, checkpoint=saved, on_epoch_end=reports.append))
    assert [report.epoch for report in reports] == [2]
    assert result.training_checkpoint.completed_updates == 6


def test_explicit_legacy_selection_online_partial_candidate_can_resume(tmp_path):
    spec = online_spec(
        checkpoint_selection=CheckpointSelection.legacy_training_batch_loss()
    )
    partial = train(replace(spec, updates=3))
    candidate = save_training_checkpoint(
        tmp_path / "selected", partial.selected_checkpoint
    )
    resumed = train(replace(spec, checkpoint=candidate))
    assert resumed.training_checkpoint.completed_updates == 6


@pytest.mark.parametrize(
    "fields",
    [
        {"train": "bad"},
        {"validation": "bad"},
        {"train": "online", "initial": ("train",)},
        {"train": "none", "initial": ("train",)},
        {"initial": ("train", "train")},
        {"initial": "train"},
        {"validation": "none", "initial": ("validation",)},
        {"test": "epoch"},
        {"update_history": 1},
    ],
)
def test_invalid_metric_policies(fields):
    with pytest.raises((ValueError, TypeError)):
        TrainingMetrics(**fields)


def test_online_reporting_policy_does_not_change_optimizer_trajectory():
    baseline = train(
        replace(
            request(),
            metrics=TrainingMetrics(
                train="evaluate", initial=("train", "validation"), update_history=True
            ),
        )
    )
    measured = train(online_spec())
    for name in baseline.parameters:
        torch.testing.assert_close(
            baseline.parameters[name], measured.parameters[name], rtol=0, atol=0
        )
    for name in baseline.optimizer_state:
        for key in baseline.optimizer_state[name]:
            torch.testing.assert_close(
                baseline.optimizer_state[name][key],
                measured.optimizer_state[name][key],
                rtol=0,
                atol=0,
            )
    assert torch.equal(
        baseline.training_checkpoint.rng_state, measured.training_checkpoint.rng_state
    )


def test_class_stratification_cannot_resume_with_changed_training_labels(tmp_path):
    spec = replace(
        request(),
        shuffle=False,
        batching=Batching(order="stratified", stratify_by="label"),
    )
    checkpoint = save_training_checkpoint(
        tmp_path / "partial", train(replace(spec, updates=1)).training_checkpoint
    )
    with pytest.raises(ValueError, match="protocol"):
        train(
            replace(
                spec,
                checkpoint=checkpoint,
                targets={"label": 1 - spec.targets["label"]},
            )
        )


def test_callback_alone_adds_no_tensor_cpu_transfers(monkeypatch):
    calls = []
    original = torch.Tensor.cpu

    def tracked(tensor, *args, **kwargs):
        calls.append(tuple(tensor.shape))
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", tracked)
    train(request())
    count = len(calls)
    calls.clear()
    train(replace(request(), on_epoch_end=lambda report: None))
    assert len(calls) == count


def test_default_metrics_match_explicit_policy_and_resume(tmp_path):
    spec = request()
    implicit = train(spec)
    explicit = train(replace(spec, metrics=TrainingMetrics()))
    assert TrainingMetrics().to_dict() == {
        "train": "online",
        "validation": "epoch",
        "initial": ["validation"],
        "update_history": False,
        "test": "final",
    }
    assert implicit.metrics["epochs"] == explicit.metrics["epochs"]
    assert implicit.metrics["updates"] == []
    assert "train_loss" not in implicit.metrics["epochs"][0]
    assert "validation_loss" in implicit.metrics["epochs"][0]
    skipped = train(replace(spec, metrics=TrainingMetrics(initial=())))
    assert "validation_loss" not in skipped.metrics["epochs"][0]
    assert skipped.metrics["epochs"][1:] == implicit.metrics["epochs"][1:]
    assert skipped.metrics["selection_record"] == implicit.metrics["selection_record"]
    assert [row.get("train_loss") for row in implicit.metrics["epochs"]] == [
        row.get("train_loss") for row in explicit.metrics["epochs"]
    ]
    partial = train(replace(spec, updates=1))
    saved = save_training_checkpoint(tmp_path / "default", partial.training_checkpoint)
    resumed = train(replace(spec, checkpoint=saved))
    assert resumed.metrics["epochs"] == implicit.metrics["epochs"]
