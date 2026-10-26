"""Read and interpret native training evidence without executing a network."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral, Real

import numpy as np

from snnlab.sim.artifacts.evidence import read_metrics
from snnlab.sim.training.policies import FrozenValues
from snnlab.sim.training.selection import CheckpointSelection


@dataclass(frozen=True)
class MetricCurve:
    """Observed epoch coordinates; NaN/mask represent unavailable measurements."""

    epochs: np.ndarray
    values: np.ndarray
    available: np.ndarray
    measurement: tuple[str | None, ...]
    split: str
    metric: str
    objective: int | None = None


@dataclass(frozen=True)
class HistoryEpoch:
    epoch: int
    splits: Mapping[str, FrozenValues]
    update: int | None
    metadata: FrozenValues


def _count(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def metric_value(values, metric, objective=None):
    """Use native objective identifiers; never guess among multiple objectives."""
    if objective is not None:
        _count(objective, "objective")
        group = {
            "accuracy": "accuracies",
            "cross_entropy": "cross_entropies",
            "loss": "components",
        }.get(metric)
        if group is None:
            raise ValueError(f"metric {metric!r} has no objective selector")
        return values.get(group, {}).get(f"objective[{objective}]")
    if metric == "loss":
        return values.get("loss")
    group = {"accuracy": "accuracies", "cross_entropy": "cross_entropies"}.get(metric)
    if group:
        objectives = values.get(group, {})
        if len(objectives) > 1:
            raise ValueError(f"{metric} requires an explicit objective")
        if len(objectives) == 1:
            return next(iter(objectives.values()))
    return values.get(metric)


def _validate_selection_metric_identities(policy):
    identities = set()
    for metric in (policy.metric, *policy.tie_break):
        if metric.kind not in {
            "loss",
            "accuracy",
            "cross_entropy",
        } or metric.direction not in {"min", "max"}:
            raise ValueError("unsupported selection metric or direction")
        _count(metric.objective, "selection objective")
        if (metric.kind == "loss" and metric.objective != 0) or (
            metric.kind,
            metric.objective,
        ) in identities:
            raise ValueError("invalid or duplicate selection metric")
        identities.add((metric.kind, metric.objective))


class TrainingHistory:
    """An independent copy of native metrics, with explicit evidence validation.

    Reading never fills missing epochs, reruns evaluation, or certifies the
    numerical implementation that produced the measurements.
    """

    def __init__(self, metrics):
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        self._metrics = copy.deepcopy(dict(metrics))
        rows = self._metrics.get("epochs", [])
        if not isinstance(rows, (list, tuple)) or any(
            not isinstance(row, Mapping) for row in rows
        ):
            raise ValueError("history epochs must be a sequence of mappings")
        self._rows = rows
        self.validate()

    @classmethod
    def read(cls, path, *, legacy=False):
        return cls.from_metrics(read_metrics(path, legacy=legacy))

    @classmethod
    def from_metrics(cls, metrics):
        return cls(metrics)

    @property
    def epochs(self):
        return tuple(
            HistoryEpoch(
                row["epoch"],
                {
                    split: FrozenValues(self._split(row, split))
                    for split in self.splits
                    if self._split(row, split)
                },
                row.get("update"),
                FrozenValues(
                    {
                        key: value
                        for key, value in row.items()
                        if not any(key.startswith(f"{split}_") for split in self.splits)
                    }
                ),
            )
            for row in self._rows
        )

    @property
    def splits(self):
        # These are native split names, not arbitrary prefixes such as epoch_updates.
        return tuple(
            split
            for split in ("train", "validation")
            if any(self._split(row, split) for row in self._rows)
        )

    @property
    def selection(self):
        value = self._metrics.get("selection_record")
        return FrozenValues(value) if value is not None else None

    @property
    def test(self):
        value = self._metrics.get("test")
        return FrozenValues(value) if value is not None else None

    @property
    def performance(self):
        return FrozenValues(self._metrics.get("performance", {}))

    @property
    def timing(self):
        value = self._metrics.get("training_timing")
        return FrozenValues(value) if value is not None else None

    @property
    def coverage(self):
        epochs = [row["epoch"] for row in self._rows]
        policy = (self._metrics.get("checkpoint_selection") or {}).get("policy", {})
        first_candidate = 0 if policy.get("include_initial") else 1
        observed_candidates = [epoch for epoch in epochs if epoch >= first_candidate]
        selection_verified = (
            bool(epochs)
            and len(observed_candidates) == max(0, epochs[-1] - first_candidate + 1)
            and all(
                epoch == index + first_candidate
                for index, epoch in enumerate(observed_candidates)
            )
        )
        return FrozenValues(
            {
                "first_epoch": self._rows[0]["epoch"] if self._rows else None,
                "last_epoch": self._rows[-1]["epoch"] if self._rows else None,
                "complete": None,
                **self._metrics.get("history_coverage", {}),
                "resumed_from_update": self._metrics.get("resumed_from_update"),
                "contiguous": all(
                    right == left + 1 for left, right in pairwise(epochs)
                ),
                "selection_verified": selection_verified
                if policy.get("mode", "epoch_metrics") == "epoch_metrics" and policy
                else False,
            }
        )

    @staticmethod
    def _split(row, split):
        return {
            key[len(split) + 1 :]: value
            for key, value in row.items()
            if key.startswith(f"{split}_")
        }

    def curve(self, *, metric, split, objective=None):
        if split not in {"train", "validation"}:
            raise ValueError(
                "curves require train or validation; test is a separate measurement"
            )
        epochs, values, semantics = [], [], []
        for row in self._rows:
            measurement = self._split(row, split)
            value = metric_value(measurement, metric, objective)
            epochs.append(row["epoch"])
            values.append(np.nan if value is None else value)
            semantics.append(measurement.get("measurement"))
        values = np.asarray(values, dtype=np.float64)
        return MetricCurve(
            np.asarray(epochs, dtype=np.int64),
            values,
            np.isfinite(values),
            tuple(semantics),
            split,
            metric,
            objective,
        )

    def curves(
        self,
        *,
        metrics=("loss", "accuracy"),
        splits=("train", "validation"),
        objective=None,
    ):
        return {
            (split, metric): self.curve(metric=metric, split=split, objective=objective)
            for split in splits
            for metric in metrics
        }

    @staticmethod
    def _validate_measurement(values, label):
        for key in ("loss", "accuracy", "cross_entropy"):
            if values.get(key) is not None:
                TrainingHistory._metric(values[key], key, label)
        for group in ("components", "accuracies", "cross_entropies"):
            entries = values.get(group, {})
            if not isinstance(entries, Mapping):
                raise ValueError(f"{label}.{group} must be a mapping")
            for key, value in entries.items():
                if value is not None:
                    TrainingHistory._metric(value, group, f"{label}.{key}")
        if "examples" in values:
            _count(values["examples"], f"{label}.examples")
        if "encoding_draws" in values:
            draws = values["encoding_draws"]
            if not isinstance(draws, (tuple, list)) or not draws:
                raise ValueError(f"{label}: encoding draws must be a nonempty sequence")
            for draw in draws:
                _count(draw, f"{label}.encoding_draw")
            if len(set(draws)) != len(draws):
                raise ValueError(f"{label}: encoding draws must be unique")
        accuracies = values.get("accuracies", {})
        # Older bare mappings can contain the scalar without the group.
        if (
            values.get("accuracy") is not None
            and (
                len(accuracies) != 1
                or values["accuracy"] != next(iter(accuracies.values()))
            )
        ) and (accuracies):
            raise ValueError(
                f"{label}: scalar accuracy disagrees with objective metrics"
            )
        semantics = values.get("measurement")
        if semantics not in {None, "online_pre_update", "fixed_weight_evaluation"}:
            raise ValueError(f"{label}: unknown measurement semantics")

    @staticmethod
    def _metric(value, kind, label):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{label}.{kind} must be a finite number")
        if kind in {"accuracy", "accuracies"} and not 0 <= value <= 1:
            raise ValueError(f"{label}: accuracy must be in [0, 1]")
        if kind in {"cross_entropy", "cross_entropies"} and value < 0:
            raise ValueError(f"{label}: cross entropy must be non-negative")

    def validate(
        self,
        *,
        expected_epochs=None,
        require_complete=False,
        selection=None,
        expected_examples=None,
        expected_encoding_draws=None,
    ):
        if type(require_complete) is not bool:
            raise TypeError("require_complete must be boolean")
        coverage = self._metrics.get("history_coverage", {})
        if not isinstance(coverage, Mapping) or (
            coverage.get("complete") is not None
            and type(coverage["complete"]) is not bool
        ):
            raise ValueError("invalid history coverage metadata")
        if "requested_epochs" in coverage:
            _count(coverage["requested_epochs"], "requested_epochs")
        previous = self._validate_epoch_rows()

        self._validate_history_coverage(
            coverage=coverage,
            expected_epochs=expected_epochs,
            require_complete=require_complete,
            previous=previous,
        )

        test = self._validate_test_measurement()

        contract, policy, record = self._resolve_history_selection(selection=selection)

        if policy is not None and policy.mode == "epoch_metrics":
            if (
                policy.split not in {"train", "validation"}
                or policy.cadence != "completed_epoch"
                or type(policy.include_initial) is not bool
            ):
                raise ValueError("unsupported epoch checkpoint selection policy")
            _validate_selection_metric_identities(policy=policy)

            self._validate_selection_record(
                record=record, policy=policy, previous=previous, coverage=coverage
            )

            self._validate_selection_winner(policy=policy, record=record)

            if (
                record is not None
                and record.get("epoch") == 0
                and not policy.include_initial
            ):
                raise ValueError("initial epoch is ineligible for checkpoint selection")
            if require_complete and not self.coverage.selection_verified:
                raise ValueError(
                    "complete selection validation requires all eligible candidates"
                )
            if record is not None and record.get("evaluation") != contract.get(
                "evaluation"
            ):
                raise ValueError(
                    "selected checkpoint evaluation provenance is inconsistent"
                )
        elif policy is not None and policy.mode == "none" and record is not None:
            raise ValueError("disabled selection cannot contain a selected checkpoint")
        elif (
            policy is not None
            and policy.mode == "legacy_training_batch_loss"
            and record is not None
        ):
            self._validate_legacy_selection(record=record, policy=policy)

        elif policy is not None and policy.mode not in {
            "none",
            "legacy_training_batch_loss",
        }:
            raise ValueError("unsupported checkpoint selection mode")
        self._validate_test_checkpoint(test=test, policy=policy, record=record)

        self._validate_expected_measurements(
            test=test,
            expected_examples=expected_examples,
            expected_encoding_draws=expected_encoding_draws,
        )

    def _validate_epoch_rows(self):
        previous = -1
        previous_update = -1
        for row in self._rows:
            epoch = row.get("epoch")
            _count(epoch, "epoch")
            if epoch <= previous:
                raise ValueError(
                    "history epochs must be strictly increasing and unique"
                )
            previous = epoch
            for key in (
                "update",
                "examples",
                "epoch_examples",
                "epoch_updates",
                "dropped_examples",
            ):
                if key in row:
                    _count(row[key], key)
            if "update" in row:
                if row["update"] < previous_update:
                    raise ValueError("history update counts cannot decrease")
                previous_update = row["update"]
            for split in ("train", "validation"):
                values = self._split(row, split)
                self._validate_measurement(values, f"epoch {epoch} {split}")
                if epoch == 0 and values.get("measurement") == "online_pre_update":
                    raise ValueError(
                        "epoch zero cannot contain online training aggregates"
                    )
                if (
                    split == "validation"
                    and values.get("measurement") == "online_pre_update"
                ):
                    raise ValueError("validation must use fixed weights")
        return previous

    def _validate_history_coverage(
        self, *, coverage, expected_epochs, require_complete, previous
    ):
        if expected_epochs is not None:
            _count(expected_epochs, "expected_epochs")
            if coverage.get("requested_epochs", expected_epochs) != expected_epochs:
                raise ValueError(
                    "recorded requested_epochs differs from expected_epochs"
                )
            if previous > expected_epochs:
                raise ValueError("history exceeds expected_epochs")
        if require_complete:
            limit = (
                expected_epochs
                if expected_epochs is not None
                else self.coverage.get("requested_epochs")
            )
            if limit is None:
                raise ValueError(
                    "complete history requires expected_epochs or recorded coverage"
                )
            positive = [row["epoch"] for row in self._rows if row["epoch"] > 0]
            if (
                self.coverage.complete is False
                or len(positive) != limit
                or any(epoch != index + 1 for index, epoch in enumerate(positive))
            ):
                raise ValueError("history is incomplete or has missing epochs")

    def _validate_test_measurement(self):
        test = self._metrics.get("test")
        if test is not None:
            if not isinstance(test, Mapping):
                raise ValueError("test must be a mapping or None")
            self._validate_measurement(test, "test")
            if test.get("measurement") == "online_pre_update":
                raise ValueError("test must use fixed weights")
            if test.get("checkpoint") not in {"selected", "final"}:
                raise ValueError("test checkpoint must be selected or final")
            for key in ("epoch", "update"):
                _count(test.get(key), f"test.{key}")
            if self.coverage.complete is False:
                raise ValueError(
                    "incomplete training cannot contain final held-out test evidence"
                )
        return test

    def _validate_selection_record(self, *, record, policy, previous, coverage):
        if record is not None:
            _count(record.get("epoch"), "selected epoch")
            _count(record.get("update"), "selected update")
            values = record.get("values")
            if not isinstance(values, (tuple, list)) or len(values) != 1 + len(
                policy.tie_break
            ):
                raise ValueError("selection scores do not match policy metrics")
            for value in values:
                self._metric(value, "selection_score", "selection")
            if record["epoch"] > max(
                previous, coverage.get("data_state", {}).get("epoch", previous)
            ):
                raise ValueError(
                    "selected checkpoint lies beyond available training progress"
                )
            if record.get(
                "phase", "initial" if record["epoch"] == 0 else "completed_epoch"
            ) != ("initial" if record["epoch"] == 0 else "completed_epoch"):
                raise ValueError("selected checkpoint phase disagrees with its epoch")

    def _validate_selection_winner(self, policy, record):
        best, best_epoch = None, None
        for row in self._rows:
            if row["epoch"] == 0 and not policy.include_initial:
                continue
            measurement = self._split(row, policy.split)
            if not measurement:
                raise ValueError("selection candidate measurement is unavailable")
            if measurement.get("measurement") == "online_pre_update":
                raise ValueError("epoch selection requires fixed-weight measurements")
            scores = policy.scores(measurement)
            if policy.better(scores, best):
                best, best_epoch = scores, row["epoch"]
        if best_epoch is not None:
            if record is None:
                raise ValueError(
                    "eligible checkpoint candidates have no selection record"
                )
            if self.coverage.selection_verified and (
                record.get("epoch") != best_epoch or record.get("values") != best
            ):
                raise ValueError(
                    "selected checkpoint disagrees with recorded candidate metrics or tie rules"
                )
            if not self.coverage.selection_verified and (
                policy.better(best, record.get("values"))
                or (
                    best == record.get("values")
                    and best_epoch < record.get("epoch", -1)
                )
            ):
                raise ValueError(
                    "selected checkpoint is worse than an available candidate"
                )
            if record.get("policy") != policy.to_dict():
                raise ValueError("selected checkpoint policy is inconsistent")
            winner = next(
                (row for row in self._rows if row["epoch"] == record["epoch"]), None
            )
            if (
                winner is not None
                and winner.get("update", record["update"]) != record["update"]
            ):
                raise ValueError("selected checkpoint update disagrees with its epoch")

    def _validate_legacy_selection(self, record, policy):
        _count(record.get("update"), "selected update")
        values = record.get("values")
        if (
            record.get("policy") != policy.to_dict()
            or not isinstance(values, (tuple, list))
            or len(values) != 1
        ):
            raise ValueError("legacy selection record disagrees with its policy")
        self._metric(values[0], "loss", "legacy selection")
        if record.get("phase") != "legacy_training_batch_loss" or record.get(
            "evaluation"
        ) != {"phase": "pre_update_batch_loss", "weights": "post_update"}:
            raise ValueError("legacy selection measurement provenance is inconsistent")
        updates = self._metrics.get("updates", ())
        best = None
        previous = 0
        for row in updates:
            _count(row.get("update"), "update history index")
            self._metric(row.get("loss"), "loss", "update history")
            if row["update"] <= previous:
                raise ValueError("update history must be strictly increasing")
            previous = row["update"]
            if best is None or row["loss"] < best["loss"]:
                best = row
            if row["loss"] < values[0] or (
                row["loss"] == values[0] and row["update"] < record["update"]
            ):
                raise ValueError("legacy winner is worse than an available update")
            if row["update"] == record["update"] and row["loss"] != values[0]:
                raise ValueError("legacy selection score disagrees with its update")
        if (
            best is not None
            and len(updates) == updates[-1]["update"]
            and best["update"] != record["update"]
        ):
            raise ValueError("legacy winner disagrees with complete update history")

    def _validate_test_checkpoint(self, *, test, policy, record):
        if test is not None:
            if test["checkpoint"] == "selected":
                legacy = (
                    policy is not None and policy.mode == "legacy_training_batch_loss"
                )
                if (
                    record is None
                    or test.get("selection_record") != record
                    or any(
                        test.get(key) != record.get(key)
                        for key in (("update",) if legacy else ("epoch", "update"))
                    )
                ):
                    raise ValueError(
                        "test provenance disagrees with selected checkpoint"
                    )
                if (
                    legacy
                    and record.get("epoch") is not None
                    and test["epoch"] not in {record["epoch"] - 1, record["epoch"]}
                ):
                    raise ValueError(
                        "legacy test checkpoint epoch disagrees with update selection"
                    )
            elif record is not None:
                raise ValueError(
                    "test uses final checkpoint despite a selected checkpoint"
                )
            elif (
                self.coverage.get("requested_epochs") is not None
                and test["epoch"] != self.coverage.requested_epochs
            ):
                raise ValueError("test final checkpoint epoch disagrees with coverage")
            if test["checkpoint"] == "final" and self._rows:
                last = self._rows[-1]
                if test["epoch"] < last["epoch"] or (
                    test["epoch"] == last["epoch"]
                    and "update" in last
                    and test["update"] != last["update"]
                ):
                    raise ValueError(
                        "test final checkpoint disagrees with epoch history"
                    )

    def _validate_expected_measurements(
        self, *, test, expected_examples, expected_encoding_draws
    ):
        for split, expected in (expected_examples or {}).items():
            _count(expected, f"expected_examples.{split}")
            measurements = (
                [test]
                if split == "test"
                else [self._split(row, split) for row in self._rows]
            )
            measurements = [value for value in measurements if value]
            if not measurements or any(
                value.get("examples") != expected for value in measurements
            ):
                raise ValueError(
                    f"{split} example count is missing or differs from expectation"
                )
        for split, expected in (expected_encoding_draws or {}).items():
            measurements = (
                [test]
                if split == "test"
                else [self._split(row, split) for row in self._rows]
            )
            measurements = [value for value in measurements if value]
            if not measurements or any(
                value.get("encoding_draws") != list(expected) for value in measurements
            ):
                raise ValueError(
                    f"{split} encoding draws are missing or differ from expectation"
                )

    def _resolve_history_selection(self, selection):
        contract = self._metrics.get("checkpoint_selection") or {}
        if not isinstance(contract, Mapping):
            raise ValueError("checkpoint selection contract must be a mapping")
        policy_dict = contract.get("policy")
        if selection is not None and not isinstance(selection, CheckpointSelection):
            raise TypeError("selection must be CheckpointSelection")
        if selection is not None and policy_dict != selection.to_dict():
            raise ValueError("checkpoint selection policy does not match expectation")
        policy = selection or (
            CheckpointSelection.from_dict(policy_dict) if policy_dict else None
        )
        record = self._metrics.get("selection_record")
        if record is not None:
            if not isinstance(record, Mapping):
                raise ValueError("selection record must be a mapping")
            for key in ("epoch", "update"):
                if record.get(key) is not None:
                    _count(record[key], f"selection.{key}")
        return contract, policy, record
