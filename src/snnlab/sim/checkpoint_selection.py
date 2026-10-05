"""Serializable policies for selecting graph-training checkpoints."""

import math
from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence


@dataclass(frozen=True)
class SelectionMetric:
    kind: Literal["cross_entropy", "accuracy"]
    objective: int = 0
    direction: Literal["min", "max"] = "min"


@dataclass(frozen=True)
class CheckpointSelection:
    split: Literal["train", "validation"] = "validation"
    cadence: Literal["completed_epoch"] = "completed_epoch"
    metric: SelectionMetric = SelectionMetric("cross_entropy")
    tie_break: Sequence[SelectionMetric] = ()
    include_initial: bool = False
    mode: Literal["epoch_metrics", "legacy_training_batch_loss"] = "epoch_metrics"

    @classmethod
    def legacy_training_batch_loss(cls):
        return cls(mode="legacy_training_batch_loss")

    def to_dict(self):
        if self.mode == "legacy_training_batch_loss":
            return {
                "mode": self.mode,
                "split": "train",
                "cadence": "update",
                "metric": {"kind": "weighted_batch_loss", "direction": "min"},
                "tie_break": [],
                "include_initial": False,
            }
        result = asdict(self)
        result["tie_break"] = list(result["tie_break"])
        return result

    @classmethod
    def from_dict(cls, value: Mapping):
        value = dict(value)
        if value.get("mode") == "legacy_training_batch_loss":
            legacy = cls.legacy_training_batch_loss()
            if any(legacy.to_dict().get(key) != item for key, item in value.items()):
                raise ValueError(
                    "legacy_training_batch_loss does not accept epoch metric settings"
                )
            return legacy
        if "metric" in value:
            value["metric"] = SelectionMetric(**value["metric"])
        value["tie_break"] = tuple(
            SelectionMetric(**m) for m in value.get("tie_break", ())
        )
        return cls(**value)

    def validate(self, training, *, epochs, has_validation):
        if self.mode == "legacy_training_batch_loss":
            if asdict(self) != asdict(self.legacy_training_batch_loss()):
                raise ValueError(
                    "legacy_training_batch_loss does not accept epoch metric settings"
                )
            return
        if self.mode != "epoch_metrics" or self.cadence != "completed_epoch":
            raise ValueError("unsupported checkpoint selection mode or cadence")
        if self.split not in {"train", "validation"}:
            raise ValueError("checkpoint selection split must be train or validation")
        if epochs <= 0:
            raise ValueError("epoch checkpoint selection requires positive epochs")
        if self.split == "validation" and not has_validation:
            raise ValueError("validation checkpoint selection requires ValidationSpec")
        if not isinstance(self.include_initial, bool):
            raise ValueError("include_initial must be boolean")
        seen = set()
        for metric in (self.metric, *self.tie_break):
            if not isinstance(metric, SelectionMetric):
                raise TypeError(
                    "checkpoint selection metrics must be SelectionMetric instances"
                )
            if metric.kind not in {
                "cross_entropy",
                "accuracy",
            } or metric.direction not in {"min", "max"}:
                raise ValueError("unsupported checkpoint selection metric or direction")
            index = metric.objective
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(training.get("objectives", ()))
            ):
                raise ValueError("checkpoint selection objective index is invalid")
            if training["objectives"][index]["kind"] != "cross_entropy":
                raise ValueError(
                    "checkpoint selection requires a cross_entropy objective"
                )
            identity = (metric.kind, index)
            if identity in seen:
                raise ValueError("duplicate checkpoint selection metric")
            seen.add(identity)

    def scores(self, metrics):
        values = []
        for metric in (self.metric, *self.tie_break):
            group = (
                "cross_entropies" if metric.kind == "cross_entropy" else "accuracies"
            )
            value = metrics.get(group, {}).get(f"objective[{metric.objective}]")
            if value is None or not math.isfinite(value):
                raise ValueError(
                    f"checkpoint selection metric {metric.kind} objective[{metric.objective}] is missing or non-finite"
                )
            values.append(float(value))
        return values

    def better(self, values, previous):
        if previous is None:
            return True
        for metric, value, old in zip(
            (self.metric, *self.tie_break), values, previous, strict=True
        ):
            if value != old:
                return value < old if metric.direction == "min" else value > old
        return False
