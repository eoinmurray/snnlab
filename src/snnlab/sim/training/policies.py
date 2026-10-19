"""Execution-only measurement, batching and immutable epoch reports."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Literal

import torch


@dataclass(frozen=True)
class TrainingMetrics:
    train: Literal["online", "evaluate", "none"] = "online"
    validation: Literal["epoch", "none"] = "epoch"
    initial: tuple[str, ...] = ("validation",)
    update_history: bool = False
    test: Literal["final", "none"] = "final"

    def __post_init__(self):
        if self.train not in {"online", "evaluate", "none"}:
            raise ValueError("metrics train must be online, evaluate or none")
        if self.validation not in {"epoch", "none"}:
            raise ValueError("metrics validation must be epoch or none")
        if self.test not in {"final", "none"}:
            raise ValueError("metrics test must be final or none")
        if (
            isinstance(self.initial, str)
            or any(split not in {"train", "validation"} for split in self.initial)
            or len(set(self.initial)) != len(self.initial)
        ):
            raise ValueError("metrics initial requires unique train/validation splits")
        object.__setattr__(self, "initial", tuple(self.initial))
        if "train" in self.initial and self.train != "evaluate":
            raise ValueError("initial train metrics require train='evaluate'")
        if "validation" in self.initial and self.validation != "epoch":
            raise ValueError("initial validation metrics require validation='epoch'")
        if type(self.update_history) is not bool:
            raise TypeError("update_history must be boolean")

    def to_dict(self):
        return {**asdict(self), "initial": list(self.initial)}


@dataclass(frozen=True)
class Batching:
    order: Literal["sequential", "shuffle", "stratified"] = "sequential"
    stratify_by: str | None = None
    remainder: Literal["keep", "drop", "merge_small"] = "keep"
    min_batch_size: int | None = None

    def __post_init__(self):
        if self.order not in {"sequential", "shuffle", "stratified"}:
            raise ValueError("batching order must be sequential, shuffle or stratified")
        if self.remainder not in {"keep", "drop", "merge_small"}:
            raise ValueError("batching remainder must be keep, drop or merge_small")
        if self.order == "stratified":
            if not isinstance(self.stratify_by, str) or not self.stratify_by:
                raise ValueError("stratified batching requires stratify_by")
        elif self.stratify_by is not None:
            raise ValueError("stratify_by requires stratified batching")
        if self.remainder == "merge_small":
            if type(self.min_batch_size) is not int or self.min_batch_size < 1:
                raise ValueError("merge_small requires a positive min_batch_size")
        elif self.min_batch_size is not None:
            raise ValueError("min_batch_size requires merge_small")

    def to_dict(self):
        return asdict(self)

    def sizes(self, count, batch_size):
        if self.min_batch_size is not None and self.min_batch_size > batch_size:
            raise ValueError("min_batch_size cannot exceed batch_size")
        full, tail = divmod(count, batch_size)
        sizes = [batch_size] * full
        if tail and self.remainder != "drop":
            if self.remainder == "merge_small" and tail < self.min_batch_size and sizes:
                sizes[-1] += tail
            else:
                sizes.append(tail)
        if not sizes:
            raise ValueError("batching drops every training example")
        return sizes

    def indices(self, count, batch_size, *, seed, labels=None):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if self.order == "stratified":
            if labels is None or labels.ndim != 1 or len(labels) != count:
                raise ValueError(
                    "stratification target must have one label per example"
                )
            labels = labels.detach().cpu()
            if (
                labels.dtype
                not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
                or (labels < 0).any()
            ):
                raise ValueError("stratification labels must be nonnegative integers")
            queues, positions = [], []
            for label in torch.unique(labels, sorted=True):
                queue = torch.where(labels == label)[0]
                queue = queue[torch.randperm(len(queue), generator=generator)]
                queues.append(queue)
                positions.append(
                    (torch.arange(len(queue), dtype=torch.float64) + 0.5) / len(queue)
                )
            indices, positions = torch.cat(queues), torch.cat(positions)
            ties = torch.randperm(count, generator=generator)
            order = indices[ties[torch.argsort(positions[ties], stable=True)]]
        elif self.order == "shuffle":
            order = torch.randperm(count, generator=generator)
        else:
            order = torch.arange(count)
        batches, start = [], 0
        for size in self.sizes(count, batch_size):
            batches.append(order[start : start + size])
            start += size
        return batches


class FrozenValues(Mapping):
    """Recursively immutable host values, with mapping and attribute access."""

    __slots__ = ("_values",)

    def __init__(self, values):
        object.__setattr__(
            self,
            "_values",
            MappingProxyType({key: freeze(value) for key, value in values.items()}),
        )

    def __setattr__(self, name, value):
        raise AttributeError("epoch reports are read-only")

    def __delattr__(self, name):
        raise AttributeError("epoch reports are read-only")

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError as error:
            raise AttributeError(name) from error


def freeze(value):
    if isinstance(value, Mapping):
        return FrozenValues(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class EpochReport:
    epoch: int
    total_epochs: int
    updates: int
    examples: int
    train: FrozenValues | None
    validation: FrozenValues | None
    observations: FrozenValues | None
    timing: FrozenValues | None
