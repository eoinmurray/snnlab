"""Immutable declarations and resolved tensors for simulation inputs and targets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class DenseArrayBinding:
    """One concrete dense tensor resolved against a named graph input."""

    input_id: str
    value: torch.Tensor
    source: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetArrayBinding:
    """One concrete target vector resolved against a named training target."""

    target_id: str
    value: torch.Tensor
    source: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EventStreamBinding:
    """Sparse binary spike events resolved against one named graph input."""

    input_id: str
    steps: torch.Tensor
    batches: torch.Tensor
    channels: torch.Tensor
    steps_count: int
    batch_size: int
    source: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PoissonInputBinding:
    """Generated Bernoulli-discretised Poisson spikes for one graph input."""

    input_id: str
    steps_count: int
    rates_hz: Sequence[float]
    seed: int
    batch_size: int = 1
    categorical: bool = False


@dataclass(frozen=True)
class DatasetEncoder:
    """Portable standard encoding recipe for an immutable dataset snapshot."""

    kind: Literal["rate_poisson", "prebinned_spikes", "event_bin", "custom"]
    duration_ms: float | None = None
    max_rate_hz: float | None = None
    seed: int = 0
    definition: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    rates_hz: Sequence[float] | None = None


@dataclass(frozen=True)
class DatasetSnapshotBinding:
    """Bind one digest-identified NPZ snapshot to a graph input and target."""

    path: Path
    input_id: str
    dataset_id: str
    split: str
    encoder: DatasetEncoder
    target_id: str | None = None
    feature_key: str = "features"
    label_key: str = "labels"
    sample_cap: int | None = None
    shuffle: bool = False
    order_seed: int = 0
    sample_ids_key: str | None = None
    sample_indices: Sequence[int] | None = None
    snapshot_digest: str | None = None


def dataset_indices(binding, count):
    """Resolve explicit subset, order and cap without changing legacy RNG draws."""
    import numpy as np

    if binding.sample_indices is None:
        base = np.arange(count, dtype=np.int64)
    else:
        raw = tuple(binding.sample_indices)
        if (
            not raw
            or any(
                isinstance(index, bool)
                or not isinstance(index, (int, np.integer))
                or not 0 <= index < count
                for index in raw
            )
            or len(set(raw)) != len(raw)
        ):
            raise ValueError("dataset sample indices must be unique, valid integers")
        base = np.asarray(raw, dtype=np.int64)
    cap = len(base) if binding.sample_cap is None else binding.sample_cap
    if (
        isinstance(cap, bool)
        or not isinstance(cap, (int, np.integer))
        or not 0 < cap <= len(base)
    ):
        raise ValueError("dataset sample cap must be within available samples")
    if binding.shuffle:
        order = torch.randperm(
            len(base),
            generator=torch.Generator(device="cpu").manual_seed(binding.order_seed),
        ).numpy()
        base = base[order]
    return base[:cap]


InputBinding = (
    DenseArrayBinding
    | EventStreamBinding
    | PoissonInputBinding
    | DatasetSnapshotBinding
)


@dataclass(frozen=True)
class ResolvedDenseInputs:
    tensors: Mapping[str, torch.Tensor]
    protocol: Mapping[str, Any]
