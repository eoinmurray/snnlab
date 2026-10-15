"""Execution requests, input binding declarations and validation policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import torch

from snnlab.sim.artifacts.snapshots import SnapshotSpec
from snnlab.sim.diagnostics.tracing import TraceSpec
from snnlab.sim.inputs.bindings import InputBinding, TargetArrayBinding
from snnlab.sim.runtime.interventions import Intervention
from snnlab.sim.runtime.recording import MeasurementWindow, RecordingSpec
from snnlab.sim.runtime.resets_decisions import DecisionSegments, ResetVoltage
from snnlab.sim.runtime.state import GraphRuntimeState
from snnlab.sim.training.observations import EpochObservations
from snnlab.sim.training.policies import Batching, EpochReport, TrainingMetrics
from snnlab.sim.training.selection import CheckpointSelection

ExecutorName = Literal["graph"]


RequestKind = Literal["build", "simulate", "train", "infer"]


@dataclass(frozen=True)
class ValidationSpec:
    """Held-out inputs and labels evaluated without optimizer updates."""

    input_bindings: Sequence[InputBinding] = field(default_factory=tuple)
    targets: Mapping[str, torch.Tensor] = field(default_factory=dict)
    target_bindings: Sequence[TargetArrayBinding] = field(default_factory=tuple)
    protocol: Mapping[str, Any] = field(default_factory=dict)
    encoding_seeds: Sequence[int] = ()


@dataclass(frozen=True)
class ExecutionSpec:
    kind: RequestKind
    executor: ExecutorName = "graph"
    bundle: Path | None = None
    graph: Mapping[str, Any] | None = None
    input_bindings: Sequence[InputBinding] = field(default_factory=tuple)
    protocol: Mapping[str, Any] = field(default_factory=dict)
    training: Mapping[str, Any] | None = None
    targets: Mapping[str, torch.Tensor] = field(default_factory=dict)
    target_bindings: Sequence[TargetArrayBinding] = field(default_factory=tuple)
    validation: ValidationSpec | None = None
    epochs: int = 0
    batch_size: int | None = None
    shuffle: bool = False
    batching: Batching | None = None
    metrics: TrainingMetrics | None = None
    on_epoch_end: Callable[[EpochReport], None] | None = None
    updates: int | None = None
    save_final_checkpoint: str | Path | None = None
    save_selected_checkpoint: str | Path | None = None
    seed: int = 0
    device: str = "auto"
    diagnostics: bool = True
    checkpoint: Path | None = None
    runtime_state: GraphRuntimeState | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    observations: EpochObservations | None = None
    checkpoint_selection: CheckpointSelection | None = None
    interventions: Sequence[Intervention] = field(default_factory=tuple)
    resets: Sequence[ResetVoltage] = field(default_factory=tuple)
    decisions: DecisionSegments | None = None
    measurement: MeasurementWindow | None = None
    recording: RecordingSpec | None = None
    retain_outputs: bool = True
    retain_parameters: bool = True
    retain_final_state: bool = True
    retain_runtime_state: bool = True
    projection_mode: Literal["fast", "strict"] = "fast"
    extension_validation: Literal["strict", "structural"] = "structural"
    compile_mode: Literal["default"] | None = None
    dataset_prefetch: int = 0
    timing_mode: Literal["synchronized", "enqueue"] = "synchronized"
    training_timing: bool = False
    tracing: TraceSpec | None = None

    snapshot: str | Path | SnapshotSpec | None = None

    def __post_init__(self):
        if self.snapshot is not None:
            if not isinstance(self.snapshot, (str, Path, SnapshotSpec)):
                raise TypeError("snapshot must be a path or SnapshotSpec")
            if isinstance(self.snapshot, (str, Path)):
                SnapshotSpec(self.snapshot)
            if self.kind == "build":
                raise ValueError("snapshots require simulate, infer or train")
        if self.tracing is not None:
            if not isinstance(self.tracing, TraceSpec):
                raise TypeError("tracing must be a TraceSpec")
            if self.kind != "train":
                raise ValueError("tracing currently requires kind='train'")
        if self.projection_mode not in {"fast", "strict"}:
            raise ValueError("projection_mode must be fast or strict")
        for name in ("retain_parameters", "retain_final_state", "retain_runtime_state"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.compile_mode not in {None, "default"}:
            raise ValueError("compile_mode must be None or default")
        if (
            type(self.dataset_prefetch) is not int
            or not 0 <= self.dataset_prefetch <= 8
        ):
            raise ValueError("dataset_prefetch must be an integer from 0 to 8")
        if not isinstance(self.training_timing, bool):
            raise TypeError("training_timing must be boolean")
        if self.training_timing and self.kind != "train":
            raise ValueError("training_timing requires kind='train'")
        if self.timing_mode not in {"synchronized", "enqueue"}:
            raise ValueError("timing_mode must be synchronized or enqueue")
        if self.extension_validation not in ("strict", "structural"):
            raise ValueError("extension_validation must be strict or structural")
        if self.executor != "graph":
            raise ValueError("only the graph executor is supported")
        if self.recording is not None and not isinstance(self.recording, RecordingSpec):
            raise TypeError(
                "recording must be RecordingSpec; legacy recording strings are unsupported"
            )
        if self.measurement is not None and not isinstance(
            self.measurement, MeasurementWindow
        ):
            raise TypeError("measurement must be MeasurementWindow")
        if not isinstance(self.retain_outputs, bool):
            raise TypeError("retain_outputs must be boolean")
