"""Keyword-only public execution functions.

ExecutionSpec is an internal handoff shared by the CLI and execution handlers.
Each public function exposes only the settings supported by that operation.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import torch

from snnlab.sim.artifacts.snapshots import SnapshotSpec
from snnlab.sim.diagnostics.tracing import TraceSpec
from snnlab.sim.inputs.bindings import InputBinding, TargetArrayBinding
from snnlab.sim.requests import handlers
from snnlab.sim.requests.performance import (
    _UNSET,
    Performance,
    _resolve_performance,
    _Unset,
)
from snnlab.sim.requests.result import ExecutionResult
from snnlab.sim.requests.spec import ExecutionSpec, TestSpec, ValidationSpec
from snnlab.sim.runtime.interventions import Intervention
from snnlab.sim.runtime.recording import MeasurementWindow, RecordingSpec
from snnlab.sim.runtime.resets_decisions import DecisionSegments, ResetVoltage
from snnlab.sim.runtime.state import GraphRuntimeState
from snnlab.sim.training.observations import EpochObservations
from snnlab.sim.training.policies import Batching, EpochReport, TrainingMetrics
from snnlab.sim.training.selection import CheckpointSelection


def build(
    *,
    bundle: Path | None = None,
    graph: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    seed: int = 0,
    device: str = "auto",
    retain_parameters: bool = True,
    performance: Performance | None = None,
    projection_mode: Literal["fast", "strict"] | _Unset = _UNSET,
    extension_validation: Literal["strict", "structural"] = "structural",
    compile_mode: Literal["default"] | None | _Unset = _UNSET,
    precision: Literal["auto", "highest"] | _Unset = _UNSET,
    compile_block_steps: int | Literal["auto"] | _Unset = _UNSET,
    cuda_graphs: Literal["auto", "off"] | _Unset = _UNSET,
) -> ExecutionResult:
    """Initialize a graph or bundle without advancing simulation time."""
    policy = _resolve_performance(
        performance,
        projection_mode=projection_mode,
        compile_mode=compile_mode,
        precision=precision,
        compile_block_steps=compile_block_steps,
        cuda_graphs=cuda_graphs,
    )
    spec = ExecutionSpec(
        **asdict(policy),
        kind="build",
        bundle=bundle,
        graph=graph,
        training=training,
        seed=seed,
        device=device,
        retain_parameters=retain_parameters,
        extension_validation=extension_validation,
    )
    return handlers.build(spec)


def simulate(
    *,
    bundle: Path | None = None,
    graph: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    input_bindings: Sequence[InputBinding] = (),
    protocol: Mapping[str, Any] | None = None,
    batch_size: int | None = None,
    seed: int = 0,
    device: str = "auto",
    diagnostics: bool = True,
    checkpoint: Path | None = None,
    runtime_state: GraphRuntimeState | None = None,
    interventions: Sequence[Intervention] = (),
    resets: Sequence[ResetVoltage] = (),
    decisions: DecisionSegments | None = None,
    measurement: MeasurementWindow | None = None,
    recording: RecordingSpec | None = None,
    retain_outputs: bool = True,
    retain_parameters: bool = True,
    retain_final_state: bool = True,
    retain_runtime_state: bool = True,
    performance: Performance | None = None,
    projection_mode: Literal["fast", "strict"] | _Unset = _UNSET,
    extension_validation: Literal["strict", "structural"] = "structural",
    compile_mode: Literal["default"] | None | _Unset = _UNSET,
    precision: Literal["auto", "highest"] | _Unset = _UNSET,
    compile_block_steps: int | Literal["auto"] | _Unset = _UNSET,
    dataset_prefetch: int | _Unset = _UNSET,
    cuda_graphs: Literal["auto", "off"] | _Unset = _UNSET,
    timing_mode: Literal["synchronized", "enqueue"] = "synchronized",
    snapshot: str | Path | SnapshotSpec | None = None,
    inference_overrides: Mapping[str, Any] | None = None,
) -> ExecutionResult:
    """Execute a graph with bound inputs and optional runtime continuation."""
    policy = _resolve_performance(
        performance,
        projection_mode=projection_mode,
        compile_mode=compile_mode,
        precision=precision,
        compile_block_steps=compile_block_steps,
        cuda_graphs=cuda_graphs,
        dataset_prefetch=dataset_prefetch,
    )
    spec = ExecutionSpec(
        **asdict(policy),
        kind="simulate",
        bundle=bundle,
        graph=graph,
        training=training,
        input_bindings=input_bindings,
        protocol=protocol if protocol is not None else {},
        batch_size=batch_size,
        seed=seed,
        device=device,
        diagnostics=diagnostics,
        checkpoint=checkpoint,
        runtime_state=runtime_state,
        interventions=interventions,
        resets=resets,
        decisions=decisions,
        measurement=measurement,
        recording=recording,
        retain_outputs=retain_outputs,
        retain_parameters=retain_parameters,
        retain_final_state=retain_final_state,
        retain_runtime_state=retain_runtime_state,
        extension_validation=extension_validation,
        timing_mode=timing_mode,
        snapshot=snapshot,
        options=(
            {"inference_overrides": dict(inference_overrides)}
            if inference_overrides is not None
            else {}
        ),
    )
    return handlers.simulate(spec)


def infer(
    *,
    bundle: Path | None = None,
    graph: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    input_bindings: Sequence[InputBinding] = (),
    protocol: Mapping[str, Any] | None = None,
    batch_size: int | None = None,
    seed: int = 0,
    device: str = "auto",
    diagnostics: bool = True,
    checkpoint: Path | None = None,
    runtime_state: GraphRuntimeState | None = None,
    interventions: Sequence[Intervention] = (),
    resets: Sequence[ResetVoltage] = (),
    decisions: DecisionSegments | None = None,
    measurement: MeasurementWindow | None = None,
    recording: RecordingSpec | None = None,
    retain_outputs: bool = True,
    retain_parameters: bool = True,
    retain_final_state: bool = True,
    retain_runtime_state: bool = True,
    performance: Performance | None = None,
    projection_mode: Literal["fast", "strict"] | _Unset = _UNSET,
    extension_validation: Literal["strict", "structural"] = "structural",
    compile_mode: Literal["default"] | None | _Unset = _UNSET,
    precision: Literal["auto", "highest"] | _Unset = _UNSET,
    compile_block_steps: int | Literal["auto"] | _Unset = _UNSET,
    dataset_prefetch: int | _Unset = _UNSET,
    cuda_graphs: Literal["auto", "off"] | _Unset = _UNSET,
    timing_mode: Literal["synchronized", "enqueue"] = "synchronized",
    snapshot: str | Path | SnapshotSpec | None = None,
    inference_overrides: Mapping[str, Any] | None = None,
) -> ExecutionResult:
    """Run inference with the same execution controls as simulate."""
    policy = _resolve_performance(
        performance,
        projection_mode=projection_mode,
        compile_mode=compile_mode,
        precision=precision,
        compile_block_steps=compile_block_steps,
        cuda_graphs=cuda_graphs,
        dataset_prefetch=dataset_prefetch,
    )
    spec = ExecutionSpec(
        **asdict(policy),
        kind="infer",
        bundle=bundle,
        graph=graph,
        training=training,
        input_bindings=input_bindings,
        protocol=protocol if protocol is not None else {},
        batch_size=batch_size,
        seed=seed,
        device=device,
        diagnostics=diagnostics,
        checkpoint=checkpoint,
        runtime_state=runtime_state,
        interventions=interventions,
        resets=resets,
        decisions=decisions,
        measurement=measurement,
        recording=recording,
        retain_outputs=retain_outputs,
        retain_parameters=retain_parameters,
        retain_final_state=retain_final_state,
        retain_runtime_state=retain_runtime_state,
        extension_validation=extension_validation,
        timing_mode=timing_mode,
        snapshot=snapshot,
        options=(
            {"inference_overrides": dict(inference_overrides)}
            if inference_overrides is not None
            else {}
        ),
    )
    return handlers.infer(spec)


def train(
    *,
    bundle: Path | None = None,
    graph: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    input_bindings: Sequence[InputBinding] = (),
    protocol: Mapping[str, Any] | None = None,
    targets: Mapping[str, torch.Tensor] | None = None,
    target_bindings: Sequence[TargetArrayBinding] = (),
    validation: ValidationSpec | None = None,
    test: TestSpec | None = None,
    epochs: int = 0,
    batch_size: int | None = None,
    shuffle: bool = False,
    batching: Batching | None = None,
    metrics: TrainingMetrics | None = None,
    on_epoch_end: Callable[[EpochReport], None] | None = None,
    updates: int | None = None,
    save_final_checkpoint: str | Path | None = None,
    save_selected_checkpoint: str | Path | None = None,
    seed: int = 0,
    device: str = "auto",
    diagnostics: bool = True,
    checkpoint: Path | None = None,
    observations: EpochObservations | None = None,
    checkpoint_selection: CheckpointSelection | None = None,
    retain_parameters: bool = True,
    retain_final_state: bool = True,
    retain_runtime_state: bool = True,
    performance: Performance | None = None,
    projection_mode: Literal["fast", "strict"] | _Unset = _UNSET,
    extension_validation: Literal["strict", "structural"] = "structural",
    compile_mode: Literal["default"] | None | _Unset = _UNSET,
    precision: Literal["auto", "highest"] | _Unset = _UNSET,
    compile_block_steps: int | Literal["auto"] | _Unset = _UNSET,
    dataset_prefetch: int | _Unset = _UNSET,
    cuda_graphs: Literal["auto", "off"] | _Unset = _UNSET,
    timing_mode: Literal["synchronized", "enqueue"] = "synchronized",
    training_timing: bool = False,
    fused_optimizer: Literal["auto", "off"] | _Unset = _UNSET,
    tracing: TraceSpec | None = None,
    snapshot: str | Path | SnapshotSpec | None = None,
) -> ExecutionResult:
    """Run native training with optional validation, testing and checkpoint resume."""
    from snnlab.sim.training.trainer import train as run_training

    policy = _resolve_performance(
        performance,
        projection_mode=projection_mode,
        compile_mode=compile_mode,
        precision=precision,
        compile_block_steps=compile_block_steps,
        cuda_graphs=cuda_graphs,
        dataset_prefetch=dataset_prefetch,
        fused_optimizer=fused_optimizer,
    )
    spec = ExecutionSpec(
        **asdict(policy),
        kind="train",
        bundle=bundle,
        graph=graph,
        training=training,
        input_bindings=input_bindings,
        protocol=protocol if protocol is not None else {},
        targets=targets if targets is not None else {},
        target_bindings=target_bindings,
        validation=validation,
        test=test,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        batching=batching,
        metrics=metrics,
        on_epoch_end=on_epoch_end,
        updates=updates,
        save_final_checkpoint=save_final_checkpoint,
        save_selected_checkpoint=save_selected_checkpoint,
        seed=seed,
        device=device,
        diagnostics=diagnostics,
        checkpoint=checkpoint,
        observations=observations,
        checkpoint_selection=checkpoint_selection,
        retain_parameters=retain_parameters,
        retain_final_state=retain_final_state,
        retain_runtime_state=retain_runtime_state,
        extension_validation=extension_validation,
        timing_mode=timing_mode,
        training_timing=training_timing,
        tracing=tracing,
        snapshot=snapshot,
    )
    return run_training(spec)
