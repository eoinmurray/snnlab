"""Tensor execution results and independent NumPy analysis views."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Any

import numpy as np
import torch
from torch import nn

from snnlab.sim.artifacts.recordings import (
    RecordingData,
    SignalInfo,
    numpy_copy,
    select_tensor,
)
from snnlab.sim.artifacts.snapshots import RunSnapshot
from snnlab.sim.requests.spec import ExecutorName
from snnlab.sim.runtime.state import GraphRuntimeState
from snnlab.sim.training.checkpoints import TrainingCheckpoint


@dataclass(frozen=True)
class NumpyExecutionResult:
    """Independent CPU arrays for plotting and analysis of an execution."""

    outputs: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]
    time_ms: np.ndarray | None
    recording: RecordingData = field(default_factory=lambda: RecordingData({}, {}))

    @property
    def recorded_signals(self):
        return self.recording.signals


@dataclass
class ExecutionResult:
    executor: ExecutorName
    outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)
    parameters: dict[str, torch.Tensor] = field(default_factory=dict)
    gradients: dict[str, torch.Tensor] = field(default_factory=dict)
    optimizer_state: dict[str, Any] = field(default_factory=dict)
    training_checkpoint: TrainingCheckpoint | None = None
    selected_checkpoint: TrainingCheckpoint | None = None
    final_state: dict[str, torch.Tensor] = field(default_factory=dict)
    runtime_state: GraphRuntimeState | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    model: nn.Module | None = None
    decisions: list[dict[str, Any]] = field(default_factory=list)
    recorded_signals: dict[str, torch.Tensor] = field(default_factory=dict)

    snapshots: dict[str, RunSnapshot] = field(default_factory=dict)

    _output_axes: dict[str, tuple[int | str, ...]] = field(
        default_factory=dict, repr=False
    )
    _diagnostic_axes: dict[str, tuple[int | str, ...]] = field(
        default_factory=dict, repr=False
    )
    _timebase: tuple[float, int, int] | None = field(default=None, repr=False)
    _batch_size: int | None = field(default=None, repr=False)
    _recording_info: dict[str, SignalInfo] = field(default_factory=dict, repr=False)

    def write_metrics(self, path, *, overwrite=False):
        """Export native metrics only; never run evaluation or selection."""
        from snnlab.sim.artifacts.evidence import write_metrics

        return write_metrics(path, self.metrics, overwrite=overwrite)

    def write_recording(self, path, *, signals=None, batch=None, overwrite=False):
        """Export retained recordings through the native lossless NPZ sink."""
        from snnlab.sim.artifacts.recordings import write_recording

        self._validate_batch(batch)
        return write_recording(
            path,
            self.recorded_signals,
            self._recording_info,
            signals=signals,
            batch=batch,
            overwrite=overwrite,
        )

    def _validate_batch(self, batch):
        if batch is not None:
            if not isinstance(batch, Integral) or isinstance(batch, bool):
                raise TypeError("batch must be an integer or None")
            if self._batch_size is None:
                raise ValueError("batch selection requires execution metadata")
            if not 0 <= batch < self._batch_size:
                raise IndexError(f"batch {batch} is outside [0, {self._batch_size})")

    def numpy(self, *, batch: int | None = None) -> NumpyExecutionResult:
        """Copy outputs, diagnostics and retained recordings to NumPy.

        Batch axes come from signal declarations, not tensor rank. Time values
        use the effective timestep and include any runtime continuation offset.
        The original tensors and their computation graphs are left intact.
        """
        self._validate_batch(batch)

        def arrays(values, axes):
            converted = {}
            for name, tensor in values.items():
                selected = tensor.detach()
                if batch is not None:
                    if name not in axes:
                        raise ValueError(f"batch axis metadata is missing for {name!r}")
                    dimensions = axes[name]
                    if len(dimensions) != tensor.ndim:
                        raise ValueError(
                            f"signal axis metadata does not match tensor {name!r}"
                        )
                    if "batch" in dimensions:
                        selected = selected.select(
                            dimensions.index("batch"), int(batch)
                        )
                converted[name] = numpy_copy(selected)
            return converted

        time_ms = None
        if self._timebase is not None:
            dt_ms, start_step, steps = self._timebase
            time_ms = (
                np.arange(start_step, start_step + steps, dtype=np.float64) * dt_ms
            )
        recordings, information = {}, {}
        for name, tensor in self.recorded_signals.items():
            if name not in self._recording_info:
                raise ValueError(f"recording metadata is missing for {name!r}")
            selected, info = select_tensor(tensor, self._recording_info[name], batch)
            recordings[name] = numpy_copy(selected)
            information[name] = replace(
                info,
                source_dtype=str(tensor.dtype).removeprefix("torch."),
                array_dtype=str(recordings[name].dtype),
                conversion="bfloat16_to_float32"
                if tensor.dtype == torch.bfloat16
                else None,
            )
        return NumpyExecutionResult(
            outputs=arrays(self.outputs, self._output_axes),
            diagnostics=arrays(self.diagnostics, self._diagnostic_axes),
            time_ms=time_ms,
            recording=RecordingData(recordings, information),
        )
