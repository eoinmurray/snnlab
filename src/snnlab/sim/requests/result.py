"""Tensor execution results and independent NumPy analysis views."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
import torch
from torch import nn

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

    def numpy(self, *, batch: int | None = None) -> NumpyExecutionResult:
        """Copy outputs/diagnostics to NumPy, optionally selecting one batch item.

        Batch axes come from signal declarations, not tensor rank. Time values
        use the effective timestep and include any runtime continuation offset.
        The original tensors and their computation graphs are left intact.
        """
        if batch is not None:
            if not isinstance(batch, Integral) or isinstance(batch, bool):
                raise TypeError("batch must be an integer or None")
            if self._batch_size is None:
                raise ValueError("batch selection requires execution metadata")
            if not 0 <= batch < self._batch_size:
                raise IndexError(f"batch {batch} is outside [0, {self._batch_size})")

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
                converted[name] = selected.cpu().numpy().copy()
            return converted

        time_ms = None
        if self._timebase is not None:
            dt_ms, start_step, steps = self._timebase
            time_ms = (
                np.arange(start_step, start_step + steps, dtype=np.float64) * dt_ms
            )
        return NumpyExecutionResult(
            outputs=arrays(self.outputs, self._output_axes),
            diagnostics=arrays(self.diagnostics, self._diagnostic_axes),
            time_ms=time_ms,
        )
