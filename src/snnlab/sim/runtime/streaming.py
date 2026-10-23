"""Prepared, fixed-shape inference sessions with privately owned model/state."""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch

from snnlab import lang
from snnlab.sim.graph.executor import GraphExecutor
from snnlab.sim.graph.plan import GraphPlan, _signal_axes, plan_graph
from snnlab.sim.requests.handlers import resolve_device
from snnlab.sim.requests.performance import Performance
from snnlab.sim.requests.result import ExecutionResult
from snnlab.sim.runtime.state import (
    GraphRuntimeState,
    runtime_state_compatibility,
    runtime_state_signature,
)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _copy_state(state, device):
    snapshot = state.detached(device=device)
    snapshot.compatibility = copy.deepcopy(state.compatibility)
    return snapshot


class StreamSession:
    """Single-caller, inference-only session; construct with :func:`prepare_stream`.

    Inputs have fixed shape/device/dtype and are copied into private buffers.
    Outputs and exported states are independent snapshots. No pacing, deadline
    scheduler, training, recording, or per-window interventions are provided.
    """

    def __init__(
        self,
        model,
        *,
        window_steps,
        batch_size,
        retain_outputs,
        warmup_windows,
        policy,
        device,
        dtype,
    ):
        self._executor = model
        self._state = None
        self._closed = False
        self._busy = False
        self._advance_probe = None
        self._window_steps = window_steps
        self._batch_size = batch_size
        self._retain_outputs = retain_outputs
        parameter = next(model.parameters(), None)
        self._device = parameter.device if parameter is not None else device
        self._dtype = parameter.dtype if parameter is not None else dtype
        self._buffers = {}
        for row in model.plan.graph.get("inputs", []):
            shape = row["shape"]
            if (
                len(shape) < 2
                or tuple(shape[:2]) != ("time", "batch")
                or any(type(n) is not int or n < 1 for n in shape[2:])
            ):
                raise ValueError(
                    f"stream input {row['id']} requires (time, batch, *fixed_features) axes"
                )
            self._buffers[row["id"]] = torch.zeros(
                (window_steps, batch_size, *shape[2:]),
                device=self._device,
                dtype=torch.bool if row.get("signal_type") == "mask" else self._dtype,
            )
        if not self._buffers:
            raise ValueError("streaming requires at least one input")
        # The model and graph are private/frozen, so these caches need no per-call
        # graph/parameter invalidation and never affect a caller's training model.
        model._stream_metadata = {
            "compatibility": runtime_state_compatibility(
                model.plan,
                projection_mode=model.projection_mode,
                threshold_inclusive=model.surrogate_threshold_inclusive,
            ),
            "signature": runtime_state_signature(
                model.plan,
                projection_mode=model.projection_mode,
                threshold_inclusive=model.surrogate_threshold_inclusive,
            ),
            "axes": _signal_axes(model.plan.graph),
            "effective_weights": {
                name: model._mask_autapses(name, value)
                for name, value in model.parameter_map().items()
            },
        }
        started = time.perf_counter()
        state = None
        with torch.inference_mode():
            for _ in range(warmup_windows):
                result = self._execute(state)
                state = result.runtime_state
        _synchronize(self._device)
        self._info = {
            "window_steps": window_steps,
            "batch_size": batch_size,
            "device": str(self._device),
            "dtype": str(self._dtype),
            "warmup_windows": warmup_windows,
            "warmup_s": time.perf_counter() - started,
            "execution_path": copy.deepcopy(result.metrics["execution_path"]),
            "compiler_wrapper": copy.deepcopy(result.metrics["compiler_wrapper"]),
            "performance": asdict(policy),
            "cuda_graphs": "off: stateful continuation uses the graph executor",
        }

    @property
    def completed_steps(self):
        return self._state.completed_steps if self._state is not None else 0

    @property
    def info(self):
        """Independent preparation metadata; includes compilation/warmup costs."""
        return copy.deepcopy(self._info)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("stream session is closed")
        if self._busy:
            raise RuntimeError(
                "stream session does not support concurrent/reentrant calls"
            )

    def _execute(self, state):
        return self._executor(
            self._buffers,
            runtime_state=state,
            diagnostics=False,
            retain_outputs=self._retain_outputs,
            retain_parameters=False,
            retain_final_state=False,
        )

    @torch.inference_mode()
    def advance(self, inputs: Mapping[str, torch.Tensor]) -> ExecutionResult:
        """Advance one window; state stays private, outputs remain independently owned."""
        self._check_open()
        probe = self._advance_probe
        if probe is not None:
            probe.mark("Input validation")
        if not isinstance(inputs, Mapping) or set(inputs) != set(self._buffers):
            raise ValueError(
                f"stream input IDs must be exactly {sorted(self._buffers)}"
            )
        # Validate the whole window before writing buffers or advancing state.
        for name, expected in self._buffers.items():
            value = inputs[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"stream input {name} must be a torch.Tensor")
            if (
                value.shape != expected.shape
                or value.dtype != expected.dtype
                or value.device != expected.device
            ):
                raise ValueError(
                    f"stream input {name} expected shape={tuple(expected.shape)}, dtype={expected.dtype}, device={expected.device}"
                )
            if value.requires_grad:
                raise ValueError(
                    "stream inputs cannot require gradients; use GraphExecutor for training"
                )
        self._busy = True
        try:
            if probe is not None:
                probe.mark("Input buffer copy")
            for name, value in inputs.items():
                self._buffers[name].copy_(value)
            if probe is not None:
                probe.mark("Executor entry / validation")
            result = self._execute(self._state)
            if probe is not None:
                probe.mark("Output copies")
            # Ordinary detached tensors can be edited or consumed by a separately
            # trained readout outside inference_mode without aliasing stream state.
            with torch.inference_mode(False), torch.no_grad():
                outputs = {
                    name: value.clone() for name, value in result.outputs.items()
                }
            if probe is not None:
                probe.mark("Session result / commit")
            metrics = {
                key: copy.deepcopy(result.metrics[key])
                for key in (
                    "execution_path",
                    "precision",
                    "compiler_fallback",
                    "compiler_wrapper",
                )
            }
            metrics["streaming"] = {
                "completed_steps": result.runtime_state.completed_steps,
                "window_steps": self._window_steps,
            }
            owned = ExecutionResult(
                executor="graph",
                outputs=outputs,
                metrics=metrics,
                _output_axes=result._output_axes,
                _timebase=result._timebase,
                _batch_size=self._batch_size,
            )
            self._state = result.runtime_state
            return owned
        finally:
            self._busy = False

    @torch.inference_mode()
    def reset(self, state: GraphRuntimeState | None = None):
        """Reset to initial state or an independent copy of a compatible snapshot.

        Restoring validates state by a discarded probe call; its simulated time
        is not committed. The imported snapshot and current session stay intact
        if validation fails.
        """
        self._check_open()
        if state is None:
            self._state = None
            return
        if not isinstance(state, GraphRuntimeState):
            raise TypeError("state must be GraphRuntimeState or None")
        if type(state.completed_steps) is not int or state.completed_steps < 0:
            raise ValueError("state completed_steps must be a nonnegative integer")
        metadata = self._executor._stream_metadata
        if (
            state.signature != metadata["signature"]
            or state.compatibility != metadata["compatibility"]
        ):
            raise ValueError("runtime state is incompatible with the prepared stream")
        self._busy = True
        try:
            candidate = _copy_state(state, self._device)
            self._execute(candidate)
            self._state = candidate
        finally:
            self._busy = False

    def export_state(self) -> GraphRuntimeState:
        """Copy dynamic state, independently of this session and its graph metadata."""
        self._check_open()
        if self._state is None:
            raise ValueError("stream has no state; advance at least one window first")
        return _copy_state(self._state, self._device)

    def close(self):
        """Release model, buffers and state; calling close again is harmless."""
        if self._busy:
            raise RuntimeError("cannot close an active stream session")
        self._state = None
        self._buffers.clear()
        self._executor = None
        self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()


def prepare_stream(
    network,
    *,
    window_steps: int,
    batch_size: int = 1,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    seed: int = 0,
    performance: Performance | None = None,
    retain_outputs: bool = True,
    warmup_windows: int = 2,
) -> StreamSession:
    """Prepare a frozen inference session from a network, graph, bundle or executor.

    A GraphExecutor supplies a snapshot of its current weights. Other inputs use
    seeded initialization. Graph/parameters are privately copied. The default
    block spans a window (at most 64 steps; four on MPS). Explicit Performance
    policies retain their block/precision choices; CUDA replay is disabled for
    stateful continuation. Preparation warms without retaining simulated state.
    """
    for name, value, minimum in (
        ("window_steps", window_steps, 1),
        ("batch_size", batch_size, 1),
        ("warmup_windows", warmup_windows, 2),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if type(retain_outputs) is not bool:
        raise TypeError("retain_outputs must be boolean")
    source = network if isinstance(network, GraphExecutor) else None
    parameter = next(source.parameters(), None) if source is not None else None
    resolved = torch.device(
        resolve_device(
            device
            if device is not None
            else parameter.device
            if parameter is not None
            else "auto"
        )
    )
    dtype = (
        dtype
        if dtype is not None
        else parameter.dtype
        if parameter is not None
        else torch.float32
    )
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("stream dtype must be a supported floating torch.dtype")
    if resolved.type == "mps" and dtype == torch.float64:
        raise ValueError("MPS does not support float64")
    if performance is not None and not isinstance(performance, Performance):
        raise TypeError("performance must be Performance or None")
    policy = performance or Performance(
        compile_mode=source.compile_mode if source is not None else "default",
        projection_mode=source.projection_mode if source is not None else "fast",
        compile_block_steps=min(window_steps, 4 if resolved.type == "mps" else 64),
        cuda_graphs="off",
        precision=source.precision if source is not None else "highest",
    )
    if source is not None:
        plan = copy.deepcopy(source.plan)
    elif isinstance(network, GraphPlan):
        plan = copy.deepcopy(network)
    else:
        if isinstance(network, lang.Network):
            graph = lang.compile(network).graph
        elif isinstance(network, lang.Bundle):
            graph = network.graph
        elif isinstance(network, (str, Path)):
            graph = lang.load_bundle(network).graph
        elif isinstance(network, Mapping):
            graph = network
        else:
            raise TypeError(
                "network must be a Network, graph, GraphPlan, Bundle, bundle path or GraphExecutor"
            )
        plan = plan_graph(copy.deepcopy(graph))
    surrogate = copy.deepcopy(source.surrogate_config) if source is not None else None
    model = (
        GraphExecutor(
            plan,
            seed=seed,
            surrogate=surrogate,
            compile_mode=policy.compile_mode,
            compile_block_steps=policy.compile_block_steps,
            precision=policy.precision,
            projection_mode=policy.projection_mode,
            cuda_graphs="off",
            extension_validation=source.extension_validation
            if source is not None
            else "structural",
        )
        .to(device=resolved, dtype=dtype)
        .eval()
    )
    if source is not None:
        model.load_state_dict(copy.deepcopy(source.state_dict()))
    model.requires_grad_(False)
    started = time.perf_counter()
    session = StreamSession(
        model,
        window_steps=window_steps,
        batch_size=batch_size,
        retain_outputs=retain_outputs,
        warmup_windows=warmup_windows,
        policy=policy,
        device=resolved,
        dtype=dtype,
    )
    session._info["session_initialization_s"] = time.perf_counter() - started
    return session
