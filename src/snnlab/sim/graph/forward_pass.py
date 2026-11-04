"""Shared graph forward execution and causal delay buffers."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from snnlab import extensions as E
from snnlab._shared import neurons as N
from snnlab._shared.gains import sample as sample_gain
from snnlab._shared.spatial import convolution_drive, pool_tensor, signal_spatial_shape
from snnlab.sim.artifacts.inference import retention_identity
from snnlab.sim.diagnostics.tracing import TraceSession, current_trace
from snnlab.sim.graph import dynamics as M
from snnlab.sim.graph import extensions as X
from snnlab.sim.graph.blocks import network_block
from snnlab.sim.graph.kernels import (
    compiler_wrapper_policy,
    precision_fallback,
)
from snnlab.sim.graph.plan import _signal_axes
from snnlab.sim.requests.result import ExecutionResult
from snnlab.sim.runtime.interventions import (
    Intervention,
    PreparedIntervention,
    prepare_interventions,
)
from snnlab.sim.runtime.recording import (
    MeasurementWindow,
    OnlineReductions,
    Recorder,
    RecordingSpec,
)
from snnlab.sim.runtime.resets_decisions import (
    DecisionSegments,
    ResetVoltage,
    SegmentAccumulator,
    segment_identity,
)
from snnlab.sim.runtime.state import (
    GraphRuntimeState,
    _compatibility_mismatch,
    runtime_state_compatibility,
    runtime_state_signature,
)

if TYPE_CHECKING:
    from snnlab.sim.graph.executor import GraphExecutor


class DelayBuffer:
    """Fixed causal delay used by recurrent and feedback projections."""

    def __init__(self, delay_steps: int, prototype: torch.Tensor):
        if delay_steps < 1:
            raise ValueError("causal delay buffer requires at least one step")
        self.delay_steps = delay_steps
        self._values = [torch.zeros_like(prototype) for _ in range(delay_steps)]

    def read(self) -> torch.Tensor:
        return self._values[0]

    def push(self, value: torch.Tensor) -> None:
        self._values.append(value)
        self._values.pop(0)

    def export(self) -> torch.Tensor:
        return torch.stack([value.detach().clone() for value in self._values])

    @classmethod
    def restore(cls, values: torch.Tensor) -> DelayBuffer:
        if values.ndim < 2 or values.shape[0] < 1:
            raise ValueError("delay history must have shape [delay, batch, ...]")
        result = cls(int(values.shape[0]), values[0])
        result._values = [value.detach().clone() for value in values.unbind(0)]
        return result


@dataclass(frozen=True, slots=True)
class _ForwardOptions:
    """Requested behavior, shared unchanged across the forward phases."""

    diagnostics: bool
    runtime_state: GraphRuntimeState | None
    interventions: Sequence[Intervention]
    resets: Sequence[ResetVoltage]
    decisions: DecisionSegments | None
    measurement: MeasurementWindow | None
    recording: RecordingSpec | None
    retain_outputs: bool
    batch_offset: int
    automatic_reductions: bool
    required_signals: Sequence[str]
    observation_populations: Sequence[str]
    retain_parameters: bool
    retain_final_state: bool
    retain_runtime_state: bool


@dataclass(slots=True)
class _ForwardContext:
    """Per-call metadata and collaborators, populated in execution order."""

    options: _ForwardOptions
    executor: GraphExecutor
    inputs: Mapping[str, torch.Tensor]
    probe: Any = field(init=False)
    trace: TraceSession | None = field(init=False)
    batch: int = field(init=False)
    steps: int = field(init=False)
    device: torch.device = field(init=False)
    parameter_dtype: torch.dtype = field(init=False)
    populations: dict[str, Any] = field(init=False)
    prepared: dict[str, Any] | None = field(init=False)
    expected_compatibility: dict[str, Any] = field(init=False)
    expected_signature: str = field(init=False)
    segments: SegmentAccumulator = field(init=False)
    prepared_interventions: list[PreparedIntervention] = field(init=False)
    intervention_request: dict[str, Any] = field(init=False)
    resolved_interventions: list[dict[str, Any]] = field(init=False)
    projection_rows: dict[str, Any] = field(init=False)
    projection_ports: dict[str, str] = field(init=False)
    continuous_delays: dict[str, int] = field(init=False)
    signal_axes: dict[str, Any] = field(init=False)
    recorder: Recorder = field(init=False)
    online: OnlineReductions = field(init=False)
    inference_only: bool = field(init=False)
    streamed_outputs: dict[str, str] = field(init=False)
    external_signals: set[str] = field(init=False)
    history_operations: list[dict[str, Any]] = field(init=False)
    needed_signals: set[str] = field(init=False)
    capture_steps: bool = field(init=False)
    effective_weights: dict[str, torch.Tensor] = field(init=False)
    projection_gains: dict[str, tuple[float, ...]] = field(init=False)
    spatial_operations: dict[str, Any] = field(init=False)
    block_reasons: list[str] = field(init=False)
    compiler_fallback: str | None = field(init=False)


@dataclass(slots=True)
class _ForwardState:
    """Mutable neural state and retained samples owned by this invocation."""

    voltage: dict[str, torch.Tensor] = field(init=False)
    refractory: dict[str, torch.Tensor] = field(init=False)
    conductance: dict[tuple[str, str], torch.Tensor] = field(init=False)
    histories: dict[str, DelayBuffer] = field(init=False)
    input_histories: dict[str, torch.Tensor] = field(init=False)
    completed_steps: int = field(init=False)
    neuron_states: dict[str, dict[str, torch.Tensor]] = field(init=False)
    synapse_states: dict[str, dict[str, torch.Tensor]] = field(init=False)
    custom_state: dict[str, torch.Tensor] = field(init=False)
    continuous_histories: dict[str, list[torch.Tensor]] = field(init=False)
    observation_counts: dict[str, torch.Tensor] = field(init=False)
    integrator_sum: dict[str, torch.Tensor] = field(init=False)
    state_traces: dict[str, list[torch.Tensor]] = field(init=False)
    spike_traces: dict[str, list[torch.Tensor]] = field(init=False)
    voltage_traces: dict[str, list[torch.Tensor]] = field(init=False)
    pre_reset_voltage_traces: dict[str, list[torch.Tensor]] = field(init=False)
    conductance_traces: dict[str, list[torch.Tensor]] = field(init=False)


def run_forward_pass(
    executor: GraphExecutor,
    inputs: Mapping[str, torch.Tensor],
    *,
    diagnostics: bool = True,
    runtime_state: GraphRuntimeState | None = None,
    interventions: Sequence[Intervention] = (),
    resets: Sequence[ResetVoltage] = (),
    decisions: DecisionSegments | None = None,
    measurement: MeasurementWindow | None = None,
    recording: RecordingSpec | None = None,
    retain_outputs: bool = True,
    batch_offset: int = 0,
    automatic_reductions: bool = False,
    required_signals: Sequence[str] = (),
    observation_populations: Sequence[str] = (),
    retain_parameters: bool = True,
    retain_final_state: bool = True,
    retain_runtime_state: bool = True,
) -> tuple[ExecutionResult, dict[str, torch.Tensor]]:
    """Execute one graph trajectory through ordered private phases."""
    options = _ForwardOptions(
        diagnostics=diagnostics,
        runtime_state=runtime_state,
        interventions=interventions,
        resets=resets,
        decisions=decisions,
        measurement=measurement,
        recording=recording,
        retain_outputs=retain_outputs,
        batch_offset=batch_offset,
        automatic_reductions=automatic_reductions,
        required_signals=required_signals,
        observation_populations=observation_populations,
        retain_parameters=retain_parameters,
        retain_final_state=retain_final_state,
        retain_runtime_state=retain_runtime_state,
    )
    forward = _ForwardContext(options=options, executor=executor, inputs=inputs)
    runtime = _ForwardState()
    _validate_forward_inputs(forward, runtime)
    _restore_runtime_state(forward, runtime)
    _prepare_forward_context(forward, runtime)
    _select_execution_path(forward, runtime)
    if not forward.block_reasons:
        _run_compiled_blocks(forward, runtime)
    else:
        _run_eager_steps(forward, runtime)
    return _assemble_forward_result(forward, runtime)


def _validate_forward_inputs(forward: _ForwardContext, runtime: _ForwardState) -> None:
    forward.probe = getattr(forward.executor, "_stream_probe", None)
    # This state machine coordinates causal updates, continuation and output ownership.
    forward.trace = current_trace()
    if forward.trace is not None and not forward.trace.active:
        forward.trace = None
    if not isinstance(forward.options.diagnostics, bool):
        raise TypeError("diagnostics must be boolean")
    for name, value in (
        ("retain_parameters", forward.options.retain_parameters),
        ("retain_final_state", forward.options.retain_final_state),
        ("retain_runtime_state", forward.options.retain_runtime_state),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
    if not forward.inputs:
        raise ValueError("graph execution requires at least one input tensor")
    first = next(iter(forward.inputs.values()))
    forward.steps, forward.batch = first.shape[:2]
    forward.device = first.device
    forward.parameter_dtype = (
        next(iter(forward.executor.weights.values())).dtype
        if forward.executor.weights
        else first.dtype
    )
    if (
        forward.executor.weights or forward.executor.plan.populations
    ) and forward.parameter_dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        raise ValueError(
            f"graph execution requires a floating dtype, got {forward.parameter_dtype}"
        )
    if forward.device.type == "mps" and forward.parameter_dtype == torch.float64:
        raise ValueError("MPS graph execution does not support torch.float64")
    for name, parameter in forward.executor.parameter_map().items():
        if (
            parameter.device != forward.device
            or parameter.dtype != forward.parameter_dtype
        ):
            raise ValueError(
                f"parameter {name} must use execution device {forward.device} and dtype {forward.parameter_dtype}"
            )
    input_specs = {
        row["id"]: row for row in forward.executor.plan.graph.get("inputs", [])
    }
    for name, value in forward.inputs.items():
        if value.shape[:2] != (forward.steps, forward.batch):
            raise ValueError(
                f"input {name} leading shape expected {(forward.steps, forward.batch)}, got {tuple(value.shape[:2])}"
            )
        if value.device != forward.device:
            raise ValueError(
                f"input {name} device expected {forward.device}, got {value.device}"
            )
        is_mask = input_specs.get(name, {}).get("signal_type") == "mask"
        if value.dtype != forward.parameter_dtype and not (
            is_mask and value.dtype == torch.bool
        ):
            raise ValueError(
                f"input {name} dtype expected {forward.parameter_dtype}, got {value.dtype}"
            )


def _restore_runtime_state(forward: _ForwardContext, runtime: _ForwardState) -> None:  # noqa: C901
    forward.populations = {p["id"]: p for p in forward.executor.plan.populations}
    # Zero-delay edges use current-step spikes, but continuation still needs
    # the last spike tensor for every population.
    population_history_lengths = {
        name: max(
            (
                p.delay_steps
                for p in forward.executor.plan.projections
                if p.root_source.startswith(name + ".") and p.delay_steps > 0
            ),
            default=1,
        )
        for name in forward.populations
    }
    input_history_lengths = {
        row["id"]: max(
            (
                p.delay_steps
                for p in forward.executor.plan.projections
                if p.root_source.partition(".")[0] == row["id"]
            ),
            default=0,
        )
        for row in forward.executor.plan.graph.get("inputs", [])
    }
    forward.prepared = getattr(forward.executor, "_stream_metadata", None)
    if forward.prepared is None:
        forward.expected_compatibility = runtime_state_compatibility(
            forward.executor.plan,
            projection_mode=forward.executor.projection_mode,
            threshold_inclusive=forward.executor.surrogate_threshold_inclusive
            if forward.executor.surrogate is None
            else True,
        )
        forward.expected_signature = runtime_state_signature(
            forward.executor.plan,
            projection_mode=forward.executor.projection_mode,
            threshold_inclusive=forward.executor.surrogate_threshold_inclusive
            if forward.executor.surrogate is None
            else True,
        )
    else:
        forward.expected_compatibility = forward.prepared["compatibility"]
        forward.expected_signature = forward.prepared["signature"]
    if forward.probe is not None:
        forward.probe.mark("State restore / init")
    if forward.options.runtime_state is None:
        runtime.voltage = {
            name: (
                torch.zeros(
                    (forward.batch, p["size"]),
                    device=forward.device,
                    dtype=forward.parameter_dtype,
                )
                if p["neuron"]["kind"] == "leaky_integrator"
                else torch.full(
                    (forward.batch, p["size"]),
                    float(p["neuron"].get("initial_voltage_mv", M.E_L))
                    if p["neuron"]["kind"] == "cuba_lif"
                    or p["neuron"]["kind"] in N.STATEFUL_KINDS
                    else M.E_L,
                    device=forward.device,
                    dtype=forward.parameter_dtype,
                )
            )
            for name, p in forward.populations.items()
        }
        runtime.refractory = {
            name: torch.zeros(
                (forward.batch, p["size"]), dtype=torch.long, device=forward.device
            )
            for name, p in forward.populations.items()
        }
        spikes = {
            name: torch.zeros(
                (forward.batch, p["size"]),
                device=forward.device,
                dtype=forward.parameter_dtype,
            )
            for name, p in forward.populations.items()
        }
        runtime.conductance = {
            (p.id, p.polarity): torch.zeros(
                (
                    forward.batch,
                    forward.populations[p.target.partition(".")[0]]["size"],
                ),
                device=forward.device,
                dtype=forward.parameter_dtype,
            )
            for p in forward.executor.plan.projections
        }
        runtime.histories = {
            name: DelayBuffer(population_history_lengths[name], value)
            for name, value in spikes.items()
        }
        runtime.input_histories = {
            name: torch.zeros(
                (length, *forward.inputs[name].shape[1:]),
                device=forward.device,
                dtype=forward.inputs[name].dtype,
            )
            for name, length in input_history_lengths.items()
            if length > 0
        }
        runtime.completed_steps = 0
    else:
        if forward.options.runtime_state.signature != forward.expected_signature:
            detail = _compatibility_mismatch(
                forward.expected_compatibility,
                forward.options.runtime_state.compatibility,
            )
            raise ValueError(
                "runtime state is incompatible with graph plan: "
                + (
                    detail
                    or f"signature expected {forward.expected_signature}, got {forward.options.runtime_state.signature}"
                )
            )

        def restore_group(
            label: str,
            values: Mapping[str, torch.Tensor],
            shapes: Mapping[str, tuple[int, ...]],
        ) -> dict[str, torch.Tensor]:
            if set(values) != set(shapes):
                raise ValueError(
                    f"runtime state {label} keys expected {sorted(shapes)}, got {sorted(values)}"
                )
            restored = {}
            for name, expected_shape in shapes.items():
                value = values[name]
                if tuple(value.shape) != expected_shape:
                    raise ValueError(
                        f"runtime state {label}.{name} shape expected {expected_shape}, got {tuple(value.shape)}"
                    )
                restored[name] = value.detach().to(forward.device).clone()
            return restored

        pop_shapes = {
            name: (forward.batch, int(row["size"]))
            for name, row in forward.populations.items()
        }
        runtime.voltage = restore_group(
            "voltages", forward.options.runtime_state.voltages, pop_shapes
        )
        for name, value in runtime.voltage.items():
            if value.dtype != forward.parameter_dtype:
                raise ValueError(
                    f"runtime state voltages.{name} dtype expected {forward.parameter_dtype}, got {value.dtype}"
                )
        runtime.refractory = restore_group(
            "refractory", forward.options.runtime_state.refractory, pop_shapes
        )
        for name, value in runtime.refractory.items():
            if value.dtype != torch.long:
                raise ValueError(
                    f"runtime state refractory.{name} dtype expected torch.int64, got {value.dtype}"
                )
        by_projection = {
            row["id"]: row for row in forward.executor.plan.graph["projections"]
        }
        conductance_by_id = {}
        for group_name, values, unit in (
            ("conductances", forward.options.runtime_state.conductances, "uS"),
            ("currents", forward.options.runtime_state.currents, "nA"),
        ):
            restored = restore_group(
                group_name,
                values,
                {
                    p.id: (
                        forward.batch,
                        int(forward.populations[p.target.partition(".")[0]]["size"]),
                    )
                    for p in forward.executor.plan.projections
                    if E.synapse_unit(by_projection[p.id]["synapse"]) == unit
                },
            )
            conductance_by_id.update(restored)
        runtime.conductance = {
            (p.id, p.polarity): conductance_by_id[p.id]
            for p in forward.executor.plan.projections
        }
        for name, value in conductance_by_id.items():
            if value.dtype != forward.parameter_dtype:
                raise ValueError(
                    f"runtime state conductances.{name} dtype expected {forward.parameter_dtype}, got {value.dtype}"
                )
        population_history_values = restore_group(
            "population_histories",
            forward.options.runtime_state.population_histories,
            {
                name: (
                    population_history_lengths[name],
                    forward.batch,
                    int(row["size"]),
                )
                for name, row in forward.populations.items()
            },
        )
        runtime.histories = {
            name: DelayBuffer.restore(value)
            for name, value in population_history_values.items()
        }
        for name, value in population_history_values.items():
            if value.dtype != forward.parameter_dtype:
                raise ValueError(
                    f"runtime state population_histories.{name} dtype expected {forward.parameter_dtype}, got {value.dtype}"
                )
        spikes = {
            name: runtime.histories[name]._values[-1].clone()
            for name in forward.populations
        }
        runtime.input_histories = restore_group(
            "input_histories",
            forward.options.runtime_state.input_histories,
            {
                name: (length, *forward.inputs[name].shape[1:])
                for name, length in input_history_lengths.items()
                if length > 0
            },
        )
        for name, value in runtime.input_histories.items():
            if value.dtype != forward.inputs[name].dtype:
                raise ValueError(
                    f"runtime state input_histories.{name} dtype expected {forward.inputs[name].dtype}, got {value.dtype}"
                )
        runtime.completed_steps = int(forward.options.runtime_state.completed_steps)


def _prepare_forward_context(forward: _ForwardContext, runtime: _ForwardState) -> None:  # noqa: C901
    if forward.probe is not None:
        forward.probe.mark("Execution setup / metadata")
    forward.segments = SegmentAccumulator(
        segment_identity(
            forward.options.resets,
            forward.options.decisions,
            graph=forward.executor.plan.graph,
            batch_size=forward.batch,
        ),
        forward.populations,
        forward.batch,
        forward.device,
        forward.options.runtime_state.segment_state
        if forward.options.runtime_state is not None
        else {},
        forward.options.runtime_state.segment_counts
        if forward.options.runtime_state is not None
        else {},
        runtime.completed_steps,
    )
    forward.prepared_interventions, forward.intervention_request = (
        prepare_interventions(
            forward.options.interventions,
            graph=forward.executor.plan.graph,
            seed=forward.executor.seed,
            start_step=runtime.completed_steps,
            steps_count=forward.steps,
            batch_size=forward.batch,
        )
    )
    forward.resolved_interventions = [
        item.metadata for item in forward.prepared_interventions
    ]
    for item in forward.prepared_interventions:
        item.stage(
            device=forward.device,
            dtype=forward.parameter_dtype,
            start_step=runtime.completed_steps,
            steps_count=forward.steps,
        )

    runtime.neuron_states = {}
    runtime.synapse_states = {}
    runtime.custom_state = {}
    forward.projection_rows = {
        row["id"]: row for row in forward.executor.plan.graph["projections"]
    }
    forward.projection_ports = {
        name: E.projection_port(row["synapse"])
        for name, row in forward.projection_rows.items()
    }
    saved_custom = (
        forward.options.runtime_state.custom_state
        if forward.options.runtime_state is not None
        else {}
    )
    for category, rows in (
        ("neuron", forward.executor.plan.populations),
        ("synapse", forward.executor.plan.graph["projections"]),
    ):
        for row in rows:
            spec = row[category]
            adaptive = category == "neuron" and spec["kind"] in N.STATEFUL_KINDS
            if not adaptive and spec["kind"] != f"custom_{category}":
                continue
            owner = row["id"]
            target = owner if category == "neuron" else row["target"].partition(".")[0]
            shape = (forward.batch, forward.populations[target]["size"])
            if adaptive:
                label = spec["kind"]
                state_key = (
                    "slow_voltage" if spec["kind"] in N.MQIF_KINDS else "adaptation"
                )
                initial_key = (
                    "initial_slow_voltage_mv"
                    if spec["kind"] in N.MQIF_KINDS
                    else "initial_adaptation_mv"
                    if spec["kind"].endswith("alif")
                    else "initial_adaptation_na"
                )
                state = {
                    "voltage": runtime.voltage[owner],
                    "refractory": runtime.refractory[owner],
                    state_key: torch.full(
                        shape,
                        float(spec[initial_key]),
                        device=forward.device,
                        dtype=forward.parameter_dtype,
                    ),
                }
            else:
                definition = (
                    forward.executor._neuron_definitions
                    if category == "neuron"
                    else forward.executor._synapse_definitions
                )[owner]
                label = definition.name
                context = E.StateContext(
                    shape,
                    forward.device,
                    forward.parameter_dtype,
                    forward.executor.plan.dt_ms,
                    spec.get("config", {}),
                )
                if definition.initialize is None:
                    state = {
                        "value": torch.zeros(
                            shape, device=forward.device, dtype=forward.parameter_dtype
                        )
                    }
                else:
                    state = X.checked_state(
                        definition.initialize(context),
                        None,
                        label,
                        required=("voltage",) if category == "neuron" else ("value",),
                    )
            if category == "neuron":
                state.setdefault(
                    "refractory",
                    torch.zeros(shape, device=forward.device, dtype=torch.long),
                )
            for key in (
                ("voltage", "refractory") if category == "neuron" else ("value",)
            ):
                X.checked_tensor(
                    state[key],
                    shape=shape,
                    device=forward.device,
                    dtype=torch.long
                    if key == "refractory"
                    else forward.parameter_dtype,
                    name=f"{label}.{key}",
                )
            for key, value in state.items():
                if value.device != torch.device(forward.device):
                    raise ValueError(
                        f"{label}.{key}: state must use execution device {forward.device}"
                    )
            for port in forward.executor._state_ports[category][owner]:
                if port not in state:
                    raise ValueError(f"{label}: declared state port {port} missing")
                X.checked_tensor(
                    state[port],
                    shape=shape,
                    device=forward.device,
                    dtype=forward.parameter_dtype,
                    name=f"{label}.{port}",
                )
            for key, value in state.items():
                state_key = f"{category}/{owner}/{key}"
                runtime.custom_state[state_key] = value
                if forward.options.runtime_state is not None:
                    if state_key not in saved_custom:
                        raise ValueError(f"runtime state missing {state_key}")
                    restored = saved_custom[state_key].to(forward.device)
                    X.checked_tensor(
                        restored,
                        shape=value.shape,
                        device=forward.device,
                        dtype=value.dtype,
                        name=state_key,
                    )
                    if category == "neuron" and key in {"voltage", "refractory"}:
                        standard = (
                            runtime.voltage[owner]
                            if key == "voltage"
                            else runtime.refractory[owner]
                        )
                        if not torch.equal(restored, standard):
                            raise ValueError(
                                f"runtime state {state_key} disagrees with standard neuron state"
                            )
                    elif category == "synapse" and key == "value":
                        standard = runtime.conductance[(owner, row["polarity"])]
                        if not torch.equal(restored, standard):
                            raise ValueError(
                                f"runtime state {state_key} disagrees with synapse output state"
                            )
                    state[key] = restored.detach().clone()
            if category == "neuron":
                runtime.neuron_states[owner] = state
                runtime.voltage[owner] = state["voltage"]
                runtime.refractory[owner] = state["refractory"]
            else:
                runtime.synapse_states[owner] = state
    forward.continuous_delays = {}
    for projection in forward.executor.plan.projections:
        if projection.continuous:
            forward.continuous_delays[projection.source] = max(
                forward.continuous_delays.get(projection.source, 0),
                projection.delay_steps,
            )

    runtime.continuous_histories = {}
    for signal, length in forward.continuous_delays.items():
        if not length:
            continue
        state_key = f"continuous_history/{signal}"
        initial = _continuous_value(runtime, signal)
        if forward.options.runtime_state is None:
            values = initial.unsqueeze(0).expand(length, *initial.shape).clone()
        else:
            if state_key not in saved_custom:
                raise ValueError(f"runtime state missing {state_key}")
            values = saved_custom[state_key].to(forward.device)
            X.checked_tensor(
                values,
                shape=(length, *initial.shape),
                device=forward.device,
                dtype=forward.parameter_dtype,
                name=state_key,
            )
        runtime.continuous_histories[signal] = list(values.unbind(0))
        runtime.custom_state[state_key] = values
    if set(saved_custom) - set(runtime.custom_state):
        raise ValueError("runtime state has unexpected custom state tensors")
    runtime.state_traces = {}
    if not isinstance(forward.options.retain_outputs, bool):
        raise TypeError("retain_outputs must be boolean")
    if (
        type(forward.options.batch_offset) is not int
        or forward.options.batch_offset < 0
    ):
        raise ValueError("batch_offset must be a non-negative integer")
    forward.signal_axes = (
        forward.prepared["axes"]
        if forward.prepared is not None
        else _signal_axes(forward.executor.plan.graph)
    )
    forward.recorder = Recorder(
        forward.options.recording,
        axes=forward.signal_axes,
        start=runtime.completed_steps,
        steps=forward.steps,
        dt_ms=forward.executor.plan.dt_ms,
        batch_offset=forward.options.batch_offset,
        graph=forward.executor.plan.graph,
        dtype=forward.parameter_dtype,
        device=forward.device,
    )
    forward.inference_only = not (
        torch.is_grad_enabled()
        and any(
            value.requires_grad
            for value in (*forward.inputs.values(), *forward.executor.weights.values())
        )
    )
    forward.streamed_outputs = {
        row["id"]: row["signal"]
        for row in forward.executor.plan.outputs
        if not forward.options.retain_outputs
        and forward.signal_axes[row["signal"]][0] == "time"
    }
    forward.online = OnlineReductions(
        forward.executor.plan.graph,
        forward.signal_axes,
        forward.executor.parameter_map(),
        enabled=forward.options.automatic_reductions and forward.inference_only,
        window=forward.options.measurement,
        start=runtime.completed_steps,
        steps=forward.steps,
        saved=forward.options.runtime_state.reduction_state
        if forward.options.runtime_state is not None
        else {},
        tensors=forward.options.runtime_state.reduction_tensors
        if forward.options.runtime_state is not None
        else {},
        recording_signals=(*forward.recorder.rows, *forward.streamed_outputs.values()),
        dtype=forward.parameter_dtype,
        device=forward.device,
    )
    forward.external_signals = set(forward.options.required_signals)
    forward.external_signals.update(
        row["signal"]
        for row in forward.executor.plan.outputs
        if row["id"] not in forward.streamed_outputs
    )
    if forward.options.diagnostics:
        forward.external_signals.update(
            row["signal"] for row in forward.executor.plan.observables
        )
    forward.history_operations = forward.online.history_operations(
        forward.external_signals
    )
    forward.needed_signals = set(forward.external_signals)
    for operation in forward.history_operations:
        forward.needed_signals.update(operation["sources"])
    forward.capture_steps = bool(
        forward.online.reductions
        or forward.online.points
        or forward.online.duration_masks
        or forward.recorder.rows
        or forward.streamed_outputs
    )
    runtime.observation_counts = {
        name: torch.zeros_like(runtime.voltage[name], dtype=torch.int64)
        for name in forward.options.observation_populations
    }
    runtime.integrator_sum: dict[str, torch.Tensor] = {}
    runtime.spike_traces: dict[str, list[torch.Tensor]] = {
        name: []
        for name in forward.populations
        if f"{name}.spikes" in forward.needed_signals
    }
    runtime.voltage_traces: dict[str, list[torch.Tensor]] = {
        name: []
        for name in forward.populations
        if f"{name}.voltage" in forward.needed_signals
    }
    runtime.pre_reset_voltage_traces: dict[str, list[torch.Tensor]] = {
        name: []
        for name, pop in forward.populations.items()
        if pop["neuron"]["kind"] == "leaky_integrator"
        and f"{name}.pre_reset_voltage" in forward.needed_signals
    }
    runtime.conductance_traces: dict[str, list[torch.Tensor]] = {
        p.id: []
        for p in forward.executor.plan.projections
        if f"{p.id}.{forward.projection_ports[p.id]}" in forward.needed_signals
    }
    forward.effective_weights = (
        forward.prepared["effective_weights"]
        if forward.prepared is not None
        else {
            name: forward.executor._mask_autapses(name, value)
            for name, value in forward.executor.parameter_map().items()
        }
    )
    forward.projection_gains = {
        projection.id: sample_gain(
            forward.projection_rows[projection.id]["modulation"],
            start_step=runtime.completed_steps,
            steps=forward.steps,
            dt_ms=forward.executor.plan.dt_ms,
        )
        for projection in forward.executor.plan.projections
        if projection.enabled and "modulation" in forward.projection_rows[projection.id]
    }
    forward.spatial_operations = {
        f"{row['id']}.value": row
        for row in forward.executor.plan.graph.get("operations", [])
        if row["kind"] == "pool2d"
    }


def _select_execution_path(forward: _ForwardContext, runtime: _ForwardState) -> None:
    forward.block_reasons = []
    forward.compiler_fallback = (
        "PyTorch 2.10 reduced-precision CPU uses eager tensor graphs"
        if forward.executor.compile_mode is not None
        and precision_fallback(forward.device, forward.parameter_dtype)
        else None
    )
    block_trainable = {
        projection.parameter
        for projection in forward.executor.plan.projections
        if projection.enabled
        and forward.effective_weights[projection.parameter].requires_grad
    }
    partial_dependencies = not forward.inference_only and any(
        forward.executor._signal_parameters.get(signal, set()) & block_trainable
        != block_trainable
        for signal in forward.external_signals
        if signal in forward.executor._signal_parameters
    )
    for condition, reason in (
        (forward.executor.compile_block_steps == 1, "block execution not requested"),
        (forward.executor.compile_mode is None, "compilation disabled"),
        (forward.compiler_fallback is not None, forward.compiler_fallback),
        (
            forward.device.type == "mps" and forward.executor.compile_block_steps > 4,
            "MPS block size above validated limit (4)",
        ),
        (
            bool(
                forward.executor._neuron_definitions
                or forward.executor._synapse_definitions
                or forward.executor.surrogate
            ),
            "registered neuron, synapse or surrogate",
        ),
        (forward.capture_steps, "online reductions or recording"),
        (bool(forward.prepared_interventions), "spike interventions"),
        (bool(forward.segments.identity), "voltage resets or decision segments"),
        (forward.trace is not None, "execution tracing"),
        (
            len(forward.inputs) > 1
            and any(value.requires_grad for value in forward.inputs.values()),
            "multiple inputs with input gradients",
        ),
        (
            partial_dependencies,
            "partial parameter dependencies in retained signals",
        ),
        (
            not forward.inference_only
            and forward.steps
            <= sum(
                p.delay_steps
                for p in forward.executor.plan.projections
                if p.enabled and p.root_source.partition(".")[0] in forward.populations
            ),
            "short delayed autograd trajectory",
        ),
    ):
        if condition:
            forward.block_reasons.append(reason)


def _run_compiled_blocks(forward: _ForwardContext, runtime: _ForwardState) -> None:  # noqa: C901
    block_signals = tuple(
        sorted(
            {
                *(
                    f"{name}.spikes"
                    for name in (*runtime.spike_traces, *runtime.observation_counts)
                ),
                *(f"{name}.voltage" for name in runtime.voltage_traces),
                *(
                    f"{name}.pre_reset_voltage"
                    for name in runtime.pre_reset_voltage_traces
                ),
                *(
                    f"{name}.{forward.projection_ports[name]}"
                    for name in runtime.conductance_traces
                ),
                *(
                    f"{name}.{port}"
                    for name in runtime.neuron_states
                    for port in forward.executor._state_ports["neuron"][name]
                    if f"{name}.{port}" in forward.needed_signals
                ),
            }
        )
    )
    if block_signals not in forward.executor._block_kernels:
        forward.executor._block_kernels[block_signals] = network_block(
            forward.executor.plan,
            spike_function=forward.executor._spike_function,
            projection_mode=forward.executor.projection_mode,
            signals=block_signals,
            compile_mode=forward.executor.compile_mode,
        )
    kernel = forward.executor._block_kernels[block_signals]
    history_values = {
        name: history._values for name, history in runtime.histories.items()
    }
    block_input_histories = runtime.input_histories
    # Prepared sessions can reuse the exact functional block and its
    # validated initial state without rebuilding a forward invocation.
    # Operation/reduction graphs retain the general executor path.
    if (
        forward.prepared is not None
        and forward.options.retain_outputs
        and not forward.executor.plan.graph.get("operations")
    ):
        forward.executor._stream_block_context = {
            "kernel": kernel,
            "state": (
                runtime.voltage,
                runtime.refractory,
                runtime.neuron_states,
                runtime.conductance,
                history_values,
                block_input_histories,
            ),
            "completed_steps": runtime.completed_steps,
            "gain_specs": {
                p.id: forward.projection_rows[p.id]["modulation"]
                for p in forward.executor.plan.projections
                if p.enabled and "modulation" in forward.projection_rows[p.id]
            },
            "ports": forward.projection_ports,
        }
    gain_tensors = {
        name: torch.as_tensor(
            values, device=forward.device, dtype=forward.parameter_dtype
        )
        for name, values in forward.projection_gains.items()
    }
    for begin in range(0, forward.steps, forward.executor.compile_block_steps):
        end = min(begin + forward.executor.compile_block_steps, forward.steps)
        if forward.probe is not None:
            forward.probe.mark("Compiled numerical block")
        (
            runtime.voltage,
            runtime.refractory,
            runtime.neuron_states,
            runtime.conductance,
            history_values,
            block_input_histories,
            spikes,
            runtime.integrator_sum,
            samples,
        ) = kernel(
            {name: value[begin:end] for name, value in forward.inputs.items()},
            runtime.voltage,
            runtime.refractory,
            runtime.neuron_states,
            runtime.conductance,
            history_values,
            block_input_histories,
            forward.effective_weights,
            {name: value[begin:end] for name, value in gain_tensors.items()},
            runtime.integrator_sum,
        )
        if forward.probe is not None:
            forward.probe.mark("Output / history assembly")
        for name in runtime.spike_traces:
            runtime.spike_traces[name].extend(samples[f"{name}.spikes"].unbind())
        for name in runtime.voltage_traces:
            runtime.voltage_traces[name].extend(samples[f"{name}.voltage"].unbind())
        for name in runtime.pre_reset_voltage_traces:
            runtime.pre_reset_voltage_traces[name].extend(
                samples[f"{name}.pre_reset_voltage"].unbind()
            )
        for name in runtime.conductance_traces:
            runtime.conductance_traces[name].extend(
                samples[f"{name}.{forward.projection_ports[name]}"].unbind()
            )
        for name in runtime.neuron_states:
            for port in forward.executor._state_ports["neuron"][name]:
                signal = f"{name}.{port}"
                if signal in samples:
                    runtime.state_traces.setdefault(signal, []).extend(
                        samples[signal].unbind()
                    )
        for name in runtime.observation_counts:
            runtime.observation_counts[name] += (
                samples[f"{name}.spikes"].detach().to(torch.int64).sum(dim=0)
            )
    for name, values in history_values.items():
        runtime.histories[name]._values = values
    for name, state in runtime.neuron_states.items():
        for key, value in state.items():
            runtime.custom_state[f"neuron/{name}/{key}"] = value


def _run_eager_steps(forward: _ForwardContext, runtime: _ForwardState) -> None:  # noqa: C901
    if forward.probe is not None:
        forward.probe.mark("Eager simulation / histories")
    for t in range(forward.steps):
        forward.segments.boundary(
            runtime.completed_steps + t, forward.executor.plan.dt_ms
        )
        forward.segments.reset(runtime.completed_steps + t, runtime.voltage)
        step_pre_reset = {}
        new_spikes: dict[str, torch.Tensor] = {}
        source_cache = {}
        # Freeze every continuous population source before any membrane
        # update. Reciprocal edges therefore share the same time origin.
        old_continuous = {
            signal: _continuous_value(runtime, signal).clone()
            for signal in forward.continuous_delays
        }

        def projection_source(
            signal,
            delay,
            *,
            source_cache=source_cache,
            old_continuous=old_continuous,
            new_spikes=new_spikes,
            t=t,
        ):
            key = (signal, delay)
            if key in source_cache:
                return source_cache[key]
            pending = []
            while signal in forward.spatial_operations:
                operation = forward.spatial_operations[signal]
                pending.append(operation)
                signal = operation["sources"][0]
            owner, _, port = signal.partition(".")
            if owner in forward.populations:
                if port != "spikes":
                    result = (
                        old_continuous[signal]
                        if delay == 0
                        else runtime.continuous_histories[signal][-delay]
                    )
                else:
                    result = (
                        new_spikes[owner]
                        if delay == 0
                        else runtime.histories[owner]._values[-delay]
                    )
            else:
                source_t = t - delay
                result = (
                    forward.inputs[owner][source_t]
                    if source_t >= 0
                    else runtime.input_histories[owner][source_t]
                )
            for operation in reversed(pending):
                result = pool_tensor(
                    result,
                    signal_spatial_shape(
                        forward.executor.plan.graph, operation["sources"][0]
                    ),
                    operation["config"],
                )
            source_cache[key] = result
            return result

        for pop in forward.executor.plan.populations:
            name = pop["id"]
            incoming = {
                "excitatory": torch.zeros_like(runtime.voltage[name]),
                "inhibitory": torch.zeros_like(runtime.voltage[name]),
            }
            for projection in forward.executor._incoming_projections[name]:
                key = (projection.id, projection.polarity)
                if not projection.enabled:
                    runtime.conductance[key].zero_()
                    continue
                source = projection_source(projection.source, projection.delay_steps)

                def project(*, projection=projection, source=source):
                    config = forward.projection_rows[projection.id].get("connectivity")
                    if config is not None:
                        return convolution_drive(
                            source,
                            forward.effective_weights[projection.parameter],
                            signal_spatial_shape(
                                forward.executor.plan.graph, projection.source
                            ),
                            config,
                        )
                    return source @ forward.effective_weights[projection.parameter]

                if forward.trace is None:
                    drive = project()
                else:
                    with forward.trace.detail(f"projection/{projection.id}"):
                        drive = project()
                if projection.id in forward.projection_gains:
                    drive = drive * forward.projection_gains[projection.id][t]
                spec = forward.projection_rows[projection.id]["synapse"]
                if spec["kind"] == "custom_synapse":
                    definition = forward.executor._synapse_definitions[projection.id]
                    previous = runtime.synapse_states[projection.id]
                    state = X.checked_state(
                        forward.executor._synapse_steps[projection.id](previous, drive),
                        previous,
                        definition.name,
                        check_values=forward.executor._check_extension_values,
                    )
                    runtime.synapse_states[projection.id] = state
                    runtime.conductance[key] = state["value"]
                else:
                    runtime.conductance[key] = forward.executor._synapse_steps[
                        projection.id
                    ](runtime.conductance[key], drive)
                incoming[projection.polarity] += runtime.conductance[key]
            neuron = pop["neuron"]
            if neuron["kind"] == "custom_neuron":
                definition = forward.executor._neuron_definitions[name]
                previous = runtime.neuron_states[name]
                response = forward.executor._neuron_steps[name](
                    previous,
                    incoming["excitatory"],
                    incoming["inhibitory"],
                )
                if not isinstance(response, tuple) or len(response) != 2:
                    raise TypeError(f"{definition.name} must return (state, spikes)")
                state, spike_values = response
                state = X.checked_state(
                    state,
                    previous,
                    definition.name,
                    check_values=forward.executor._check_extension_values,
                )
                X.checked_tensor(
                    spike_values,
                    shape=runtime.voltage[name].shape,
                    device=forward.device,
                    dtype=forward.parameter_dtype,
                    name=f"{definition.name}.spikes",
                    check_values=forward.executor._check_extension_values,
                )
                if forward.executor._check_extension_values and torch.any(
                    (spike_values != 0) & (spike_values != 1)
                ):
                    raise ValueError(f"{definition.name} spikes must be binary")
                runtime.neuron_states[name] = state
                runtime.voltage[name], runtime.refractory[name] = (
                    state["voltage"],
                    state["refractory"],
                )
                new_spikes[name] = (
                    spike_values if pop["spiking"] else torch.zeros_like(spike_values)
                )
                new_spikes[name] = _intervene(
                    forward, name, new_spikes[name], runtime.completed_steps + t
                )
                continue
            if neuron["kind"] in N.STATEFUL_KINDS:
                state, spike_values = forward.executor._neuron_steps[name](
                    runtime.neuron_states[name],
                    incoming["excitatory"],
                    incoming["inhibitory"],
                )
                runtime.neuron_states[name] = state
                runtime.voltage[name], runtime.refractory[name] = (
                    state["voltage"],
                    state["refractory"],
                )
                new_spikes[name] = _intervene(
                    forward,
                    name,
                    spike_values if pop["spiking"] else torch.zeros_like(spike_values),
                    runtime.completed_steps + t,
                )
                continue
            if neuron["kind"] == "cuba_lif":
                runtime.voltage[name], new_spikes[name], runtime.refractory[name] = (
                    forward.executor._neuron_steps[name](
                        runtime.voltage[name],
                        runtime.refractory[name],
                        incoming["excitatory"],
                        incoming["inhibitory"],
                    )
                )
                new_spikes[name] = _intervene(
                    forward, name, new_spikes[name], runtime.completed_steps + t
                )
                continue
            if neuron["kind"] == "leaky_integrator":
                pre_reset, post_reset, new_spikes[name] = (
                    forward.executor._neuron_steps[name](
                        runtime.voltage[name], incoming["excitatory"]
                    )
                )
                if name in runtime.pre_reset_voltage_traces:
                    runtime.pre_reset_voltage_traces[name].append(pre_reset)
                step_pre_reset[f"{name}.pre_reset_voltage"] = pre_reset
                runtime.integrator_sum[name] = (
                    runtime.integrator_sum.get(name, torch.zeros_like(pre_reset))
                    + pre_reset
                )
                runtime.voltage[name] = post_reset
                new_spikes[name] = _intervene(
                    forward, name, new_spikes[name], runtime.completed_steps + t
                )
                continue
            runtime.voltage[name], new_spikes[name], runtime.refractory[name] = (
                forward.executor._neuron_steps[name](
                    runtime.voltage[name],
                    runtime.refractory[name],
                    incoming["excitatory"],
                    incoming["inhibitory"],
                )
            )
            new_spikes[name] = _intervene(
                forward, name, new_spikes[name], runtime.completed_steps + t
            )
        for category, states in (
            ("neuron", runtime.neuron_states),
            ("synapse", runtime.synapse_states),
        ):
            for owner, state in states.items():
                for key, value in state.items():
                    runtime.custom_state[f"{category}/{owner}/{key}"] = value
                for port in forward.executor._state_ports[category][owner]:
                    signal = f"{owner}.{port}"
                    if signal in forward.needed_signals:
                        runtime.state_traces.setdefault(signal, []).append(state[port])
        if forward.capture_steps:
            step_signals = {
                f"{name}.value": value[t] for name, value in forward.inputs.items()
            }
            step_signals.update(
                {f"{name}.spikes": value for name, value in new_spikes.items()}
            )
            step_signals.update(
                {f"{name}.voltage": value for name, value in runtime.voltage.items()}
            )
            step_signals.update(step_pre_reset)
            step_signals.update(
                {
                    f"{p.id}.{forward.projection_ports[p.id]}": runtime.conductance[
                        (p.id, p.polarity)
                    ]
                    for p in forward.executor.plan.projections
                }
            )
            for category, states in (
                ("neuron", runtime.neuron_states),
                ("synapse", runtime.synapse_states),
            ):
                for owner, state in states.items():
                    for port in forward.executor._state_ports[category][owner]:
                        step_signals[f"{owner}.{port}"] = state[port]
            step_signals = forward.online.samples(step_signals)
            forward.online.observe(step_signals, runtime.completed_steps + t)
            forward.recorder.observe(step_signals, runtime.completed_steps + t)
            for output_id, signal in forward.streamed_outputs.items():
                forward.recorder.output_step(
                    output_id, step_signals[signal], runtime.completed_steps + t
                )
        if forward.trace is not None:
            forward.trace.observe_spikes(new_spikes)
        spikes = new_spikes
        forward.segments.observe(spikes)
        for name in runtime.observation_counts:
            runtime.observation_counts[name] += spikes[name].detach().to(torch.int64)
        for name in runtime.spike_traces:
            runtime.spike_traces[name].append(spikes[name])
        for name in runtime.voltage_traces:
            runtime.voltage_traces[name].append(runtime.voltage[name])
        for name in forward.populations:
            runtime.histories[name].push(spikes[name])
        for signal, values in runtime.continuous_histories.items():
            values.append(old_continuous[signal])
            del values[0]
            runtime.custom_state[f"continuous_history/{signal}"] = torch.stack(values)
        for projection_id in runtime.conductance_traces:
            projection = next(
                p for p in forward.executor.plan.projections if p.id == projection_id
            )
            runtime.conductance_traces[projection_id].append(
                runtime.conductance[(projection.id, projection.polarity)]
            )


def _assemble_forward_result(  # noqa: C901
    forward: _ForwardContext, runtime: _ForwardState
) -> tuple[ExecutionResult, dict[str, torch.Tensor]]:
    if forward.probe is not None:
        forward.probe.mark("Output / history assembly")
    outputs: dict[str, torch.Tensor] = {}
    signal_values: dict[str, torch.Tensor] = {
        f"{name}.value": value for name, value in forward.inputs.items()
    }
    for name, values in runtime.spike_traces.items():
        signal_values[f"{name}.spikes"] = torch.stack(values)
    for name, values in runtime.voltage_traces.items():
        signal_values[f"{name}.voltage"] = torch.stack(values)
    for name, values in runtime.pre_reset_voltage_traces.items():
        signal_values[f"{name}.pre_reset_voltage"] = torch.stack(values)
    for name, values in runtime.conductance_traces.items():
        signal_values[f"{name}.{forward.projection_ports[name]}"] = torch.stack(values)
    for signal, values in runtime.state_traces.items():
        signal_values[signal] = torch.stack(values)

    def time_mask(
        mask: torch.Tensor, *, target: torch.Tensor, op_id: str
    ) -> torch.Tensor:
        if mask.shape[:2] != target.shape[:2]:
            raise ValueError(
                f"{op_id}: valid-time mask leading shape expected {tuple(target.shape[:2])}, got {tuple(mask.shape[:2])}"
            )
        if mask.ndim != 2:
            raise ValueError(f"{op_id}: valid-time mask must have shape [time, batch]")
        mask_value = mask.to(device=target.device, dtype=target.dtype)
        return mask_value.reshape(
            mask_value.shape[0], mask_value.shape[1], *([1] * (target.ndim - 2))
        )

    def reduce_time(
        source: torch.Tensor,
        *,
        kind: str,
        mask: torch.Tensor | None,
        op_id: str,
        sequential: bool = False,
    ) -> torch.Tensor:
        if sequential:
            # Match legacy pre-reset accumulation order, including all-true masks.
            weights = (
                time_mask(mask, target=source, op_id=op_id)
                if mask is not None
                else None
            )
            samples = source if weights is None else source * weights
            numerator = torch.zeros_like(source[0])
            for sample in samples:
                numerator = numerator + sample
            if kind == "reduce_sum":
                return numerator
            counts = source.shape[0] if weights is None else weights.sum(dim=0)
            if weights is not None and torch.any(counts <= 0):
                raise ValueError(
                    f"{op_id}: valid-time mask contains an empty reduction window"
                )
            return numerator / counts
        if mask is None:
            return source.sum(dim=0) if kind == "reduce_sum" else source.mean(dim=0)
        weights = time_mask(mask, target=source, op_id=op_id)
        numerator = (source * weights).sum(dim=0)
        if kind == "reduce_sum":
            return numerator
        counts = weights.sum(dim=0)
        if torch.any(counts <= 0):
            raise ValueError(
                f"{op_id}: valid-time mask contains an empty reduction window"
            )
        return numerator / counts

    signal_values.update(forward.online.finish(runtime.completed_steps + forward.steps))
    remaining_ops = list(forward.history_operations)
    while remaining_ops:
        ready_index = next(
            (
                index
                for index, op in enumerate(remaining_ops)
                if all(source in signal_values for source in op["sources"])
            ),
            None,
        )
        if ready_index is None:
            unresolved = {
                op["id"]: [
                    source for source in op["sources"] if source not in signal_values
                ]
                for op in remaining_ops
            }
            raise ValueError(f"operation dependencies are unresolved: {unresolved}")
        op = remaining_ops.pop(ready_index)
        sources = [signal_values[source] for source in op["sources"]]
        kind = op["kind"]
        if kind == "linear":
            parameter = op["parameters"][0].replace(".", "__")
            signal_values[f"{op['id']}.value"] = (
                sources[0] @ forward.executor.weights[parameter]
            )
        elif kind == "pool2d":
            signal_values[f"{op['id']}.value"] = pool_tensor(
                sources[0],
                signal_spatial_shape(forward.executor.plan.graph, op["sources"][0]),
                op["config"],
            )
        elif kind in {"reduce_mean", "reduce_sum"}:
            mask_name = op.get("config", {}).get("mask")
            mask = signal_values.get(mask_name) if mask_name else None
            source_id = op["sources"][0]
            owner, _, port = source_id.partition(".")
            if (
                "voltage_sampling" not in forward.executor.plan.graph
                and kind == "reduce_mean"
                and mask is None
                and port == "voltage"
                and owner in runtime.integrator_sum
            ):
                signal_values[f"{op['id']}.value"] = (
                    runtime.integrator_sum[owner] / forward.steps
                )
            else:
                signal_values[f"{op['id']}.value"] = reduce_time(
                    sources[0],
                    kind=kind,
                    mask=mask,
                    op_id=op["id"],
                    sequential=port == "pre_reset_voltage",
                )
        elif kind == "select_final":
            signal_values[f"{op['id']}.value"] = sources[0][-1]
        elif kind == "duration_normalise":
            config = op.get("config", {})
            mask_name = config.get("mask")
            if mask_name:
                mask = signal_values[mask_name]
                if mask.ndim != 2:
                    raise ValueError(
                        f"{op['id']}: valid-time mask must have shape [time, batch]"
                    )
                duration_key = f"{op['id']}.value/duration"
                if duration_key in forward.online.tensors:
                    mask_seconds = forward.online.tensors[duration_key].to(
                        sources[0].dtype
                    ) * (forward.executor.plan.dt_ms / 1000.0)
                else:
                    mask_seconds = mask.to(
                        device=sources[0].device, dtype=sources[0].dtype
                    ).sum(dim=0) * (forward.executor.plan.dt_ms / 1000.0)
                mask_seconds = mask_seconds.reshape(
                    mask_seconds.shape[0], *([1] * (sources[0].ndim - 1))
                )
                if torch.any(mask_seconds <= 0) and (
                    forward.options.measurement is None
                    or runtime.completed_steps + forward.steps
                    >= forward.options.measurement.end_step
                ):
                    raise ValueError(
                        f"{op['id']}: valid-time mask contains zero valid duration"
                    )
                signal_values[f"{op['id']}.value"] = sources[0] / torch.where(
                    mask_seconds > 0, mask_seconds, torch.ones_like(mask_seconds)
                )
            else:
                duration_s = float(config["duration"])
                if duration_s <= 0:
                    raise ValueError(
                        f"{op['id']}: spike-rate duration must be positive seconds"
                    )
                signal_values[f"{op['id']}.value"] = sources[0] / duration_s
        elif kind == "divide":
            signal_values[f"{op['id']}.value"] = sources[0] / sources[1]
        elif kind == "custom_operation":
            definition, update = forward.executor._operation_steps[op["id"]]
            parameters = {
                name: forward.executor.parameter_map()[name]
                for name in op.get("parameters", [])
            }
            value = update(tuple(sources), parameters)
            shape = tuple(
                forward.steps if d == "time" else forward.batch if d == "batch" else d
                for d in op["shape"]
            )
            signal_values[f"{op['id']}.value"] = X.checked_tensor(
                value,
                shape=shape,
                device=forward.device,
                dtype=forward.parameter_dtype,
                name=definition.name,
            )
        elif kind == "cumulative_sum":
            signal_values[f"{op['id']}.value"] = sources[0].cumsum(dim=0)
        else:
            raise ValueError(f"{op['id']}: unsupported operation {kind}")
    for output in forward.executor.plan.outputs:
        if output["id"] in forward.streamed_outputs:
            continue
        value = signal_values[output["signal"]]
        forward.recorder.output(output["id"], value, forward.options.measurement)
        if forward.options.retain_outputs:
            outputs[output["id"]] = value
    packed = (
        {
            row["id"]: signal_values[row["signal"]].detach().clone()
            for row in forward.executor.plan.observables
        }
        if forward.options.diagnostics
        else {}
    )
    forward.segments.boundary(
        runtime.completed_steps + forward.steps, forward.executor.plan.dt_ms
    )
    forward.segments.finish_records()
    if forward.probe is not None:
        forward.probe.mark("Runtime-state snapshot")
    next_runtime_state = None
    if forward.options.retain_runtime_state:
        next_input_histories = {
            name: (
                forward.inputs[name][-history.shape[0] :]
                if forward.inputs[name].shape[0] >= history.shape[0]
                else torch.cat((history, forward.inputs[name]), dim=0)[
                    -history.shape[0] :
                ]
            )
            .detach()
            .clone()
            for name, history in runtime.input_histories.items()
        }
        next_runtime_state = GraphRuntimeState(
            signature=forward.expected_signature,
            compatibility=forward.expected_compatibility,
            completed_steps=runtime.completed_steps + forward.steps,
            voltages={
                name: value.detach().clone() for name, value in runtime.voltage.items()
            },
            refractory={
                name: value.detach().clone()
                for name, value in runtime.refractory.items()
            },
            conductances={
                p.id: runtime.conductance[(p.id, p.polarity)].detach().clone()
                for p in forward.executor.plan.projections
                if forward.projection_ports[p.id] == "conductance"
            },
            currents={
                p.id: runtime.conductance[(p.id, p.polarity)].detach().clone()
                for p in forward.executor.plan.projections
                if forward.projection_ports[p.id] == "current"
            },
            population_histories={
                name: history.export() for name, history in runtime.histories.items()
            },
            input_histories=next_input_histories,
            custom_state={
                key: value.detach().clone()
                for key, value in runtime.custom_state.items()
            },
            segment_state=forward.segments.state(),
            segment_counts=forward.segments.counts,
            reduction_state=forward.online.state(
                runtime.completed_steps + forward.steps
            ),
            reduction_tensors={
                name: value.detach().clone()
                for name, value in forward.online.tensors.items()
            }
            if forward.options.measurement is not None
            else {},
        )
    if forward.probe is not None:
        forward.probe.mark("Executor result / metadata")
    result = ExecutionResult(
        executor="graph",
        outputs=outputs,
        diagnostics=packed,
        parameters={k: v.detach().clone() for k, v in forward.effective_weights.items()}
        if forward.options.retain_parameters
        else {},
        final_state={
            f"{k}.voltage": v.detach().clone() for k, v in runtime.voltage.items()
        }
        if forward.options.retain_final_state
        else {},
        runtime_state=next_runtime_state,
        metrics={
            "recording_units": {
                name: forward.recorder.units.get(name) for name in forward.recorder.rows
            },
            "online_reductions": list(forward.online.reductions),
            "retained_signal_samples": {
                name: len(values)
                for name, values in (
                    *runtime.spike_traces.items(),
                    *runtime.voltage_traces.items(),
                    *runtime.pre_reset_voltage_traces.items(),
                    *runtime.conductance_traces.items(),
                    *runtime.state_traces.items(),
                )
            },
            "measurement_complete": forward.options.measurement is None
            or runtime.completed_steps + forward.steps
            >= forward.options.measurement.end_step,
            "inference_retention": retention_identity(
                graph=forward.executor.plan.graph,
                start_step=runtime.completed_steps,
                steps_count=forward.steps,
                batch_size=forward.batch,
                batch_offset=forward.options.batch_offset,
                diagnostics=forward.options.diagnostics,
                retain_outputs=forward.options.retain_outputs,
                retain_parameters=forward.options.retain_parameters,
                retain_final_state=forward.options.retain_final_state,
                retain_runtime_state=forward.options.retain_runtime_state,
                projection_mode=forward.executor.projection_mode,
                measurement=forward.options.measurement,
                recording=forward.options.recording,
            ),
            "segment_policy": forward.segments.identity,
            "segment_window": {
                "start_step": runtime.completed_steps,
                "steps_count": forward.steps,
            }
            if forward.segments.identity
            else None,
            "decisions": forward.segments.records,
            "resolved_interventions": forward.resolved_interventions,
            "refractory": dict(forward.executor.plan.refractory),
            "autapses": forward.executor.autapse_metadata,
            "extension_validation": forward.executor.extension_validation,
            "projection_mode": forward.executor.projection_mode,
            "compile_mode": forward.executor.compile_mode,
            "compiler_wrapper": compiler_wrapper_policy(
                forward.device, forward.parameter_dtype, forward.executor.compile_mode
            ),
            "compiler_precision_policy": "snnlab.eager-precision-casts/v1"
            if forward.executor.compile_mode is not None
            else None,
            "compiler_fallback": forward.compiler_fallback,
            "execution_path": {
                "schema": "snnlab.network-blocks/v1",
                "selected": "kernel_loop"
                if forward.block_reasons
                else "compiled_blocks",
                "requested_block_steps": forward.executor.requested_block_steps,
                "block_steps": 1
                if forward.block_reasons
                else forward.executor.compile_block_steps,
                "fallback_reasons": forward.block_reasons,
            },
            "surrogate": copy.deepcopy(forward.executor.surrogate_config),
            "result_snapshots": {
                "parameters": forward.options.retain_parameters,
                "final_state": forward.options.retain_final_state,
                "runtime_state": forward.options.retain_runtime_state,
            },
            "inference_interventions": forward.intervention_request
            if forward.options.interventions
            else None,
            **(
                {"population_observations": runtime.observation_counts}
                if runtime.observation_counts
                else {}
            ),
        },
        decisions=forward.segments.records,
        recorded_signals=forward.recorder.finish(forward.batch),
        _recording_info=forward.recorder.information(forward.batch),
        _output_axes={
            row["id"]: forward.signal_axes[row["signal"]]
            for row in forward.executor.plan.outputs
        },
        _diagnostic_axes={
            row["id"]: forward.signal_axes[row["signal"]]
            for row in forward.executor.plan.observables
            if forward.options.diagnostics
        },
        _timebase=(forward.executor.plan.dt_ms, runtime.completed_steps, forward.steps),
        _batch_size=forward.batch,
        model=forward.executor,
    )
    if result.metrics["execution_path"] not in forward.executor._execution_paths:
        forward.executor._execution_paths.append(
            copy.deepcopy(result.metrics["execution_path"])
        )
    return result, signal_values


def _intervene(
    forward: _ForwardContext, name: str, emitted: torch.Tensor, absolute_step: int
) -> torch.Tensor:
    for index, item in enumerate(forward.prepared_interventions):
        if item.metadata["population_id"] == name:
            emitted = item.apply(emitted, absolute_step, index)
    return emitted


def _continuous_value(runtime: _ForwardState, signal: str) -> torch.Tensor:
    owner, _, port = signal.partition(".")
    return (
        runtime.voltage[owner]
        if port == "voltage"
        else runtime.neuron_states[owner][port]
    )
