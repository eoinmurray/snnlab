"""Shared graph forward execution and causal delay buffers."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from snnlab import extensions as E
from snnlab._shared import neurons as N
from snnlab._shared.gains import sample as sample_gain
from snnlab._shared.spatial import convolution_drive, pool_tensor, signal_spatial_shape
from snnlab.sim.artifacts.inference import retention_identity
from snnlab.sim.diagnostics.tracing import current_trace
from snnlab.sim.graph import dynamics as M
from snnlab.sim.graph import extensions as X
from snnlab.sim.graph.blocks import network_block
from snnlab.sim.graph.kernels import (
    compiler_wrapper_policy,
    precision_fallback,
)
from snnlab.sim.graph.plan import _signal_axes
from snnlab.sim.requests.result import ExecutionResult
from snnlab.sim.runtime.interventions import Intervention, prepare_interventions
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


def run_forward_pass(  # noqa: C901
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
    probe = getattr(executor, "_stream_probe", None)
    # This state machine coordinates causal updates, continuation and output ownership.
    trace = current_trace()
    if trace is not None and not trace.active:
        trace = None
    if not isinstance(diagnostics, bool):
        raise TypeError("diagnostics must be boolean")
    for name, value in (
        ("retain_parameters", retain_parameters),
        ("retain_final_state", retain_final_state),
        ("retain_runtime_state", retain_runtime_state),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
    if not inputs:
        raise ValueError("graph execution requires at least one input tensor")
    first = next(iter(inputs.values()))
    steps, batch = first.shape[:2]
    device = first.device
    parameter_dtype = (
        next(iter(executor.weights.values())).dtype if executor.weights else first.dtype
    )
    if (executor.weights or executor.plan.populations) and parameter_dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        raise ValueError(
            f"graph execution requires a floating dtype, got {parameter_dtype}"
        )
    if device.type == "mps" and parameter_dtype == torch.float64:
        raise ValueError("MPS graph execution does not support torch.float64")
    for name, parameter in executor.parameter_map().items():
        if parameter.device != device or parameter.dtype != parameter_dtype:
            raise ValueError(
                f"parameter {name} must use execution device {device} and dtype {parameter_dtype}"
            )
    input_specs = {row["id"]: row for row in executor.plan.graph.get("inputs", [])}
    for name, value in inputs.items():
        if value.shape[:2] != (steps, batch):
            raise ValueError(
                f"input {name} leading shape expected {(steps, batch)}, got {tuple(value.shape[:2])}"
            )
        if value.device != device:
            raise ValueError(
                f"input {name} device expected {device}, got {value.device}"
            )
        is_mask = input_specs.get(name, {}).get("signal_type") == "mask"
        if value.dtype != parameter_dtype and not (
            is_mask and value.dtype == torch.bool
        ):
            raise ValueError(
                f"input {name} dtype expected {parameter_dtype}, got {value.dtype}"
            )
    populations = {p["id"]: p for p in executor.plan.populations}
    # Zero-delay edges use current-step spikes, but continuation still needs
    # the last spike tensor for every population.
    population_history_lengths = {
        name: max(
            (
                p.delay_steps
                for p in executor.plan.projections
                if p.root_source.startswith(name + ".") and p.delay_steps > 0
            ),
            default=1,
        )
        for name in populations
    }
    input_history_lengths = {
        row["id"]: max(
            (
                p.delay_steps
                for p in executor.plan.projections
                if p.root_source.partition(".")[0] == row["id"]
            ),
            default=0,
        )
        for row in executor.plan.graph.get("inputs", [])
    }
    prepared = getattr(executor, "_stream_metadata", None)
    if prepared is None:
        expected_compatibility = runtime_state_compatibility(
            executor.plan,
            projection_mode=executor.projection_mode,
            threshold_inclusive=executor.surrogate_threshold_inclusive
            if executor.surrogate is None
            else True,
        )
        expected_signature = runtime_state_signature(
            executor.plan,
            projection_mode=executor.projection_mode,
            threshold_inclusive=executor.surrogate_threshold_inclusive
            if executor.surrogate is None
            else True,
        )
    else:
        expected_compatibility = prepared["compatibility"]
        expected_signature = prepared["signature"]
    if probe is not None:
        probe.mark("State restore / init")
    if runtime_state is None:
        voltage = {
            name: (
                torch.zeros((batch, p["size"]), device=device, dtype=parameter_dtype)
                if p["neuron"]["kind"] == "leaky_integrator"
                else torch.full(
                    (batch, p["size"]),
                    float(p["neuron"].get("initial_voltage_mv", M.E_L))
                    if p["neuron"]["kind"] == "cuba_lif"
                    or p["neuron"]["kind"] in N.ADAPTIVE_KINDS
                    else M.E_L,
                    device=device,
                    dtype=parameter_dtype,
                )
            )
            for name, p in populations.items()
        }
        refractory = {
            name: torch.zeros((batch, p["size"]), dtype=torch.long, device=device)
            for name, p in populations.items()
        }
        spikes = {
            name: torch.zeros((batch, p["size"]), device=device, dtype=parameter_dtype)
            for name, p in populations.items()
        }
        conductance = {
            (p.id, p.polarity): torch.zeros(
                (batch, populations[p.target.partition(".")[0]]["size"]),
                device=device,
                dtype=parameter_dtype,
            )
            for p in executor.plan.projections
        }
        histories = {
            name: DelayBuffer(population_history_lengths[name], value)
            for name, value in spikes.items()
        }
        input_histories = {
            name: torch.zeros(
                (length, *inputs[name].shape[1:]),
                device=device,
                dtype=inputs[name].dtype,
            )
            for name, length in input_history_lengths.items()
            if length > 0
        }
        completed_steps = 0
    else:
        if runtime_state.signature != expected_signature:
            detail = _compatibility_mismatch(
                expected_compatibility, runtime_state.compatibility
            )
            raise ValueError(
                "runtime state is incompatible with graph plan: "
                + (
                    detail
                    or f"signature expected {expected_signature}, got {runtime_state.signature}"
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
                restored[name] = value.detach().to(device).clone()
            return restored

        pop_shapes = {
            name: (batch, int(row["size"])) for name, row in populations.items()
        }
        voltage = restore_group("voltages", runtime_state.voltages, pop_shapes)
        for name, value in voltage.items():
            if value.dtype != parameter_dtype:
                raise ValueError(
                    f"runtime state voltages.{name} dtype expected {parameter_dtype}, got {value.dtype}"
                )
        refractory = restore_group("refractory", runtime_state.refractory, pop_shapes)
        for name, value in refractory.items():
            if value.dtype != torch.long:
                raise ValueError(
                    f"runtime state refractory.{name} dtype expected torch.int64, got {value.dtype}"
                )
        by_projection = {row["id"]: row for row in executor.plan.graph["projections"]}
        conductance_by_id = {}
        for group_name, values, unit in (
            ("conductances", runtime_state.conductances, "uS"),
            ("currents", runtime_state.currents, "nA"),
        ):
            restored = restore_group(
                group_name,
                values,
                {
                    p.id: (
                        batch,
                        int(populations[p.target.partition(".")[0]]["size"]),
                    )
                    for p in executor.plan.projections
                    if E.synapse_unit(by_projection[p.id]["synapse"]) == unit
                },
            )
            conductance_by_id.update(restored)
        conductance = {
            (p.id, p.polarity): conductance_by_id[p.id]
            for p in executor.plan.projections
        }
        for name, value in conductance_by_id.items():
            if value.dtype != parameter_dtype:
                raise ValueError(
                    f"runtime state conductances.{name} dtype expected {parameter_dtype}, got {value.dtype}"
                )
        population_history_values = restore_group(
            "population_histories",
            runtime_state.population_histories,
            {
                name: (population_history_lengths[name], batch, int(row["size"]))
                for name, row in populations.items()
            },
        )
        histories = {
            name: DelayBuffer.restore(value)
            for name, value in population_history_values.items()
        }
        for name, value in population_history_values.items():
            if value.dtype != parameter_dtype:
                raise ValueError(
                    f"runtime state population_histories.{name} dtype expected {parameter_dtype}, got {value.dtype}"
                )
        spikes = {name: histories[name]._values[-1].clone() for name in populations}
        input_histories = restore_group(
            "input_histories",
            runtime_state.input_histories,
            {
                name: (length, *inputs[name].shape[1:])
                for name, length in input_history_lengths.items()
                if length > 0
            },
        )
        for name, value in input_histories.items():
            if value.dtype != inputs[name].dtype:
                raise ValueError(
                    f"runtime state input_histories.{name} dtype expected {inputs[name].dtype}, got {value.dtype}"
                )
        completed_steps = int(runtime_state.completed_steps)
    if probe is not None:
        probe.mark("Execution setup / metadata")
    segments = SegmentAccumulator(
        segment_identity(
            resets, decisions, graph=executor.plan.graph, batch_size=batch
        ),
        populations,
        batch,
        device,
        runtime_state.segment_state if runtime_state is not None else {},
        runtime_state.segment_counts if runtime_state is not None else {},
        completed_steps,
    )
    prepared_interventions, intervention_request = prepare_interventions(
        interventions,
        graph=executor.plan.graph,
        seed=executor.seed,
        start_step=completed_steps,
        steps_count=steps,
        batch_size=batch,
    )
    resolved_interventions = [item.metadata for item in prepared_interventions]
    for item in prepared_interventions:
        item.stage(
            device=device,
            dtype=parameter_dtype,
            start_step=completed_steps,
            steps_count=steps,
        )

    def intervene(name, emitted, absolute_step):
        for index, item in enumerate(prepared_interventions):
            if item.metadata["population_id"] == name:
                emitted = item.apply(emitted, absolute_step, index)
        return emitted

    neuron_states = {}
    synapse_states = {}
    custom_state = {}
    projection_rows = {row["id"]: row for row in executor.plan.graph["projections"]}
    projection_ports = {
        name: E.projection_port(row["synapse"]) for name, row in projection_rows.items()
    }
    saved_custom = runtime_state.custom_state if runtime_state is not None else {}
    for category, rows in (
        ("neuron", executor.plan.populations),
        ("synapse", executor.plan.graph["projections"]),
    ):
        for row in rows:
            spec = row[category]
            adaptive = category == "neuron" and spec["kind"] in N.ADAPTIVE_KINDS
            if not adaptive and spec["kind"] != f"custom_{category}":
                continue
            owner = row["id"]
            target = owner if category == "neuron" else row["target"].partition(".")[0]
            shape = (batch, populations[target]["size"])
            if adaptive:
                label = spec["kind"]
                initial_key = (
                    "initial_adaptation_mv"
                    if spec["kind"].endswith("alif")
                    else "initial_adaptation_na"
                )
                state = {
                    "voltage": voltage[owner],
                    "refractory": refractory[owner],
                    "adaptation": torch.full(
                        shape,
                        float(spec[initial_key]),
                        device=device,
                        dtype=parameter_dtype,
                    ),
                }
            else:
                definition = (
                    executor._neuron_definitions
                    if category == "neuron"
                    else executor._synapse_definitions
                )[owner]
                label = definition.name
                context = E.StateContext(
                    shape,
                    device,
                    parameter_dtype,
                    executor.plan.dt_ms,
                    spec.get("config", {}),
                )
                if definition.initialize is None:
                    state = {
                        "value": torch.zeros(
                            shape, device=device, dtype=parameter_dtype
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
                    torch.zeros(shape, device=device, dtype=torch.long),
                )
            for key in (
                ("voltage", "refractory") if category == "neuron" else ("value",)
            ):
                X.checked_tensor(
                    state[key],
                    shape=shape,
                    device=device,
                    dtype=torch.long if key == "refractory" else parameter_dtype,
                    name=f"{label}.{key}",
                )
            for key, value in state.items():
                if value.device != torch.device(device):
                    raise ValueError(
                        f"{label}.{key}: state must use execution device {device}"
                    )
            for port in executor._state_ports[category][owner]:
                if port not in state:
                    raise ValueError(f"{label}: declared state port {port} missing")
                X.checked_tensor(
                    state[port],
                    shape=shape,
                    device=device,
                    dtype=parameter_dtype,
                    name=f"{label}.{port}",
                )
            for key, value in state.items():
                state_key = f"{category}/{owner}/{key}"
                custom_state[state_key] = value
                if runtime_state is not None:
                    if state_key not in saved_custom:
                        raise ValueError(f"runtime state missing {state_key}")
                    restored = saved_custom[state_key].to(device)
                    X.checked_tensor(
                        restored,
                        shape=value.shape,
                        device=device,
                        dtype=value.dtype,
                        name=state_key,
                    )
                    if category == "neuron" and key in {"voltage", "refractory"}:
                        standard = (
                            voltage[owner] if key == "voltage" else refractory[owner]
                        )
                        if not torch.equal(restored, standard):
                            raise ValueError(
                                f"runtime state {state_key} disagrees with standard neuron state"
                            )
                    elif category == "synapse" and key == "value":
                        standard = conductance[(owner, row["polarity"])]
                        if not torch.equal(restored, standard):
                            raise ValueError(
                                f"runtime state {state_key} disagrees with synapse output state"
                            )
                    state[key] = restored.detach().clone()
            if category == "neuron":
                neuron_states[owner] = state
                voltage[owner] = state["voltage"]
                refractory[owner] = state["refractory"]
            else:
                synapse_states[owner] = state
    continuous_delays = {}
    for projection in executor.plan.projections:
        if projection.continuous:
            continuous_delays[projection.source] = max(
                continuous_delays.get(projection.source, 0), projection.delay_steps
            )

    def continuous_value(signal):
        owner, _, port = signal.partition(".")
        return voltage[owner] if port == "voltage" else neuron_states[owner][port]

    continuous_histories = {}
    for signal, length in continuous_delays.items():
        if not length:
            continue
        state_key = f"continuous_history/{signal}"
        initial = continuous_value(signal)
        if runtime_state is None:
            values = initial.unsqueeze(0).expand(length, *initial.shape).clone()
        else:
            if state_key not in saved_custom:
                raise ValueError(f"runtime state missing {state_key}")
            values = saved_custom[state_key].to(device)
            X.checked_tensor(
                values,
                shape=(length, *initial.shape),
                device=device,
                dtype=parameter_dtype,
                name=state_key,
            )
        continuous_histories[signal] = list(values.unbind(0))
        custom_state[state_key] = values
    if set(saved_custom) - set(custom_state):
        raise ValueError("runtime state has unexpected custom state tensors")
    state_traces = {}
    if not isinstance(retain_outputs, bool):
        raise TypeError("retain_outputs must be boolean")
    if type(batch_offset) is not int or batch_offset < 0:
        raise ValueError("batch_offset must be a non-negative integer")
    signal_axes = (
        prepared["axes"] if prepared is not None else _signal_axes(executor.plan.graph)
    )
    recorder = Recorder(
        recording,
        axes=signal_axes,
        start=completed_steps,
        steps=steps,
        dt_ms=executor.plan.dt_ms,
        batch_offset=batch_offset,
        graph=executor.plan.graph,
        dtype=parameter_dtype,
        device=device,
    )
    inference_only = not (
        torch.is_grad_enabled()
        and any(
            value.requires_grad
            for value in (*inputs.values(), *executor.weights.values())
        )
    )
    streamed_outputs = {
        row["id"]: row["signal"]
        for row in executor.plan.outputs
        if not retain_outputs and signal_axes[row["signal"]][0] == "time"
    }
    online = OnlineReductions(
        executor.plan.graph,
        signal_axes,
        executor.parameter_map(),
        enabled=automatic_reductions and inference_only,
        window=measurement,
        start=completed_steps,
        steps=steps,
        saved=runtime_state.reduction_state if runtime_state is not None else {},
        tensors=runtime_state.reduction_tensors if runtime_state is not None else {},
        recording_signals=(*recorder.rows, *streamed_outputs.values()),
        dtype=parameter_dtype,
        device=device,
    )
    external_signals = set(required_signals)
    external_signals.update(
        row["signal"]
        for row in executor.plan.outputs
        if row["id"] not in streamed_outputs
    )
    if diagnostics:
        external_signals.update(row["signal"] for row in executor.plan.observables)
    history_operations = online.history_operations(external_signals)
    needed_signals = set(external_signals)
    for operation in history_operations:
        needed_signals.update(operation["sources"])
    capture_steps = bool(
        online.reductions
        or online.points
        or online.duration_masks
        or recorder.rows
        or streamed_outputs
    )
    observation_counts = {
        name: torch.zeros_like(voltage[name], dtype=torch.int64)
        for name in observation_populations
    }
    integrator_sum: dict[str, torch.Tensor] = {}
    spike_traces: dict[str, list[torch.Tensor]] = {
        name: [] for name in populations if f"{name}.spikes" in needed_signals
    }
    voltage_traces: dict[str, list[torch.Tensor]] = {
        name: [] for name in populations if f"{name}.voltage" in needed_signals
    }
    pre_reset_voltage_traces: dict[str, list[torch.Tensor]] = {
        name: []
        for name, pop in populations.items()
        if pop["neuron"]["kind"] == "leaky_integrator"
        and f"{name}.pre_reset_voltage" in needed_signals
    }
    conductance_traces: dict[str, list[torch.Tensor]] = {
        p.id: []
        for p in executor.plan.projections
        if f"{p.id}.{projection_ports[p.id]}" in needed_signals
    }
    effective_weights = (
        prepared["effective_weights"]
        if prepared is not None
        else {
            name: executor._mask_autapses(name, value)
            for name, value in executor.parameter_map().items()
        }
    )
    projection_gains = {
        projection.id: sample_gain(
            projection_rows[projection.id]["modulation"],
            start_step=completed_steps,
            steps=steps,
            dt_ms=executor.plan.dt_ms,
        )
        for projection in executor.plan.projections
        if projection.enabled and "modulation" in projection_rows[projection.id]
    }
    spatial_operations = {
        f"{row['id']}.value": row
        for row in executor.plan.graph.get("operations", [])
        if row["kind"] == "pool2d"
    }
    block_reasons = []
    compiler_fallback = (
        "PyTorch 2.10 reduced-precision CPU uses eager tensor graphs"
        if executor.compile_mode is not None
        and precision_fallback(device, parameter_dtype)
        else None
    )
    block_trainable = {
        projection.parameter
        for projection in executor.plan.projections
        if projection.enabled and effective_weights[projection.parameter].requires_grad
    }
    partial_dependencies = not inference_only and any(
        executor._signal_parameters.get(signal, set()) & block_trainable
        != block_trainable
        for signal in external_signals
        if signal in executor._signal_parameters
    )
    for condition, reason in (
        (executor.compile_block_steps == 1, "block execution not requested"),
        (executor.compile_mode is None, "compilation disabled"),
        (compiler_fallback is not None, compiler_fallback),
        (
            device.type == "mps" and executor.compile_block_steps > 4,
            "MPS block size above validated limit (4)",
        ),
        (
            bool(
                executor._neuron_definitions
                or executor._synapse_definitions
                or executor.surrogate
            ),
            "registered neuron, synapse or surrogate",
        ),
        (capture_steps, "online reductions or recording"),
        (bool(prepared_interventions), "spike interventions"),
        (bool(segments.identity), "voltage resets or decision segments"),
        (trace is not None, "execution tracing"),
        (
            len(inputs) > 1 and any(value.requires_grad for value in inputs.values()),
            "multiple inputs with input gradients",
        ),
        (
            partial_dependencies,
            "partial parameter dependencies in retained signals",
        ),
        (
            not inference_only
            and steps
            <= sum(
                p.delay_steps
                for p in executor.plan.projections
                if p.enabled and p.root_source.partition(".")[0] in populations
            ),
            "short delayed autograd trajectory",
        ),
    ):
        if condition:
            block_reasons.append(reason)
    if not block_reasons:
        block_signals = tuple(
            sorted(
                {
                    *(
                        f"{name}.spikes"
                        for name in (*spike_traces, *observation_counts)
                    ),
                    *(f"{name}.voltage" for name in voltage_traces),
                    *(f"{name}.pre_reset_voltage" for name in pre_reset_voltage_traces),
                    *(
                        f"{name}.{projection_ports[name]}"
                        for name in conductance_traces
                    ),
                    *(
                        f"{name}.{port}"
                        for name in neuron_states
                        for port in executor._state_ports["neuron"][name]
                        if f"{name}.{port}" in needed_signals
                    ),
                }
            )
        )
        if block_signals not in executor._block_kernels:
            executor._block_kernels[block_signals] = network_block(
                executor.plan,
                spike_function=executor._spike_function,
                projection_mode=executor.projection_mode,
                signals=block_signals,
                compile_mode=executor.compile_mode,
            )
        kernel = executor._block_kernels[block_signals]
        history_values = {name: history._values for name, history in histories.items()}
        block_input_histories = input_histories
        # Prepared sessions can reuse the exact functional block and its
        # validated initial state without rebuilding a forward invocation.
        # Operation/reduction graphs retain the general executor path.
        if (
            prepared is not None
            and retain_outputs
            and not executor.plan.graph.get("operations")
        ):
            executor._stream_block_context = {
                "kernel": kernel,
                "state": (
                    voltage,
                    refractory,
                    neuron_states,
                    conductance,
                    history_values,
                    block_input_histories,
                ),
                "completed_steps": completed_steps,
                "gain_specs": {
                    p.id: projection_rows[p.id]["modulation"]
                    for p in executor.plan.projections
                    if p.enabled and "modulation" in projection_rows[p.id]
                },
                "ports": projection_ports,
            }
        gain_tensors = {
            name: torch.as_tensor(values, device=device, dtype=parameter_dtype)
            for name, values in projection_gains.items()
        }
        for begin in range(0, steps, executor.compile_block_steps):
            end = min(begin + executor.compile_block_steps, steps)
            if probe is not None:
                probe.mark("Compiled numerical block")
            (
                voltage,
                refractory,
                neuron_states,
                conductance,
                history_values,
                block_input_histories,
                spikes,
                integrator_sum,
                samples,
            ) = kernel(
                {name: value[begin:end] for name, value in inputs.items()},
                voltage,
                refractory,
                neuron_states,
                conductance,
                history_values,
                block_input_histories,
                effective_weights,
                {name: value[begin:end] for name, value in gain_tensors.items()},
                integrator_sum,
            )
            if probe is not None:
                probe.mark("Output / history assembly")
            for name in spike_traces:
                spike_traces[name].extend(samples[f"{name}.spikes"].unbind())
            for name in voltage_traces:
                voltage_traces[name].extend(samples[f"{name}.voltage"].unbind())
            for name in pre_reset_voltage_traces:
                pre_reset_voltage_traces[name].extend(
                    samples[f"{name}.pre_reset_voltage"].unbind()
                )
            for name in conductance_traces:
                conductance_traces[name].extend(
                    samples[f"{name}.{projection_ports[name]}"].unbind()
                )
            for name in neuron_states:
                for port in executor._state_ports["neuron"][name]:
                    signal = f"{name}.{port}"
                    if signal in samples:
                        state_traces.setdefault(signal, []).extend(
                            samples[signal].unbind()
                        )
            for name in observation_counts:
                observation_counts[name] += (
                    samples[f"{name}.spikes"].detach().to(torch.int64).sum(dim=0)
                )
        for name, values in history_values.items():
            histories[name]._values = values
        for name, state in neuron_states.items():
            for key, value in state.items():
                custom_state[f"neuron/{name}/{key}"] = value
    else:
        if probe is not None:
            probe.mark("Eager simulation / histories")
        for t in range(steps):
            segments.boundary(completed_steps + t, executor.plan.dt_ms)
            segments.reset(completed_steps + t, voltage)
            step_pre_reset = {}
            new_spikes: dict[str, torch.Tensor] = {}
            source_cache = {}
            # Freeze every continuous population source before any membrane
            # update. Reciprocal edges therefore share the same time origin.
            old_continuous = {
                signal: continuous_value(signal).clone() for signal in continuous_delays
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
                while signal in spatial_operations:
                    operation = spatial_operations[signal]
                    pending.append(operation)
                    signal = operation["sources"][0]
                owner, _, port = signal.partition(".")
                if owner in populations:
                    if port != "spikes":
                        result = (
                            old_continuous[signal]
                            if delay == 0
                            else continuous_histories[signal][-delay]
                        )
                    else:
                        result = (
                            new_spikes[owner]
                            if delay == 0
                            else histories[owner]._values[-delay]
                        )
                else:
                    source_t = t - delay
                    result = (
                        inputs[owner][source_t]
                        if source_t >= 0
                        else input_histories[owner][source_t]
                    )
                for operation in reversed(pending):
                    result = pool_tensor(
                        result,
                        signal_spatial_shape(
                            executor.plan.graph, operation["sources"][0]
                        ),
                        operation["config"],
                    )
                source_cache[key] = result
                return result

            for pop in executor.plan.populations:
                name = pop["id"]
                incoming = {
                    "excitatory": torch.zeros_like(voltage[name]),
                    "inhibitory": torch.zeros_like(voltage[name]),
                }
                for projection in executor._incoming_projections[name]:
                    key = (projection.id, projection.polarity)
                    if not projection.enabled:
                        conductance[key].zero_()
                        continue
                    source = projection_source(
                        projection.source, projection.delay_steps
                    )

                    def project(*, projection=projection, source=source):
                        config = projection_rows[projection.id].get("connectivity")
                        if config is not None:
                            return convolution_drive(
                                source,
                                effective_weights[projection.parameter],
                                signal_spatial_shape(
                                    executor.plan.graph, projection.source
                                ),
                                config,
                            )
                        return source @ effective_weights[projection.parameter]

                    if trace is None:
                        drive = project()
                    else:
                        with trace.detail(f"projection/{projection.id}"):
                            drive = project()
                    if projection.id in projection_gains:
                        drive = drive * projection_gains[projection.id][t]
                    spec = projection_rows[projection.id]["synapse"]
                    if spec["kind"] == "custom_synapse":
                        definition = executor._synapse_definitions[projection.id]
                        previous = synapse_states[projection.id]
                        state = X.checked_state(
                            executor._synapse_steps[projection.id](previous, drive),
                            previous,
                            definition.name,
                            check_values=executor._check_extension_values,
                        )
                        synapse_states[projection.id] = state
                        conductance[key] = state["value"]
                    else:
                        conductance[key] = executor._synapse_steps[projection.id](
                            conductance[key], drive
                        )
                    incoming[projection.polarity] += conductance[key]
                neuron = pop["neuron"]
                if neuron["kind"] == "custom_neuron":
                    definition = executor._neuron_definitions[name]
                    previous = neuron_states[name]
                    response = executor._neuron_steps[name](
                        previous,
                        incoming["excitatory"],
                        incoming["inhibitory"],
                    )
                    if not isinstance(response, tuple) or len(response) != 2:
                        raise TypeError(
                            f"{definition.name} must return (state, spikes)"
                        )
                    state, spike_values = response
                    state = X.checked_state(
                        state,
                        previous,
                        definition.name,
                        check_values=executor._check_extension_values,
                    )
                    X.checked_tensor(
                        spike_values,
                        shape=voltage[name].shape,
                        device=device,
                        dtype=parameter_dtype,
                        name=f"{definition.name}.spikes",
                        check_values=executor._check_extension_values,
                    )
                    if executor._check_extension_values and torch.any(
                        (spike_values != 0) & (spike_values != 1)
                    ):
                        raise ValueError(f"{definition.name} spikes must be binary")
                    neuron_states[name] = state
                    voltage[name], refractory[name] = (
                        state["voltage"],
                        state["refractory"],
                    )
                    new_spikes[name] = (
                        spike_values
                        if pop["spiking"]
                        else torch.zeros_like(spike_values)
                    )
                    new_spikes[name] = intervene(
                        name, new_spikes[name], completed_steps + t
                    )
                    continue
                if neuron["kind"] in N.ADAPTIVE_KINDS:
                    state, spike_values = executor._neuron_steps[name](
                        neuron_states[name],
                        incoming["excitatory"],
                        incoming["inhibitory"],
                    )
                    neuron_states[name] = state
                    voltage[name], refractory[name] = (
                        state["voltage"],
                        state["refractory"],
                    )
                    new_spikes[name] = intervene(
                        name,
                        spike_values
                        if pop["spiking"]
                        else torch.zeros_like(spike_values),
                        completed_steps + t,
                    )
                    continue
                if neuron["kind"] == "cuba_lif":
                    voltage[name], new_spikes[name], refractory[name] = (
                        executor._neuron_steps[name](
                            voltage[name],
                            refractory[name],
                            incoming["excitatory"],
                            incoming["inhibitory"],
                        )
                    )
                    new_spikes[name] = intervene(
                        name, new_spikes[name], completed_steps + t
                    )
                    continue
                if neuron["kind"] == "leaky_integrator":
                    pre_reset, post_reset, new_spikes[name] = executor._neuron_steps[
                        name
                    ](voltage[name], incoming["excitatory"])
                    if name in pre_reset_voltage_traces:
                        pre_reset_voltage_traces[name].append(pre_reset)
                    step_pre_reset[f"{name}.pre_reset_voltage"] = pre_reset
                    integrator_sum[name] = (
                        integrator_sum.get(name, torch.zeros_like(pre_reset))
                        + pre_reset
                    )
                    voltage[name] = post_reset
                    new_spikes[name] = intervene(
                        name, new_spikes[name], completed_steps + t
                    )
                    continue
                voltage[name], new_spikes[name], refractory[name] = (
                    executor._neuron_steps[name](
                        voltage[name],
                        refractory[name],
                        incoming["excitatory"],
                        incoming["inhibitory"],
                    )
                )
                new_spikes[name] = intervene(
                    name, new_spikes[name], completed_steps + t
                )
            for category, states in (
                ("neuron", neuron_states),
                ("synapse", synapse_states),
            ):
                for owner, state in states.items():
                    for key, value in state.items():
                        custom_state[f"{category}/{owner}/{key}"] = value
                    for port in executor._state_ports[category][owner]:
                        signal = f"{owner}.{port}"
                        if signal in needed_signals:
                            state_traces.setdefault(signal, []).append(state[port])
            if capture_steps:
                step_signals = {
                    f"{name}.value": value[t] for name, value in inputs.items()
                }
                step_signals.update(
                    {f"{name}.spikes": value for name, value in new_spikes.items()}
                )
                step_signals.update(
                    {f"{name}.voltage": value for name, value in voltage.items()}
                )
                step_signals.update(step_pre_reset)
                step_signals.update(
                    {
                        f"{p.id}.{projection_ports[p.id]}": conductance[
                            (p.id, p.polarity)
                        ]
                        for p in executor.plan.projections
                    }
                )
                for category, states in (
                    ("neuron", neuron_states),
                    ("synapse", synapse_states),
                ):
                    for owner, state in states.items():
                        for port in executor._state_ports[category][owner]:
                            step_signals[f"{owner}.{port}"] = state[port]
                step_signals = online.samples(step_signals)
                online.observe(step_signals, completed_steps + t)
                recorder.observe(step_signals, completed_steps + t)
                for output_id, signal in streamed_outputs.items():
                    recorder.output_step(
                        output_id, step_signals[signal], completed_steps + t
                    )
            if trace is not None:
                trace.observe_spikes(new_spikes)
            spikes = new_spikes
            segments.observe(spikes)
            for name in observation_counts:
                observation_counts[name] += spikes[name].detach().to(torch.int64)
            for name in spike_traces:
                spike_traces[name].append(spikes[name])
            for name in voltage_traces:
                voltage_traces[name].append(voltage[name])
            for name in populations:
                histories[name].push(spikes[name])
            for signal, values in continuous_histories.items():
                values.append(old_continuous[signal])
                del values[0]
                custom_state[f"continuous_history/{signal}"] = torch.stack(values)
            for projection_id in conductance_traces:
                projection = next(
                    p for p in executor.plan.projections if p.id == projection_id
                )
                conductance_traces[projection_id].append(
                    conductance[(projection.id, projection.polarity)]
                )
    if probe is not None:
        probe.mark("Output / history assembly")
    outputs: dict[str, torch.Tensor] = {}
    signal_values: dict[str, torch.Tensor] = {
        f"{name}.value": value for name, value in inputs.items()
    }
    for name, values in spike_traces.items():
        signal_values[f"{name}.spikes"] = torch.stack(values)
    for name, values in voltage_traces.items():
        signal_values[f"{name}.voltage"] = torch.stack(values)
    for name, values in pre_reset_voltage_traces.items():
        signal_values[f"{name}.pre_reset_voltage"] = torch.stack(values)
    for name, values in conductance_traces.items():
        signal_values[f"{name}.{projection_ports[name]}"] = torch.stack(values)
    for signal, values in state_traces.items():
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

    signal_values.update(online.finish(completed_steps + steps))
    remaining_ops = list(history_operations)
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
                sources[0] @ executor.weights[parameter]
            )
        elif kind == "pool2d":
            signal_values[f"{op['id']}.value"] = pool_tensor(
                sources[0],
                signal_spatial_shape(executor.plan.graph, op["sources"][0]),
                op["config"],
            )
        elif kind in {"reduce_mean", "reduce_sum"}:
            mask_name = op.get("config", {}).get("mask")
            mask = signal_values.get(mask_name) if mask_name else None
            source_id = op["sources"][0]
            owner, _, port = source_id.partition(".")
            if (
                "voltage_sampling" not in executor.plan.graph
                and kind == "reduce_mean"
                and mask is None
                and port == "voltage"
                and owner in integrator_sum
            ):
                signal_values[f"{op['id']}.value"] = integrator_sum[owner] / steps
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
                if duration_key in online.tensors:
                    mask_seconds = online.tensors[duration_key].to(sources[0].dtype) * (
                        executor.plan.dt_ms / 1000.0
                    )
                else:
                    mask_seconds = mask.to(
                        device=sources[0].device, dtype=sources[0].dtype
                    ).sum(dim=0) * (executor.plan.dt_ms / 1000.0)
                mask_seconds = mask_seconds.reshape(
                    mask_seconds.shape[0], *([1] * (sources[0].ndim - 1))
                )
                if torch.any(mask_seconds <= 0) and (
                    measurement is None
                    or completed_steps + steps >= measurement.end_step
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
            definition, update = executor._operation_steps[op["id"]]
            parameters = {
                name: executor.parameter_map()[name]
                for name in op.get("parameters", [])
            }
            value = update(tuple(sources), parameters)
            shape = tuple(
                steps if d == "time" else batch if d == "batch" else d
                for d in op["shape"]
            )
            signal_values[f"{op['id']}.value"] = X.checked_tensor(
                value,
                shape=shape,
                device=device,
                dtype=parameter_dtype,
                name=definition.name,
            )
        elif kind == "cumulative_sum":
            signal_values[f"{op['id']}.value"] = sources[0].cumsum(dim=0)
        else:
            raise ValueError(f"{op['id']}: unsupported operation {kind}")
    for output in executor.plan.outputs:
        if output["id"] in streamed_outputs:
            continue
        value = signal_values[output["signal"]]
        recorder.output(output["id"], value, measurement)
        if retain_outputs:
            outputs[output["id"]] = value
    packed = (
        {
            row["id"]: signal_values[row["signal"]].detach().clone()
            for row in executor.plan.observables
        }
        if diagnostics
        else {}
    )
    segments.boundary(completed_steps + steps, executor.plan.dt_ms)
    segments.finish_records()
    if probe is not None:
        probe.mark("Runtime-state snapshot")
    next_runtime_state = None
    if retain_runtime_state:
        next_input_histories = {
            name: (
                inputs[name][-history.shape[0] :]
                if inputs[name].shape[0] >= history.shape[0]
                else torch.cat((history, inputs[name]), dim=0)[-history.shape[0] :]
            )
            .detach()
            .clone()
            for name, history in input_histories.items()
        }
        next_runtime_state = GraphRuntimeState(
            signature=expected_signature,
            compatibility=expected_compatibility,
            completed_steps=completed_steps + steps,
            voltages={name: value.detach().clone() for name, value in voltage.items()},
            refractory={
                name: value.detach().clone() for name, value in refractory.items()
            },
            conductances={
                p.id: conductance[(p.id, p.polarity)].detach().clone()
                for p in executor.plan.projections
                if projection_ports[p.id] == "conductance"
            },
            currents={
                p.id: conductance[(p.id, p.polarity)].detach().clone()
                for p in executor.plan.projections
                if projection_ports[p.id] == "current"
            },
            population_histories={
                name: history.export() for name, history in histories.items()
            },
            input_histories=next_input_histories,
            custom_state={
                key: value.detach().clone() for key, value in custom_state.items()
            },
            segment_state=segments.state(),
            segment_counts=segments.counts,
            reduction_state=online.state(completed_steps + steps),
            reduction_tensors={
                name: value.detach().clone() for name, value in online.tensors.items()
            }
            if measurement is not None
            else {},
        )
    if probe is not None:
        probe.mark("Executor result / metadata")
    result = ExecutionResult(
        executor="graph",
        outputs=outputs,
        diagnostics=packed,
        parameters={k: v.detach().clone() for k, v in effective_weights.items()}
        if retain_parameters
        else {},
        final_state={f"{k}.voltage": v.detach().clone() for k, v in voltage.items()}
        if retain_final_state
        else {},
        runtime_state=next_runtime_state,
        metrics={
            "recording_units": {
                name: recorder.units.get(name) for name in recorder.rows
            },
            "online_reductions": list(online.reductions),
            "retained_signal_samples": {
                name: len(values)
                for name, values in (
                    *spike_traces.items(),
                    *voltage_traces.items(),
                    *pre_reset_voltage_traces.items(),
                    *conductance_traces.items(),
                    *state_traces.items(),
                )
            },
            "measurement_complete": measurement is None
            or completed_steps + steps >= measurement.end_step,
            "inference_retention": retention_identity(
                graph=executor.plan.graph,
                start_step=completed_steps,
                steps_count=steps,
                batch_size=batch,
                batch_offset=batch_offset,
                diagnostics=diagnostics,
                retain_outputs=retain_outputs,
                retain_parameters=retain_parameters,
                retain_final_state=retain_final_state,
                retain_runtime_state=retain_runtime_state,
                projection_mode=executor.projection_mode,
                measurement=measurement,
                recording=recording,
            ),
            "segment_policy": segments.identity,
            "segment_window": {"start_step": completed_steps, "steps_count": steps}
            if segments.identity
            else None,
            "decisions": segments.records,
            "resolved_interventions": resolved_interventions,
            "refractory": dict(executor.plan.refractory),
            "autapses": executor.autapse_metadata,
            "extension_validation": executor.extension_validation,
            "projection_mode": executor.projection_mode,
            "compile_mode": executor.compile_mode,
            "compiler_wrapper": compiler_wrapper_policy(
                device, parameter_dtype, executor.compile_mode
            ),
            "compiler_precision_policy": "snnlab.eager-precision-casts/v1"
            if executor.compile_mode is not None
            else None,
            "compiler_fallback": compiler_fallback,
            "execution_path": {
                "schema": "snnlab.network-blocks/v1",
                "selected": "kernel_loop" if block_reasons else "compiled_blocks",
                "requested_block_steps": executor.requested_block_steps,
                "block_steps": 1 if block_reasons else executor.compile_block_steps,
                "fallback_reasons": block_reasons,
            },
            "surrogate": copy.deepcopy(executor.surrogate_config),
            "result_snapshots": {
                "parameters": retain_parameters,
                "final_state": retain_final_state,
                "runtime_state": retain_runtime_state,
            },
            "inference_interventions": intervention_request if interventions else None,
            **(
                {"population_observations": observation_counts}
                if observation_counts
                else {}
            ),
        },
        decisions=segments.records,
        recorded_signals=recorder.finish(batch),
        _recording_info=recorder.information(batch),
        _output_axes={
            row["id"]: signal_axes[row["signal"]] for row in executor.plan.outputs
        },
        _diagnostic_axes={
            row["id"]: signal_axes[row["signal"]]
            for row in executor.plan.observables
            if diagnostics
        },
        _timebase=(executor.plan.dt_ms, completed_steps, steps),
        _batch_size=batch,
        model=executor,
    )
    if result.metrics["execution_path"] not in executor._execution_paths:
        executor._execution_paths.append(
            copy.deepcopy(result.metrics["execution_path"])
        )
    return result, signal_values
