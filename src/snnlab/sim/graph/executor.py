"""The graph-native PyTorch forward executor and causal delay buffers."""

from __future__ import annotations

import copy
import math
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import nn

from snnlab import extensions as E
from snnlab._shared import neurons as N
from snnlab.sim.artifacts.inference import retention_identity
from snnlab.sim.diagnostics.tracing import current_trace
from snnlab.sim.graph import dynamics as M
from snnlab.sim.graph import extensions as X
from snnlab.sim.graph.kernels import neuron_kernel
from snnlab.sim.graph.plan import GraphPlan, PlannedProjection, _signal_axes
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


class GraphExecutor(nn.Module):
    """Dense graph executor whose graph topology is lowered before simulation."""

    def __init__(
        self,
        plan: GraphPlan,
        *,
        seed: int = 0,
        trainable_parameters: Sequence[str] = (),
        surrogate_slope: float = M.SURROGATE_SLOPE,
        surrogate: Mapping[str, Any] | None = None,
        extension_validation: Literal["strict", "structural"] = "structural",
        compile_mode: Literal["default"] | None = None,
        projection_mode: Literal["fast", "strict"] = "fast",
    ):
        super().__init__()
        if projection_mode not in {"fast", "strict"}:
            raise ValueError("projection_mode must be fast or strict")
        self.projection_mode = projection_mode
        if compile_mode not in {None, "default"}:
            raise ValueError("compile_mode must be None or default")
        self.compile_mode = compile_mode
        if extension_validation not in ("strict", "structural"):
            raise ValueError("extension_validation must be strict or structural")
        self.extension_validation = extension_validation
        self._check_extension_values = extension_validation == "strict"
        self.plan = plan
        self.seed = int(seed)
        self.surrogate = (
            surrogate if (surrogate or {}).get("kind") == "custom_surrogate" else None
        )
        builtin = surrogate or {}
        if builtin and builtin.get("kind") not in {"fast_sigmoid", "custom_surrogate"}:
            raise ValueError("unsupported surrogate kind")
        self.surrogate_slope = (
            builtin.get("slope", surrogate_slope)
            if self.surrogate is None
            else surrogate_slope
        )
        self.surrogate_gradient_scale = (
            builtin.get("gradient_scale", 1.0) if self.surrogate is None else 1.0
        )
        self.surrogate_threshold_inclusive = (
            builtin.get("threshold_inclusive", True) if self.surrogate is None else True
        )
        for name, value, positive in (
            ("slope", self.surrogate_slope, True),
            ("gradient_scale", self.surrogate_gradient_scale, False),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (value <= 0 if positive else value < 0)
            ):
                domain = "positive" if positive else "nonnegative"
                raise ValueError(f"surrogate {name} must be finite and {domain}")
        if not isinstance(self.surrogate_threshold_inclusive, bool):
            raise ValueError("surrogate threshold_inclusive must be boolean")
        self.surrogate_slope = float(self.surrogate_slope)
        self.surrogate_config = (
            copy.deepcopy(self.surrogate)
            if self.surrogate is not None
            else {
                "kind": "fast_sigmoid",
                "slope": self.surrogate_slope,
                "gradient_scale": self.surrogate_gradient_scale,
                "threshold_inclusive": self.surrogate_threshold_inclusive,
            }
        )
        self._spike_function = X.make_spike(
            slope=self.surrogate_slope,
            gradient_scale=self.surrogate_gradient_scale,
            threshold_inclusive=self.surrogate_threshold_inclusive,
            custom=self.surrogate,
            check_values=self._check_extension_values,
        )
        # Registered surrogates/callbacks retain eager dispatch and validation.
        self._neuron_steps = {
            row["id"]: neuron_kernel(
                row,
                dt_ms=plan.dt_ms,
                refractory=plan.refractory.get(row["id"], {}),
                spike_function=self._spike_function,
                compile_mode=compile_mode if self.surrogate is None else None,
            )
            for row in plan.populations
        }
        self._incoming_projections = {
            row["id"]: tuple(
                p for p in plan.projections if p.target.partition(".")[0] == row["id"]
            )
            for row in plan.populations
        }
        torch.manual_seed(seed)
        rows = {row["id"]: row for row in plan.graph.get("parameters", [])}
        self.weights = nn.ParameterDict()
        self.initialization_metadata: dict[str, dict[str, Any]] = {}
        self.autapse_metadata = dict(plan.autapses)
        self._excluded_autapse_parameters = {
            p.parameter
            for p in plan.projections
            if plan.autapses[p.id]["effective"] == "excluded"
        }
        explicit_parameters = {
            p["parameters"][0]
            for p in plan.graph.get("projections", [])
            if "allow_autapses" in p
        }
        pop_ids = {p["id"] for p in plan.populations}
        trainable = set(trainable_parameters)

        def initialise(
            row: Mapping[str, Any],
            *,
            runtime_shape: tuple[int, ...],
            scale_by_fanin: bool,
        ) -> torch.Tensor:
            scaling = row.get(
                "initialization_scaling",
                "fan_in_normalized" if scale_by_fanin else "direct",
            )
            if scaling not in ("direct", "fan_in_normalized"):
                raise ValueError(
                    f"{row['id']}: invalid initialization_scaling {scaling!r}"
                )
            scale_by_fanin = scaling == "fan_in_normalized"
            exclude_autapses = row["id"] in self._excluded_autapse_parameters
            eligible_fan_in = runtime_shape[0] - int(exclude_autapses)
            init = row["initializer"]
            kind = init["kind"]
            if kind == "custom_initializer":
                definition = E.resolve("initializer", init)
                value = definition.function(
                    runtime_shape,
                    init.get("config", {}),
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                X.checked_tensor(
                    value,
                    shape=runtime_shape,
                    device="cpu",
                    dtype=torch.float32,
                    name=definition.name,
                )
            elif kind in {"normal", "lower_clamped_normal"}:
                value = (
                    torch.randn(*runtime_shape)
                    .mul_(float(init["std"]))
                    .add_(float(init["mean"]))
                    .clamp_(min=0)
                )
            elif kind == "signed_normal":
                value = (
                    torch.randn(*runtime_shape)
                    .mul_(float(init["std"]))
                    .add_(float(init["mean"]))
                )
            elif kind == "uniform":
                value = (
                    torch.rand(*runtime_shape)
                    .mul_(float(init["high"]) - float(init["low"]))
                    .add_(float(init["low"]))
                )
            elif init["kind"] == "constant":
                value = torch.full(runtime_shape, float(init["value"]))
            elif kind == "zeros":
                value = torch.zeros(runtime_shape)
            else:
                raise ValueError(f"{row['id']}: unsupported initializer {init['kind']}")
            zero_fraction = float(init.get("initial_zero_fraction", 0.0))
            if not 0 <= zero_fraction < 1:
                raise ValueError(
                    f"{row['id']}: initial_zero_fraction must satisfy 0 <= fraction < 1"
                )
            zeroing = init.get("zeroing", "bernoulli")
            if zero_fraction and eligible_fan_in:
                if zeroing == "exact_k":
                    if len(runtime_shape) != 2:
                        raise ValueError(
                            f"{row['id']}: exact_k zeroing requires a matrix"
                        )
                    fan_in, fan_out = runtime_shape
                    kept = max(1, int(round((1.0 - zero_fraction) * eligible_fan_in)))
                    mask = torch.zeros(runtime_shape)
                    for column in range(fan_out):
                        partners = torch.randperm(eligible_fan_in)[:kept]
                        if exclude_autapses:
                            partners = partners + (partners >= column).long()
                        mask[partners, column] = 1
                    value = value * mask * (eligible_fan_in / kept)
                elif zeroing == "bernoulli":
                    value = (
                        value
                        * (torch.rand(*runtime_shape) > zero_fraction)
                        / (1 - zero_fraction)
                    )
                else:
                    raise ValueError(
                        f"{row['id']}: unsupported initializer zeroing {zeroing}"
                    )
            constraint = row.get("constraint")
            if scale_by_fanin:
                value = value / max(1, eligible_fan_in)
            if constraint:
                value = X.apply_constraint(value, constraint)
            if exclude_autapses:
                value = value.clone().fill_diagonal_(0)
            flat = value.reshape(-1)
            self.initialization_metadata[row["id"]] = {
                "initializer": dict(init),
                "constraint": dict(constraint) if constraint else None,
                "unit": row.get("unit"),
                "runtime_shape": list(runtime_shape),
                "scaling": "fan_in_normalized" if scale_by_fanin else "direct",
                "statistics": {
                    "count": flat.numel(),
                    "zero_fraction": float((flat == 0).float().mean()),
                    "mean": float(flat.mean()),
                    "std": float(flat.std(unbiased=False)),
                    "min": float(flat.min()),
                    "max": float(flat.max()),
                },
            }
            if row["id"] in explicit_parameters:
                self.initialization_metadata[row["id"]]["autapses"] = {
                    p.id: dict(plan.autapses[p.id])
                    for p in plan.projections
                    if p.parameter == row["id"]
                }
            return value

        def init_priority(projection: PlannedProjection) -> tuple[int, str]:
            source = projection.source.partition(".")[0]
            target = projection.target.partition(".")[0]
            target_kind = next(
                p["neuron"]["kind"] for p in plan.populations if p["id"] == target
            )
            if source not in pop_ids:
                return (0, projection.id)
            if target_kind == "leaky_integrator":
                return (1, projection.id)
            if projection.polarity == "excitatory":
                return (2, projection.id)
            return (3, projection.id)

        realised: dict[str, torch.Tensor] = {}
        for projection in sorted(plan.projections, key=init_priority):
            row = rows[projection.parameter]
            # Old shared projections retain their historical repeated draws.
            if projection.parameter in realised and "initialization_scaling" in row:
                continue
            shape = tuple(reversed(row["shape"]))  # runtime is [source, target]
            realised[projection.parameter] = initialise(
                row, runtime_shape=shape, scale_by_fanin=True
            )
        for operation in plan.graph.get("operations", []):
            for parameter in operation.get("parameters", []):
                if parameter in realised:
                    continue
                row = rows[parameter]
                shape = (
                    tuple(reversed(row["shape"]))
                    if operation["kind"] == "linear"
                    else tuple(row["shape"])
                )
                realised[parameter] = initialise(
                    row, runtime_shape=shape, scale_by_fanin=False
                )
        for projection in plan.projections:
            parameter_id = projection.parameter
            self.weights[projection.parameter.replace(".", "__")] = nn.Parameter(
                realised[projection.parameter], requires_grad=parameter_id in trainable
            )
        for operation in plan.graph.get("operations", []):
            for parameter in operation.get("parameters", []):
                self.weights[parameter.replace(".", "__")] = nn.Parameter(
                    realised[parameter], requires_grad=parameter in trainable
                )
        self._autapse_gradient_hooks = []
        self._register_autapse_gradient_hooks()
        self.register_state_dict_pre_hook(
            lambda module, prefix, keep_vars: module._enforce_autapses()
        )
        self.register_load_state_dict_post_hook(
            lambda module, incompatible: module._restore_autapses()
        )

    def _register_autapse_gradient_hooks(self) -> None:
        for handle in self._autapse_gradient_hooks:
            handle.remove()
        self._autapse_gradient_hooks = []
        for name in self._excluded_autapse_parameters:
            parameter = self.weights[name.replace(".", "__")]
            if parameter.requires_grad:
                self._autapse_gradient_hooks.append(
                    parameter.register_hook(
                        lambda gradient, name=name: self._mask_autapses(name, gradient)
                    )
                )

    def _restore_autapses(self) -> None:
        self._enforce_autapses()
        # Native load_state_dict(assign=True) replaces parameter objects.
        self._register_autapse_gradient_hooks()

    def _mask_autapses(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if name in self._excluded_autapse_parameters:
            return value.clone().fill_diagonal_(0)
        return value

    def _enforce_autapses(self) -> None:
        with torch.no_grad():
            for name in self._excluded_autapse_parameters:
                self.weights[name.replace(".", "__")].fill_diagonal_(0)

    def get_extra_state(self) -> dict[str, Any]:
        """Include the declared connectivity contract in native checkpoints."""
        return {"autapses": copy.deepcopy(self.autapse_metadata)}

    def set_extra_state(self, state: Any) -> None:
        if state != self.get_extra_state():
            raise ValueError("checkpoint autapse policy does not match graph")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key = prefix + "_extra_state"
        if key not in state_dict:
            if self._excluded_autapse_parameters:
                error_msgs.append(
                    "checkpoint has no autapse policy; load it with its historical "
                    "graph rather than silently changing connectivity"
                )
            state_dict[key] = self.get_extra_state()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def parameter_map(self) -> dict[str, torch.Tensor]:
        return {name.replace("__", "."): value for name, value in self.weights.items()}

    def enforce_constraints(self) -> None:
        """Apply declared constraints after an external optimizer step."""
        rows = {row["id"]: row for row in self.plan.graph.get("parameters", [])}
        with torch.no_grad():
            for name, parameter in self.parameter_map().items():
                parameter.copy_(
                    X.apply_constraint(parameter, rows[name].get("constraint"))
                )
        self._enforce_autapses()

    def forward(
        self,
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
        retain_parameters: bool = True,
        retain_final_state: bool = True,
        retain_runtime_state: bool = True,
        batch_offset: int = 0,
    ) -> ExecutionResult:
        result, _ = self._forward(
            inputs,
            diagnostics=diagnostics,
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
            batch_offset=batch_offset,
            automatic_reductions=True,
        )
        return result

    def _forward(
        self,
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
            next(iter(self.weights.values())).dtype if self.weights else first.dtype
        )
        input_specs = {row["id"]: row for row in self.plan.graph.get("inputs", [])}
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
        populations = {p["id"]: p for p in self.plan.populations}
        # Zero-delay edges use current-step spikes, but continuation still needs
        # the last spike tensor for every population.
        population_history_lengths = {
            name: max(
                (
                    p.delay_steps
                    for p in self.plan.projections
                    if p.source.startswith(name + ".") and p.delay_steps > 0
                ),
                default=1,
            )
            for name in populations
        }
        input_history_lengths = {
            row["id"]: max(
                (
                    p.delay_steps
                    for p in self.plan.projections
                    if p.source.partition(".")[0] == row["id"]
                ),
                default=0,
            )
            for row in self.plan.graph.get("inputs", [])
        }
        expected_compatibility = runtime_state_compatibility(
            self.plan,
            projection_mode=self.projection_mode,
            threshold_inclusive=self.surrogate_threshold_inclusive
            if self.surrogate is None
            else True,
        )
        expected_signature = runtime_state_signature(
            self.plan,
            projection_mode=self.projection_mode,
            threshold_inclusive=self.surrogate_threshold_inclusive
            if self.surrogate is None
            else True,
        )
        if runtime_state is None:
            voltage = {
                name: (
                    torch.zeros((batch, p["size"]), device=device)
                    if p["neuron"]["kind"] == "leaky_integrator"
                    else torch.full(
                        (batch, p["size"]),
                        float(p["neuron"].get("initial_voltage_mv", M.E_L))
                        if p["neuron"]["kind"] == "cuba_lif"
                        or p["neuron"]["kind"] in N.ADAPTIVE_KINDS
                        else M.E_L,
                        device=device,
                    )
                )
                for name, p in populations.items()
            }
            refractory = {
                name: torch.zeros((batch, p["size"]), dtype=torch.long, device=device)
                for name, p in populations.items()
            }
            spikes = {
                name: torch.zeros((batch, p["size"]), device=device)
                for name, p in populations.items()
            }
            conductance = {
                (p.id, p.polarity): torch.zeros(
                    (batch, populations[p.target.partition(".")[0]]["size"]),
                    device=device,
                )
                for p in self.plan.projections
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
            refractory = restore_group(
                "refractory", runtime_state.refractory, pop_shapes
            )
            for name, value in refractory.items():
                if value.dtype != torch.long:
                    raise ValueError(
                        f"runtime state refractory.{name} dtype expected torch.int64, got {value.dtype}"
                    )
            by_projection = {row["id"]: row for row in self.plan.graph["projections"]}
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
                        for p in self.plan.projections
                        if E.synapse_unit(by_projection[p.id]["synapse"]) == unit
                    },
                )
                conductance_by_id.update(restored)
            conductance = {
                (p.id, p.polarity): conductance_by_id[p.id]
                for p in self.plan.projections
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
        segments = SegmentAccumulator(
            segment_identity(
                resets, decisions, graph=self.plan.graph, batch_size=batch
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
            graph=self.plan.graph,
            seed=self.seed,
            start_step=completed_steps,
            steps_count=steps,
            batch_size=batch,
        )
        resolved_interventions = [item.metadata for item in prepared_interventions]

        def intervene(name, emitted, absolute_step):
            for index, item in enumerate(prepared_interventions):
                if item.metadata["population_id"] == name:
                    emitted = item.apply(emitted, absolute_step, index)
            return emitted

        neuron_states = {}
        synapse_states = {}
        custom_state = {}
        projection_rows = {row["id"]: row for row in self.plan.graph["projections"]}
        projection_ports = {
            name: E.projection_port(row["synapse"])
            for name, row in projection_rows.items()
        }
        saved_custom = runtime_state.custom_state if runtime_state is not None else {}
        for category, rows in (
            ("neuron", self.plan.populations),
            ("synapse", self.plan.graph["projections"]),
        ):
            for row in rows:
                spec = row[category]
                adaptive = category == "neuron" and spec["kind"] in N.ADAPTIVE_KINDS
                if not adaptive and spec["kind"] != f"custom_{category}":
                    continue
                owner = row["id"]
                target = (
                    owner if category == "neuron" else row["target"].partition(".")[0]
                )
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
                    definition = E.resolve(category, spec)
                    label = definition.name
                    context = E.StateContext(
                        shape,
                        device,
                        parameter_dtype,
                        self.plan.dt_ms,
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
                            required=("voltage",)
                            if category == "neuron"
                            else ("value",),
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
                for port in E.state_units(category, spec):
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
                                voltage[owner]
                                if key == "voltage"
                                else refractory[owner]
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
        if set(saved_custom) - set(custom_state):
            raise ValueError("runtime state has unexpected custom state tensors")
        state_traces = {}
        if not isinstance(retain_outputs, bool):
            raise TypeError("retain_outputs must be boolean")
        if type(batch_offset) is not int or batch_offset < 0:
            raise ValueError("batch_offset must be a non-negative integer")
        signal_axes = _signal_axes(self.plan.graph)
        recorder = Recorder(
            recording,
            axes=signal_axes,
            start=completed_steps,
            steps=steps,
            dt_ms=self.plan.dt_ms,
            batch_offset=batch_offset,
            graph=self.plan.graph,
            dtype=parameter_dtype,
        )
        inference_only = not (
            torch.is_grad_enabled()
            and any(
                value.requires_grad
                for value in (*inputs.values(), *self.weights.values())
            )
        )
        streamed_outputs = {
            row["id"]: row["signal"]
            for row in self.plan.outputs
            if not retain_outputs and signal_axes[row["signal"]][0] == "time"
        }
        online = OnlineReductions(
            self.plan.graph,
            signal_axes,
            self.parameter_map(),
            enabled=automatic_reductions and inference_only,
            window=measurement,
            start=completed_steps,
            steps=steps,
            saved=runtime_state.reduction_state if runtime_state is not None else {},
            tensors=runtime_state.reduction_tensors
            if runtime_state is not None
            else {},
            recording_signals=(*recorder.rows, *streamed_outputs.values()),
            dtype=parameter_dtype,
        )
        external_signals = set(required_signals)
        external_signals.update(
            row["signal"]
            for row in self.plan.outputs
            if row["id"] not in streamed_outputs
        )
        if diagnostics:
            external_signals.update(row["signal"] for row in self.plan.observables)
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
            for p in self.plan.projections
            if f"{p.id}.{projection_ports[p.id]}" in needed_signals
        }

        effective_weights = {
            name: self._mask_autapses(name, value)
            for name, value in self.parameter_map().items()
        }
        for t in range(steps):
            segments.boundary(completed_steps + t, self.plan.dt_ms)
            segments.reset(completed_steps + t, voltage)
            step_pre_reset = {}
            new_spikes: dict[str, torch.Tensor] = {}
            for pop in self.plan.populations:
                name = pop["id"]
                incoming = {
                    "excitatory": torch.zeros_like(voltage[name]),
                    "inhibitory": torch.zeros_like(voltage[name]),
                }
                for projection in self._incoming_projections[name]:
                    key = (projection.id, projection.polarity)
                    if not projection.enabled:
                        conductance[key].zero_()
                        continue
                    source_owner = projection.source.partition(".")[0]
                    if source_owner in populations:
                        if projection.delay_steps == 0:
                            source = new_spikes[source_owner]
                        else:
                            history = histories[source_owner]._values
                            source = history[-projection.delay_steps]
                    else:
                        source_t = t - projection.delay_steps
                        source = (
                            inputs[source_owner][source_t]
                            if source_t >= 0
                            else input_histories[source_owner][source_t]
                        )
                    if trace is None:
                        drive = source @ effective_weights[projection.parameter]
                    else:
                        with trace.detail(f"projection/{projection.id}"):
                            drive = source @ effective_weights[projection.parameter]
                    spec = projection_rows[projection.id]["synapse"]
                    if spec["kind"] == "custom_synapse":
                        definition = E.resolve("synapse", spec)
                        previous = synapse_states[projection.id]
                        context = E.SynapseContext(
                            previous, drive, self.plan.dt_ms, spec.get("config", {})
                        )
                        state = X.checked_state(
                            definition.function(context),
                            previous,
                            definition.name,
                            check_values=self._check_extension_values,
                        )
                        synapse_states[projection.id] = state
                        conductance[key] = state["value"]
                    elif self.projection_mode == "fast" and projection.decay == 0.0:
                        conductance[key] = drive
                    else:
                        conductance[key] = conductance[key] * projection.decay + drive
                    incoming[projection.polarity] += conductance[key]
                neuron = pop["neuron"]

                spike_function = self._spike_function

                if neuron["kind"] == "custom_neuron":
                    definition = E.resolve("neuron", neuron)
                    previous = neuron_states[name]
                    context = E.NeuronContext(
                        previous,
                        incoming["excitatory"],
                        incoming["inhibitory"],
                        self.plan.dt_ms,
                        neuron.get("config", {}),
                        spike_function,
                    )
                    response = definition.function(context)
                    if not isinstance(response, tuple) or len(response) != 2:
                        raise TypeError(
                            f"{definition.name} must return (state, spikes)"
                        )
                    state, spike_values = response
                    state = X.checked_state(
                        state,
                        previous,
                        definition.name,
                        check_values=self._check_extension_values,
                    )
                    X.checked_tensor(
                        spike_values,
                        shape=voltage[name].shape,
                        device=device,
                        dtype=parameter_dtype,
                        name=f"{definition.name}.spikes",
                        check_values=self._check_extension_values,
                    )
                    if self._check_extension_values and torch.any(
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
                    state, spike_values = self._neuron_steps[name](
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
                        self._neuron_steps[name](
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
                    pre_reset, post_reset, new_spikes[name] = self._neuron_steps[name](
                        voltage[name], incoming["excitatory"]
                    )
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
                voltage[name], new_spikes[name], refractory[name] = self._neuron_steps[
                    name
                ](
                    voltage[name],
                    refractory[name],
                    incoming["excitatory"],
                    incoming["inhibitory"],
                )
                new_spikes[name] = intervene(
                    name, new_spikes[name], completed_steps + t
                )
            for category, states, rows in (
                ("neuron", neuron_states, populations),
                ("synapse", synapse_states, projection_rows),
            ):
                for owner, state in states.items():
                    spec = rows[owner][category]
                    for key, value in state.items():
                        custom_state[f"{category}/{owner}/{key}"] = value
                    for port in E.state_units(category, spec):
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
                        for p in self.plan.projections
                    }
                )
                for category, states in (
                    ("neuron", neuron_states),
                    ("synapse", synapse_states),
                ):
                    for owner, state in states.items():
                        spec = (
                            populations[owner][category]
                            if category == "neuron"
                            else projection_rows[owner][category]
                        )
                        for port in E.state_units(category, spec):
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
            for projection_id in conductance_traces:
                projection = next(
                    p for p in self.plan.projections if p.id == projection_id
                )
                conductance_traces[projection_id].append(
                    conductance[(projection.id, projection.polarity)]
                )

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
                raise ValueError(
                    f"{op_id}: valid-time mask must have shape [time, batch]"
                )
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
                        source
                        for source in op["sources"]
                        if source not in signal_values
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
                    sources[0] @ self.weights[parameter]
                )
            elif kind in {"reduce_mean", "reduce_sum"}:
                mask_name = op.get("config", {}).get("mask")
                mask = signal_values.get(mask_name) if mask_name else None
                source_id = op["sources"][0]
                owner, _, port = source_id.partition(".")
                if (
                    "voltage_sampling" not in self.plan.graph
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
                        mask_seconds = online.tensors[duration_key].to(
                            sources[0].dtype
                        ) * (self.plan.dt_ms / 1000.0)
                    else:
                        mask_seconds = mask.to(
                            device=sources[0].device, dtype=sources[0].dtype
                        ).sum(dim=0) * (self.plan.dt_ms / 1000.0)
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
                config = op.get("config", {})
                definition = E.resolve(
                    "operation",
                    {
                        "definition": config["definition"],
                        "config": config.get("settings", {}),
                    },
                )
                parameters = {
                    name: self.parameter_map()[name]
                    for name in op.get("parameters", [])
                }
                value = definition.function(
                    tuple(sources), parameters, config.get("settings", {})
                )
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
        for output in self.plan.outputs:
            if output["id"] in streamed_outputs:
                continue
            value = signal_values[output["signal"]]
            recorder.output(output["id"], value, measurement)
            if retain_outputs:
                outputs[output["id"]] = value
        packed = (
            {
                row["id"]: signal_values[row["signal"]].detach().clone()
                for row in self.plan.observables
            }
            if diagnostics
            else {}
        )
        segments.boundary(completed_steps + steps, self.plan.dt_ms)
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
                voltages={
                    name: value.detach().clone() for name, value in voltage.items()
                },
                refractory={
                    name: value.detach().clone() for name, value in refractory.items()
                },
                conductances={
                    p.id: conductance[(p.id, p.polarity)].detach().clone()
                    for p in self.plan.projections
                    if projection_ports[p.id] == "conductance"
                },
                currents={
                    p.id: conductance[(p.id, p.polarity)].detach().clone()
                    for p in self.plan.projections
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
                    name: value.detach().clone()
                    for name, value in online.tensors.items()
                }
                if measurement is not None
                else {},
            )
        signal_axes = _signal_axes(self.plan.graph)
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
                    graph=self.plan.graph,
                    start_step=completed_steps,
                    steps_count=steps,
                    batch_size=batch,
                    batch_offset=batch_offset,
                    diagnostics=diagnostics,
                    retain_outputs=retain_outputs,
                    retain_parameters=retain_parameters,
                    retain_final_state=retain_final_state,
                    retain_runtime_state=retain_runtime_state,
                    projection_mode=self.projection_mode,
                    measurement=measurement,
                    recording=recording,
                ),
                "segment_policy": segments.identity,
                "segment_window": {"start_step": completed_steps, "steps_count": steps}
                if segments.identity
                else None,
                "decisions": segments.records,
                "resolved_interventions": resolved_interventions,
                "refractory": dict(self.plan.refractory),
                "autapses": self.autapse_metadata,
                "extension_validation": self.extension_validation,
                "projection_mode": self.projection_mode,
                "compile_mode": self.compile_mode,
                "surrogate": copy.deepcopy(self.surrogate_config),
                "result_snapshots": {
                    "parameters": retain_parameters,
                    "final_state": retain_final_state,
                    "runtime_state": retain_runtime_state,
                },
                "inference_interventions": intervention_request
                if interventions
                else None,
                **(
                    {"population_observations": observation_counts}
                    if observation_counts
                    else {}
                ),
            },
            decisions=segments.records,
            recorded_signals=recorder.finish(batch),
            _output_axes={
                row["id"]: signal_axes[row["signal"]] for row in self.plan.outputs
            },
            _diagnostic_axes={
                row["id"]: signal_axes[row["signal"]]
                for row in self.plan.observables
                if diagnostics
            },
            _timebase=(self.plan.dt_ms, completed_steps, steps),
            _batch_size=batch,
            model=self,
        )
        return result, signal_values
