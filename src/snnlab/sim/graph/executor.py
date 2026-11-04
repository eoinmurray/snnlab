"""The graph-native PyTorch forward executor and causal delay buffers."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any, Literal

import torch
from torch import nn

from snnlab import extensions as E
from snnlab.sim.graph import dynamics as M
from snnlab.sim.graph import extensions as X
from snnlab.sim.graph.blocks import signal_parameters
from snnlab.sim.graph.forward_pass import DelayBuffer as DelayBuffer
from snnlab.sim.graph.forward_pass import run_forward_pass
from snnlab.sim.graph.kernels import (
    constraint_kernel,
    neuron_kernel,
    operation_kernel,
    synapse_kernel,
)
from snnlab.sim.graph.plan import GraphPlan, PlannedProjection
from snnlab.sim.requests.result import ExecutionResult
from snnlab.sim.runtime.interventions import Intervention
from snnlab.sim.runtime.recording import (
    MeasurementWindow,
    RecordingSpec,
)
from snnlab.sim.runtime.resets_decisions import (
    DecisionSegments,
    ResetVoltage,
)
from snnlab.sim.runtime.state import (
    GraphRuntimeState,
)


def _apply_initial_sparsity(
    *, row, init, eligible_fan_in, exclude_autapses, runtime_shape, value
):
    zero_fraction = float(init.get("initial_zero_fraction", 0.0))
    if not 0 <= zero_fraction < 1:
        raise ValueError(
            f"{row['id']}: initial_zero_fraction must satisfy 0 <= fraction < 1"
        )
    zeroing = init.get("zeroing", "bernoulli")
    if zero_fraction and eligible_fan_in:
        if zeroing == "exact_k":
            if len(runtime_shape) != 2:
                raise ValueError(f"{row['id']}: exact_k zeroing requires a matrix")
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
            raise ValueError(f"{row['id']}: unsupported initializer zeroing {zeroing}")
    return value


def _projection_initialization_order(plan, pop_ids):
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

    return init_priority


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
        compile_mode: Literal["default"] | None = "default",
        compile_block_steps: int | Literal["auto"] = "auto",
        projection_mode: Literal["fast", "strict"] = "fast",
        precision: Literal["auto", "highest"] = "auto",
        cuda_graphs: Literal["auto", "off"] = "auto",
    ):
        super().__init__()
        if projection_mode not in {"fast", "strict"}:
            raise ValueError("projection_mode must be fast or strict")
        self.projection_mode = projection_mode
        if compile_mode not in {None, "default"}:
            raise ValueError("compile_mode must be None or default")
        self.compile_mode = compile_mode
        if compile_block_steps != "auto" and (
            type(compile_block_steps) is not int or not 1 <= compile_block_steps <= 64
        ):
            raise ValueError("compile_block_steps must be an integer from 1 to 64")
        from snnlab.sim.graph.replay import ForwardReplay

        if cuda_graphs not in {"auto", "off"}:
            raise ValueError("cuda_graphs must be auto or off")
        self.cuda_graphs = cuda_graphs
        self._forward_replay = ForwardReplay()
        self.requested_block_steps = compile_block_steps
        self.precision = precision
        if precision not in {"auto", "highest"}:
            raise ValueError("precision must be auto or highest")
        self._block_kernels = {}
        self._execution_paths = []
        if extension_validation not in ("strict", "structural"):
            raise ValueError("extension_validation must be strict or structural")
        self.extension_validation = extension_validation
        self._check_extension_values = extension_validation == "strict"
        self.plan = plan
        self._signal_parameters = signal_parameters(plan)
        self.seed = int(seed)
        self._configure_surrogate(
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            compile_mode=compile_mode,
        )

        self._neuron_definitions = {
            row["id"]: E.resolve("neuron", row["neuron"])
            for row in plan.populations
            if row["neuron"]["kind"] == "custom_neuron"
        }
        self._synapse_definitions = {
            row["id"]: E.resolve("synapse", row["synapse"])
            for row in plan.graph["projections"]
            if row["synapse"]["kind"] == "custom_synapse"
        }
        self._state_ports = {
            category: {
                row["id"]: tuple(E.state_units(category, row[category])) for row in rows
            }
            for category, rows in (
                ("neuron", plan.populations),
                ("synapse", plan.graph["projections"]),
            )
        }
        self._neuron_steps = {
            row["id"]: neuron_kernel(
                row,
                dt_ms=plan.dt_ms,
                refractory=plan.refractory.get(row["id"], {}),
                spike_function=self._spike_function,
                compile_mode=compile_mode,
                definition=self._neuron_definitions.get(row["id"]),
                # Strict custom surrogate diagnostics require host value reads.
                fullgraph=not (self.surrogate and self._check_extension_values),
            )
            for row in plan.populations
        }
        projection_rows = {row["id"]: row for row in plan.graph["projections"]}
        self._synapse_steps = {
            row.id: synapse_kernel(
                row,
                spec=projection_rows[row.id]["synapse"],
                dt_ms=plan.dt_ms,
                compile_mode=compile_mode,
                projection_mode=projection_mode,
                definition=self._synapse_definitions.get(row.id),
            )
            for row in plan.projections
        }
        self._operation_steps = {
            row["id"]: operation_kernel(row, compile_mode=compile_mode)
            for row in plan.graph.get("operations", [])
            if row["kind"] == "custom_operation"
        }
        self._incoming_projections = {
            row["id"]: tuple(
                p for p in plan.projections if p.target.partition(".")[0] == row["id"]
            )
            for row in plan.populations
        }
        torch.manual_seed(seed)
        rows = {row["id"]: row for row in plan.graph.get("parameters", [])}
        self._constraints = {
            name: constraint_kernel(
                row["constraint"], compile_mode=compile_mode, check_values=False
            )
            for name, row in rows.items()
            if row.get("constraint")
        }
        self._custom_constraint_names = {
            name: E.resolve("constraint", row["constraint"]).name
            for name, row in rows.items()
            if (row.get("constraint") or {}).get("kind") == "custom_constraint"
        }
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

        initialise = self._parameter_initializer(
            plan=plan, explicit_parameters=explicit_parameters
        )

        init_priority = _projection_initialization_order(plan=plan, pop_ids=pop_ids)

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
        with torch.no_grad():
            parameters = self.parameter_map()
            updates = {
                name: constraint(parameters[name])
                for name, constraint in self._constraints.items()
            }
            flags = {}
            for name in self._custom_constraint_names:
                value = updates[name]
                flags.setdefault(value.device, []).append(
                    (name, torch.isfinite(value).all())
                )
            for group in flags.values():
                if not torch.stack([flag for _, flag in group]).all():
                    for name, flag in group:
                        if not flag:
                            raise ValueError(
                                f"{self._custom_constraint_names[name]} tensor must be finite"
                            )
            for name, value in updates.items():
                parameters[name].copy_(value)
        self._enforce_autapses()

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        diagnostics: bool = True,
        runtime_state: GraphRuntimeState | None = None,
        initial_state=None,
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
            **({"initial_state": initial_state} if initial_state is not None else {}),
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

    @property
    def compile_block_steps(self):
        if self.requested_block_steps != "auto":
            return self.requested_block_steps
        parameter = next(self.parameters(), None)
        device = parameter.device.type if parameter is not None else "cpu"
        return {"cuda": 16, "mps": 4}.get(device, 1)

    def _forward(self, inputs, **kwargs):
        from snnlab.sim.graph.precision import execution_precision

        parameter = next(self.parameters(), None)
        automatic_inference = (
            torch.is_grad_enabled()
            and kwargs.get("runtime_state") is None
            and not any(
                value.requires_grad for value in (*inputs.values(), *self.parameters())
            )
            and all(
                E.get(row["category"], row["definition"]).capture_safe
                for row in E.required(self.plan.graph)
                if row["category"]
                in {"neuron", "synapse", "surrogate", "operation", "constraint"}
            )
            and (
                self.surrogate is None
                or E.resolve("surrogate", self.surrogate).capture_safe
            )
        )
        with (
            execution_precision(self.precision, parameter),
            torch.no_grad() if automatic_inference else nullcontext(),
        ):
            captured = (
                self._forward_replay.run(
                    self,
                    inputs,
                    kwargs,
                    lambda values: self._forward_impl(values, **kwargs),
                )
                if not torch.is_grad_enabled()
                else None
            )
            result, signals = (
                captured
                if captured is not None
                else self._forward_impl(inputs, **kwargs)
            )
        result.metrics["cuda_graphs"] = self._forward_replay.summary()
        result.metrics["precision"] = {
            "policy": self.precision,
            "autocast": "bfloat16" if self._amp_enabled(parameter) else "off",
            "matmul": "high"
            if self.precision == "auto"
            and parameter is not None
            and parameter.device.type == "cuda"
            else "highest"
            if parameter is not None and parameter.device.type == "cuda"
            else "caller",
        }
        if parameter is not None and parameter.device.type == "cuda":
            result.metrics["inference_retention"]["precision"] = dict(
                result.metrics["precision"]
            )
        return result, signals

    def _amp_enabled(self, parameter):
        from snnlab.sim.graph.precision import amp_enabled

        return amp_enabled(self.precision, parameter)

    def _forward_impl(  # noqa: C901
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        diagnostics: bool = True,
        runtime_state: GraphRuntimeState | None = None,
        initial_state=None,
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
        return run_forward_pass(
            self,
            inputs,
            diagnostics=diagnostics,
            runtime_state=runtime_state,
            initial_state=initial_state,
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

    def _configure_surrogate(self, *, surrogate, surrogate_slope, compile_mode):
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
            compile_mode=compile_mode,
        )

    def _parameter_initializer(self, plan, explicit_parameters):
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
            value = _apply_initial_sparsity(
                row=row,
                init=init,
                eligible_fan_in=eligible_fan_in,
                exclude_autapses=exclude_autapses,
                runtime_shape=runtime_shape,
                value=value,
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

        return initialise
