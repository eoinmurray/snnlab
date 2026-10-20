"""Build, simulation, inference and typed request dispatch."""

from __future__ import annotations

import copy
import math
import os
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch

from snnlab import extensions as E
from snnlab._shared import neurons as N
from snnlab.sim.artifacts.bundle import load_graph_bundle, load_training_recipe
from snnlab.sim.artifacts.integrity import _json_digest
from snnlab.sim.artifacts.snapshots import SnapshotSession
from snnlab.sim.graph import dynamics as M
from snnlab.sim.graph.executor import GraphExecutor
from snnlab.sim.graph.plan import plan_graph
from snnlab.sim.inputs.bindings import ResolvedDenseInputs
from snnlab.sim.inputs.provider import DatasetProvider, simulate_dataset
from snnlab.sim.inputs.resolution import _split_input_bindings, resolve_input_bindings
from snnlab.sim.requests.performance import _performance_report
from snnlab.sim.requests.result import ExecutionResult
from snnlab.sim.requests.spec import ExecutionSpec
from snnlab.sim.runtime.state import RUNTIME_STATE_SCHEMA, GraphRuntimeState
from snnlab.sim.training.checkpoints import (
    TRAINING_CHECKPOINT_SCHEMA,
    load_training_checkpoint,
)
from snnlab.sim.training.observations import evaluation_rng

INFERENCE_OVERRIDE_SCHEMA = "tools/snnsim.inference-overrides/v1"


def _projection_protocol(protocol, mode):
    result = dict(protocol)
    if result.get("projection_mode", mode) != mode:
        raise ValueError("execution protocol projection_mode does not match request")
    if mode == "fast":
        result["projection_mode"] = "fast"
    else:
        result.pop("projection_mode", None)
    return result


def _compilation_protocol(protocol, spec):
    """Bind resolved block policy for optimizer/checkpoint replay."""
    block_steps = spec.compile_block_steps
    if block_steps == "auto":
        block_steps = {"cuda": 16, "mps": 4}.get(
            torch.device(resolve_device(spec.device)).type, 1
        )
    result = dict(protocol)
    precision = (
        {"schema": "snnlab.cuda-precision/v1", "policy": "auto"}
        if torch.device(resolve_device(spec.device)).type == "cuda"
        and spec.precision == "auto"
        else None
    )
    if result.get("execution_precision", precision) != precision:
        raise ValueError("execution protocol precision does not match request")
    if precision is not None:
        result["execution_precision"] = precision
    else:
        result.pop("execution_precision", None)
    expected = (
        {
            "schema": "snnlab.network-blocks/v1",
            "compile_mode": spec.compile_mode,
            "block_steps": block_steps,
        }
        if block_steps > 1
        else None
    )
    if result.get("network_blocks", expected) != expected:
        raise ValueError("execution protocol network_blocks does not match request")
    if expected is not None:
        result["network_blocks"] = expected
    else:
        result.pop("network_blocks", None)
    return result


def _synchronize_device(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def resolve_device(requested: str | torch.device = "auto") -> str:
    """Resolve an explicit device or select the fastest available accelerator."""
    name = str(requested).lower()
    if name == "auto":
        forced = os.environ.get("SNNLAB_DEVICE")
        if forced:
            return resolve_device(forced)
        if torch.cuda.is_available():
            return "cuda"
        # Graph execution launches several small kernels from Python per timestep.
        # On the representative 800E/200I graph MPS is slower than CPU, so keep it
        # available explicitly without selecting it automatically.
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError(
            "MPS was requested but torch.backends.mps.is_available() is false"
        )
    if (
        name != "cpu"
        and name != "cuda"
        and name != "mps"
        and not name.startswith("cuda:")
    ):
        raise ValueError(
            f"device expected auto, cpu, cuda, cuda:N, or mps; got {requested!r}"
        )
    if name.startswith("cuda:") and not torch.cuda.is_available():
        raise ValueError(f"{name} was requested but torch.cuda.is_available() is false")
    return name


def build(spec: ExecutionSpec) -> ExecutionResult:
    graph = spec.graph
    training = spec.training
    if graph is None and spec.bundle is not None:
        manifest, graph = load_graph_bundle(spec.bundle)
        if spec.kind == "train" and training is None:
            training = load_training_recipe(spec.bundle, manifest, graph)
    if graph is None:
        raise ValueError("graph execution requires graph data or a bundle")
    if training is not None:
        E.restore(training)
    device = resolve_device(spec.device)
    started = time.perf_counter()
    trainable = (
        training.get("resolved_parameters", {}).get("trainable", []) if training else []
    )
    surrogate = (training or {}).get("surrogate") or {}
    surrogate_slope = float(surrogate.get("slope", M.SURROGATE_SLOPE))
    model = GraphExecutor(
        plan_graph(graph),
        seed=spec.seed,
        trainable_parameters=trainable,
        surrogate_slope=surrogate_slope,
        surrogate=surrogate,
        extension_validation=spec.extension_validation,
        compile_mode=spec.compile_mode,
        precision=spec.precision,
        cuda_graphs=spec.cuda_graphs,
        compile_block_steps=spec.compile_block_steps,
        projection_mode=spec.projection_mode,
    ).to(device)
    result = ExecutionResult(
        executor="graph",
        model=model,
        parameters=model.parameter_map() if spec.retain_parameters else {},
        metrics={
            "build_s": time.perf_counter() - started,
            "initialization": model.initialization_metadata,
            "refractory": dict(model.plan.refractory),
            "autapses": model.autapse_metadata,
            "extension_validation": model.extension_validation,
            "projection_mode": model.projection_mode,
            "compile_mode": model.compile_mode,
            "compile_block_steps": model.compile_block_steps,
            "precision": spec.precision,
            "surrogate": copy.deepcopy(model.surrogate_config),
            "compiled_populations": [
                row["id"]
                for row in model.plan.populations
                if model.compile_mode is not None
                and model._neuron_steps[row["id"]] is not None
            ],
            "compiled_projections": (
                list(model._synapse_steps) if model.compile_mode is not None else []
            ),
            "training_schema": training.get("schema") if training else None,
        },
    )

    result.metrics["performance"] = _performance_report(
        spec, model, result.metrics, device=device, executed=False
    )
    return result


def simulate(
    spec: ExecutionSpec, *, runtime_state: GraphRuntimeState | None = None
) -> ExecutionResult:
    if "inference_interventions" in spec.options:
        raise ValueError(
            "move options['inference_interventions'] to ExecutionSpec.interventions using typed intervention objects"
        )
    sources = _split_input_bindings(spec.input_bindings)
    overrides = dict(spec.options.get("inference_overrides", {}))
    interventions = tuple(spec.interventions)
    allowed_overrides = {
        "duration_ms",
        "input_rate_hz",
        "projection_scales",
        "timestep_ms",
    }
    unknown_overrides = sorted(set(overrides) - allowed_overrides)
    if unknown_overrides:
        raise ValueError(f"unsupported inference overrides: {unknown_overrides}")
    source_graph: Mapping[str, Any] | None = None
    source_graph_digest: str | None = None
    source_dt_ms: float | None = None
    build_spec = spec
    if "timestep_ms" in overrides:
        timestep_ms = float(overrides["timestep_ms"])
        if not math.isfinite(timestep_ms) or timestep_ms <= 0:
            raise ValueError("inference timestep must be finite and positive")
        if runtime_state is not None or spec.runtime_state is not None:
            raise ValueError(
                "inference timestep recompilation cannot convert runtime state"
            )
        if not sources.poisson or sources.dense or sources.events:
            raise ValueError(
                "inference timestep recompilation requires resampleable Poisson input bindings"
            )
        if spec.graph is not None:
            source_graph = spec.graph
        elif spec.bundle is not None:
            _, source_graph = load_graph_bundle(spec.bundle)
        else:
            raise ValueError("graph execution requires graph data or a bundle")
        source_graph_digest = _json_digest(source_graph)
        source_dt_ms = float(source_graph["timebase"]["dt"]["value"])
        recompiled_graph = copy.deepcopy(source_graph)
        recompiled_graph = N.migrate_refractory(recompiled_graph)
        recompiled_graph["timebase"]["dt"] = {
            "value": timestep_ms,
            "unit": "ms",
        }
        build_spec = replace(spec, graph=recompiled_graph, bundle=None)
    built = build(build_spec)
    assert isinstance(built.model, GraphExecutor)
    device = resolve_device(spec.device)
    poisson_bindings = sources.poisson
    if (
        "duration_ms" in overrides
        or "input_rate_hz" in overrides
        or "timestep_ms" in overrides
    ):
        if not poisson_bindings or sources.dense or sources.events:
            raise ValueError(
                "duration and input-rate inference overrides require Poisson input bindings"
            )
        dt_ms = built.model.plan.dt_ms
        duration_ms = float(
            overrides.get(
                "duration_ms",
                poisson_bindings[0].steps_count * (source_dt_ms or dt_ms),
            )
        )
        raw_steps = duration_ms / dt_ms
        if duration_ms <= 0 or not math.isclose(
            raw_steps, round(raw_steps), abs_tol=1e-9
        ):
            raise ValueError(
                f"inference duration {duration_ms} ms must be a positive integer multiple of dt={dt_ms} ms"
            )
        rate = overrides.get("input_rate_hz")
        if rate is not None and (not math.isfinite(float(rate)) or float(rate) < 0):
            raise ValueError("inference input rate must be finite and non-negative")
        poisson_bindings = tuple(
            replace(
                binding,
                steps_count=int(round(raw_steps)),
                rates_hz=(float(rate),) if rate is not None else binding.rates_hz,
                categorical=False if rate is not None else binding.categorical,
            )
            for binding in poisson_bindings
        )
    checkpoint_provenance = None
    if spec.checkpoint:
        checkpoint_path = Path(spec.checkpoint)
        if checkpoint_path.is_dir():
            checkpoint = load_training_checkpoint(checkpoint_path, device=device)
            graph_digest = source_graph_digest or _json_digest(built.model.plan.graph)
            if checkpoint.graph_digest != graph_digest:
                raise ValueError(
                    f"inference checkpoint graph digest expected {graph_digest}, got {checkpoint.graph_digest}"
                )
            parameter_map = built.model.parameter_map()
            if set(checkpoint.parameters) != set(parameter_map):
                raise ValueError(
                    "inference checkpoint parameter names do not match graph"
                )
            with torch.no_grad():
                for name, parameter in parameter_map.items():
                    restored = checkpoint.parameters[name]
                    if (
                        restored.shape != parameter.shape
                        or restored.dtype != parameter.dtype
                    ):
                        raise ValueError(
                            f"inference checkpoint parameter {name} expected shape={list(parameter.shape)} dtype={parameter.dtype}, "
                            f"got shape={list(restored.shape)} dtype={restored.dtype}"
                        )
                    parameter.copy_(restored)
            built.model._enforce_autapses()
            checkpoint_provenance = {
                "format": TRAINING_CHECKPOINT_SCHEMA,
                "path": str(checkpoint_path),
                "graph_digest": checkpoint.graph_digest,
                "training_digest": checkpoint.training_digest,
                "completed_updates": checkpoint.completed_updates,
                "selected_loss": checkpoint.selected_loss,
            }
        else:
            state_dict = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
            checkpoint_provenance = {
                "format": "graph_torch_state_dict",
                "path": str(checkpoint_path),
            }
            built.model.load_state_dict(state_dict)
    scales = dict(overrides.get("projection_scales", {}))
    if scales:
        projection_parameters = {
            row["id"]: row["parameters"][0]
            for row in built.model.plan.graph.get("projections", [])
        }
        unknown = sorted(set(scales) - set(projection_parameters))
        if unknown:
            raise ValueError(
                f"inference projection scales target unknown projections: {unknown}"
            )
        with torch.no_grad():
            parameters = built.model.parameter_map()
            for projection_id, factor in sorted(scales.items()):
                factor = float(factor)
                if not math.isfinite(factor) or factor < 0:
                    raise ValueError(
                        f"inference projection scale {projection_id} must be finite and non-negative"
                    )
                parameters[projection_parameters[projection_id]].mul_(factor)
    if spec.timing_mode == "synchronized":
        _synchronize_device(device)
    tracemalloc.start()
    started = time.perf_counter()
    if sources.dataset is not None:
        with evaluation_rng(spec.seed):
            resolved_inputs = DatasetProvider(
                built.model.plan.graph,
                sources.dataset,
                device=device,
                execution_seed=spec.seed,
                protocol=spec.protocol,
            )
    else:
        resolved_inputs = resolve_input_bindings(
            built.model.plan.graph,
            dense_bindings=sources.dense,
            event_bindings=sources.events,
            poisson_bindings=poisson_bindings,
            device=device,
            seed=spec.seed,
            protocol=spec.protocol,
        )
    protocol = _projection_protocol(resolved_inputs.protocol, spec.projection_mode)
    protocol = _compilation_protocol(protocol, spec)
    resolved_inputs = (
        resolved_inputs.with_protocol(protocol)
        if isinstance(resolved_inputs, DatasetProvider)
        else ResolvedDenseInputs(resolved_inputs.tensors, protocol)
    )
    snapshot_runtime_state = (
        runtime_state if runtime_state is not None else spec.runtime_state
    )
    snapshots = SnapshotSession(
        spec,
        built.model,
        kind="simulate",
        protocol=protocol,
        device_resolved=device,
        checkpoint=checkpoint_provenance,
        inference_overrides=overrides,
        effective_poisson_bindings=poisson_bindings,
        runtime_state_signature=snapshot_runtime_state.signature
        if snapshot_runtime_state is not None
        else None,
        runtime_state_completed_steps=snapshot_runtime_state.completed_steps
        if snapshot_runtime_state is not None
        else None,
        initialization=built.model.initialization_metadata,
    )
    snapshots.capture("start")
    if isinstance(resolved_inputs, DatasetProvider):
        if spec.resets or spec.decisions is not None:
            raise ValueError(
                "reset/decision streams require direct dense, event or Poisson bindings; dataset batches are independent trajectories"
            )
        result = simulate_dataset(
            built.model,
            resolved_inputs,
            batch_size=spec.batch_size
            if spec.batch_size is not None
            else min(32, resolved_inputs.sample_count),
            diagnostics=spec.diagnostics,
            interventions=interventions,
            measurement=spec.measurement,
            recording=spec.recording,
            retain_outputs=spec.retain_outputs,
            retain_parameters=spec.retain_parameters,
            retain_final_state=spec.retain_final_state,
            retain_runtime_state=spec.retain_runtime_state,
            prefetch=spec.dataset_prefetch,
            runtime_state=runtime_state
            if runtime_state is not None
            else spec.runtime_state,
        )
    else:
        result = built.model(
            resolved_inputs.tensors,
            diagnostics=spec.diagnostics,
            runtime_state=runtime_state
            if runtime_state is not None
            else spec.runtime_state,
            interventions=interventions,
            resets=spec.resets,
            decisions=spec.decisions,
            measurement=spec.measurement,
            recording=spec.recording,
            retain_outputs=spec.retain_outputs,
            retain_parameters=spec.retain_parameters,
            retain_final_state=spec.retain_final_state,
            retain_runtime_state=spec.retain_runtime_state,
        )
    if spec.timing_mode == "synchronized":
        _synchronize_device(device)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result.metrics.pop("resolved_interventions", None)
    intervention_request = result.metrics.get("inference_interventions")
    result.metrics.update(
        {
            "simulate_s"
            if spec.timing_mode == "synchronized"
            else "simulate_enqueue_s": elapsed,
            "timing_mode": spec.timing_mode,
            "timing_scope": "input_resolution_and_execution_excluding_build_and_checkpoint_loading",
            "dataset_prefetch": spec.dataset_prefetch,
            "peak_python_bytes": peak,
            "device": device,
            "diagnostics": spec.diagnostics,
            "execution_protocol": resolved_inputs.protocol,
            "checkpoint": checkpoint_provenance,
            "inference_overrides": {
                "schema": INFERENCE_OVERRIDE_SCHEMA,
                "requested": overrides,
                "resolved": {
                    "duration_ms": resolved_inputs.protocol["timing"]["duration_ms"],
                    "timestep_ms": built.model.plan.dt_ms,
                    "projection_scales": scales,
                    **(
                        {"input_rate_hz": float(overrides["input_rate_hz"])}
                        if "input_rate_hz" in overrides
                        else {}
                    ),
                },
            }
            if overrides
            else None,
            "inference_interventions": intervention_request,
            "source_graph_digest": source_graph_digest,
            "effective_graph_digest": _json_digest(built.model.plan.graph),
            **built.metrics,
        }
    )
    if result.runtime_state is not None:
        result.metrics.update(
            {
                "runtime_state_schema": RUNTIME_STATE_SCHEMA,
                "runtime_state_signature": result.runtime_state.signature,
                "completed_steps": result.runtime_state.completed_steps,
            }
        )
    result.metrics["performance"] = _performance_report(
        spec,
        built.model,
        result.metrics,
        device=device,
        executed=True,
        dataset=isinstance(resolved_inputs, DatasetProvider),
    )
    snapshots.capture("end")
    result.snapshots = snapshots.snapshots
    return result


def execute_request(spec: ExecutionSpec) -> ExecutionResult:
    """Dispatch one graph execution request."""
    from snnlab.sim.training.trainer import train

    handlers = {"build": build, "simulate": simulate, "train": train}
    try:
        handler = handlers[spec.kind]
    except KeyError:
        raise ValueError(f"unsupported execution request: {spec.kind}") from None
    return handler(spec)
