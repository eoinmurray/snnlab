"""Native training orchestration, evaluation, reporting and checkpoint selection."""

from __future__ import annotations

import copy
import hashlib
import math
import time
from contextlib import nullcontext
from dataclasses import replace
from numbers import Integral
from typing import Any, Mapping

import torch

from snnlab import extensions as E
from snnlab.sim.artifacts.bundle import load_graph_bundle, load_training_recipe
from snnlab.sim.artifacts.integrity import _json_digest
from snnlab.sim.artifacts.snapshots import SnapshotSession
from snnlab.sim.diagnostics.tracing import TraceSession, current_trace
from snnlab.sim.graph import extensions as X
from snnlab.sim.graph.executor import GraphExecutor
from snnlab.sim.inputs.bindings import (
    DatasetSnapshotBinding,
    ResolvedDenseInputs,
    TargetArrayBinding,
)
from snnlab.sim.inputs.provider import DatasetProvider, iter_batches
from snnlab.sim.inputs.provider import sample_count as dataset_sample_count
from snnlab.sim.inputs.provider import steps_count as dataset_steps_count
from snnlab.sim.inputs.resolution import (
    _split_input_bindings,
    resolve_input_bindings,
    resolve_target_array_bindings,
)
from snnlab.sim.requests.handlers import (
    _compilation_protocol,
    _projection_protocol,
    _synchronize_device,
    build,
    resolve_device,
)
from snnlab.sim.requests.performance import _performance_report
from snnlab.sim.requests.result import ExecutionResult
from snnlab.sim.requests.spec import ExecutionSpec, TestSpec, ValidationSpec
from snnlab.sim.training.checkpoints import (
    TRAINING_CHECKPOINT_SCHEMA,
    TrainingCheckpoint,
    capture_training_rng_state,
    load_training_checkpoint,
    restore_training_rng_state,
    save_training_checkpoint,
)
from snnlab.sim.training.observations import (
    ActivityMeasurements,
    EpochObservations,
    GradientMeasurements,
    ObservationProbe,
    evaluation_rng,
    finite_norm,
    validate_observation_state,
)
from snnlab.sim.training.policies import (
    Batching,
    EpochReport,
    FrozenValues,
    TrainingMetrics,
)
from snnlab.sim.training.replay import TrainingReplay
from snnlab.sim.training.selection import CheckpointSelection, SelectionMetric


def _host_metrics(values):
    """Materialise scalar reporting tensors with one transfer per device."""
    result = {}
    groups = {}
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            groups.setdefault(
                (value.device, value.dtype if value.device.type == "mps" else None), []
            ).append((name, value.detach()))
        else:
            result[name] = float(value)
    for group in groups.values():
        if group[0][1].device.type == "mps":
            host = torch.stack([value for _, value in group]).cpu().double().tolist()
        else:
            host = torch.stack([value.double() for _, value in group]).cpu().tolist()
        result.update((name, value) for (name, _), value in zip(group, host))
    return result


def _reporting_scalar(value):
    """CUDA/CPU keep double accumulators; MPS lacks float64 device tensors."""
    value = value.detach()
    if value.device.type == "mps":
        value = value.cpu()
    return value.double()


class _TrainingTiming:
    """Optional, disjoint wall phases; fences occur at phase/epoch boundaries."""

    def __init__(self, spec):
        self.enabled = spec.training_timing
        self.mode = spec.timing_mode
        self.device = resolve_device(spec.device) if self.enabled else None
        self.totals = {}
        self.epochs = {}
        self.phase = None
        self.epoch = None
        self.started = None
        if self.enabled:
            self.switch("setup")

    def switch(self, phase, epoch=None):
        if not self.enabled:
            return
        if self.mode == "synchronized":
            _synchronize_device(self.device)
        now = time.perf_counter()
        if self.phase is not None:
            seconds = now - self.started
            self.totals[self.phase] = self.totals.get(self.phase, 0.0) + seconds
            if self.epoch is not None:
                row = self.epochs.setdefault(self.epoch, {})
                row[self.phase] = row.get(self.phase, 0.0) + seconds
        self.phase, self.epoch, self.started = phase, epoch, now

    def finish(self):
        self.switch(None)
        phases = (
            "setup",
            "training",
            "train_evaluation",
            "validation",
            "test",
            "checkpoint",
            "reporting",
        )
        return {
            "mode": self.mode,
            "scope": "complete train call after argument validation; excludes caller preparation and persistence unless save_*_checkpoint is requested",
            "training_scope": "minibatch preparation, forward/loss/backward, optimizer/constraints, update bookkeeping and in-loop checkpoint selection; excludes epoch evaluations",
            "total_s": sum(self.totals.values()),
            **{f"{phase}_s": self.totals.get(phase, 0.0) for phase in phases},
            "epochs": [
                {
                    "epoch": epoch,
                    **{f"{phase}_s": row.get(phase, 0.0) for phase in phases},
                }
                for epoch, row in sorted(self.epochs.items())
            ],
        }


def train(spec: ExecutionSpec) -> ExecutionResult:
    """Execute native training, optionally collecting bounded diagnostic traces."""
    # Validate split declarations before build/performance reporting inspects
    # their bindings; preserve the documented public TypeError on invalid input.
    if spec.validation is not None and not isinstance(spec.validation, ValidationSpec):
        raise TypeError("validation must be a ValidationSpec")
    if spec.test is not None and not isinstance(spec.test, TestSpec):
        raise TypeError("test must be a TestSpec")
    trace = current_trace()
    if trace is not None:
        if getattr(trace, "training_started", False):
            raise ValueError("nested training during tracing is not supported")
        trace.training_started = True
    if spec.tracing is not None and spec.tracing.mode != "off":
        if trace is not None:
            raise ValueError("nested traced training is not supported")
        with TraceSession(spec.tracing, resolve_device(spec.device)) as session:
            result = train(replace(spec, tracing=None))
        result.metrics["tracing"] = session.summary
        return result
    if (
        spec.executor == "graph"
        and isinstance(spec.epochs, Integral)
        and not isinstance(spec.epochs, bool)
        and spec.epochs == 0
        and any(
            isinstance(binding, DatasetSnapshotBinding)
            for binding in spec.input_bindings
        )
    ):
        spec = replace(spec, epochs=1)
    if (
        spec.measurement is not None
        or spec.recording is not None
        or not spec.retain_outputs
    ):
        raise ValueError(
            "measurement/recording retention controls require graph inference"
        )
    if spec.resets or spec.decisions is not None:
        raise ValueError(
            "resets and decisions are supported only for graph simulation/inference"
        )
    if spec.interventions or "inference_interventions" in spec.options:
        raise ValueError(
            "interventions are supported only for graph simulation/inference"
        )
    if spec.observations is not None and (spec.executor != "graph" or spec.epochs <= 0):
        raise ValueError(
            "epoch observations require graph training with positive epochs"
        )
    if spec.observations is not None and not isinstance(
        spec.observations, EpochObservations
    ):
        raise TypeError("observations must be an EpochObservations specification")
    moved_options = set(spec.options) & {
        "epochs",
        "batch_size",
        "shuffle",
        "updates",
        "save_final_checkpoint",
        "save_selected_checkpoint",
        "checkpoint_selection",
        "metrics",
        "batching",
        "on_epoch_end",
        "test",
    }
    if moved_options:
        raise ValueError(
            f"training settings must be ExecutionSpec fields, not options: {sorted(moved_options)}"
        )
    for name, minimum in (("epochs", 0), ("batch_size", 1), ("updates", 1)):
        value = getattr(spec, name)
        if value is None and name != "epochs":
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < minimum
        ):
            raise ValueError(f"training {name} must be an integer >= {minimum}")
    if not isinstance(spec.shuffle, bool):
        raise TypeError("training shuffle must be boolean")
    if spec.metrics is None and spec.epochs > 0:
        spec = replace(spec, metrics=TrainingMetrics())
    metric_policy = spec.metrics or TrainingMetrics()
    if not isinstance(metric_policy, TrainingMetrics):
        raise TypeError("metrics must be TrainingMetrics")
    # Final testing is independent of the authenticated training/resume policy.
    training_metric_contract = metric_policy.to_dict()
    training_metric_contract.pop("test")
    if spec.batching is not None and not isinstance(spec.batching, Batching):
        raise TypeError("batching must be Batching")
    if spec.batching is not None and spec.shuffle and spec.batching.order != "shuffle":
        raise ValueError("shuffle=True conflicts with batching order")
    if spec.on_epoch_end is not None and not callable(spec.on_epoch_end):
        raise TypeError("on_epoch_end must be callable")
    if spec.epochs <= 0 and (
        spec.metrics is not None
        or spec.batching is not None
        or spec.on_epoch_end is not None
    ):
        raise ValueError(
            "training policies and epoch callbacks require positive epochs"
        )
    persist_reporting = (
        spec.metrics is not None
        or spec.batching is not None
        or spec.on_epoch_end is not None
    )
    timing = _TrainingTiming(spec)
    sources = _split_input_bindings(spec.input_bindings)
    built = build(spec)
    assert isinstance(built.model, GraphExecutor)
    model = built.model
    trace = current_trace()
    if trace is not None:
        trace.bind(model)
    graph = model.plan.graph
    training = spec.training
    if training is None and spec.bundle is not None:
        manifest, _ = load_graph_bundle(spec.bundle)
        training = load_training_recipe(spec.bundle, manifest, graph)
    if training is None:
        raise ValueError("graph training requires a training recipe or training bundle")
    if (
        not spec.targets
        and not spec.target_bindings
        and not (sources.dataset and sources.dataset.target_id)
    ):
        raise ValueError("graph training requires external target tensors")
    device = resolve_device(spec.device)
    dataset_epochs = spec.epochs
    dataset_mode = dataset_epochs > 0
    if spec.validation is not None:
        if not isinstance(spec.validation, ValidationSpec):
            raise TypeError("validation must be a ValidationSpec")
        if not dataset_mode:
            raise ValueError("validation requires training with positive epochs")
    if spec.test is not None:
        if not isinstance(spec.test, TestSpec):
            raise TypeError("test must be a TestSpec")
        if not dataset_mode:
            raise ValueError("test requires training with positive epochs")

    def resolve_data(
        data: ExecutionSpec | ValidationSpec | ObservationProbe,
        *,
        observation_only=False,
    ):
        bindings = _split_input_bindings(data.input_bindings)
        dataset_targets: tuple[TargetArrayBinding, ...] = ()
        if bindings.dataset is not None:
            if not observation_only and (data.target_bindings or data.targets):
                raise ValueError(
                    "dataset snapshot binding cannot be combined with other input or target bindings"
                )
            with evaluation_rng(data.seed if observation_only else spec.seed):
                inputs = DatasetProvider(
                    graph,
                    bindings.dataset,
                    device=device,
                    execution_seed=data.seed if observation_only else spec.seed,
                    protocol=data.protocol,
                    encoding_seeds=getattr(data, "encoding_seeds", ()),
                )
            dataset_targets = inputs.targets
        else:
            if getattr(data, "encoding_seeds", ()):
                raise ValueError("encoding_seeds requires a dataset binding")
            inputs = resolve_input_bindings(
                graph,
                dense_bindings=bindings.dense,
                event_bindings=bindings.events,
                poisson_bindings=bindings.poisson,
                device=device,
                seed=data.seed if observation_only else spec.seed,
                protocol=data.protocol,
            )
        sample_count = dataset_sample_count(inputs)
        if not isinstance(inputs, DatasetProvider) and any(
            value.shape[1] != sample_count for value in inputs.tensors.values()
        ):
            raise ValueError("graph training inputs must share one dataset sample axis")
        if observation_only:
            return inputs, {}, ()
        targets, rows = resolve_target_array_bindings(
            training,
            bindings=dataset_targets or data.target_bindings,
            targets=data.targets,
            sample_count=sample_count,
            device=device,
        )
        protocol = _projection_protocol(inputs.protocol, spec.projection_mode)
        protocol = _compilation_protocol(protocol, spec)
        inputs = (
            inputs.with_protocol(protocol)
            if isinstance(inputs, DatasetProvider)
            else ResolvedDenseInputs(inputs.tensors, protocol)
        )
        return inputs, targets, rows

    resolved_inputs, resolved_targets, target_rows = resolve_data(spec)
    dataset_size = dataset_sample_count(resolved_inputs)
    validation_data = (
        resolve_data(spec.validation) if spec.validation is not None else None
    )
    selection = spec.checkpoint_selection or (
        CheckpointSelection(metric=SelectionMetric("loss"))
        if validation_data is not None and metric_policy.validation == "epoch"
        else CheckpointSelection.legacy_training_batch_loss()
        if metric_policy.update_history
        else CheckpointSelection.disabled()
    )
    if not isinstance(selection, CheckpointSelection):
        raise TypeError("checkpoint_selection must be a CheckpointSelection")
    selection.validate(
        training, epochs=dataset_epochs, has_validation=validation_data is not None
    )
    epoch_selection = selection.mode == "epoch_metrics"
    update_selection = selection.mode == "legacy_training_batch_loss"
    if selection.mode == "none" and spec.save_selected_checkpoint is not None:
        raise ValueError("save_selected_checkpoint requires checkpoint selection")
    if epoch_selection:
        if (selection.split == "train" and metric_policy.train != "evaluate") or (
            selection.split == "validation" and metric_policy.validation != "epoch"
        ):
            raise ValueError(
                "checkpoint selection requires enabled fixed-weight split evaluation"
            )
        if selection.include_initial and selection.split not in metric_policy.initial:
            raise ValueError(
                "initial checkpoint selection requires initial split metrics"
            )
    if spec.observations is not None and (
        spec.observations.population_rates or spec.observations.output_activity
    ):
        if metric_policy.train != "evaluate" or (
            validation_data is not None and metric_policy.validation != "epoch"
        ):
            raise ValueError(
                "split activity observations require enabled split evaluation; use separate probes for online training"
            )
    batch_size = (
        spec.batch_size
        if spec.batch_size is not None
        else (
            min(32, dataset_size)
            if isinstance(resolved_inputs, DatasetProvider)
            else dataset_size
        )
    )
    if batch_size <= 0:
        raise ValueError("graph training batch size must be positive")
    observation_spec = spec.observations
    resolved_probes = {}
    if observation_spec is not None:
        observation_spec.validate(graph, model.parameter_map())
        for name, probe in sorted(observation_spec.probes.items()):
            with evaluation_rng(probe.seed):
                resolved_probes[name] = resolve_data(probe, observation_only=True)[0]
    for data in (
        resolved_inputs,
        *((validation_data[0],) if validation_data is not None else ()),
        *resolved_probes.values(),
    ):
        if isinstance(data, DatasetProvider):
            data.protocol["dataset"]["batch_size"] = batch_size
    observation_contract = (
        None
        if observation_spec is None
        else {
            "schema": "snnlab.epoch-observations/v1",
            "population_rates": list(observation_spec.population_rates),
            "parameter_norms": list(observation_spec.parameter_norms),
            "output_activity": list(observation_spec.output_activity),
            "gradient_norms": list(observation_spec.gradient_norms),
            "evaluation_seed": spec.seed,
            "batch_size": batch_size,
            "validation": None
            if validation_data is None
            else {
                "inputs": validation_data[0].protocol,
                "targets": validation_data[2],
            },
            "probes": {
                name: {
                    "seed": observation_spec.probes[name].seed,
                    "split": observation_spec.probes[name].split,
                    "inputs": data.protocol,
                }
                for name, data in resolved_probes.items()
            },
        }
    )
    batching = spec.batching or Batching(
        order="shuffle" if spec.shuffle else "sequential"
    )
    batch_sizes = batching.sizes(dataset_size, batch_size)
    batches_per_epoch = len(batch_sizes)
    stratify_labels = None
    if batching.order == "stratified":
        if batching.stratify_by not in resolved_targets:
            raise ValueError("stratify_by must name a training target")
        stratify_labels = resolved_targets[batching.stratify_by].detach().cpu()
        batching.indices(
            dataset_size, batch_size, seed=spec.seed, labels=stratify_labels
        )
    shuffle = batching.order == "shuffle"
    protocol = {**resolved_inputs.protocol, "targets": target_rows}
    if dataset_mode:
        protocol = {
            **protocol,
            "dataset": {
                **protocol["dataset"],
                "sample_cap": dataset_size,
                "batch_size": batch_size,
                "shuffle": shuffle,
            },
            "training_iteration": {
                "schema": "tools/snnsim.dataset-iteration/v1",
                "epochs": dataset_epochs,
                "drop_last": batching.remainder == "drop",
                "order_seed": int(spec.seed),
            },
        }
    if spec.metrics is not None:
        protocol["training_metrics"] = training_metric_contract
    if spec.batching is not None:
        protocol["training_iteration"]["batching"] = {
            **batching.to_dict(),
            "algorithm": "fractional_class_positions_seeded_ties/v1"
            if batching.order == "stratified"
            else batching.order,
            "batches_per_epoch": batches_per_epoch,
            "batch_sizes": batch_sizes,
            "examples_per_epoch": sum(batch_sizes),
            "dropped_per_epoch": dataset_size - sum(batch_sizes),
            "stratification_labels_sha256": hashlib.sha256(
                stratify_labels.contiguous().numpy().tobytes()
            ).hexdigest()
            if stratify_labels is not None
            else None,
        }
    resolved_inputs = (
        resolved_inputs.with_protocol(protocol)
        if isinstance(resolved_inputs, DatasetProvider)
        else ResolvedDenseInputs(resolved_inputs.tensors, protocol)
    )
    parameter_map = model.parameter_map()
    graph_digest = training["graph_digest"]
    training_digest = _json_digest(training)
    resumed: TrainingCheckpoint | None = None
    completed_updates = 0
    data_state: dict[str, Any] = {"epoch": 0, "batch": 0} if dataset_mode else {}
    if spec.checkpoint is not None:
        resumed = load_training_checkpoint(spec.checkpoint, device=device)
        if resumed.graph_digest != graph_digest:
            raise ValueError(
                f"training checkpoint graph digest expected {graph_digest}, got {resumed.graph_digest}"
            )
        if resumed.training_digest != training_digest:
            raise ValueError(
                f"training checkpoint recipe digest expected {training_digest}, got {resumed.training_digest}"
            )
        if resumed.execution_protocol != resolved_inputs.protocol:
            raise ValueError(
                "training checkpoint execution protocol does not match request"
            )
        if resumed.initialization != built.metrics["initialization"]:
            raise ValueError(
                "training checkpoint initializer metadata does not match graph build"
            )
        if set(resumed.parameters) != set(parameter_map):
            missing = sorted(set(parameter_map) - set(resumed.parameters))
            extra = sorted(set(resumed.parameters) - set(parameter_map))
            raise ValueError(
                f"training checkpoint parameter names mismatch; missing={missing}, extra={extra}"
            )
        with torch.no_grad():
            for name, parameter in parameter_map.items():
                restored = resumed.parameters[name]
                if (
                    restored.shape != parameter.shape
                    or restored.dtype != parameter.dtype
                ):
                    raise ValueError(
                        f"training checkpoint parameter {name} expected shape={list(parameter.shape)} dtype={parameter.dtype}, "
                        f"got shape={list(restored.shape)} dtype={restored.dtype}"
                    )
                parameter.copy_(restored)
        model._enforce_autapses()
        completed_updates = resumed.completed_updates
        if dataset_mode:
            if set(resumed.data_state) != {"epoch", "batch"}:
                raise ValueError(
                    "dataset training checkpoint requires epoch and batch data-order state"
                )
            data_state = {
                "epoch": int(resumed.data_state["epoch"]),
                "batch": int(resumed.data_state["batch"]),
            }
            epoch = data_state["epoch"]
            batch = data_state["batch"]
            if (
                epoch < 0
                or epoch > dataset_epochs
                or batch < 0
                or batch >= batches_per_epoch
                or (epoch == dataset_epochs and batch != 0)
            ):
                raise ValueError(
                    f"dataset training checkpoint has invalid data-order state {data_state}"
                )
        elif resumed.data_state:
            raise ValueError(
                "single-batch training cannot resume a dataset-iteration checkpoint"
            )
    groups = []
    for group in sorted(
        training.get("parameter_groups", []), key=lambda row: row["id"]
    ):
        if group.get("frozen"):
            continue
        groups.append(
            {
                "params": [parameter_map[name] for name in sorted(group["parameters"])],
                "lr": float(group["lr"]),
                "name": group["id"],
            }
        )
    if not groups:
        raise ValueError(
            "graph training requires at least one trainable parameter group"
        )
    from snnlab.sim.graph.precision import (
        automatic_fused_optimizer,
        execution_precision,
    )

    optimizer_spec = training.get("optimizer", {})
    if optimizer_spec.get("kind") == "custom_optimizer":
        definition = E.resolve("optimizer", optimizer_spec)
        optimizer = definition.function(groups, optimizer_spec.get("config", {}))
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"{definition.name} must return torch.optim.Optimizer")
    elif optimizer_spec.get("kind") == "adamw":
        optimizer = torch.optim.AdamW(groups, **dict(optimizer_spec.get("config", {})))
    else:
        raise ValueError(
            f"graph training unsupported optimizer {optimizer_spec.get('kind')}"
        )
    if spec.fused_optimizer == "auto":
        automatic_fused_optimizer(optimizer)

    fused_optimizer = any(group.get("fused", False) for group in optimizer.param_groups)

    if resumed is not None:
        trainable_names = {
            name for name, parameter in parameter_map.items() if parameter.requires_grad
        }
        if set(resumed.optimizer_state) != trainable_names:
            missing = sorted(trainable_names - set(resumed.optimizer_state))
            extra = sorted(set(resumed.optimizer_state) - trainable_names)
            raise ValueError(
                f"training checkpoint optimizer parameter names mismatch; missing={missing}, extra={extra}"
            )
        for name in sorted(trainable_names):
            optimizer.state[parameter_map[name]] = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in resumed.optimizer_state[name].items()
            }
        restore_training_rng_state(resumed, device)
    output_ids = {row["signal"]: row["id"] for row in graph.get("outputs", [])}
    updates_option = spec.updates
    updates = int(
        updates_option
        if updates_option is not None
        else (dataset_epochs * batches_per_epoch if dataset_mode else 1)
    )
    if updates <= 0:
        raise ValueError("graph training updates must be positive")
    observation_saved = resumed.observation_state if resumed is not None else None
    if (
        resumed is not None
        and observation_contract is not None
        and observation_saved is None
    ):
        raise ValueError(
            "checkpoint has no epoch observation history; cannot reconstruct missing observations"
        )
    if observation_saved is not None:
        if observation_contract is None:
            raise ValueError(
                "checkpoint epoch observation configuration does not match request"
            )
        validate_observation_state(
            observation_saved, observation_contract, data_state, completed_updates
        )
    epoch_history = (
        copy.deepcopy(observation_saved["history"])
        if observation_saved is not None
        else []
    )
    gradient_measurements = GradientMeasurements(
        observation_spec.gradient_norms if observation_spec is not None else (),
        copy.deepcopy(observation_saved["gradients"])
        if observation_saved is not None
        else None,
    )
    reporting_saved = resumed.reporting_state if resumed is not None else None
    if reporting_saved is not None:
        epoch_history = copy.deepcopy(reporting_saved["history"])
        persist_reporting = True
    elif resumed is not None and spec.metrics is not None:
        raise ValueError("checkpoint has no reporting state for requested metrics")
    online_sums = {
        name: torch.as_tensor(
            value,
            device=device,
            dtype=torch.float64
            if torch.device(device).type != "mps"
            else torch.float32,
        )
        for name, value in (reporting_saved or {}).get("online_sums", {}).items()
    }
    online_count = (reporting_saved or {}).get("online_count", 0)
    executed_updates = 0
    examples_consumed = (reporting_saved or {}).get(
        "examples",
        data_state.get("epoch", 0) * sum(batch_sizes)
        + sum(batch_sizes[: data_state.get("batch", 0)]),
    )
    epoch_examples = (reporting_saved or {}).get(
        "epoch_examples", sum(batch_sizes[: data_state.get("batch", 0)])
    )
    epoch_updates = (reporting_saved or {}).get(
        "epoch_updates", data_state.get("batch", 0)
    )
    last_loss = None
    history = []
    last_gradients: dict[str, torch.Tensor] = {}
    final_forward: ExecutionResult | None = None

    def evaluation_tensor_identity(tensors):
        if isinstance(tensors, DatasetProvider):
            return {"provider": tensors.protocol}
        if isinstance(tensors, ResolvedDenseInputs):
            tensors = tensors.tensors
        return {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(
                    value.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest(),
            }
            for name, value in sorted(tensors.items())
        }

    reporting_contract = None
    if persist_reporting:
        bind_data = spec.metrics is not None or spec.batching is not None
        reporting_contract = {
            "metrics": training_metric_contract,
            "train_inputs": evaluation_tensor_identity(resolved_inputs)
            if bind_data
            else resolved_inputs.protocol,
            "train_targets": evaluation_tensor_identity(resolved_targets)
            if bind_data
            else list(target_rows),
            "validation": None
            if validation_data is None or metric_policy.validation == "none"
            else {
                "protocol": validation_data[0].protocol,
                "inputs": evaluation_tensor_identity(validation_data[0])
                if bind_data
                else validation_data[0].protocol,
                "targets": evaluation_tensor_identity(validation_data[1])
                if bind_data
                else list(validation_data[2]),
            },
        }
    if reporting_saved is not None:
        if reporting_saved.get("contract") != reporting_contract:
            raise ValueError(
                "checkpoint reporting metric/data identity does not match request"
            )
        expected_partial = sum(batch_sizes[: data_state["batch"]])
        if (
            examples_consumed
            != data_state["epoch"] * sum(batch_sizes) + expected_partial
            or epoch_examples != expected_partial
            or epoch_updates != data_state["batch"]
            or online_count
            != (expected_partial if metric_policy.train == "online" else 0)
            or completed_updates
            != data_state["epoch"] * batches_per_epoch + data_state["batch"]
            or not epoch_history
            or [row.get("epoch") for row in epoch_history]
            != list(range(epoch_history[0]["epoch"], data_state["epoch"] + 1))
        ):
            raise ValueError("checkpoint reporting coordinates do not match schedule")
        if online_count:
            expected_keys = (
                {"loss"}
                | {
                    f"component/objective[{i}]"
                    for i in range(len(training.get("objectives", [])))
                }
                | {
                    f"component/regularizer[{i}]"
                    for i in range(len(training.get("regularizers", [])))
                }
                | {
                    f"{prefix}/objective[{i}]"
                    for i, objective in enumerate(training.get("objectives", []))
                    if X.classification(objective)
                    for prefix in ("ce", "correct")
                }
            )
            if set(online_sums) != expected_keys:
                raise ValueError(
                    "checkpoint online metric accumulator keys do not match recipe"
                )
        elif online_sums:
            raise ValueError("checkpoint contains online sums without examples")

    selection_contract = {
        "policy": selection.to_dict(),
        "evaluation": None
        if not epoch_selection
        else {
            "seed": spec.seed,
            "batch_size": batch_size,
            "inputs": (
                validation_data[0]
                if selection.split == "validation"
                else resolved_inputs
            ).protocol,
            "targets": list(
                validation_data[2] if selection.split == "validation" else target_rows
            ),
            "input_tensors": evaluation_tensor_identity(
                (
                    validation_data[0]
                    if selection.split == "validation"
                    else resolved_inputs
                )
            ),
            "target_tensors": evaluation_tensor_identity(
                validation_data[1]
                if selection.split == "validation"
                else resolved_targets
            ),
            "aggregation": "sample_weighted_mean",
        },
    }
    selected_checkpoint: TrainingCheckpoint | None = None
    if resumed is not None:
        if resumed.selection_contract is None:
            if epoch_selection:
                raise ValueError(
                    "checkpoint has no selection history; cannot reconstruct missing candidates"
                )
        elif resumed.selection_contract != selection_contract:
            raise ValueError(
                "checkpoint selection policy or evaluation identity does not match request"
            )
        else:
            selected_checkpoint = resumed.best_checkpoint
            if selected_checkpoint is None and resumed.selection_record is not None:
                selected_checkpoint = replace(resumed, best_checkpoint=None)
            if selected_checkpoint is not None:
                record = selected_checkpoint.selection_record
                if (
                    not isinstance(record, Mapping)
                    or record.get("update") != selected_checkpoint.completed_updates
                ):
                    raise ValueError(
                        "checkpoint selection record coordinates do not match candidate"
                    )
                if record.get("policy") != selection.to_dict():
                    raise ValueError(
                        "checkpoint selection record policy does not match request"
                    )
                values = record.get("values", [])
                expected_count = 1 + len(selection.tie_break) if epoch_selection else 1
                if len(values) != expected_count or any(
                    not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in values
                ):
                    raise ValueError(
                        "checkpoint selection record metrics are missing or non-finite"
                    )
                if (
                    epoch_selection
                    and record.get("evaluation") != selection_contract["evaluation"]
                ):
                    raise ValueError(
                        "checkpoint selection record evaluation identity does not match request"
                    )
                if set(selected_checkpoint.parameters) != set(parameter_map) or any(
                    value.shape != parameter_map[name].shape
                    or value.dtype != parameter_map[name].dtype
                    for name, value in selected_checkpoint.parameters.items()
                    if name in parameter_map
                ):
                    raise ValueError(
                        "checkpoint selected candidate parameters do not match graph"
                    )
                if epoch_selection and record.get(
                    "epoch"
                ) != selected_checkpoint.data_state.get("epoch"):
                    raise ValueError(
                        "checkpoint selection record epoch does not match candidate"
                    )
                if (
                    selected_checkpoint.graph_digest != graph_digest
                    or selected_checkpoint.training_digest != training_digest
                    or selected_checkpoint.selection_contract != selection_contract
                    or selected_checkpoint.selection_record != resumed.selection_record
                    or selected_checkpoint.completed_updates > completed_updates
                ):
                    raise ValueError(
                        "checkpoint selected candidate identity does not match training state"
                    )

    def optimizer_state_by_name() -> dict[str, dict[str, Any]]:
        packed = {}
        for name, parameter in parameter_map.items():
            if parameter not in optimizer.state:
                if parameter.requires_grad:
                    packed[name] = {}
                continue
            packed[name] = {
                key: value.detach().clone()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in optimizer.state[parameter].items()
            }
        return packed

    def checkpoint_at(
        update_count: int, loss_value: float | None, next_data_state: Mapping[str, Any]
    ) -> TrainingCheckpoint:
        rng_backend, accelerator_rng_states = capture_training_rng_state(device)
        return TrainingCheckpoint(
            graph_digest=graph_digest,
            training_digest=training_digest,
            completed_updates=update_count,
            selected_loss=loss_value,
            selection_contract=copy.deepcopy(selection_contract),
            selection_record=copy.deepcopy(selected_checkpoint.selection_record)
            if selected_checkpoint is not None
            else None,
            execution_protocol=resolved_inputs.protocol,
            initialization=built.metrics["initialization"],
            parameters={
                name: value.detach().clone() for name, value in parameter_map.items()
            },
            optimizer_state=optimizer_state_by_name(),
            rng_state=torch.get_rng_state().clone(),
            rng_backend=rng_backend,
            accelerator_rng_states=accelerator_rng_states,
            data_state=dict(next_data_state),
            reporting_state={
                "contract": reporting_contract,
                "history": copy.deepcopy(epoch_history),
                "online_sums": _host_metrics(online_sums),
                "online_count": online_count,
                "examples": examples_consumed,
                "epoch_examples": epoch_examples,
                "epoch_updates": epoch_updates,
            }
            if persist_reporting
            else None,
            observation_state=copy.deepcopy(
                {
                    "contract": observation_contract,
                    "history": epoch_history,
                    "gradients": gradient_measurements.state,
                }
            )
            if observation_contract is not None
            else None,
        )

    def compute_loss(forward, signal_values, batch_targets, input_protocol):
        components: dict[str, torch.Tensor] = {}
        loss = torch.zeros((), device=device)
        for index, objective in enumerate(training.get("objectives", [])):
            output_id = output_ids.get(objective["prediction"])
            if output_id is None:
                raise ValueError(
                    f"objective[{index}] prediction {objective['prediction']} is not a graph output"
                )
            target_name = objective["target"]
            if target_name not in batch_targets:
                raise ValueError(
                    f"objective[{index}] missing target tensor {target_name}"
                )
            target = batch_targets[target_name].to(device=device)
            prediction = forward.outputs[output_id]
            if objective.get("kind") == "custom_objective":
                definition = E.resolve("objective", objective)
                raw = definition.function(
                    prediction, target, objective.get("config", {})
                )
                X.checked_tensor(
                    raw,
                    shape=(),
                    device=prediction.device,
                    dtype=prediction.dtype,
                    name=definition.name,
                )
            elif objective.get("kind") == "cross_entropy":
                raw = torch.nn.functional.cross_entropy(
                    prediction, target.to(dtype=torch.long)
                )
            else:
                raise ValueError(
                    f"objective[{index}] unsupported kind {objective.get('kind')}"
                )
            value = raw * float(objective.get("weight", 1.0))
            components[f"objective[{index}]"] = value
            loss = loss + value
        duration = training.get("presentation_duration")
        duration_s = (
            float(duration["value"]) / 1000.0
            if duration
            else input_protocol["timing"]["duration_ms"] / 1000.0
        )
        for index, regularizer in enumerate(training.get("regularizers", [])):
            if regularizer.get("kind") == "custom_regularizer":
                definition = E.resolve("regularizer", regularizer)
                values = tuple(signal_values[name] for name in regularizer["signals"])
                raw = definition.function(
                    values, duration_s, regularizer.get("config", {})
                )
                X.checked_tensor(
                    raw,
                    shape=(),
                    device=values[0].device,
                    dtype=values[0].dtype,
                    name=definition.name,
                )
                value = float(regularizer["strength"]) * raw
                components[f"regularizer[{index}]"] = value
                loss = loss + value
                continue
            if regularizer.get("kind") != "spike_budget":
                raise ValueError(
                    f"regularizer[{index}] unsupported kind {regularizer.get('kind')}"
                )
            ceiling = float(regularizer["config"]["ceiling"]["value"])
            penalties = []
            for signal in regularizer["signals"]:
                spikes = signal_values.get(signal)
                if spikes is None:
                    raise ValueError(
                        f"regularizer[{index}] spike signal {signal} is unavailable"
                    )
                sample_rates = spikes.sum(dim=0).mean(dim=1) / duration_s
                penalties.append(torch.relu(sample_rates - ceiling).square())
            value = float(regularizer["strength"]) * torch.stack(penalties).mean()
            components[f"regularizer[{index}]"] = value
            loss = loss + value
        return loss, components

    required_signals = tuple(
        signal
        for regularizer in training.get("regularizers", [])
        for signal in regularizer["signals"]
    )
    observation_populations = (
        ()
        if observation_spec is None
        else tuple(
            sorted(
                set(observation_spec.population_rates)
                | set(observation_spec.output_activity)
            )
        )
    )

    def evaluate_split(inputs, targets, *, probe_name=None, test=False):
        sample_count = dataset_sample_count(inputs)
        draws = inputs.draw_seeds if isinstance(inputs, DatasetProvider) else (None,)
        presentation_count = sample_count * len(draws)
        phase = (
            "test"
            if test
            else (
                "train_evaluation"
                if inputs is resolved_inputs
                else ("validation" if probe_name is None else f"probe:{probe_name}")
            )
        )
        total_loss = torch.zeros(
            (),
            device="cpu" if torch.device(device).type == "mps" else device,
            dtype=torch.float64,
        )
        cross_entropies = {}
        components_total: dict[str, float] = {}
        correct = {
            f"objective[{i}]": 0
            for i, row in enumerate(training.get("objectives", []))
            if X.classification(row)
        }
        measurements = (
            ActivityMeasurements(
                observation_spec,
                {p["id"]: p for p in graph["populations"]},
                dataset_steps_count(inputs) * model.plan.dt_ms / 1000,
                presentation_count,
            )
            if observation_spec is not None and not test
            else None
        )
        was_training = model.training
        model.eval()
        try:
            seed = (
                spec.seed
                if probe_name is None
                else observation_spec.probes[probe_name].seed
            )
            with evaluation_rng(seed), torch.no_grad():
                for draw_seed in draws:
                    requests = (
                        (
                            torch.arange(start, min(start + batch_size, sample_count)),
                            phase,
                            0,
                            draw_seed,
                        )
                        for start in range(0, sample_count, batch_size)
                    )
                    for request, evaluation_inputs, _ in iter_batches(
                        inputs, requests, prefetch=spec.dataset_prefetch
                    ):
                        indices = request[0]
                        start, end = int(indices[0]), int(indices[-1]) + 1
                        evaluation_targets = {
                            name: value[start:end] for name, value in targets.items()
                        }
                        forward, signals = model._forward(
                            evaluation_inputs,
                            diagnostics=False,
                            required_signals=required_signals
                            if probe_name is None
                            else (),
                            observation_populations=()
                            if test
                            else observation_populations,
                            automatic_reductions=True,
                            retain_parameters=False,
                            retain_final_state=False,
                            retain_runtime_state=False,
                        )
                        if measurements is not None:
                            measurements.add(
                                forward.metrics.get("population_observations", {})
                            )
                        if probe_name is not None:
                            continue
                        loss, components = compute_loss(
                            forward, signals, evaluation_targets, inputs.protocol
                        )
                        count = end - start
                        for i, objective in enumerate(training.get("objectives", [])):
                            if X.classification(objective):
                                prediction = forward.outputs[
                                    output_ids[objective["prediction"]]
                                ]
                                raw = torch.nn.functional.cross_entropy(
                                    prediction,
                                    evaluation_targets[objective["target"]].long(),
                                )
                                name = f"objective[{i}]"
                                cross_entropies[name] = (
                                    cross_entropies.get(name, 0.0)
                                    + _reporting_scalar(raw) * count
                                )
                        total_loss += _reporting_scalar(loss) * count
                        for name, value in components.items():
                            components_total[name] = (
                                components_total.get(name, 0.0)
                                + _reporting_scalar(value) * count
                            )
                        for i, objective in enumerate(training.get("objectives", [])):
                            if not X.classification(objective):
                                continue
                            prediction = forward.outputs[
                                output_ids[objective["prediction"]]
                            ].argmax(dim=-1)
                            correct[f"objective[{i}]"] += (
                                prediction == evaluation_targets[objective["target"]]
                            ).sum()
        finally:
            model.train(was_training)
        packed = _host_metrics(
            {
                "loss": total_loss,
                **{f"ce/{k}": v for k, v in cross_entropies.items()},
                **{f"component/{k}": v for k, v in components_total.items()},
                **{
                    f"correct/{k}": torch.as_tensor(v, device=device)
                    for k, v in correct.items()
                },
            }
        )
        metrics = {
            "loss": packed["loss"] / presentation_count,
            "cross_entropies": {
                name: packed[f"ce/{name}"] / presentation_count
                for name in cross_entropies
            },
            "components": {
                name: packed[f"component/{name}"] / presentation_count
                for name in components_total
            },
            "accuracies": {
                name: packed[f"correct/{name}"] / presentation_count for name in correct
            },
        }
        if len(correct) == 1:
            metrics["accuracy"] = (
                packed[f"correct/{next(iter(correct))}"] / presentation_count
            )
        if measurements is not None:
            metrics["observation"] = measurements.finish()
            metrics["observation"]["input_protocol"] = inputs.protocol
            metrics["observation"]["evaluation_seed"] = seed
        if isinstance(inputs, DatasetProvider):
            metrics["encoding_draws"] = list(draws)
        return metrics

    def record_epoch(epoch, *, emit=True):
        nonlocal \
            gradient_measurements, \
            selected_checkpoint, \
            online_sums, \
            online_count, \
            epoch_examples, \
            epoch_updates
        if (observation_spec is not None or persist_reporting) and any(
            row["epoch"] == epoch for row in epoch_history
        ):
            return
        timing.switch("reporting", epoch)
        row = {"epoch": epoch}
        observations = {}
        selection_metrics = None
        splits = {}
        if metric_policy.train == "evaluate" and (
            epoch > 0 or "train" in metric_policy.initial
        ):
            splits["train"] = (resolved_inputs, resolved_targets)
        if (
            validation_data is not None
            and metric_policy.validation == "epoch"
            and (epoch > 0 or "validation" in metric_policy.initial)
        ):
            splits["validation"] = validation_data[:2]
        if metric_policy.train == "online" and epoch > 0:
            packed = _host_metrics(online_sums)
            online = {"loss": packed["loss"] / online_count}
            for group, prefix in (
                ("components", "component/"),
                ("cross_entropies", "ce/"),
                ("accuracies", "correct/"),
            ):
                online[group] = {
                    name.removeprefix(prefix): value / online_count
                    for name, value in packed.items()
                    if name.startswith(prefix)
                }
            if len(online["accuracies"]) == 1:
                online["accuracy"] = next(iter(online["accuracies"].values()))
            row.update({f"train_{key}": value for key, value in online.items()})
            row["train_measurement"] = "online_pre_update"
            row["train_examples"] = online_count
            online_sums, online_count = {}, 0
        for split, (inputs, targets) in splits.items():
            timing.switch(
                "train_evaluation" if split == "train" else "validation", epoch
            )
            metrics = evaluate_split(inputs, targets)
            timing.switch("reporting", epoch)
            if split == selection.split:
                selection_metrics = metrics
            if "observation" in metrics:
                observations[split] = metrics.pop("observation")
                observations[split]["split"] = split
            row.update({f"{split}_{key}": value for key, value in metrics.items()})
            if spec.metrics is not None:
                row[f"{split}_measurement"] = "fixed_weight_evaluation"
                row[f"{split}_examples"] = dataset_sample_count(inputs) * (
                    len(inputs.draw_seeds) if isinstance(inputs, DatasetProvider) else 1
                )
        if observation_spec is not None:
            observations["probes"] = {}
            for name, inputs in resolved_probes.items():
                timing.switch("validation", epoch)
                measurement = evaluate_split(inputs, {}, probe_name=name)["observation"]
                timing.switch("reporting", epoch)
                observations["probes"][name] = {
                    **measurement,
                    "split": observation_spec.probes[name].split,
                    "draw_id": name,
                }
            observations["parameters"] = {
                "l2_norm": {
                    name: finite_norm(parameter_map[name])
                    for name in observation_spec.parameter_norms
                },
                "state": "initial" if epoch == 0 else "completed_epoch",
                "aggregation": "flattened_l2",
                "units": {
                    name: next(
                        p["unit"] for p in graph["parameters"] if p["id"] == name
                    )
                    for name in observation_spec.parameter_norms
                },
            }
            observations["gradients"] = gradient_measurements.finish()
            row.update(
                {
                    "observations": observations,
                    "update": completed_updates + executed_updates,
                    "phase": "initial" if epoch == 0 else "completed_epoch",
                }
            )
            gradient_measurements = GradientMeasurements(
                observation_spec.gradient_norms
            )
        if persist_reporting:
            row.update(
                {
                    "phase": "initial" if epoch == 0 else "completed_epoch",
                    "update": completed_updates + executed_updates,
                    "examples": examples_consumed,
                    "epoch_examples": epoch_examples,
                    "epoch_updates": epoch_updates,
                }
            )
            if epoch > 0:
                row["dropped_examples"] = dataset_size - sum(batch_sizes)
        epoch_examples, epoch_updates = 0, 0
        epoch_history.append(row)
        if epoch_selection and (epoch > 0 or selection.include_initial):
            values = selection.scores(selection_metrics)
            previous = (
                selected_checkpoint.selection_record["values"]
                if selected_checkpoint is not None
                else None
            )
            if selection.better(values, previous):
                candidate = checkpoint_at(
                    completed_updates + executed_updates,
                    None,
                    {"epoch": epoch, "batch": 0},
                )
                selected_checkpoint = replace(
                    candidate,
                    selection_record={
                        "policy": selection.to_dict(),
                        "values": values,
                        "epoch": epoch,
                        "update": candidate.completed_updates,
                        "evaluation": copy.deepcopy(selection_contract["evaluation"]),
                        "phase": "initial" if epoch == 0 else "completed_epoch",
                    },
                )

        if emit and spec.on_epoch_end is not None and epoch > 0:
            timing.switch("reporting", epoch)
            epoch_timing = None
            if timing.enabled:
                values = timing.epochs.get(epoch, {})
                epoch_timing = {
                    "mode": timing.mode,
                    "total_s": sum(values.values()),
                    **{
                        f"{phase}_s": values.get(phase, 0.0)
                        for phase in (
                            "training",
                            "train_evaluation",
                            "validation",
                            "checkpoint",
                            "reporting",
                        )
                    },
                }
                row["timing"] = epoch_timing

            def split_snapshot(split):
                values = {
                    key.removeprefix(f"{split}_"): value
                    for key, value in row.items()
                    if key.startswith(f"{split}_")
                }
                return FrozenValues(values) if values else None

            report = EpochReport(
                epoch,
                dataset_epochs,
                completed_updates + executed_updates,
                examples_consumed,
                split_snapshot("train"),
                split_snapshot("validation"),
                FrozenValues(row["observations"]) if "observations" in row else None,
                FrozenValues(epoch_timing) if epoch_timing is not None else None,
            )
            with evaluation_rng():
                spec.on_epoch_end(report)

    scheduled: list[tuple[int, int, torch.Tensor]] = []
    if dataset_mode:
        for epoch in range(data_state["epoch"], dataset_epochs):
            batches = batching.indices(
                dataset_size, batch_size, seed=spec.seed + epoch, labels=stratify_labels
            )
            first_batch = data_state["batch"] if epoch == data_state["epoch"] else 0
            for batch in range(first_batch, batches_per_epoch):
                scheduled.append((epoch, batch, batches[batch]))
        scheduled = scheduled[:updates]
        if not scheduled:
            raise ValueError(
                "dataset training checkpoint is already at the requested end epoch"
            )
    else:
        scheduled = [
            (-1, update, torch.arange(dataset_size)) for update in range(updates)
        ]

    replay = TrainingReplay(spec, model, training, scheduled, tracing=trace is not None)

    def numerical_step(batch_inputs, batch_targets):
        forward, signals = model._forward(
            batch_inputs,
            diagnostics=spec.diagnostics,
            required_signals=required_signals,
            retain_parameters=False,
            retain_final_state=False,
            retain_runtime_state=False,
        )
        loss, components = compute_loss(
            forward, signals, batch_targets, resolved_inputs.protocol
        )
        with execution_precision(spec.precision, next(model.parameters(), None)):
            loss.backward()
        return forward, signals, loss, components

    snapshots = SnapshotSession(
        spec,
        model,
        kind="train",
        protocol=resolved_inputs.protocol,
        training=training,
        device_resolved=device,
        batch_size=batch_size,
        batching=batching,
        metrics=metric_policy,
        checkpoint_selection=selection.to_dict(),
        initialization=model.initialization_metadata,
    )
    snapshots.capture("start", completed_updates=completed_updates)

    if dataset_mode and data_state["batch"] == 0:
        record_epoch(data_state["epoch"], emit=False)

    timing.switch("training", scheduled[0][0] + 1 if dataset_mode else None)
    requests = (
        (indices, "train", epoch if dataset_mode else completed_updates + index, None)
        for index, (epoch, _, indices) in enumerate(scheduled)
    )
    batches = iter(
        iter_batches(resolved_inputs, requests, prefetch=spec.dataset_prefetch)
    )
    for update in range(len(scheduled)):
        if trace is not None:
            trace.begin_update(
                update,
                scheduled[update][0] + 1,
                scheduled[update][1] + 1,
                parameter_map,
            )
        with trace.region("batch_prepare_wait") if trace is not None else nullcontext():
            _, batch_inputs, encoding_metadata = next(batches)
        if trace is not None:
            trace.observe_inputs(batch_inputs)
        epoch, batch, sample_indices = scheduled[update]
        batch_targets = {
            name: value.index_select(0, sample_indices.to(value.device))
            for name, value in resolved_targets.items()
        }
        replayed = replay.run(batch_inputs, batch_targets, numerical_step)
        if replayed is not None:
            forward, signal_values, loss, components = replayed
        else:
            optimizer.zero_grad(set_to_none=True)
            with trace.region("forward") if trace is not None else nullcontext():
                forward, signal_values = model._forward(
                    batch_inputs,
                    diagnostics=spec.diagnostics,
                    required_signals=required_signals,
                    retain_parameters=False,
                    retain_final_state=False,
                    retain_runtime_state=False,
                )
            with trace.region("loss") if trace is not None else nullcontext():
                loss, components = compute_loss(
                    forward, signal_values, batch_targets, resolved_inputs.protocol
                )
        if metric_policy.train == "online":
            count = len(sample_indices)
            values = {
                "loss": loss.detach() * count,
                **{
                    f"component/{name}": value.detach() * count
                    for name, value in components.items()
                },
            }
            with torch.no_grad():
                for index, objective in enumerate(training.get("objectives", [])):
                    if X.classification(objective):
                        prediction = forward.outputs[
                            output_ids[objective["prediction"]]
                        ].detach()
                        target = batch_targets[objective["target"]].long()
                        values[f"ce/objective[{index}]"] = (
                            torch.nn.functional.cross_entropy(
                                prediction, target, reduction="sum"
                            )
                        )
                        values[f"correct/objective[{index}]"] = (
                            prediction.argmax(-1) == target
                        ).sum()
                for name, value in values.items():
                    value = value.to(
                        dtype=torch.float64
                        if torch.device(device).type != "mps"
                        else torch.float32
                    )
                    if name not in online_sums:
                        online_sums[name] = torch.zeros_like(value)
                    online_sums[name].add_(value)
            online_count += count
        with trace.region("backward") if trace is not None else nullcontext():
            if replayed is None:
                with execution_precision(
                    spec.precision, next(model.parameters(), None)
                ):
                    loss.backward()
        if trace is not None:
            trace.observe_gradients(parameter_map)
        gradient_names = (
            set(parameter_map)
            if update == len(scheduled) - 1
            else set(observation_spec.gradient_norms)
            if observation_spec is not None
            else set()
        )
        before_clip = {
            name: parameter.grad.detach().clone()
            for name, parameter in parameter_map.items()
            if name in gradient_names and parameter.grad is not None
        }
        if update == len(scheduled) - 1:
            last_gradients = before_clip
        clip = training.get("gradient_clip")
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in groups for parameter in group["params"]],
                float(clip),
            )
        if observation_spec is not None:
            gradient_measurements.add(before_clip, parameter_map)
        with (
            trace.region("optimizer_constraints")
            if trace is not None
            else nullcontext()
        ):
            optimizer.step()
            model.enforce_constraints()
        if trace is not None:
            trace.end_update(parameter_map, last=update + 1 == len(scheduled))
        executed_updates = update + 1
        examples_consumed += len(sample_indices)
        epoch_examples += len(sample_indices)
        epoch_updates += 1
        absolute_update = completed_updates + executed_updates
        next_data_state: dict[str, Any] = {}
        if dataset_mode:
            next_data_state = {"epoch": epoch, "batch": batch + 1}
            if next_data_state["batch"] == batches_per_epoch:
                next_data_state = {"epoch": epoch + 1, "batch": 0}
        last_loss = loss.detach()
        loss_value = (
            float(loss.detach())
            if update_selection
            else (loss.detach().clone() if metric_policy.update_history else None)
        )
        if metric_policy.update_history:
            history.append(
                {
                    "update": absolute_update,
                    **(
                        {"encoding": encoding_metadata}
                        if isinstance(resolved_inputs, DatasetProvider)
                        else {}
                    ),
                    **(
                        {"epoch": epoch + 1, "batch": batch + 1} if dataset_mode else {}
                    ),
                    "loss": loss_value,
                    "components": {
                        name: value.detach().clone()
                        for name, value in components.items()
                    },
                }
            )
        if (
            (observation_spec is not None or epoch_selection or persist_reporting)
            and dataset_mode
            and next_data_state["batch"] == 0
        ):
            record_epoch(next_data_state["epoch"])
        if update_selection:
            if selected_checkpoint is None or loss_value < float(
                selected_checkpoint.selected_loss
            ):
                candidate = checkpoint_at(absolute_update, loss_value, next_data_state)
                selected_checkpoint = replace(
                    candidate,
                    selection_record={
                        "policy": selection.to_dict(),
                        "values": [loss_value],
                        "epoch": epoch + 1 if dataset_mode else None,
                        "update": absolute_update,
                        "evaluation": {
                            "phase": "pre_update_batch_loss",
                            "weights": "post_update",
                        },
                        "phase": "legacy_training_batch_loss",
                    },
                )
        final_forward = forward
        if (
            observation_spec is None
            and not persist_reporting
            and not epoch_selection
            and dataset_mode
            and next_data_state["batch"] == 0
        ):
            record_epoch(next_data_state["epoch"])
        if dataset_mode and next_data_state["batch"] == 0:
            if update + 1 < len(scheduled):
                timing.switch("training", scheduled[update + 1][0] + 1)
            else:
                timing.switch("reporting")
    batches.close()
    timing.switch("reporting")
    assert final_forward is not None
    # Reporting reads are grouped after execution; selection reads remain explicit.
    reporting = {}
    for index, row in enumerate(history):
        if isinstance(row["loss"], torch.Tensor):
            reporting[f"{index}/loss"] = row["loss"]
        for name, value in row["components"].items():
            reporting[f"{index}/{name}"] = value
    materialized = _host_metrics(reporting)
    for index, row in enumerate(history):
        row["loss"] = materialized.get(f"{index}/loss", row["loss"])
        row["components"] = {
            name: materialized[f"{index}/{name}"] for name in row["components"]
        }
    timing.switch("checkpoint")
    completed_this_call = len(scheduled)
    final_checkpoint = checkpoint_at(
        completed_updates + completed_this_call,
        float(last_loss),
        next_data_state,
    )
    final_checkpoint = replace(final_checkpoint, best_checkpoint=selected_checkpoint)
    if save_final := spec.save_final_checkpoint:
        save_training_checkpoint(save_final, final_checkpoint)
    if save_selected := spec.save_selected_checkpoint:
        if selected_checkpoint is None:
            raise ValueError(
                "no eligible checkpoint selection candidate; complete an epoch or include_initial"
            )
        save_training_checkpoint(save_selected, selected_checkpoint)

    test_metrics = None
    if (
        spec.test is not None
        and metric_policy.test == "final"
        and next_data_state == {"epoch": dataset_epochs, "batch": 0}
    ):
        timing.switch("test")
        # The test split is never used for selection, and is not even resolved
        # for an interrupted run. Preserve the final model for exact resume.
        with evaluation_rng(spec.seed):
            test_inputs, test_targets, test_target_rows = resolve_data(spec.test)
            if isinstance(test_inputs, DatasetProvider):
                test_inputs.protocol["dataset"]["batch_size"] = batch_size
            candidate = selected_checkpoint or final_checkpoint
            try:
                with torch.no_grad():
                    for name, parameter in parameter_map.items():
                        parameter.copy_(candidate.parameters[name])
                test_metrics = {
                    **evaluate_split(test_inputs, test_targets, test=True),
                    "measurement": "fixed_weight_evaluation",
                    "examples": dataset_sample_count(test_inputs)
                    * (
                        len(test_inputs.draw_seeds)
                        if isinstance(test_inputs, DatasetProvider)
                        else 1
                    ),
                    "checkpoint": "selected"
                    if selected_checkpoint is not None
                    else "final",
                    "epoch": candidate.data_state["epoch"],
                    "update": candidate.completed_updates,
                    "selection_record": copy.deepcopy(candidate.selection_record),
                    "evaluation_seed": spec.seed,
                    "protocol": {**test_inputs.protocol, "targets": test_target_rows},
                }
            finally:
                with torch.no_grad():
                    for name, parameter in parameter_map.items():
                        parameter.copy_(final_checkpoint.parameters[name])

    timing.switch("reporting")
    result = ExecutionResult(
        executor="graph",
        outputs=final_forward.outputs,
        diagnostics=final_forward.diagnostics,
        _output_axes=final_forward._output_axes,
        _diagnostic_axes=final_forward._diagnostic_axes,
        _timebase=final_forward._timebase,
        _batch_size=final_forward._batch_size,
        parameters={
            name: value.detach().clone() for name, value in parameter_map.items()
        }
        if spec.retain_parameters
        else {},
        gradients=last_gradients,
        optimizer_state=optimizer_state_by_name(),
        training_checkpoint=final_checkpoint,
        selected_checkpoint=selected_checkpoint,
        model=model,
        metrics={
            **built.metrics,
            "diagnostics": spec.diagnostics,
            "dataset_prefetch": spec.dataset_prefetch,
            "cuda_graphs": replay.summary(),
            "execution_paths": copy.deepcopy(model._execution_paths),
            "updates": history,
            "epochs": epoch_history,
            "test": test_metrics,
            "history_coverage": {
                "requested_epochs": dataset_epochs,
                "complete": next_data_state == {"epoch": dataset_epochs, "batch": 0} if dataset_mode else None,
                "data_state": dict(next_data_state),
            },
            "validation_protocol": (
                {**validation_data[0].protocol, "targets": validation_data[2]}
                if validation_data is not None
                else None
            ),
            "execution_protocol": resolved_inputs.protocol,
            "trainable_parameters": sorted(last_gradients),
            "optimizer": optimizer_spec,
            "fused_optimizer": fused_optimizer,
            "precision": {
                "policy": spec.precision,
                "autocast": "bfloat16"
                if model._amp_enabled(next(model.parameters(), None))
                else "off",
            },
            "training_checkpoint_schema": TRAINING_CHECKPOINT_SCHEMA,
            "training_checkpoint_rng": {
                "backend": final_checkpoint.rng_backend,
                "devices": sorted(final_checkpoint.accelerator_rng_states),
            },
            "checkpoint_selection": selection_contract,
            "selection_record": final_checkpoint.selection_record,
            "resumed_from_update": completed_updates,
        },
    )

    if timing.enabled:
        result.metrics["training_timing"] = timing.finish()
    result.metrics["performance"] = _performance_report(
        spec, model, result.metrics, device=device, executed=True
    )
    snapshots.capture("end", completed_updates=final_checkpoint.completed_updates)
    result.snapshots = snapshots.snapshots
    return result
