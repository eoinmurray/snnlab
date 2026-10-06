"""Typed execution seam and graph-native forward executor.

The bundle is data.  This module intentionally has no dependency on snnlang.
Legacy requests continue to route through the existing CLI handlers; graph
requests are planned once and execute a fixed vectorised schedule per step.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field, replace
from numbers import Integral
from pathlib import Path
from types import EllipsisType
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from snnlab import extensions as E
from snnlab.sim import extensions as X
from snnlab.sim import models as M
from snnlab.sim.bundle import load_graph_bundle, load_training_recipe
from snnlab.sim.checkpoint_selection import CheckpointSelection
from snnlab.sim.checkpoint_selection import SelectionMetric as SelectionMetric
from snnlab.sim.dataset_provider import (
    DatasetProvider,
    batch_tensors,
    simulate_dataset,
)
from snnlab.sim.dataset_provider import (
    sample_count as dataset_sample_count,
)
from snnlab.sim.dataset_provider import (
    steps_count as dataset_steps_count,
)
from snnlab.sim.epoch_observations import (
    ActivityMeasurements,
    EpochObservations,
    GradientMeasurements,
    ObservationProbe,
    evaluation_rng,
    finite_norm,
    validate_observation_state,
)
from snnlab.sim.interventions import (
    AddPoissonSpikes as AddPoissonSpikes,
)
from snnlab.sim.interventions import (
    DenseSpikeReplay as DenseSpikeReplay,
)
from snnlab.sim.interventions import (
    DropSpikes as DropSpikes,
)
from snnlab.sim.interventions import (
    Intervention,
    parse_intervention,
    prepare_interventions,
)
from snnlab.sim.interventions import (
    ReplaySpikes as ReplaySpikes,
)
from snnlab.sim.interventions import (
    SparseSpikeReplay as SparseSpikeReplay,
)
from snnlab.sim.interventions import (
    intervention_identity as intervention_identity,
)
from snnlab.sim.interventions import (
    load_spike_replay as load_spike_replay,
)
from snnlab.sim.interventions import (
    save_spike_replay as save_spike_replay,
)

ExecutorName = Literal["legacy", "graph"]
RequestKind = Literal["build", "simulate", "train", "infer"]


DENSE_ARRAY_BINDING_SCHEMA = "tools/snnsim.dense-array-binding/v1"
EVENT_STREAM_BINDING_SCHEMA = "tools/snnsim.event-stream-binding/v1"
MIXED_INPUT_BINDING_SCHEMA = "tools/snnsim.mixed-input-bindings/v1"
POISSON_INPUT_BINDING_SCHEMA = "tools/snnsim.poisson-input-binding/v1"
DATASET_SNAPSHOT_BINDING_SCHEMA = "tools/snnsim.dataset-snapshot-binding/v1"
EXECUTION_PROTOCOL_SCHEMA = "tools/snnsim.execution-protocol/v1"
INFERENCE_OVERRIDE_SCHEMA = "tools/snnsim.inference-overrides/v1"
INFERENCE_INTERVENTION_SCHEMA = "tools/snnsim.inference-interventions/v1"
INFERENCE_ARTIFACT_SCHEMA = "tools/snnsim.inference-artifacts/v1"
DERIVED_INFERENCE_SCHEMA = "tools/snnsim.derived-inference/v1"
TRAINING_CHECKPOINT_SCHEMA = "tools/snnsim.training-checkpoint/v1"
LEGACY_PARAMETER_INTERCHANGE_SCHEMA = "tools/snnsim.legacy-parameter-interchange/v1"


@dataclass(frozen=True)
class DenseArrayBinding:
    """One concrete dense tensor resolved against a named graph input."""

    input_id: str
    value: torch.Tensor
    source: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetArrayBinding:
    """One concrete target vector resolved against a named training target."""

    target_id: str
    value: torch.Tensor
    source: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EventStreamBinding:
    """Sparse binary spike events resolved against one named graph input."""

    input_id: str
    steps: torch.Tensor
    batches: torch.Tensor
    channels: torch.Tensor
    steps_count: int
    batch_size: int
    source: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PoissonInputBinding:
    """Generated Bernoulli-discretised Poisson spikes for one graph input."""

    input_id: str
    steps_count: int
    rates_hz: Sequence[float]
    seed: int
    batch_size: int = 1
    categorical: bool = False


@dataclass(frozen=True)
class DatasetEncoder:
    """Portable standard encoding recipe for an immutable dataset snapshot."""

    kind: Literal["rate_poisson", "prebinned_spikes", "event_bin", "custom"]
    duration_ms: float | None = None
    max_rate_hz: float | None = None
    seed: int = 0
    definition: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    rates_hz: Sequence[float] | None = None


@dataclass(frozen=True)
class DatasetSnapshotBinding:
    """Bind one digest-identified NPZ snapshot to a graph input and target."""

    path: Path
    input_id: str
    dataset_id: str
    split: str
    encoder: DatasetEncoder
    target_id: str | None = None
    feature_key: str = "features"
    label_key: str = "labels"
    sample_cap: int | None = None
    shuffle: bool = False
    order_seed: int = 0


InputBinding = (
    DenseArrayBinding
    | EventStreamBinding
    | PoissonInputBinding
    | DatasetSnapshotBinding
)


@dataclass(frozen=True)
class _InputSources:
    dense: tuple[DenseArrayBinding, ...]
    events: tuple[EventStreamBinding, ...]
    poisson: tuple[PoissonInputBinding, ...]
    dataset: DatasetSnapshotBinding | None


def _split_input_bindings(bindings: Sequence[InputBinding]) -> _InputSources:
    dense, events, poisson, datasets = [], [], [], []
    input_ids: set[str] = set()
    for binding in bindings:
        if isinstance(binding, DenseArrayBinding):
            dense.append(binding)
        elif isinstance(binding, EventStreamBinding):
            events.append(binding)
        elif isinstance(binding, PoissonInputBinding):
            poisson.append(binding)
        elif isinstance(binding, DatasetSnapshotBinding):
            datasets.append(binding)
        else:
            raise TypeError(f"unsupported input binding type: {type(binding).__name__}")
        if binding.input_id in input_ids:
            raise ValueError(
                f"duplicate input binding for graph input {binding.input_id}"
            )
        input_ids.add(binding.input_id)
    if datasets and len(bindings) != 1:
        raise ValueError(
            "dataset snapshot binding cannot be combined with other input bindings"
        )
    if poisson and (dense or events):
        raise ValueError("Poisson bindings cannot yet be mixed with replay bindings")
    return _InputSources(
        tuple(dense), tuple(events), tuple(poisson), datasets[0] if datasets else None
    )


@dataclass(frozen=True)
class ResolvedDenseInputs:
    tensors: Mapping[str, torch.Tensor]
    protocol: Mapping[str, Any]


@dataclass(frozen=True)
class ValidationSpec:
    """Held-out inputs and labels evaluated without optimizer updates."""

    input_bindings: Sequence[InputBinding] = field(default_factory=tuple)
    targets: Mapping[str, torch.Tensor] = field(default_factory=dict)
    target_bindings: Sequence[TargetArrayBinding] = field(default_factory=tuple)
    protocol: Mapping[str, Any] = field(default_factory=dict)
    encoding_seeds: Sequence[int] = ()


@dataclass(frozen=True)
class ExecutionSpec:
    kind: RequestKind
    executor: ExecutorName = "graph"
    bundle: Path | None = None
    graph: Mapping[str, Any] | None = None
    input_bindings: Sequence[InputBinding] = field(default_factory=tuple)
    protocol: Mapping[str, Any] = field(default_factory=dict)
    training: Mapping[str, Any] | None = None
    targets: Mapping[str, torch.Tensor] = field(default_factory=dict)
    target_bindings: Sequence[TargetArrayBinding] = field(default_factory=tuple)
    validation: ValidationSpec | None = None
    epochs: int = 0
    batch_size: int | None = None
    shuffle: bool = False
    updates: int | None = None
    save_final_checkpoint: str | Path | None = None
    save_selected_checkpoint: str | Path | None = None
    seed: int = 0
    device: str = "auto"
    diagnostics: bool = True
    checkpoint: Path | None = None
    runtime_state: GraphRuntimeState | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    observations: EpochObservations | None = None
    checkpoint_selection: CheckpointSelection | None = None
    interventions: Sequence[Intervention] = field(default_factory=tuple)


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


@dataclass(frozen=True)
class CapabilityIssue:
    element: str
    capability: str
    message: str


@dataclass
class TrainingCheckpoint:
    graph_digest: str
    training_digest: str
    completed_updates: int
    execution_protocol: Mapping[str, Any]
    initialization: Mapping[str, Any]
    parameters: dict[str, torch.Tensor]
    optimizer_state: dict[str, dict[str, Any]]
    rng_state: torch.Tensor
    rng_backend: str = "cpu"
    accelerator_rng_states: dict[str, torch.Tensor] = field(default_factory=dict)
    data_state: Mapping[str, Any] = field(default_factory=dict)
    selected_loss: float | None = None
    observation_state: Mapping[str, Any] | None = None
    selection_contract: Mapping[str, Any] | None = None
    selection_record: Mapping[str, Any] | None = None
    best_checkpoint: TrainingCheckpoint | None = None


@dataclass(frozen=True)
class ParameterInterchange:
    parameters: dict[str, torch.Tensor]
    provenance: Mapping[str, Any]


GRAPH_CAPABILITIES_V1 = {
    "schema": "tools/snnsim.capabilities/v1",
    "neurons": {"coba_lif", "cuba_lif", "leaky_integrator", "custom_neuron"},
    "synapses": {
        "ampa",
        "gaba",
        "leaky_integrator",
        "exponential_current",
        "custom_synapse",
    },
    "operations": {
        "linear",
        "reduce_mean",
        "reduce_sum",
        "select_final",
        "duration_normalise",
        "cumulative_sum",
        "custom_operation",
        "divide",
    },
    "connections": {"feedforward", "recurrent", "feedback"},
    "recordings": {"spikes", "voltage"},
    "delays": "integer_steps",
    "training": {
        "objectives": {"cross_entropy", "custom_objective"},
        "regularizers": {"spike_budget", "custom_regularizer"},
        "optimizers": {"adamw", "custom_optimizer"},
        "parameter_groups": "named_trainable_and_frozen",
        "updates": "deterministic_epochs_and_minibatches",
        "targets": "named_classification_or_regression_arrays",
    },
}


def load_dense_array_bindings(
    path: str | Path, graph: Mapping[str, Any]
) -> tuple[DenseArrayBinding, ...]:
    """Load a replayable NPY/NPZ file without weakening named-input semantics."""
    source_path = Path(path)
    digest = "sha256:" + hashlib.sha256(source_path.read_bytes()).hexdigest()
    loaded = np.load(source_path, allow_pickle=False)
    input_ids = [row["id"] for row in graph.get("inputs", [])]
    source_base = {
        "kind": "file",
        "path": str(source_path),
        "digest": digest,
    }
    if isinstance(loaded, np.ndarray):
        if len(input_ids) != 1:
            raise ValueError(
                "a dense NPY can bind only a graph with exactly one input; "
                f"graph inputs are {input_ids}"
            )
        return (
            DenseArrayBinding(
                input_ids[0],
                torch.as_tensor(loaded),
                {**source_base, "array": None},
            ),
        )
    try:
        arrays = {key: loaded[key] for key in loaded.files}
    finally:
        loaded.close()
    if set(arrays) == {"input_spikes"} and len(input_ids) == 1:
        arrays = {input_ids[0]: arrays["input_spikes"]}
        keys = {input_ids[0]: "input_spikes"}
    else:
        keys = {key: key for key in arrays}
    return tuple(
        DenseArrayBinding(
            input_id,
            torch.as_tensor(value),
            {**source_base, "array": keys[input_id]},
        )
        for input_id, value in arrays.items()
    )


def load_target_array_bindings(
    path: str | Path, training: Mapping[str, Any]
) -> tuple[TargetArrayBinding, ...]:
    """Load NPY/NPZ class targets against the recipe's named objectives."""
    source_path = Path(path)
    digest = "sha256:" + hashlib.sha256(source_path.read_bytes()).hexdigest()
    target_ids = sorted(
        {row["target"] for row in training.get("objectives", []) if "target" in row}
    )
    loaded = np.load(source_path, allow_pickle=False)
    source = {"kind": "file", "path": str(source_path), "digest": digest}
    if isinstance(loaded, np.ndarray):
        if len(target_ids) != 1:
            raise ValueError(
                f"a target NPY requires exactly one recipe target; targets are {target_ids}"
            )
        arrays = {target_ids[0]: loaded}
        keys = {target_ids[0]: None}
    else:
        try:
            arrays = {key: loaded[key] for key in loaded.files}
        finally:
            loaded.close()
        keys = {key: key for key in arrays}
    if set(arrays) != set(target_ids):
        raise ValueError(
            f"target array ids do not match recipe targets; expected={target_ids}, got={sorted(arrays)}"
        )
    return tuple(
        TargetArrayBinding(
            target_id,
            torch.as_tensor(arrays[target_id]),
            {**source, "array": keys[target_id]},
        )
        for target_id in target_ids
    )


def resolve_target_array_bindings(
    training: Mapping[str, Any],
    *,
    bindings: Sequence[TargetArrayBinding] = (),
    targets: Mapping[str, torch.Tensor] | None = None,
    sample_count: int,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Validate named classification/regression targets and retain provenance."""
    if bindings and targets:
        raise ValueError("provide target bindings or target tensors, not both")
    if not bindings:
        bindings = tuple(
            TargetArrayBinding(name, value, {"kind": "memory"})
            for name, value in (targets or {}).items()
        )
    expected = {
        row["target"] for row in training.get("objectives", []) if "target" in row
    }
    by_name = {binding.target_id: binding for binding in bindings}
    if len(by_name) != len(bindings):
        raise ValueError("duplicate target array binding")
    if set(by_name) != expected:
        raise ValueError(
            f"target ids do not match recipe; missing={sorted(expected - set(by_name))}, unexpected={sorted(set(by_name) - expected)}"
        )
    resolved = {}
    rows = []
    integer_dtypes = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
    for target_id in sorted(by_name):
        binding = by_name[target_id]
        value = binding.value
        classification = any(
            X.classification(row)
            for row in training.get("objectives", [])
            if row.get("target") == target_id
        )
        if (
            value.ndim < 1
            or value.shape[0] != sample_count
            or (classification and value.ndim != 1)
        ):
            raise ValueError(
                f"training target {target_id} expected shape [{sample_count}], got {list(value.shape)}"
            )
        if classification and value.dtype not in integer_dtypes:
            raise ValueError(f"training target {target_id} must use an integer dtype")
        if not classification and (
            not (value.is_floating_point() or value.dtype in integer_dtypes)
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"training target {target_id} must use finite real values")
        resolved[target_id] = value.to(
            device=device, dtype=torch.long if classification else value.dtype
        )
        rows.append(
            {
                "target_id": target_id,
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "source": dict(binding.source),
            }
        )
    return resolved, rows


def load_event_stream_bindings(
    path: str | Path, graph: Mapping[str, Any]
) -> tuple[EventStreamBinding, ...]:
    """Load named sparse spike coordinates from a replayable NPZ file."""
    source_path = Path(path)
    if source_path.suffix.lower() != ".npz":
        raise ValueError("event-stream replay requires an NPZ file")
    digest = "sha256:" + hashlib.sha256(source_path.read_bytes()).hexdigest()
    loaded = np.load(source_path, allow_pickle=False)
    try:
        arrays = {key: loaded[key] for key in loaded.files}
    finally:
        loaded.close()
    input_ids = [row["id"] for row in graph.get("inputs", [])]
    fields = ("steps", "batches", "channels", "steps_count", "batch_size")
    plain = set(fields)
    use_plain = len(input_ids) == 1 and set(arrays) == plain
    expected = (
        plain
        if use_plain
        else {f"{input_id}.{field}" for input_id in input_ids for field in fields}
    )
    if set(arrays) != expected:
        raise ValueError(
            "event-stream NPZ keys do not match graph inputs; "
            f"expected={sorted(expected)}, got={sorted(arrays)}"
        )
    source_base = {"kind": "file", "path": str(source_path), "digest": digest}
    bindings = []
    for input_id in input_ids:

        def key(field: str) -> str:
            return field if use_plain else f"{input_id}.{field}"

        steps_count_value = np.asarray(arrays[key("steps_count")])
        batch_size_value = np.asarray(arrays[key("batch_size")])
        if steps_count_value.size != 1 or batch_size_value.size != 1:
            raise ValueError(
                f"event input {input_id} steps_count and batch_size must be scalars"
            )
        if not np.issubdtype(steps_count_value.dtype, np.integer) or not np.issubdtype(
            batch_size_value.dtype, np.integer
        ):
            raise ValueError(
                f"event input {input_id} steps_count and batch_size must use integer dtypes"
            )
        bindings.append(
            EventStreamBinding(
                input_id=input_id,
                steps=torch.as_tensor(arrays[key("steps")]),
                batches=torch.as_tensor(arrays[key("batches")]),
                channels=torch.as_tensor(arrays[key("channels")]),
                steps_count=int(steps_count_value.item()),
                batch_size=int(batch_size_value.item()),
                source={
                    **source_base,
                    "arrays": {field: key(field) for field in fields},
                },
            )
        )
    return tuple(bindings)


def resolve_dense_array_bindings(
    graph: Mapping[str, Any],
    *,
    bindings: Sequence[DenseArrayBinding] = (),
    inputs: Mapping[str, torch.Tensor] | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    protocol: Mapping[str, Any] | None = None,
) -> ResolvedDenseInputs:
    """Validate dense arrays, resolve symbolic axes, and freeze run provenance."""
    if bindings and inputs:
        raise ValueError("provide dense input bindings or input tensors, not both")
    if not bindings:
        bindings = tuple(
            DenseArrayBinding(name, value, {"kind": "memory"})
            for name, value in (inputs or {}).items()
        )
    specs = {row["id"]: row for row in graph.get("inputs", [])}
    by_name: dict[str, DenseArrayBinding] = {}
    for binding in bindings:
        if binding.input_id in by_name:
            raise ValueError(f"duplicate dense binding for input {binding.input_id}")
        by_name[binding.input_id] = binding
    if set(by_name) != set(specs):
        missing = sorted(set(specs) - set(by_name))
        unexpected = sorted(set(by_name) - set(specs))
        raise ValueError(
            f"dense input ids do not match graph inputs; missing={missing}, unexpected={unexpected}"
        )

    resolved: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    leading_shape: tuple[int, int] | None = None
    masks: list[str] = []
    for input_id in sorted(specs):
        spec = specs[input_id]
        binding = by_name[input_id]
        value = binding.value
        declared = spec.get("shape", [])
        if len(declared) < 2 or declared[:2] != ["time", "batch"]:
            raise ValueError(
                f"input {input_id} dense binding requires declared shape beginning with ['time', 'batch']"
            )
        if value.ndim == len(declared) - 1 and declared[1] == "batch":
            value = value.unsqueeze(1)
        if value.ndim != len(declared):
            raise ValueError(
                f"input {input_id} rank expected {len(declared)}, got {value.ndim}"
            )
        expected_tail = tuple(int(axis) for axis in declared[2:])
        if tuple(value.shape[2:]) != expected_tail:
            raise ValueError(
                f"input {input_id} trailing shape expected {expected_tail}, got {tuple(value.shape[2:])}"
            )
        current_leading = (int(value.shape[0]), int(value.shape[1]))
        if leading_shape is None:
            leading_shape = current_leading
        elif current_leading != leading_shape:
            raise ValueError(
                f"input {input_id} leading shape expected {leading_shape}, got {current_leading}"
            )
        signal_type = spec.get("signal_type")
        if signal_type == "mask":
            if value.dtype != torch.bool:
                if not torch.all((value == 0) | (value == 1)):
                    raise ValueError(
                        f"input {input_id} mask values must be boolean or zero/one"
                    )
                value = value.bool()
            masks.append(input_id)
        elif not (value.is_floating_point() or value.dtype == torch.bool):
            value = value.float()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"input {input_id} contains non-finite values")
        if signal_type == "spikes" and not torch.all((value == 0) | (value == 1)):
            raise ValueError(
                f"input {input_id} spike values must be boolean or zero/one"
            )
        if signal_type != "mask":
            value = value.float()
        value = value.to(device)
        resolved[input_id] = value
        rows.append(
            {
                "input_id": input_id,
                "representation": "dense_array",
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "signal_type": signal_type,
                "unit": spec.get("unit"),
                "source": dict(binding.source),
            }
        )
    assert leading_shape is not None
    dt_ms = float(graph["timebase"]["dt"]["value"])
    supplied = dict(protocol or {})
    dataset = dict(supplied.pop("dataset", {}))
    reserved = {
        "schema",
        "binding_schema",
        "representation",
        "inputs",
        "timing",
        "masks",
        "seeds",
    }
    if reserved & supplied.keys():
        raise ValueError(
            f"execution protocol cannot override reserved fields {sorted(reserved & supplied.keys())}"
        )
    dataset.setdefault("identity", None)
    dataset.setdefault("split", None)
    dataset.setdefault("sample_cap", leading_shape[1])
    dataset.setdefault("batch_size", leading_shape[1])
    dataset.setdefault("shuffle", None)
    execution_protocol = {
        "schema": EXECUTION_PROTOCOL_SCHEMA,
        "binding_schema": DENSE_ARRAY_BINDING_SCHEMA,
        "representation": "dense_array",
        "inputs": rows,
        "dataset": dataset,
        "timing": {
            "dt_ms": dt_ms,
            "steps": leading_shape[0],
            "duration_ms": leading_shape[0] * dt_ms,
        },
        "masks": masks,
        "seeds": {"execution": int(seed)},
        **supplied,
    }
    try:
        json.dumps(execution_protocol, sort_keys=True)
    except TypeError as exc:
        raise ValueError(
            f"execution protocol must be JSON-serializable: {exc}"
        ) from exc
    return ResolvedDenseInputs(tensors=resolved, protocol=execution_protocol)


def resolve_event_stream_bindings(
    graph: Mapping[str, Any],
    *,
    bindings: Sequence[EventStreamBinding],
    device: str | torch.device = "cpu",
    seed: int = 0,
    protocol: Mapping[str, Any] | None = None,
) -> ResolvedDenseInputs:
    """Validate sparse spike coordinates and materialize binary graph inputs."""
    specs = {row["id"]: row for row in graph.get("inputs", [])}
    by_name: dict[str, EventStreamBinding] = {}
    for binding in bindings:
        if binding.input_id in by_name:
            raise ValueError(f"duplicate event binding for input {binding.input_id}")
        by_name[binding.input_id] = binding
    if set(by_name) != set(specs):
        missing = sorted(set(specs) - set(by_name))
        unexpected = sorted(set(by_name) - set(specs))
        raise ValueError(
            f"event input ids do not match graph inputs; missing={missing}, unexpected={unexpected}"
        )

    resolved: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    leading_shape: tuple[int, int] | None = None
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    for input_id in sorted(specs):
        spec = specs[input_id]
        binding = by_name[input_id]
        declared = spec.get("shape", [])
        if declared[:2] != ["time", "batch"] or len(declared) != 3:
            raise ValueError(
                f"input {input_id} event binding requires shape ['time', 'batch', channels]"
            )
        if spec.get("signal_type") != "spikes":
            raise ValueError(
                f"input {input_id} event binding requires signal_type spikes"
            )
        channels_count = int(declared[2])
        if (
            not isinstance(binding.steps_count, Integral)
            or isinstance(binding.steps_count, bool)
            or not isinstance(binding.batch_size, Integral)
            or isinstance(binding.batch_size, bool)
        ):
            raise ValueError(
                f"event input {input_id} steps_count and batch_size must be integers"
            )
        steps_count = int(binding.steps_count)
        batch_size = int(binding.batch_size)
        if steps_count <= 0 or batch_size <= 0:
            raise ValueError(
                f"event input {input_id} steps_count and batch_size must be positive"
            )
        coordinates = (binding.steps, binding.batches, binding.channels)
        if any(value.ndim != 1 for value in coordinates):
            raise ValueError(
                f"event input {input_id} coordinates must be one-dimensional"
            )
        lengths = {int(value.numel()) for value in coordinates}
        if len(lengths) != 1:
            raise ValueError(f"event input {input_id} coordinate lengths must match")
        if any(value.dtype not in integer_dtypes for value in coordinates):
            raise ValueError(
                f"event input {input_id} coordinates must use integer dtypes"
            )
        steps = binding.steps.to(dtype=torch.int64, device="cpu")
        batches = binding.batches.to(dtype=torch.int64, device="cpu")
        channels = binding.channels.to(dtype=torch.int64, device="cpu")
        bounds = (
            ("step", steps, steps_count),
            ("batch", batches, batch_size),
            ("channel", channels, channels_count),
        )
        for label, values, upper in bounds:
            if torch.any(values < 0) or torch.any(values >= upper):
                raise ValueError(
                    f"event input {input_id} {label} coordinates must be in [0, {upper})"
                )
        flat = (steps * batch_size + batches) * channels_count + channels
        if flat.numel() > 1:
            differences = flat[1:] - flat[:-1]
            if torch.any(differences < 0):
                raise ValueError(
                    f"event input {input_id} coordinates must be ordered by step, batch, channel"
                )
            if torch.any(differences == 0):
                raise ValueError(
                    f"event input {input_id} contains duplicate coordinates"
                )
        current_leading = (steps_count, batch_size)
        if leading_shape is None:
            leading_shape = current_leading
        elif current_leading != leading_shape:
            raise ValueError(
                f"event input {input_id} leading shape expected {leading_shape}, got {current_leading}"
            )
        value = torch.zeros(
            (steps_count, batch_size, channels_count),
            dtype=torch.float32,
            device=device,
        )
        if flat.numel():
            value[
                steps.to(device=device),
                batches.to(device=device),
                channels.to(device=device),
            ] = 1.0
        resolved[input_id] = value
        rows.append(
            {
                "input_id": input_id,
                "representation": "event_stream",
                "shape": list(value.shape),
                "dtype": "float32",
                "signal_type": "spikes",
                "unit": spec.get("unit"),
                "event_count": int(flat.numel()),
                "source": dict(binding.source),
            }
        )
    if leading_shape is None:
        raise ValueError("graph execution requires at least one event input binding")
    dt_ms = float(graph["timebase"]["dt"]["value"])
    supplied = dict(protocol or {})
    dataset = dict(supplied.pop("dataset", {}))
    reserved = {
        "schema",
        "binding_schema",
        "representation",
        "inputs",
        "dataset",
        "timing",
        "masks",
        "seeds",
        "resolution",
    }
    if reserved & supplied.keys():
        raise ValueError(
            f"execution protocol cannot override reserved fields {sorted(reserved & supplied.keys())}"
        )
    dataset.setdefault("identity", None)
    dataset.setdefault("split", None)
    dataset.setdefault("sample_cap", leading_shape[1])
    dataset.setdefault("batch_size", leading_shape[1])
    dataset.setdefault("shuffle", None)
    execution_protocol = {
        "schema": EXECUTION_PROTOCOL_SCHEMA,
        "binding_schema": EVENT_STREAM_BINDING_SCHEMA,
        "representation": "event_stream",
        "inputs": rows,
        "dataset": dataset,
        "timing": {
            "dt_ms": dt_ms,
            "steps": leading_shape[0],
            "duration_ms": leading_shape[0] * dt_ms,
        },
        "masks": [],
        "seeds": {"execution": int(seed)},
        "resolution": {
            "coordinates": "zero_based_integer_steps",
            "ordering": "step,batch,channel",
            "duplicates": "reject",
            "materialization": "binary_dense",
        },
        **supplied,
    }
    try:
        json.dumps(execution_protocol, sort_keys=True)
    except TypeError as exc:
        raise ValueError(
            f"execution protocol must be JSON-serializable: {exc}"
        ) from exc
    return ResolvedDenseInputs(tensors=resolved, protocol=execution_protocol)


def resolve_input_bindings(
    graph: Mapping[str, Any],
    *,
    dense_bindings: Sequence[DenseArrayBinding] = (),
    event_bindings: Sequence[EventStreamBinding] = (),
    poisson_bindings: Sequence[PoissonInputBinding] = (),
    inputs: Mapping[str, torch.Tensor] | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    protocol: Mapping[str, Any] | None = None,
) -> ResolvedDenseInputs:
    """Resolve dense, event-stream, generated-Poisson, or mixed graph inputs."""
    if dense_bindings and inputs:
        raise ValueError("provide dense input bindings or input tensors, not both")
    if inputs:
        dense_bindings = tuple(
            DenseArrayBinding(name, value, {"kind": "memory"})
            for name, value in inputs.items()
        )
    dense_ids = {binding.input_id for binding in dense_bindings}
    event_ids = {binding.input_id for binding in event_bindings}
    poisson_ids = {binding.input_id for binding in poisson_bindings}
    overlap = sorted(
        (dense_ids & event_ids) | (dense_ids & poisson_ids) | (event_ids & poisson_ids)
    )
    if overlap:
        raise ValueError(
            f"graph inputs cannot have dense and event bindings: {overlap}"
        )
    graph_ids = {row["id"] for row in graph.get("inputs", [])}
    if dense_ids | event_ids | poisson_ids != graph_ids:
        missing = sorted(graph_ids - dense_ids - event_ids - poisson_ids)
        unexpected = sorted((dense_ids | event_ids | poisson_ids) - graph_ids)
        raise ValueError(
            f"input ids do not match graph inputs; missing={missing}, unexpected={unexpected}"
        )
    if poisson_bindings:
        if dense_bindings or event_bindings:
            raise ValueError(
                "Poisson bindings cannot yet be mixed with replay bindings"
            )
        return resolve_poisson_input_bindings(
            graph,
            bindings=poisson_bindings,
            device=device,
            seed=seed,
            protocol=protocol,
        )
    if not event_bindings:
        return resolve_dense_array_bindings(
            graph,
            bindings=dense_bindings,
            device=device,
            seed=seed,
            protocol=protocol,
        )
    if not dense_bindings:
        return resolve_event_stream_bindings(
            graph,
            bindings=event_bindings,
            device=device,
            seed=seed,
            protocol=protocol,
        )

    def graph_with_inputs(input_ids: set[str]) -> dict[str, Any]:
        return {
            **graph,
            "inputs": [
                row for row in graph.get("inputs", []) if row["id"] in input_ids
            ],
        }

    dense = resolve_dense_array_bindings(
        graph_with_inputs(dense_ids),
        bindings=dense_bindings,
        device=device,
        seed=seed,
        protocol=protocol,
    )
    events = resolve_event_stream_bindings(
        graph_with_inputs(event_ids),
        bindings=event_bindings,
        device=device,
        seed=seed,
        protocol=protocol,
    )
    if dense.protocol["timing"] != events.protocol["timing"]:
        raise ValueError(
            "dense and event input bindings must resolve to the same timestep, duration, and batch shape"
        )
    execution_protocol = {
        "schema": EXECUTION_PROTOCOL_SCHEMA,
        "binding_schema": MIXED_INPUT_BINDING_SCHEMA,
        "representation": "mixed",
        "inputs": sorted(
            [*dense.protocol["inputs"], *events.protocol["inputs"]],
            key=lambda row: row["input_id"],
        ),
        "dataset": dense.protocol["dataset"],
        "timing": dense.protocol["timing"],
        "masks": dense.protocol["masks"],
        "seeds": dense.protocol["seeds"],
        "resolution": {"event_stream": events.protocol["resolution"]},
        **{
            key: value
            for key, value in dense.protocol.items()
            if key
            not in {
                "schema",
                "binding_schema",
                "representation",
                "inputs",
                "dataset",
                "timing",
                "masks",
                "seeds",
                "resolution",
            }
        },
    }
    return ResolvedDenseInputs(
        tensors={**dense.tensors, **events.tensors}, protocol=execution_protocol
    )


def resolve_poisson_input_bindings(
    graph: Mapping[str, Any],
    *,
    bindings: Sequence[PoissonInputBinding],
    device: str | torch.device = "cpu",
    seed: int = 0,
    protocol: Mapping[str, Any] | None = None,
) -> ResolvedDenseInputs:
    """Generate reproducible fixed or per-presentation categorical Poisson spikes."""
    specs = {row["id"]: row for row in graph.get("inputs", [])}
    by_name = {binding.input_id: binding for binding in bindings}
    if len(by_name) != len(bindings):
        raise ValueError("duplicate Poisson binding for a graph input")
    if set(by_name) != set(specs):
        missing = sorted(set(specs) - set(by_name))
        unexpected = sorted(set(by_name) - set(specs))
        raise ValueError(
            f"Poisson input ids do not match graph inputs; missing={missing}, unexpected={unexpected}"
        )
    dt_ms = float(graph["timebase"]["dt"]["value"])
    tensors: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    leading_shape: tuple[int, int] | None = None
    for input_id in sorted(specs):
        spec = specs[input_id]
        binding = by_name[input_id]
        declared = spec.get("shape", [])
        if declared[:2] != ["time", "batch"] or len(declared) != 3:
            raise ValueError(
                f"input {input_id} Poisson binding requires shape ['time', 'batch', channels]"
            )
        if spec.get("signal_type") != "spikes":
            raise ValueError(
                f"input {input_id} Poisson binding requires signal_type spikes"
            )
        if binding.steps_count <= 0 or binding.batch_size <= 0:
            raise ValueError(
                f"Poisson input {input_id} steps_count and batch_size must be positive"
            )
        rates = tuple(float(rate) for rate in binding.rates_hz)
        if not rates or any(not math.isfinite(rate) or rate < 0 for rate in rates):
            raise ValueError(
                f"Poisson input {input_id} rates must be finite and non-negative"
            )
        if not binding.categorical and len(rates) != 1:
            raise ValueError(
                f"fixed-rate Poisson input {input_id} requires exactly one rate"
            )
        if max(rates) * dt_ms / 1000.0 > 1.0:
            raise ValueError(
                f"Poisson input {input_id} rate times dt exceeds probability one"
            )
        current_leading = (int(binding.steps_count), int(binding.batch_size))
        if leading_shape is None:
            leading_shape = current_leading
        elif current_leading != leading_shape:
            raise ValueError(
                f"Poisson input {input_id} leading shape expected {leading_shape}, got {current_leading}"
            )
        generator = torch.Generator(device="cpu").manual_seed(int(binding.seed))
        if binding.categorical:
            indices = torch.randint(
                len(rates), (binding.batch_size,), generator=generator
            )
            realized = torch.tensor(rates, dtype=torch.float32)[indices]
        else:
            realized = torch.full((binding.batch_size,), rates[0], dtype=torch.float32)
        probability = realized.reshape(1, -1, 1) * dt_ms / 1000.0
        value = (
            (
                torch.rand(
                    binding.steps_count,
                    binding.batch_size,
                    int(declared[2]),
                    generator=generator,
                )
                < probability
            )
            .float()
            .to(device)
        )
        tensors[input_id] = value
        rows.append(
            {
                "input_id": input_id,
                "representation": "poisson",
                "shape": list(value.shape),
                "dtype": "float32",
                "signal_type": "spikes",
                "unit": spec.get("unit"),
                "protocol": "categorical_rate" if binding.categorical else "fixed_rate",
                "rates_hz": list(rates),
                "realized_rates_hz": realized.tolist(),
                "seed": int(binding.seed),
                "selection": "uniform_independent_per_presentation"
                if binding.categorical
                else "constant",
            }
        )
    assert leading_shape is not None
    supplied = dict(protocol or {})
    dataset = dict(supplied.pop("dataset", {}))
    if {
        "schema",
        "binding_schema",
        "representation",
        "inputs",
        "timing",
        "seeds",
    } & supplied.keys():
        raise ValueError("execution protocol cannot override reserved Poisson fields")
    dataset.setdefault("identity", None)
    dataset.setdefault("split", None)
    dataset.setdefault("sample_cap", leading_shape[1])
    dataset.setdefault("batch_size", leading_shape[1])
    dataset.setdefault("shuffle", None)
    execution_protocol = {
        "schema": EXECUTION_PROTOCOL_SCHEMA,
        "binding_schema": POISSON_INPUT_BINDING_SCHEMA,
        "representation": "poisson",
        "inputs": rows,
        "dataset": dataset,
        "timing": {
            "dt_ms": dt_ms,
            "steps": leading_shape[0],
            "duration_ms": leading_shape[0] * dt_ms,
        },
        "masks": [],
        "seeds": {
            "execution": int(seed),
            "poisson": {row["input_id"]: row["seed"] for row in rows},
        },
        "resolution": {
            "distribution": "Bernoulli discretization of homogeneous Poisson",
            "rate_selection": "per_presentation",
        },
        **supplied,
    }
    json.dumps(execution_protocol, sort_keys=True)
    return ResolvedDenseInputs(tensors=tensors, protocol=execution_protocol)


def resolve_dataset_snapshot_binding(
    graph: Mapping[str, Any],
    binding: DatasetSnapshotBinding,
    *,
    device: str | torch.device = "cpu",
    execution_seed: int = 0,
    protocol: Mapping[str, Any] | None = None,
    _snapshot=None,
    _indices=None,
    _spike_seed=None,
    _rate_seed=None,
    _validated=False,
) -> tuple[ResolvedDenseInputs, tuple[TargetArrayBinding, ...]]:
    """Load, select, and encode one immutable MNIST/SHD-style NPZ snapshot."""
    specs = {row["id"]: row for row in graph.get("inputs", [])}
    if set(specs) != {binding.input_id}:
        raise ValueError(
            "dataset snapshot binding requires exactly its one named graph input"
        )
    spec = specs[binding.input_id]
    declared = spec.get("shape", [])
    if declared[:2] != ["time", "batch"] or len(declared) != 3:
        raise ValueError(
            "dataset snapshot binding requires graph input shape ['time', 'batch', channels]"
        )
    if spec.get("signal_type") != "spikes":
        raise ValueError("dataset snapshot binding requires a spike graph input")
    source_path = Path(binding.path)
    if source_path.suffix.lower() != ".npz":
        raise ValueError("dataset snapshot binding requires an NPZ file")
    if _snapshot is None:
        source_digest = _file_digest(source_path)
        with np.load(source_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
    else:
        arrays, source_digest = _snapshot
    if not binding.dataset_id or not binding.split:
        raise ValueError("dataset snapshot identity and split must be non-empty")
    if binding.label_key not in arrays:
        raise ValueError(
            f"dataset snapshot is missing labels key {binding.label_key!r}"
        )
    labels = np.asarray(arrays[binding.label_key])
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            "dataset snapshot labels must be a one-dimensional integer array"
        )
    sample_count = int(labels.shape[0])
    if sample_count <= 0:
        raise ValueError("dataset snapshot must contain at least one sample")
    cap = sample_count if binding.sample_cap is None else int(binding.sample_cap)
    if cap <= 0 or cap > sample_count:
        raise ValueError(f"dataset snapshot sample cap must be in [1, {sample_count}]")
    if _indices is None:
        order = torch.arange(sample_count)
        if binding.shuffle:
            generator = torch.Generator(device="cpu").manual_seed(
                int(binding.order_seed)
            )
            order = torch.randperm(sample_count, generator=generator)
        selected = order[:cap].numpy()
    else:
        selected = np.asarray(_indices, dtype=np.int64)
        cap = len(selected)
    selected_labels = labels[selected].astype(np.int64, copy=False)
    encoder = binding.encoder
    dt_ms = float(graph["timebase"]["dt"]["value"])
    channels = int(declared[2])
    encoder_row: dict[str, Any]
    if encoder.kind == "rate_poisson" and (
        encoder.duration_ms is None
        or (encoder.max_rate_hz is None and encoder.rates_hz is None)
    ):
        raise ValueError(
            "rate-Poisson encoder requires explicit duration_ms and max_rate_hz"
        )
    if encoder.kind != "rate_poisson" and encoder.rates_hz is not None:
        raise ValueError("rates_hz is supported only by rate_poisson")
    if encoder.kind == "prebinned_spikes" and (
        encoder.duration_ms is not None
        or encoder.max_rate_hz is not None
        or encoder.seed != 0
    ):
        raise ValueError(
            "prebinned encoder does not accept duration, max rate, or a stochastic seed"
        )
    if encoder.kind == "event_bin" and (
        encoder.duration_ms is None
        or encoder.max_rate_hz is not None
        or encoder.seed != 0
    ):
        raise ValueError(
            "event-bin encoder requires duration_ms and does not accept max rate or a stochastic seed"
        )
    if encoder.kind in {"rate_poisson", "prebinned_spikes"}:
        if binding.feature_key not in arrays:
            raise ValueError(
                f"dataset snapshot is missing feature key {binding.feature_key!r}"
            )
        features = np.asarray(arrays[binding.feature_key])
    if encoder.kind == "rate_poisson":
        if features.shape != (sample_count, channels):
            raise ValueError(
                f"rate-Poisson dataset features expected shape {(sample_count, channels)}, got {features.shape}"
            )
        if not np.issubdtype(features.dtype, np.floating):
            raise ValueError("rate-Poisson dataset features must use a floating dtype")
        if not _validated and (
            not np.isfinite(features).all()
            or np.any(features < 0)
            or np.any(features > 1)
        ):
            raise ValueError("rate-Poisson dataset features must be finite in [0, 1]")
        duration_ms = float(encoder.duration_ms or 0)
        if encoder.rates_hz is not None:
            if encoder.max_rate_hz is not None or not encoder.rates_hz:
                raise ValueError(
                    "rate-Poisson requires either max_rate_hz or a non-empty rates_hz sequence"
                )
            choices = tuple(float(rate) for rate in encoder.rates_hz)
            if any(not math.isfinite(rate) or rate < 0 for rate in choices):
                raise ValueError("categorical rates_hz must be finite and non-negative")
            max_rate_hz = max(choices)
        else:
            choices = None
            max_rate_hz = float(encoder.max_rate_hz or 0)
        raw_steps = duration_ms / dt_ms
        if (
            duration_ms <= 0
            or not math.isclose(raw_steps, round(raw_steps), abs_tol=1e-9)
            or not math.isfinite(max_rate_hz)
            or max_rate_hz < 0
            or max_rate_hz * dt_ms / 1000.0 > 1
        ):
            raise ValueError(
                "rate-Poisson encoder requires timestep-aligned positive duration and a supported finite non-negative max rate"
            )
        rates = torch.as_tensor(features[selected], dtype=torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(
            int(encoder.seed if _spike_seed is None else _spike_seed)
        )
        if choices is not None:
            rate_generator = torch.Generator(device="cpu").manual_seed(
                int(encoder.seed if _rate_seed is None else _rate_seed)
            )
            selected_rates = torch.tensor(choices, dtype=torch.float32)[
                torch.randint(len(choices), (cap,), generator=rate_generator)
            ]
            probability = (
                rates.unsqueeze(0) * selected_rates.reshape(1, cap, 1) * dt_ms / 1000.0
            )
        else:
            selected_rates = None
            probability = rates.unsqueeze(0) * max_rate_hz * dt_ms / 1000.0
        spikes = (
            torch.rand(
                int(round(raw_steps)),
                cap,
                channels,
                generator=generator,
                dtype=torch.float32,
            )
            < probability
        ).float()
        encoder_row = {
            "kind": encoder.kind,
            "duration_ms": duration_ms,
            "max_rate_hz": max_rate_hz,
            "seed": int(encoder.seed),
            "distribution": "Bernoulli discretization of feature-scaled homogeneous Poisson",
            **(
                {
                    "rates_hz": list(choices),
                    "selected_rates_hz": selected_rates.tolist(),
                }
                if choices is not None
                else {}
            ),
        }
    elif encoder.kind == "prebinned_spikes":
        if features.ndim != 3 or features.shape[1:] != (sample_count, channels):
            raise ValueError(
                f"prebinned dataset features expected [time, {sample_count}, {channels}]"
            )
        if not _validated and not np.all((features == 0) | (features == 1)):
            raise ValueError("prebinned dataset spikes must be binary")
        spikes = torch.as_tensor(features[:, selected, :], dtype=torch.float32)
        encoder_row = {
            "kind": encoder.kind,
            "duration_ms": int(features.shape[0]) * dt_ms,
            "seed": None,
        }
    elif encoder.kind == "event_bin":
        event_keys = ("event_sample", "event_time_ms", "event_channel")
        missing = [key for key in event_keys if key not in arrays]
        if missing:
            raise ValueError(f"event dataset snapshot is missing keys {missing}")
        samples = np.asarray(arrays["event_sample"])
        times = np.asarray(arrays["event_time_ms"])
        event_channels = np.asarray(arrays["event_channel"])
        if (
            samples.ndim != 1
            or times.ndim != 1
            or event_channels.ndim != 1
            or not (len(samples) == len(times) == len(event_channels))
            or not np.issubdtype(samples.dtype, np.integer)
            or not np.issubdtype(event_channels.dtype, np.integer)
            or not np.issubdtype(times.dtype, np.floating)
        ):
            raise ValueError(
                "event dataset coordinates must be equal-length sample/channel integer and time floating arrays"
            )
        duration_ms = float(encoder.duration_ms or 0)
        raw_steps = duration_ms / dt_ms
        if duration_ms <= 0 or not math.isclose(
            raw_steps, round(raw_steps), abs_tol=1e-9
        ):
            raise ValueError(
                "event-bin encoder duration must be a positive integer number of timesteps"
            )
        if (
            not np.isfinite(times).all()
            or np.any(samples < 0)
            or np.any(samples >= sample_count)
            or np.any(event_channels < 0)
            or np.any(event_channels >= channels)
            or np.any(times < 0)
            or np.any(times >= duration_ms)
        ):
            raise ValueError("event dataset coordinates are out of snapshot bounds")
        selected_position = {
            int(sample): index for index, sample in enumerate(selected)
        }
        spikes = torch.zeros(int(round(raw_steps)), cap, channels)
        retained = 0
        collisions = 0
        for sample, time_ms, channel in zip(
            samples, times, event_channels, strict=True
        ):
            batch = selected_position.get(int(sample))
            if batch is None:
                continue
            step = min(int(float(time_ms) / dt_ms), spikes.shape[0] - 1)
            collisions += int(spikes[step, batch, int(channel)] != 0)
            spikes[step, batch, int(channel)] = 1
            retained += 1
        encoder_row = {
            "kind": encoder.kind,
            "duration_ms": duration_ms,
            "seed": None,
            "timestamp_unit": "ms",
            "binning": "floor_left_closed_right_open",
            "retained_events": retained,
            "binary_collisions": collisions,
        }
    elif encoder.kind == "custom":
        spec_row = {"definition": encoder.definition, "config": dict(encoder.config)}
        definition = E.resolve("encoder", spec_row)
        spikes = definition.function(
            arrays,
            selected,
            dt_ms=dt_ms,
            channels=channels,
            config=encoder.config,
            seed=encoder.seed if _spike_seed is None else _spike_seed,
        )
        if (
            not isinstance(spikes, torch.Tensor)
            or spikes.ndim != 3
            or spikes.shape[1:] != (cap, channels)
        ):
            raise ValueError(
                f"{definition.name} must return (time, selected_samples, channels) spikes"
            )
        encoder_row = {
            "kind": "custom",
            "definition": encoder.definition,
            "config": dict(encoder.config),
            "seed": encoder.seed,
        }
    else:
        raise ValueError(f"unsupported dataset encoder {encoder.kind!r}")
    source = {
        "kind": "dataset_snapshot",
        "path": str(source_path),
        "digest": source_digest,
        "arrays": {
            "labels": binding.label_key,
            **(
                {"features": binding.feature_key}
                if encoder.kind != "event_bin"
                else {
                    "sample": "event_sample",
                    "time_ms": "event_time_ms",
                    "channel": "event_channel",
                }
            ),
        },
    }
    resolved = resolve_dense_array_bindings(
        graph,
        bindings=(DenseArrayBinding(binding.input_id, spikes, source),),
        device=device,
        seed=execution_seed,
        protocol={
            "dataset": {
                "identity": binding.dataset_id,
                "split": binding.split,
                "sample_cap": cap,
                "batch_size": cap,
                "shuffle": bool(binding.shuffle),
            }
        },
    )
    supplied = dict(protocol or {})
    supplied_dataset = dict(supplied.pop("dataset", {}))
    expected_dataset = dict(resolved.protocol["dataset"])
    unknown_dataset = sorted(set(supplied_dataset) - set(expected_dataset))
    conflicts = sorted(
        key
        for key, value in supplied_dataset.items()
        if value is not None and value != expected_dataset[key]
    )
    if unknown_dataset or conflicts:
        raise ValueError(
            "dataset execution protocol metadata does not match binding; "
            f"unknown={unknown_dataset}, conflicts={conflicts}"
        )
    reserved = {
        "schema",
        "binding_schema",
        "representation",
        "inputs",
        "timing",
        "masks",
        "seeds",
        "dataset_binding",
    }
    if reserved & supplied.keys():
        raise ValueError(
            f"dataset execution protocol cannot override reserved fields {sorted(reserved & supplied.keys())}"
        )
    protocol = {
        **resolved.protocol,
        "binding_schema": DATASET_SNAPSHOT_BINDING_SCHEMA,
        "representation": "dataset_snapshot",
        "dataset_binding": {
            "source": source,
            "sample_count": sample_count,
            "selected_indices": selected.tolist(),
            "order_seed": int(binding.order_seed),
            "encoder": encoder_row,
            "target_id": binding.target_id,
        },
        "seeds": {
            **resolved.protocol["seeds"],
            "dataset_order": int(binding.order_seed),
            "encoder": int(encoder.seed) if encoder.kind == "rate_poisson" else None,
        },
        **supplied,
    }
    targets = (
        (
            TargetArrayBinding(
                binding.target_id,
                torch.as_tensor(selected_labels),
                source={**source, "array": binding.label_key},
            ),
        )
        if binding.target_id
        else ()
    )
    return ResolvedDenseInputs(resolved.tensors, protocol), targets


def graph_capability_issues(graph: Mapping[str, Any]) -> list[CapabilityIssue]:
    """Return precise graph-executor capability failures."""
    issues: list[CapabilityIssue] = []
    neuron_capabilities: set[str] = {
        "coba_lif",
        "cuba_lif",
        "leaky_integrator",
        "custom_neuron",
    }
    synapse_capabilities: set[str] = {
        "ampa",
        "gaba",
        "leaky_integrator",
        "exponential_current",
        "custom_synapse",
    }
    operation_capabilities: set[str] = {
        "linear",
        "reduce_mean",
        "reduce_sum",
        "select_final",
        "duration_normalise",
        "cumulative_sum",
        "custom_operation",
        "divide",
    }
    connection_capabilities: set[str] = {"feedforward", "recurrent", "feedback"}
    for pop in graph.get("populations", []):
        if pop.get("neuron", {}).get("kind") == "custom_neuron":
            E.resolve("neuron", pop["neuron"])
        kind = pop.get("neuron", {}).get("kind")
        if kind not in neuron_capabilities:
            issues.append(
                CapabilityIssue(pop["id"], f"neuron:{kind}", "unsupported neuron kind")
            )
    for projection in graph.get("projections", []):
        if projection.get("synapse", {}).get("kind") == "custom_synapse":
            E.resolve("synapse", projection["synapse"])
        synapse = projection.get("synapse", {}).get("kind")
        if synapse not in synapse_capabilities:
            issues.append(
                CapabilityIssue(
                    projection["id"], f"synapse:{synapse}", "unsupported synapse kind"
                )
            )
        connection = projection.get("connection")
        if connection not in connection_capabilities:
            issues.append(
                CapabilityIssue(
                    projection["id"],
                    f"connection:{connection}",
                    "unsupported connection kind",
                )
            )
    for operation in graph.get("operations", []):
        if operation.get("kind") == "custom_operation":
            E.resolve(
                "operation",
                {
                    "definition": operation["config"]["definition"],
                    "config": operation["config"].get("settings", {}),
                },
            )
        kind = operation.get("kind")
        if kind not in operation_capabilities:
            issues.append(
                CapabilityIssue(
                    operation["id"], f"operation:{kind}", "unsupported operation kind"
                )
            )
    return issues


@dataclass(frozen=True)
class PlannedProjection:
    id: str
    source: str
    target: str
    polarity: str
    decay: float
    delay_steps: int
    parameter: str
    enabled: bool


@dataclass(frozen=True)
class GraphPlan:
    graph: Mapping[str, Any]
    dt_ms: float
    populations: tuple[Mapping[str, Any], ...]
    projections: tuple[PlannedProjection, ...]
    observables: tuple[Mapping[str, Any], ...]
    outputs: tuple[Mapping[str, Any], ...]


def _signal_axes(graph: Mapping[str, Any]) -> dict[str, tuple[int | str, ...]]:
    axes = {
        f"{row['id']}.value": tuple(row["shape"])
        for row in (*graph.get("inputs", []), *graph.get("operations", []))
    }
    populations = {row["id"]: row for row in graph["populations"]}
    for name, population in populations.items():
        shape = ("time", "batch", population["size"])
        axes[f"{name}.voltage"] = shape
        if population["neuron"]["kind"] == "leaky_integrator":
            axes[f"{name}.pre_reset_voltage"] = shape
        if population["spiking"]:
            axes[f"{name}.spikes"] = shape
    for row in graph.get("projections", []):
        target = row["target"].partition(".")[0]
        axes[f"{row['id']}.{E.projection_port(row['synapse'])}"] = (
            "time",
            "batch",
            populations[target]["size"],
        )
    for row in graph.get("populations", []):
        if row["neuron"]["kind"] == "custom_neuron":
            for port in E.resolve("neuron", row["neuron"]).state_units:
                axes[f"{row['id']}.{port}"] = ("time", "batch", row["size"])
    for row in graph.get("projections", []):
        if row["synapse"]["kind"] == "custom_synapse":
            size = populations[row["target"].partition(".")[0]]["size"]
            for port in E.resolve("synapse", row["synapse"]).state_units:
                axes[f"{row['id']}.{port}"] = ("time", "batch", size)
    return axes


RUNTIME_STATE_SCHEMA = "tools/snnsim.graph-runtime-state/v1"


@dataclass
class GraphRuntimeState:
    """Complete dynamic state required to continue one graph trajectory.

    Static parameters are deliberately excluded: weight checkpoints and runtime
    state are orthogonal, allowing a mature state to branch across compatible
    parameterisations of the same graph structure.
    """

    signature: str
    compatibility: dict[str, Any]
    completed_steps: int
    voltages: dict[str, torch.Tensor]
    refractory: dict[str, torch.Tensor]
    conductances: dict[str, torch.Tensor]
    population_histories: dict[str, torch.Tensor]
    input_histories: dict[str, torch.Tensor]
    custom_state: dict[str, torch.Tensor] = field(default_factory=dict)
    currents: dict[str, torch.Tensor] = field(default_factory=dict)

    def detached(self, *, device: str | torch.device = "cpu") -> GraphRuntimeState:
        def moved(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {
                name: value.detach().to(device).clone()
                for name, value in values.items()
            }

        return GraphRuntimeState(
            signature=self.signature,
            compatibility=self.compatibility,
            completed_steps=self.completed_steps,
            voltages=moved(self.voltages),
            refractory=moved(self.refractory),
            conductances=moved(self.conductances),
            population_histories=moved(self.population_histories),
            input_histories=moved(self.input_histories),
            custom_state=moved(self.custom_state),
            currents=moved(self.currents),
        )


def runtime_state_compatibility(plan: GraphPlan) -> dict[str, Any]:
    """Describe state-layout and dynamical semantics, excluding parameter values."""
    parameters = {row["id"]: row for row in plan.graph.get("parameters", [])}
    return {
        "schema": RUNTIME_STATE_SCHEMA,
        "dt_ms": plan.dt_ms,
        "populations": [
            {"id": row["id"], "size": row["size"], "neuron": row["neuron"]}
            for row in plan.populations
        ],
        "inputs": [
            {
                "id": row["id"],
                "shape": row["shape"],
                "signal_type": row.get("signal_type"),
            }
            for row in plan.graph.get("inputs", [])
        ],
        "projections": [
            {
                "id": row.id,
                "source": row.source,
                "target": row.target,
                "polarity": row.polarity,
                "synapse": next(
                    item["synapse"]
                    for item in plan.graph.get("projections", [])
                    if item["id"] == row.id
                ),
                "delay_steps": row.delay_steps,
                "parameter": row.parameter,
                "parameter_shape": parameters[row.parameter]["shape"],
                "enabled": row.enabled,
            }
            for row in plan.projections
        ],
    }


def runtime_state_signature(plan: GraphPlan) -> str:
    payload = runtime_state_compatibility(plan)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _compatibility_mismatch(
    expected: Any, actual: Any, path: str = "graph"
) -> str | None:
    if type(expected) is not type(actual):
        return f"{path} expected {expected!r}, got {actual!r}"
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return f"{path} keys expected {sorted(expected)}, got {sorted(actual)}"
        for key in expected:
            mismatch = _compatibility_mismatch(
                expected[key], actual[key], f"{path}.{key}"
            )
            if mismatch:
                return mismatch
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path} length expected {len(expected)}, got {len(actual)}"
        for index, (left, right) in enumerate(zip(expected, actual)):
            mismatch = _compatibility_mismatch(left, right, f"{path}[{index}]")
            if mismatch:
                return mismatch
    elif expected != actual:
        return f"{path} expected {expected!r}, got {actual!r}"
    return None


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def save_runtime_state(path: str | Path, state: GraphRuntimeState) -> Path:
    """Atomically publish a portable JSON/NPZ graph-runtime state directory."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    groups = {
        "voltages": state.voltages,
        "refractory": state.refractory,
        "conductances": state.conductances,
        "population_histories": state.population_histories,
        "input_histories": state.input_histories,
        "custom_state": state.custom_state,
        "currents": state.currents,
    }
    arrays: dict[str, np.ndarray] = {}
    tensors: list[dict[str, Any]] = []
    for group_name, values in groups.items():
        for name in sorted(values):
            key = f"tensor_{len(arrays):04d}"
            value = values[name].detach().cpu().contiguous()
            arrays[key] = value.numpy()
            tensors.append(
                {
                    "group": group_name,
                    "name": name,
                    "key": key,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                }
            )
    fd, temporary_name = tempfile.mkstemp(prefix=".tensors-", suffix=".npz", dir=root)
    os.close(fd)
    temporary_tensors = Path(temporary_name)
    try:
        np.savez_compressed(temporary_tensors, **arrays)
        tensors_digest = _file_digest(temporary_tensors)
        os.replace(temporary_tensors, root / "tensors.npz")
    finally:
        temporary_tensors.unlink(missing_ok=True)
    manifest = {
        "schema": RUNTIME_STATE_SCHEMA,
        "schema_version": 2 if state.custom_state or state.currents else 1,
        "signature": state.signature,
        "compatibility": state.compatibility,
        "completed_steps": state.completed_steps,
        "tensors_file": "tensors.npz",
        "tensors_digest": tensors_digest,
        "tensors": tensors,
    }
    fd, temporary_name = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=root)
    temporary_manifest = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary_manifest, root / "manifest.json")
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return root


def load_runtime_state(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> GraphRuntimeState:
    """Load and authenticate a portable graph-runtime state artifact."""
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != RUNTIME_STATE_SCHEMA or manifest.get(
        "schema_version"
    ) not in {1, 2}:
        raise ValueError(f"unsupported runtime-state schema: {manifest.get('schema')}")
    tensors_path = root / manifest.get("tensors_file", "tensors.npz")
    actual_digest = _file_digest(tensors_path)
    if actual_digest != manifest.get("tensors_digest"):
        raise ValueError(
            f"runtime-state tensors digest expected {manifest.get('tensors_digest')}, got {actual_digest}"
        )
    groups: dict[str, dict[str, torch.Tensor]] = {
        "voltages": {},
        "refractory": {},
        "conductances": {},
        "population_histories": {},
        "input_histories": {},
        "custom_state": {},
        "currents": {},
    }
    with np.load(tensors_path, allow_pickle=False) as archive:
        expected_keys = {row["key"] for row in manifest["tensors"]}
        if set(archive.files) != expected_keys:
            raise ValueError(
                f"runtime-state tensor keys expected {sorted(expected_keys)}, got {sorted(archive.files)}"
            )
        for row in manifest["tensors"]:
            array = archive[row["key"]]
            if list(array.shape) != row["shape"] or str(array.dtype) != row["dtype"]:
                raise ValueError(
                    f"runtime-state tensor {row['group']}.{row['name']} metadata does not match tensors.npz"
                )
            groups[row["group"]][row["name"]] = torch.from_numpy(array.copy()).to(
                device
            )
    return GraphRuntimeState(
        signature=manifest["signature"],
        compatibility=manifest["compatibility"],
        completed_steps=int(manifest["completed_steps"]),
        voltages=groups["voltages"],
        refractory=groups["refractory"],
        conductances=groups["conductances"],
        population_histories=groups["population_histories"],
        input_histories=groups["input_histories"],
        custom_state=groups["custom_state"],
        currents=groups["currents"],
    )


def _json_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return "sha256:" + hashlib.sha256(encoded + b"\n").hexdigest()


def write_inference_artifacts(
    path: str | Path,
    result: ExecutionResult,
    *,
    graph: Mapping[str, Any],
    seed: int,
) -> Mapping[str, Any]:
    """Persist graph inference tensors and a digest-bound cache manifest."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "recording.npz": result.diagnostics,
        "outputs.npz": result.outputs,
        "parameters.npz": result.parameters,
    }
    files = []
    for filename, tensors in payloads.items():
        arrays = {
            name: value.detach().cpu().numpy()
            for name, value in sorted(tensors.items())
        }
        destination = root / filename
        np.savez_compressed(destination, **arrays)
        files.append(
            {
                "path": filename,
                "digest": _file_digest(destination),
                "arrays": [
                    {
                        "name": name,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                    for name, value in arrays.items()
                ],
            }
        )
    metrics_path = root / "metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2, sort_keys=True) + "\n")
    files.append({"path": "metrics.json", "digest": _file_digest(metrics_path)})
    request = {
        "seed": int(seed),
        "execution_protocol": result.metrics.get("execution_protocol"),
        "checkpoint": result.metrics.get("checkpoint"),
        "inference_overrides": result.metrics.get("inference_overrides"),
        "inference_interventions": result.metrics.get("inference_interventions"),
        "diagnostics": result.metrics.get("diagnostics"),
        "device": result.metrics.get("device"),
        "source_graph_digest": result.metrics.get("source_graph_digest"),
        "effective_graph_digest": result.metrics.get("effective_graph_digest"),
    }
    manifest = {
        "schema": INFERENCE_ARTIFACT_SCHEMA,
        "schema_version": 1,
        "graph_digest": _json_digest(graph),
        "request_seed": int(seed),
        "request_digest": _json_digest(request),
        "files": files,
    }
    manifest["artifact_digest"] = _json_digest(manifest)
    (root / "inference-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def validate_inference_artifacts(
    path: str | Path,
    *,
    graph: Mapping[str, Any] | None = None,
    seed: int | None = None,
    expected_interventions: Mapping[str, Any] | None | EllipsisType = ...,
) -> Mapping[str, Any]:
    """Authenticate a persisted graph inference artifact set before reuse."""
    root = Path(path)
    manifest_path = root / "inference-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != INFERENCE_ARTIFACT_SCHEMA
        or manifest.get("schema_version") != 1
    ):
        raise ValueError(
            f"unsupported inference-artifact schema: {manifest.get('schema')}"
        )
    claimed_artifact_digest = manifest.get("artifact_digest")
    unsigned = dict(manifest)
    unsigned.pop("artifact_digest", None)
    actual_artifact_digest = _json_digest(unsigned)
    if claimed_artifact_digest != actual_artifact_digest:
        raise ValueError(
            f"inference artifact manifest digest expected {claimed_artifact_digest}, got {actual_artifact_digest}"
        )
    if graph is not None:
        expected_graph_digest = _json_digest(graph)
        if manifest.get("graph_digest") != expected_graph_digest:
            raise ValueError(
                f"inference artifact graph digest expected {expected_graph_digest}, got {manifest.get('graph_digest')}"
            )
    if seed is not None and manifest.get("request_seed") != int(seed):
        raise ValueError(
            f"inference artifact request seed expected {int(seed)}, got {manifest.get('request_seed')}"
        )
    expected_files = {
        "recording.npz",
        "outputs.npz",
        "parameters.npz",
        "metrics.json",
    }
    rows = manifest.get("files", [])
    if {row.get("path") for row in rows} != expected_files:
        raise ValueError("inference artifact manifest file set is incomplete")
    for row in rows:
        filename = row["path"]
        if Path(filename).name != filename:
            raise ValueError(f"inference artifact path must be a filename: {filename}")
        payload_path = root / filename
        actual_digest = _file_digest(payload_path)
        if actual_digest != row.get("digest"):
            raise ValueError(
                f"inference artifact {filename} digest expected {row.get('digest')}, got {actual_digest}"
            )
        if filename.endswith(".npz"):
            loaded = np.load(payload_path, allow_pickle=False)
            try:
                actual_arrays = {
                    name: {
                        "name": name,
                        "shape": list(loaded[name].shape),
                        "dtype": str(loaded[name].dtype),
                    }
                    for name in sorted(loaded.files)
                }
            finally:
                loaded.close()
            expected_arrays = {row["name"]: row for row in row.get("arrays", [])}
            if actual_arrays != expected_arrays:
                raise ValueError(
                    f"inference artifact {filename} array inventory does not match manifest"
                )
    metrics = json.loads((root / "metrics.json").read_text())
    if (
        expected_interventions is not Ellipsis
        and metrics.get("inference_interventions") != expected_interventions
    ):
        raise ValueError(
            "inference artifact interventions do not match expected identity"
        )
    request = {
        "seed": int(manifest["request_seed"]),
        "execution_protocol": metrics.get("execution_protocol"),
        "checkpoint": metrics.get("checkpoint"),
        "inference_overrides": metrics.get("inference_overrides"),
        "inference_interventions": metrics.get("inference_interventions"),
        "diagnostics": metrics.get("diagnostics"),
        "device": metrics.get("device"),
        "source_graph_digest": metrics.get("source_graph_digest"),
        "effective_graph_digest": metrics.get("effective_graph_digest"),
    }
    actual_request_digest = _json_digest(request)
    if actual_request_digest != manifest.get("request_digest"):
        raise ValueError(
            f"inference artifact request digest expected {manifest.get('request_digest')}, got {actual_request_digest}"
        )
    return manifest


def derive_inference_products(
    source: str | Path,
    destination: str | Path,
    *,
    logits_id: str,
    labels: np.ndarray | torch.Tensor,
    spike_recordings: Sequence[str] = (),
) -> Mapping[str, Any]:
    """Derive named accuracy, per-cell rates, and sparse rasters from a cache."""
    source_root = Path(source)
    source_manifest = validate_inference_artifacts(source_root)
    output_file = np.load(source_root / "outputs.npz", allow_pickle=False)
    recording_file = np.load(source_root / "recording.npz", allow_pickle=False)
    try:
        if logits_id not in output_file.files:
            raise ValueError(f"inference outputs do not contain logits {logits_id!r}")
        logits = np.asarray(output_file[logits_id])
        if logits.ndim != 2 or not np.issubdtype(logits.dtype, np.floating):
            raise ValueError(
                f"inference logits {logits_id} must have floating shape [batch, classes]"
            )
        label_values = (
            labels.detach().cpu().numpy()
            if isinstance(labels, torch.Tensor)
            else np.asarray(labels)
        )
        if label_values.ndim != 1 or label_values.shape[0] != logits.shape[0]:
            raise ValueError(
                f"inference labels shape expected [{logits.shape[0]}], got {list(label_values.shape)}"
            )
        if not np.issubdtype(label_values.dtype, np.integer):
            raise ValueError("inference labels must use an integer dtype")
        if np.any(label_values < 0) or np.any(label_values >= logits.shape[1]):
            raise ValueError(f"inference labels must be in [0, {logits.shape[1]})")
        metrics = json.loads((source_root / "metrics.json").read_text())
        timing = metrics.get("execution_protocol", {}).get("timing", {})
        duration_s = float(timing.get("duration_ms", 0.0)) / 1000.0
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError(
                "inference artifact requires a positive execution duration"
            )
        rates: dict[str, np.ndarray] = {}
        rasters: dict[str, np.ndarray] = {}
        recording_rows = []
        for recording_id in spike_recordings:
            if recording_id not in recording_file.files:
                raise ValueError(
                    f"inference recordings do not contain spikes {recording_id!r}"
                )
            spikes = np.asarray(recording_file[recording_id])
            if (
                spikes.ndim != 3
                or spikes.shape[1] != logits.shape[0]
                or not np.all((spikes == 0) | (spikes == 1))
            ):
                raise ValueError(
                    f"inference spike recording {recording_id} must be binary [time, batch, cells]"
                )
            rates[recording_id] = spikes.sum(axis=0, dtype=np.float64) / duration_s
            coordinates = np.argwhere(spikes != 0)
            rasters[f"{recording_id}.steps"] = coordinates[:, 0].astype(np.int64)
            rasters[f"{recording_id}.batches"] = coordinates[:, 1].astype(np.int64)
            rasters[f"{recording_id}.cells"] = coordinates[:, 2].astype(np.int64)
            rasters[f"{recording_id}.shape"] = np.asarray(spikes.shape, dtype=np.int64)
            recording_rows.append(
                {
                    "id": recording_id,
                    "spike_shape": list(spikes.shape),
                    "rate_shape": list(rates[recording_id].shape),
                    "spike_count": int(coordinates.shape[0]),
                }
            )
    finally:
        output_file.close()
        recording_file.close()

    root = Path(destination)
    if root.exists():
        raise ValueError(f"derived inference destination already exists: {root}")
    root.mkdir(parents=True)
    predictions = logits.argmax(axis=1).astype(np.int64)
    np.save(root / "labels.npy", label_values.astype(np.int64, copy=False))
    np.save(root / "predictions.npy", predictions)
    np.savez_compressed(root / "rates.npz", **rates)
    np.savez_compressed(root / "rasters.npz", **rasters)
    summary = {
        "schema": DERIVED_INFERENCE_SCHEMA,
        "source_artifact_digest": source_manifest["artifact_digest"],
        "logits_id": logits_id,
        "logits_shape": list(logits.shape),
        "labels_shape": list(label_values.shape),
        "accuracy": float(np.mean(predictions == label_values)),
        "duration_s": duration_s,
        "spike_recordings": recording_rows,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    files = []
    for filename in (
        "labels.npy",
        "predictions.npy",
        "rates.npz",
        "rasters.npz",
        "summary.json",
    ):
        path = root / filename
        files.append({"path": filename, "digest": _file_digest(path)})
    manifest = {
        "schema": DERIVED_INFERENCE_SCHEMA,
        "schema_version": 1,
        "source_artifact_digest": source_manifest["artifact_digest"],
        "files": files,
    }
    manifest["artifact_digest"] = _json_digest(manifest)
    (root / "derived-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return {**summary, "artifact_digest": manifest["artifact_digest"]}


def validate_derived_inference_products(
    path: str | Path, *, source_artifact_digest: str | None = None
) -> Mapping[str, Any]:
    """Authenticate one derived-inference directory before downstream reuse."""
    root = Path(path)
    manifest = json.loads((root / "derived-manifest.json").read_text())
    if (
        manifest.get("schema") != DERIVED_INFERENCE_SCHEMA
        or manifest.get("schema_version") != 1
    ):
        raise ValueError(
            f"unsupported derived-inference schema: {manifest.get('schema')}"
        )
    claimed = manifest.get("artifact_digest")
    unsigned = dict(manifest)
    unsigned.pop("artifact_digest", None)
    actual = _json_digest(unsigned)
    if claimed != actual:
        raise ValueError(
            f"derived inference manifest digest expected {claimed}, got {actual}"
        )
    if (
        source_artifact_digest is not None
        and manifest.get("source_artifact_digest") != source_artifact_digest
    ):
        raise ValueError(
            "derived inference source artifact digest does not match expected cache"
        )
    expected_files = {
        "labels.npy",
        "predictions.npy",
        "rates.npz",
        "rasters.npz",
        "summary.json",
    }
    rows = manifest.get("files", [])
    if {row.get("path") for row in rows} != expected_files:
        raise ValueError("derived inference manifest file set is incomplete")
    for row in rows:
        path = root / row["path"]
        actual_digest = _file_digest(path)
        if actual_digest != row.get("digest"):
            raise ValueError(
                f"derived inference {row['path']} digest expected {row.get('digest')}, got {actual_digest}"
            )
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("schema") != DERIVED_INFERENCE_SCHEMA or summary.get(
        "source_artifact_digest"
    ) != manifest.get("source_artifact_digest"):
        raise ValueError("derived inference summary identity does not match manifest")
    return manifest


def save_training_checkpoint(path: str | Path, checkpoint: TrainingCheckpoint) -> Path:
    """Atomically write a named, authenticated graph-training checkpoint."""
    if checkpoint.rng_state.dtype != torch.uint8 or checkpoint.rng_state.ndim != 1:
        raise ValueError(
            "training checkpoint CPU RNG state must be one-dimensional uint8"
        )
    if checkpoint.rng_backend not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            f"training checkpoint has unsupported RNG backend {checkpoint.rng_backend!r}"
        )
    devices = sorted(checkpoint.accelerator_rng_states)
    if checkpoint.rng_backend == "cpu" and devices:
        raise ValueError(
            "CPU training checkpoint cannot contain accelerator RNG states"
        )
    if checkpoint.rng_backend == "cuda" and (
        not devices or devices != [f"cuda:{index}" for index in range(len(devices))]
    ):
        raise ValueError(
            "CUDA training checkpoint RNG devices must be contiguous from cuda:0"
        )
    if checkpoint.rng_backend == "mps" and devices != ["mps"]:
        raise ValueError("MPS training checkpoint requires exactly the mps RNG state")
    for name, state in checkpoint.accelerator_rng_states.items():
        if state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError(
                f"training checkpoint {name} RNG state must be one-dimensional uint8"
            )
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    tensors: list[dict[str, Any]] = []

    def append(group: str, name: str, value: torch.Tensor, state: str | None = None):
        key = f"tensor_{len(arrays):04d}"
        tensor = value.detach().cpu().contiguous()
        arrays[key] = tensor.numpy()
        row = {
            "group": group,
            "name": name,
            "key": key,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
        if state is not None:
            row["state"] = state
        tensors.append(row)

    for name in sorted(checkpoint.parameters):
        append("parameters", name, checkpoint.parameters[name])
    optimizer_scalars: dict[str, dict[str, Any]] = {}
    for name in sorted(checkpoint.optimizer_state):
        optimizer_scalars[name] = {}
        for state, value in sorted(checkpoint.optimizer_state[name].items()):
            if isinstance(value, torch.Tensor):
                append("optimizer", name, value, state)
            else:
                optimizer_scalars[name][state] = value
    append("rng", "cpu", checkpoint.rng_state)
    for name in sorted(checkpoint.accelerator_rng_states):
        append("rng", name, checkpoint.accelerator_rng_states[name])
    fd, temporary_name = tempfile.mkstemp(prefix=".tensors-", suffix=".npz", dir=root)
    os.close(fd)
    temporary_tensors = Path(temporary_name)
    try:
        np.savez_compressed(temporary_tensors, **arrays)
        tensors_digest = _file_digest(temporary_tensors)
        os.replace(temporary_tensors, root / "tensors.npz")
    finally:
        temporary_tensors.unlink(missing_ok=True)
    best_digest = None
    if checkpoint.best_checkpoint is not None:
        if checkpoint.best_checkpoint.best_checkpoint is not None:
            raise ValueError(
                "selected checkpoint must not contain another selected checkpoint"
            )
        save_training_checkpoint(root / "selected", checkpoint.best_checkpoint)
        best_digest = _file_digest(root / "selected" / "manifest.json")
    selection_metadata = {
        "contract": checkpoint.selection_contract,
        "record": checkpoint.selection_record,
        "best_manifest_digest": best_digest,
    }
    manifest = {
        "schema": TRAINING_CHECKPOINT_SCHEMA,
        "schema_version": 4
        if checkpoint.selection_contract is not None
        and checkpoint.selection_contract["policy"]["mode"] == "epoch_metrics"
        else (3 if checkpoint.observation_state is not None else 2),
        "backend": "tools/snnsim.graph-training/v1",
        "rng_backend": checkpoint.rng_backend,
        "accelerator_rng_devices": sorted(checkpoint.accelerator_rng_states),
        "graph_digest": checkpoint.graph_digest,
        "training_digest": checkpoint.training_digest,
        "completed_updates": checkpoint.completed_updates,
        "selected_loss": checkpoint.selected_loss,
        "execution_protocol": checkpoint.execution_protocol,
        "initialization": checkpoint.initialization,
        "data_state": checkpoint.data_state,
        "selection_metadata": selection_metadata,
        "selection_metadata_digest": _json_digest(selection_metadata),
        "observation_state": checkpoint.observation_state,
        "observation_state_digest": _json_digest(checkpoint.observation_state),
        "optimizer_scalars": optimizer_scalars,
        "tensors_file": "tensors.npz",
        "tensors_digest": tensors_digest,
        "tensors": tensors,
    }
    fd, temporary_name = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=root)
    temporary_manifest = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary_manifest, root / "manifest.json")
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return root


def load_training_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> TrainingCheckpoint:
    """Load and authenticate a portable named graph-training checkpoint."""
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != TRAINING_CHECKPOINT_SCHEMA or manifest.get(
        "schema_version"
    ) not in {1, 2, 3, 4}:
        raise ValueError(
            f"unsupported training-checkpoint schema: {manifest.get('schema')}"
        )
    tensors_path = root / manifest.get("tensors_file", "tensors.npz")
    actual_digest = _file_digest(tensors_path)
    if actual_digest != manifest.get("tensors_digest"):
        raise ValueError(
            f"training-checkpoint tensors digest expected {manifest.get('tensors_digest')}, got {actual_digest}"
        )
    parameters: dict[str, torch.Tensor] = {}
    optimizer_state: dict[str, dict[str, Any]] = {
        name: dict(values)
        for name, values in manifest.get("optimizer_scalars", {}).items()
    }
    rng_state = None
    accelerator_rng_states: dict[str, torch.Tensor] = {}
    with np.load(tensors_path, allow_pickle=False) as archive:
        expected_keys = {row["key"] for row in manifest["tensors"]}
        if set(archive.files) != expected_keys:
            raise ValueError(
                f"training-checkpoint tensor keys expected {sorted(expected_keys)}, got {sorted(archive.files)}"
            )
        for row in manifest["tensors"]:
            array = archive[row["key"]]
            if list(array.shape) != row["shape"] or str(array.dtype) != row["dtype"]:
                raise ValueError(
                    f"training-checkpoint tensor {row['group']}.{row['name']} metadata does not match tensors.npz"
                )
            value = torch.from_numpy(array.copy())
            if row["group"] == "parameters":
                parameters[row["name"]] = value.to(device)
            elif row["group"] == "optimizer":
                optimizer_state.setdefault(row["name"], {})[row["state"]] = value.to(
                    device
                )
            elif row["group"] == "rng" and row["name"] == "cpu":
                if value.dtype != torch.uint8 or value.ndim != 1:
                    raise ValueError(
                        "training checkpoint CPU RNG state must be one-dimensional uint8"
                    )
                rng_state = value.cpu()
            elif row["group"] == "rng":
                if value.dtype != torch.uint8 or value.ndim != 1:
                    raise ValueError(
                        f"training checkpoint {row['name']} RNG state must be one-dimensional uint8"
                    )
                accelerator_rng_states[row["name"]] = value.cpu()
            else:
                raise ValueError(
                    f"unsupported training-checkpoint tensor group {row['group']}"
                )
    if rng_state is None:
        raise ValueError("training checkpoint is missing CPU RNG state")
    schema_version = int(manifest["schema_version"])
    rng_backend = manifest.get("rng_backend", "cpu")
    declared_devices = set(manifest.get("accelerator_rng_devices", []))
    if schema_version in {2, 3, 4}:
        if rng_backend not in {"cpu", "cuda", "mps"}:
            raise ValueError(
                f"training checkpoint has unsupported RNG backend {rng_backend!r}"
            )
        if declared_devices != set(accelerator_rng_states):
            raise ValueError(
                "training checkpoint accelerator RNG device inventory does not match tensors"
            )
        if rng_backend == "cpu" and accelerator_rng_states:
            raise ValueError(
                "CPU training checkpoint cannot contain accelerator RNG states"
            )
        if rng_backend == "cuda" and not accelerator_rng_states:
            raise ValueError(
                "CUDA training checkpoint is missing accelerator RNG states"
            )
        if rng_backend == "cuda":
            expected_cuda = [
                f"cuda:{index}" for index in range(len(accelerator_rng_states))
            ]
            if sorted(accelerator_rng_states) != expected_cuda:
                raise ValueError(
                    "CUDA training checkpoint RNG devices must be contiguous from cuda:0"
                )
        if rng_backend == "mps" and set(accelerator_rng_states) != {"mps"}:
            raise ValueError(
                "MPS training checkpoint requires exactly the mps RNG state"
            )
    observation_state = manifest.get("observation_state")
    if schema_version == 3 and observation_state is None:
        raise ValueError("audited training checkpoint is missing observation history")
    if "observation_state" in manifest and manifest.get(
        "observation_state_digest"
    ) != _json_digest(observation_state):
        raise ValueError("training checkpoint observation history digest mismatch")
    selection_metadata = manifest.get("selection_metadata", {})
    if "selection_metadata" in manifest and manifest.get(
        "selection_metadata_digest"
    ) != _json_digest(selection_metadata):
        raise ValueError("training checkpoint selection metadata digest mismatch")
    if schema_version == 4 and selection_metadata.get("contract") is None:
        raise ValueError("training checkpoint is missing selection contract")
    best_checkpoint = None
    if selection_metadata.get("best_manifest_digest") is not None:
        best_root = root / "selected"
        if (
            _file_digest(best_root / "manifest.json")
            != selection_metadata["best_manifest_digest"]
        ):
            raise ValueError(
                "training checkpoint selected candidate manifest digest mismatch"
            )
        best_manifest = json.loads((best_root / "manifest.json").read_text())
        if (
            best_manifest.get("selection_metadata", {}).get("best_manifest_digest")
            is not None
        ):
            raise ValueError("nested selected checkpoint is invalid")
        best_checkpoint = load_training_checkpoint(best_root, device=device)
    return TrainingCheckpoint(
        selection_contract=selection_metadata.get("contract"),
        selection_record=selection_metadata.get("record"),
        best_checkpoint=best_checkpoint,
        graph_digest=manifest["graph_digest"],
        training_digest=manifest["training_digest"],
        completed_updates=int(manifest["completed_updates"]),
        selected_loss=manifest.get("selected_loss"),
        execution_protocol=manifest["execution_protocol"],
        initialization=manifest["initialization"],
        parameters=parameters,
        optimizer_state=optimizer_state,
        rng_state=rng_state,
        rng_backend=rng_backend,
        accelerator_rng_states=accelerator_rng_states,
        data_state=manifest.get("data_state", {}),
        observation_state=observation_state,
    )


def capture_training_rng_state(
    device: str | torch.device,
) -> tuple[str, dict[str, torch.Tensor]]:
    """Capture every stochastic stream required for exact resume on one backend."""
    name = str(device).lower()
    if name.startswith("cuda"):
        states = torch.cuda.get_rng_state_all()
        expected = torch.cuda.device_count()
        if len(states) != expected or expected <= 0:
            raise ValueError(
                f"CUDA RNG capture expected {expected} device states, got {len(states)}"
            )
        return "cuda", {
            f"cuda:{index}": state.detach().cpu().clone()
            for index, state in enumerate(states)
        }
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS RNG capture requires an available MPS backend")
        return "mps", {"mps": torch.mps.get_rng_state().detach().cpu().clone()}
    return "cpu", {}


def restore_training_rng_state(
    checkpoint: TrainingCheckpoint, device: str | torch.device
) -> None:
    """Restore CPU and exact-matching accelerator streams or fail closed."""
    name = str(device).lower()
    requested_backend = (
        "cuda" if name.startswith("cuda") else "mps" if name == "mps" else "cpu"
    )
    if checkpoint.rng_backend != requested_backend:
        raise ValueError(
            f"training checkpoint RNG backend {checkpoint.rng_backend} cannot resume on {requested_backend}"
        )
    if requested_backend == "cuda":
        expected = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        if sorted(checkpoint.accelerator_rng_states) != expected:
            raise ValueError(
                "training checkpoint CUDA RNG topology does not match available devices"
            )
        torch.set_rng_state(checkpoint.rng_state.cpu())
        torch.cuda.set_rng_state_all(
            [checkpoint.accelerator_rng_states[key].cpu() for key in expected]
        )
    elif requested_backend == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS RNG restore requires an available MPS backend")
        if set(checkpoint.accelerator_rng_states) != {"mps"}:
            raise ValueError("training checkpoint MPS RNG state is missing")
        torch.set_rng_state(checkpoint.rng_state.cpu())
        torch.mps.set_rng_state(checkpoint.accelerator_rng_states["mps"].cpu())
    else:
        if checkpoint.accelerator_rng_states:
            raise ValueError("CPU resume cannot restore accelerator RNG states")
        torch.set_rng_state(checkpoint.rng_state.cpu())


def legacy_parameter_map_v1(graph: Mapping[str, Any]) -> dict[str, str]:
    """Map the supported one-layer legacy COBANet by semantic graph roles."""
    populations = {row["id"]: row for row in graph.get("populations", [])}
    input_ids = {row["id"] for row in graph.get("inputs", [])}
    mapping: dict[str, str] = {}
    recurrent_index = 1
    for projection in graph.get("projections", []):
        parameters = projection.get("parameters", [])
        if len(parameters) != 1:
            raise ValueError(
                f"legacy parameter mapping requires one parameter on projection {projection.get('id')}"
            )
        parameter = parameters[0]
        source = projection["source"].partition(".")[0]
        target = projection["target"].partition(".")[0]
        if source in input_ids:
            legacy = "W_ff.0"
        elif populations[target]["neuron"]["kind"] == "leaky_integrator":
            legacy = "W_ff.1"
        elif projection.get("connection") == "recurrent":
            source_size = int(populations[source]["size"])
            target_size = int(populations[target]["size"])
            if source == target and projection["polarity"] == "excitatory":
                legacy = f"W_ee.{recurrent_index}"
            elif source == target and projection["polarity"] == "inhibitory":
                legacy = f"W_ii.{recurrent_index}"
            elif source_size >= target_size and projection["polarity"] == "excitatory":
                legacy = f"W_ei.{recurrent_index}"
            elif source_size <= target_size and projection["polarity"] == "inhibitory":
                legacy = f"W_ie.{recurrent_index}"
            else:
                raise ValueError(
                    f"legacy parameter mapping cannot classify recurrent projection {projection['id']}"
                )
        else:
            raise ValueError(
                f"legacy parameter mapping cannot classify projection {projection['id']}"
            )
        if legacy in mapping.values():
            raise ValueError(f"legacy parameter mapping duplicates role {legacy}")
        mapping[parameter] = legacy
    for operation in graph.get("operations", []):
        if operation.get("kind") != "linear":
            continue
        for parameter in operation.get("parameters", []):
            if parameter in mapping:
                continue
            legacy = "W_ff.1"
            if legacy in mapping.values():
                raise ValueError(f"legacy parameter mapping duplicates role {legacy}")
            mapping[parameter] = legacy
    graph_parameters = {row["id"] for row in graph.get("parameters", [])}
    if set(mapping) != graph_parameters:
        missing = sorted(graph_parameters - set(mapping))
        extra = sorted(set(mapping) - graph_parameters)
        raise ValueError(
            f"legacy parameter mapping must be complete; missing={missing}, extra={extra}"
        )
    return dict(sorted(mapping.items()))


def _validate_interchange_parameters(
    graph: Mapping[str, Any], parameters: Mapping[str, torch.Tensor]
) -> None:
    rows = {row["id"]: row for row in graph.get("parameters", [])}
    if set(parameters) != set(rows):
        raise ValueError(
            f"graph parameter interchange must be complete; missing={sorted(set(rows) - set(parameters))}, extra={sorted(set(parameters) - set(rows))}"
        )
    for name, value in parameters.items():
        runtime_shape = tuple(reversed(rows[name]["shape"]))
        if tuple(value.shape) != runtime_shape:
            raise ValueError(
                f"graph parameter {name} expected runtime shape {runtime_shape}, got {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise ValueError(f"graph parameter {name} must use a floating dtype")


def import_legacy_parameters_v1(
    graph: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    *,
    device: str | torch.device = "cpu",
) -> ParameterInterchange:
    """Import the exact supported one-layer legacy parameter state by semantic name."""
    mapping = legacy_parameter_map_v1(graph)
    reverse = {legacy: graph_name for graph_name, legacy in mapping.items()}
    if set(state_dict) != set(reverse):
        raise ValueError(
            f"legacy parameter interchange requires exact keys; missing={sorted(set(reverse) - set(state_dict))}, extra={sorted(set(state_dict) - set(reverse))}"
        )
    parameters = {
        reverse[name]: value.detach().clone().to(device)
        for name, value in state_dict.items()
    }
    _validate_interchange_parameters(graph, parameters)
    return ParameterInterchange(
        parameters=dict(sorted(parameters.items())),
        provenance={
            "schema": LEGACY_PARAMETER_INTERCHANGE_SCHEMA,
            "mapping_version": 1,
            "direction": "legacy_to_graph",
            "mapping": mapping,
        },
    )


def export_legacy_parameters_v1(
    graph: Mapping[str, Any], parameters: Mapping[str, torch.Tensor]
) -> ParameterInterchange:
    """Export a complete supported graph parameter set under legacy state keys."""
    _validate_interchange_parameters(graph, parameters)
    mapping = legacy_parameter_map_v1(graph)
    exported = {
        mapping[name]: value.detach().clone() for name, value in parameters.items()
    }
    return ParameterInterchange(
        parameters=dict(sorted(exported.items())),
        provenance={
            "schema": LEGACY_PARAMETER_INTERCHANGE_SCHEMA,
            "mapping_version": 1,
            "direction": "graph_to_legacy",
            "mapping": mapping,
        },
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


def plan_graph(graph: Mapping[str, Any]) -> GraphPlan:
    if "voltage_sampling" in graph and graph["voltage_sampling"] != "explicit":
        raise ValueError("unsupported voltage_sampling contract")
    populations = {p["id"]: p for p in graph.get("populations", [])}
    signal_references = [
        r["signal"] for r in (*graph.get("outputs", []), *graph.get("observables", []))
    ]
    signal_references.extend(
        source for op in graph.get("operations", []) for source in op["sources"]
    )
    for signal in signal_references:
        owner, _, port = signal.partition(".")
        if (
            port == "pre_reset_voltage"
            and populations.get(owner, {}).get("neuron", {}).get("kind")
            != "leaky_integrator"
        ):
            raise ValueError(f"{signal}: pre_reset_voltage requires a leaky integrator")
    issues = graph_capability_issues(graph)
    if issues:
        detail = "; ".join(
            f"{x.element} requires {x.capability}: {x.message}" for x in issues
        )
        raise ValueError(f"graph executor capability failure: {detail}")
    dt = float(graph["timebase"]["dt"]["value"])
    if dt <= 0:
        raise ValueError("graph timebase dt must be positive")
    parameter_rows = {row["id"]: row for row in graph.get("parameters", [])}
    planned = []
    for row in graph.get("projections", []):
        expected_unit = E.synapse_unit(row["synapse"])
        target_neuron = next(
            p["neuron"]
            for p in graph["populations"]
            if p["id"] == row["target"].partition(".")[0]
        )
        if expected_unit != E.neuron_unit(target_neuron):
            raise ValueError(
                f"{row['id']}: synapse unit {expected_unit} incompatible with neuron input unit {E.neuron_unit(target_neuron)}"
            )
        for parameter_id in row.get("parameters", []):
            unit = parameter_rows.get(parameter_id, {}).get("unit")
            if unit != expected_unit:
                raise ValueError(
                    f"{row['id']}: projection parameter {parameter_id} requires unit {expected_unit}, got {unit}"
                )
        delay = row.get("delay")
        delay_ms = 0.0 if delay is None else float(delay["value"])
        raw_steps = delay_ms / dt
        steps = int(round(raw_steps))
        if not math.isclose(raw_steps, steps, abs_tol=1e-9):
            raise ValueError(
                f"{row['id']}: delay {delay_ms} ms is not an integer number of dt={dt} ms steps"
            )
        source_owner = row["source"].partition(".")[0]
        # Recurrent/feedback edges are causal even when the author declares
        # zero additional delay. Feedforward population edges may consume the
        # current-step source after topological scheduling.
        if (
            source_owner in {p["id"] for p in graph.get("populations", [])}
            and row.get("connection") != "feedforward"
        ):
            steps = max(1, steps)
        tau = float(row["synapse"].get("tau", {"value": 1.0})["value"])
        if tau <= 0 or not math.isfinite(tau):
            raise ValueError(f"{row['id']}: synapse tau must be finite and positive")
        decay = (
            0.0 if row["synapse"]["kind"] == "leaky_integrator" else math.exp(-dt / tau)
        )
        planned.append(
            PlannedProjection(
                id=row["id"],
                source=row["source"],
                target=row["target"],
                polarity=row["polarity"],
                decay=decay,
                delay_steps=steps,
                parameter=row["parameters"][0],
                enabled=row.get("enabled", True),
            )
        )
    populations = list(graph.get("populations", []))
    population_ids = {p["id"] for p in populations}
    zero_edges = [
        (p.source.partition(".")[0], p.target.partition(".")[0])
        for p in planned
        if p.enabled
        and p.delay_steps == 0
        and p.source.partition(".")[0] in population_ids
    ]
    ordered: list[Mapping[str, Any]] = []
    remaining = {p["id"]: p for p in populations}
    while remaining:
        ready = sorted(
            name
            for name in remaining
            if not any(dst == name and src in remaining for src, dst in zero_edges)
        )
        if not ready:
            raise ValueError(
                "zero-delay population projections form an algebraic cycle"
            )
        for name in ready:
            ordered.append(remaining.pop(name))
    return GraphPlan(
        graph=graph,
        dt_ms=dt,
        populations=tuple(ordered),
        projections=tuple(planned),
        observables=tuple(graph.get("observables", [])),
        outputs=tuple(graph.get("outputs", [])),
    )


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
    ):
        super().__init__()
        self.plan = plan
        self.seed = int(seed)
        self.surrogate = (
            surrogate if (surrogate or {}).get("kind") == "custom_surrogate" else None
        )
        if self.surrogate is not None:
            E.resolve("surrogate", self.surrogate)
        self.surrogate_slope = float(surrogate_slope)
        torch.manual_seed(seed)
        rows = {row["id"]: row for row in plan.graph.get("parameters", [])}
        self.weights = nn.ParameterDict()
        self.initialization_metadata: dict[str, dict[str, Any]] = {}
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
            if zero_fraction:
                if zeroing == "exact_k":
                    if len(runtime_shape) != 2:
                        raise ValueError(
                            f"{row['id']}: exact_k zeroing requires a matrix"
                        )
                    fan_in, fan_out = runtime_shape
                    kept = max(1, int(round((1.0 - zero_fraction) * fan_in)))
                    mask = torch.zeros(runtime_shape)
                    for column in range(fan_out):
                        mask[torch.randperm(fan_in)[:kept], column] = 1
                    value = value * mask * (fan_in / kept)
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
                value = value / runtime_shape[0]
            if constraint:
                value = X.apply_constraint(value, constraint)
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

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        diagnostics: bool = True,
        runtime_state: GraphRuntimeState | None = None,
        interventions: Sequence[Intervention] = (),
    ) -> ExecutionResult:
        result, _ = self._forward(
            inputs,
            diagnostics=diagnostics,
            runtime_state=runtime_state,
            interventions=interventions,
        )
        return result

    def _forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        diagnostics: bool = True,
        runtime_state: GraphRuntimeState | None = None,
        interventions: Sequence[Intervention] = (),
        required_signals: Sequence[str] = (),
        observation_populations: Sequence[str] = (),
    ) -> tuple[ExecutionResult, dict[str, torch.Tensor]]:
        if not isinstance(diagnostics, bool):
            raise TypeError("diagnostics must be boolean")
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
        expected_compatibility = runtime_state_compatibility(self.plan)
        expected_signature = runtime_state_signature(self.plan)
        if runtime_state is None:
            voltage = {
                name: (
                    torch.zeros((batch, p["size"]), device=device)
                    if p["neuron"]["kind"] == "leaky_integrator"
                    else torch.full(
                        (batch, p["size"]),
                        float(p["neuron"].get("initial_voltage_mv", M.E_L))
                        if p["neuron"]["kind"] == "cuba_lif"
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
                if spec["kind"] != f"custom_{category}":
                    continue
                definition = E.resolve(category, spec)
                owner = row["id"]
                target = (
                    owner if category == "neuron" else row["target"].partition(".")[0]
                )
                shape = (batch, populations[target]["size"])
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
                        definition.name,
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
                        name=f"{definition.name}.{key}",
                    )
                for key, value in state.items():
                    if value.device != torch.device(device):
                        raise ValueError(
                            f"{definition.name}.{key}: state must use execution device {device}"
                        )
                for port in definition.state_units:
                    if port not in state:
                        raise ValueError(
                            f"{definition.name}: declared state port {port} missing"
                        )
                    X.checked_tensor(
                        state[port],
                        shape=shape,
                        device=device,
                        dtype=parameter_dtype,
                        name=f"{definition.name}.{port}",
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
        needed_signals = set(required_signals)
        needed_signals.update(row["signal"] for row in self.plan.outputs)
        if diagnostics:
            needed_signals.update(row["signal"] for row in self.plan.observables)
        for operation in self.plan.graph.get("operations", []):
            needed_signals.update(operation["sources"])
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

        for t in range(steps):
            new_spikes: dict[str, torch.Tensor] = {}
            for pop in self.plan.populations:
                name = pop["id"]
                incoming = {
                    "excitatory": torch.zeros_like(voltage[name]),
                    "inhibitory": torch.zeros_like(voltage[name]),
                }
                for projection in self.plan.projections:
                    if projection.target.partition(".")[0] != name:
                        continue
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
                    drive = (
                        source @ self.weights[projection.parameter.replace(".", "__")]
                    )
                    spec = projection_rows[projection.id]["synapse"]
                    if spec["kind"] == "custom_synapse":
                        definition = E.resolve("synapse", spec)
                        previous = synapse_states[projection.id]
                        context = E.SynapseContext(
                            previous, drive, self.plan.dt_ms, spec.get("config", {})
                        )
                        state = X.checked_state(
                            definition.function(context), previous, definition.name
                        )
                        synapse_states[projection.id] = state
                        conductance[key] = state["value"]
                    else:
                        conductance[key] = conductance[key] * projection.decay + drive
                    incoming[projection.polarity] += conductance[key]
                neuron = pop["neuron"]

                def spike_function(value):
                    return X.spike(
                        value, slope=self.surrogate_slope, custom=self.surrogate
                    )

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
                    state = X.checked_state(state, previous, definition.name)
                    X.checked_tensor(
                        spike_values,
                        shape=voltage[name].shape,
                        device=device,
                        dtype=parameter_dtype,
                        name=f"{definition.name}.spikes",
                    )
                    if torch.any((spike_values != 0) & (spike_values != 1)):
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
                if neuron["kind"] == "cuba_lif":
                    voltage[name], new_spikes[name], refractory[name] = X.current_lif(
                        {"voltage": voltage[name], "refractory": refractory[name]},
                        incoming["excitatory"],
                        incoming["inhibitory"],
                        dt_ms=self.plan.dt_ms,
                        config=neuron,
                        spike_function=spike_function,
                    )
                    new_spikes[name] = intervene(
                        name, new_spikes[name], completed_steps + t
                    )
                    continue
                if neuron["kind"] == "leaky_integrator":
                    beta = math.exp(-self.plan.dt_ms / float(neuron["tau"]["value"]))
                    voltage[name] = (
                        beta * voltage[name]
                        + (1.0 - beta) / self.plan.dt_ms * incoming["excitatory"]
                    )
                    new_spikes[name] = torch.zeros_like(spikes[name])
                    if name in pre_reset_voltage_traces:
                        pre_reset_voltage_traces[name].append(voltage[name])
                    integrator_sum[name] = (
                        integrator_sum.get(name, torch.zeros_like(voltage[name]))
                        + voltage[name]
                    )
                    threshold = neuron.get("soft_reset_threshold")
                    if threshold is not None:
                        reset = M.fast_sigmoid_spike(
                            voltage[name] - float(threshold),
                            float(neuron.get("surrogate_slope", M.SURROGATE_SLOPE)),
                        )
                        if pop.get("spiking"):
                            new_spikes[name] = reset
                        voltage[name] = voltage[name] - reset * float(threshold)
                    new_spikes[name] = intervene(
                        name, new_spikes[name], completed_steps + t
                    )
                    continue
                tau_mem = float(neuron["tau_mem"]["value"])
                c_m = float(neuron.get("capacitance_nf", 1.0 if tau_mem >= 15 else 0.5))
                g_l = float(neuron.get("leak_us", c_m / tau_mem))
                ref_steps = int(
                    neuron.get(
                        "refractory_steps",
                        max(
                            1,
                            round(
                                (M.ref_ms_E if tau_mem >= 15 else M.ref_ms_I)
                                / self.plan.dt_ms
                            ),
                        ),
                    )
                )
                dampen = float(neuron.get("voltage_grad_dampen", M.V_GRAD_DAMPEN))
                threshold = float(neuron.get("threshold_mv", M.V_th))
                voltage[name], new_spikes[name], refractory[name] = M.lif_step_expeuler(
                    voltage[name],
                    refractory[name],
                    incoming["excitatory"],
                    incoming["inhibitory"],
                    c_m,
                    g_l,
                    ref_steps,
                    lambda value, threshold_offset=0.0, threshold=threshold: (
                        spike_function(value - threshold - threshold_offset)
                    ),
                    dt_override=self.plan.dt_ms,
                    v_grad_dampen=dampen,
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
                    for port in E.resolve(category, spec).state_units:
                        signal = f"{owner}.{port}"
                        if signal in needed_signals:
                            state_traces.setdefault(signal, []).append(state[port])
            spikes = new_spikes
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

        remaining_ops = list(self.plan.graph.get("operations", []))
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
                    mask_seconds = mask.to(
                        device=sources[0].device, dtype=sources[0].dtype
                    ).sum(dim=0) * (self.plan.dt_ms / 1000.0)
                    mask_seconds = mask_seconds.reshape(
                        mask_seconds.shape[0], *([1] * (sources[0].ndim - 1))
                    )
                    if torch.any(mask_seconds <= 0):
                        raise ValueError(
                            f"{op['id']}: valid-time mask contains zero valid duration"
                        )
                    signal_values[f"{op['id']}.value"] = sources[0] / mask_seconds
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
            outputs[output["id"]] = signal_values[output["signal"]]
        packed = (
            {
                row["id"]: signal_values[row["signal"]].detach().clone()
                for row in self.plan.observables
            }
            if diagnostics
            else {}
        )
        next_input_histories = {
            name: torch.cat((history, inputs[name]), dim=0)[-history.shape[0] :]
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
        )
        signal_axes = _signal_axes(self.plan.graph)
        result = ExecutionResult(
            executor="graph",
            outputs=outputs,
            diagnostics=packed,
            parameters={k: v.detach().clone() for k, v in self.parameter_map().items()},
            final_state={
                f"{k}.voltage": v.detach().clone() for k, v in voltage.items()
            },
            runtime_state=next_runtime_state,
            metrics={
                "resolved_interventions": resolved_interventions,
                "inference_interventions": intervention_request
                if interventions
                else None,
                **(
                    {"population_observations": observation_counts}
                    if observation_counts
                    else {}
                ),
            },
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


def build(spec: ExecutionSpec) -> ExecutionResult:
    if spec.executor == "legacy":
        return ExecutionResult(
            executor="legacy", metrics={"request": "build", "routing": "legacy"}
        )
    graph = spec.graph
    training = spec.training
    if graph is None and spec.bundle is not None:
        manifest, graph = load_graph_bundle(spec.bundle)
        if spec.kind == "train" and training is None:
            training = load_training_recipe(spec.bundle, manifest, graph)
    if graph is None:
        raise ValueError("graph execution requires graph data or a bundle")
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
    ).to(device)
    return ExecutionResult(
        executor="graph",
        model=model,
        parameters=model.parameter_map(),
        metrics={
            "build_s": time.perf_counter() - started,
            "initialization": model.initialization_metadata,
            "training_schema": training.get("schema") if training else None,
        },
    )


def simulate(
    spec: ExecutionSpec, *, runtime_state: GraphRuntimeState | None = None
) -> ExecutionResult:
    if "inference_interventions" in spec.options:
        raise ValueError(
            "move options['inference_interventions'] to ExecutionSpec.interventions using typed intervention objects"
        )
    if spec.executor != "graph" and spec.interventions:
        raise ValueError("interventions require the graph executor")
    if spec.executor != "graph":
        return ExecutionResult(
            executor="legacy", metrics={"request": "simulate", "routing": "legacy"}
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
            if "W_ff.0" in state_dict:
                imported = import_legacy_parameters_v1(
                    built.model.plan.graph, state_dict, device=device
                )
                with torch.no_grad():
                    for name, parameter in built.model.parameter_map().items():
                        parameter.copy_(imported.parameters[name])
                checkpoint_provenance.update(
                    format="legacy_torch_state_dict",
                    interchange=imported.provenance,
                )
            else:
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
    if isinstance(resolved_inputs, DatasetProvider):
        result = simulate_dataset(
            built.model,
            resolved_inputs,
            batch_size=spec.batch_size
            if spec.batch_size is not None
            else min(32, resolved_inputs.sample_count),
            diagnostics=spec.diagnostics,
            interventions=interventions,
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
        )
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result.metrics.pop("resolved_interventions", None)
    intervention_request = result.metrics.get("inference_interventions")
    result.metrics.update(
        {
            "simulate_s": elapsed,
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
    return result


def train(spec: ExecutionSpec) -> ExecutionResult:
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
    if spec.executor != "graph":
        return ExecutionResult(
            executor="legacy", metrics={"request": "train", "routing": "legacy"}
        )
    moved_options = set(spec.options) & {
        "epochs",
        "batch_size",
        "shuffle",
        "updates",
        "save_final_checkpoint",
        "save_selected_checkpoint",
        "checkpoint_selection",
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
    sources = _split_input_bindings(spec.input_bindings)
    built = build(spec)
    assert isinstance(built.model, GraphExecutor)
    model = built.model
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
        return inputs, targets, rows

    resolved_inputs, resolved_targets, target_rows = resolve_data(spec)
    dataset_size = dataset_sample_count(resolved_inputs)
    validation_data = (
        resolve_data(spec.validation) if spec.validation is not None else None
    )
    selection = (
        spec.checkpoint_selection or CheckpointSelection.legacy_training_batch_loss()
    )
    if not isinstance(selection, CheckpointSelection):
        raise TypeError("checkpoint_selection must be a CheckpointSelection")
    selection.validate(
        training, epochs=dataset_epochs, has_validation=validation_data is not None
    )
    epoch_selection = selection.mode == "epoch_metrics"
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
    batches_per_epoch = math.ceil(dataset_size / batch_size)
    shuffle = spec.shuffle
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
                "drop_last": False,
                "order_seed": int(spec.seed),
            },
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
        else (
            dataset_epochs * math.ceil(dataset_size / batch_size) if dataset_mode else 1
        )
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
                    device=device,
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
                    device=device,
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

    def evaluate_split(inputs, targets, *, probe_name=None):
        sample_count = dataset_sample_count(inputs)
        draws = inputs.draw_seeds if isinstance(inputs, DatasetProvider) else (None,)
        presentation_count = sample_count * len(draws)
        phase = (
            "train_evaluation"
            if inputs is resolved_inputs
            else ("validation" if probe_name is None else f"probe:{probe_name}")
        )
        total_loss = 0.0
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
            if observation_spec is not None
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
                    for start in range(0, sample_count, batch_size):
                        end = min(start + batch_size, sample_count)
                        evaluation_targets = {
                            name: value[start:end] for name, value in targets.items()
                        }
                        forward, signals = model._forward(
                            batch_tensors(
                                inputs,
                                torch.arange(start, end),
                                phase=phase,
                                draw_seed=draw_seed,
                            ),
                            diagnostics=False,
                            required_signals=required_signals
                            if probe_name is None
                            else (),
                            observation_populations=observation_populations,
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
                            if objective["kind"] == "cross_entropy":
                                prediction = forward.outputs[
                                    output_ids[objective["prediction"]]
                                ]
                                raw = torch.nn.functional.cross_entropy(
                                    prediction,
                                    evaluation_targets[objective["target"]].long(),
                                )
                                name = f"objective[{i}]"
                                cross_entropies[name] = (
                                    cross_entropies.get(name, 0.0) + float(raw) * count
                                )
                        total_loss += float(loss) * count
                        for name, value in components.items():
                            components_total[name] = (
                                components_total.get(name, 0.0) + float(value) * count
                            )
                        for i, objective in enumerate(training.get("objectives", [])):
                            if not X.classification(objective):
                                continue
                            prediction = forward.outputs[
                                output_ids[objective["prediction"]]
                            ].argmax(dim=-1)
                            correct[f"objective[{i}]"] += int(
                                (
                                    prediction
                                    == evaluation_targets[objective["target"]]
                                ).sum()
                            )
        finally:
            model.train(was_training)
        metrics = {
            "loss": total_loss / presentation_count,
            "cross_entropies": {
                name: value / presentation_count
                for name, value in cross_entropies.items()
            },
            "components": {
                name: value / presentation_count
                for name, value in components_total.items()
            },
            "accuracies": {
                name: value / presentation_count for name, value in correct.items()
            },
        }
        if len(correct) == 1:
            metrics["accuracy"] = next(iter(correct.values())) / presentation_count
        if measurements is not None:
            metrics["observation"] = measurements.finish()
            metrics["observation"]["input_protocol"] = inputs.protocol
            metrics["observation"]["evaluation_seed"] = seed
        if isinstance(inputs, DatasetProvider):
            metrics["encoding_draws"] = list(draws)
        return metrics

    def record_epoch(epoch):
        nonlocal gradient_measurements, selected_checkpoint
        if observation_spec is not None and any(
            row["epoch"] == epoch for row in epoch_history
        ):
            return
        row = {"epoch": epoch}
        observations = {}
        selection_metrics = None
        splits = {"train": (resolved_inputs, resolved_targets)}
        if validation_data is not None:
            splits["validation"] = validation_data[:2]
        for split, (inputs, targets) in splits.items():
            metrics = evaluate_split(inputs, targets)
            if split == selection.split:
                selection_metrics = metrics
            if "observation" in metrics:
                observations[split] = metrics.pop("observation")
                observations[split]["split"] = split
            row.update({f"{split}_{key}": value for key, value in metrics.items()})
        if observation_spec is not None:
            observations["probes"] = {}
            for name, inputs in resolved_probes.items():
                measurement = evaluate_split(inputs, {}, probe_name=name)["observation"]
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
                    "update": completed_updates + len(history),
                    "phase": "initial" if epoch == 0 else "completed_epoch",
                }
            )
            gradient_measurements = GradientMeasurements(
                observation_spec.gradient_norms
            )
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
                    completed_updates + len(history), None, {"epoch": epoch, "batch": 0}
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

    scheduled: list[tuple[int, int, torch.Tensor]] = []
    if dataset_mode:
        for epoch in range(data_state["epoch"], dataset_epochs):
            generator = torch.Generator(device="cpu").manual_seed(spec.seed + epoch)
            order = (
                torch.randperm(dataset_size, generator=generator)
                if shuffle
                else torch.arange(dataset_size)
            )
            first_batch = data_state["batch"] if epoch == data_state["epoch"] else 0
            for batch in range(first_batch, batches_per_epoch):
                scheduled.append(
                    (epoch, batch, order[batch * batch_size : (batch + 1) * batch_size])
                )
        scheduled = scheduled[:updates]
        if not scheduled:
            raise ValueError(
                "dataset training checkpoint is already at the requested end epoch"
            )
    else:
        scheduled = [
            (-1, update, torch.arange(dataset_size)) for update in range(updates)
        ]

    if dataset_mode and data_state["batch"] == 0:
        record_epoch(data_state["epoch"])

    for update, (epoch, batch, sample_indices) in enumerate(scheduled):
        optimizer.zero_grad(set_to_none=True)
        batch_inputs = batch_tensors(
            resolved_inputs,
            sample_indices,
            phase="train",
            epoch=epoch if dataset_mode else completed_updates + update,
        )
        batch_targets = {
            name: value.index_select(0, sample_indices.to(value.device))
            for name, value in resolved_targets.items()
        }
        forward, signal_values = model._forward(
            batch_inputs,
            diagnostics=spec.diagnostics,
            required_signals=required_signals,
        )
        loss, components = compute_loss(
            forward, signal_values, batch_targets, resolved_inputs.protocol
        )
        loss.backward()
        last_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in parameter_map.items()
            if parameter.grad is not None
        }
        clip = training.get("gradient_clip")
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in groups for parameter in group["params"]],
                float(clip),
            )
        if observation_spec is not None:
            gradient_measurements.add(last_gradients, parameter_map)
        optimizer.step()
        rows = {row["id"]: row for row in graph.get("parameters", [])}
        with torch.no_grad():
            for name, parameter in parameter_map.items():
                constraint = rows[name].get("constraint")
                if constraint:
                    parameter.copy_(X.apply_constraint(parameter, constraint))
        absolute_update = completed_updates + update + 1
        next_data_state: dict[str, Any] = {}
        if dataset_mode:
            next_data_state = {"epoch": epoch, "batch": batch + 1}
            if next_data_state["batch"] == batches_per_epoch:
                next_data_state = {"epoch": epoch + 1, "batch": 0}
        loss_value = float(loss.detach())
        history.append(
            {
                "update": absolute_update,
                **(
                    {"encoding": copy.deepcopy(resolved_inputs.last_batch)}
                    if isinstance(resolved_inputs, DatasetProvider)
                    else {}
                ),
                **({"epoch": epoch + 1, "batch": batch + 1} if dataset_mode else {}),
                "loss": loss_value,
                "components": {
                    name: float(value.detach()) for name, value in components.items()
                },
            }
        )
        if (
            (observation_spec is not None or epoch_selection)
            and dataset_mode
            and next_data_state["batch"] == 0
        ):
            record_epoch(next_data_state["epoch"])
        if not epoch_selection:
            candidate = checkpoint_at(absolute_update, loss_value, next_data_state)
            if selected_checkpoint is None or loss_value < float(
                selected_checkpoint.selected_loss
            ):
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
            and not epoch_selection
            and dataset_mode
            and next_data_state["batch"] == 0
        ):
            record_epoch(next_data_state["epoch"])
    assert final_forward is not None
    completed_this_call = len(scheduled)
    final_checkpoint = checkpoint_at(
        completed_updates + completed_this_call,
        history[-1]["loss"],
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

    return ExecutionResult(
        executor="graph",
        outputs=final_forward.outputs,
        diagnostics=final_forward.diagnostics,
        _output_axes=final_forward._output_axes,
        _diagnostic_axes=final_forward._diagnostic_axes,
        _timebase=final_forward._timebase,
        _batch_size=final_forward._batch_size,
        parameters={
            name: value.detach().clone() for name, value in parameter_map.items()
        },
        gradients=last_gradients,
        optimizer_state=optimizer_state_by_name(),
        training_checkpoint=final_checkpoint,
        selected_checkpoint=selected_checkpoint,
        model=model,
        metrics={
            **built.metrics,
            "diagnostics": spec.diagnostics,
            "updates": history,
            "epochs": epoch_history,
            "validation_protocol": (
                {**validation_data[0].protocol, "targets": validation_data[2]}
                if validation_data is not None
                else None
            ),
            "execution_protocol": resolved_inputs.protocol,
            "trainable_parameters": sorted(last_gradients),
            "optimizer": optimizer_spec,
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


def infer(spec: ExecutionSpec) -> ExecutionResult:
    return (
        simulate(spec)
        if spec.executor == "graph"
        else ExecutionResult(
            executor="legacy", metrics={"request": "infer", "routing": "legacy"}
        )
    )


def execution_spec_from_args(
    args: Any, *, kind: RequestKind | None = None
) -> ExecutionSpec:
    """Compatibility adapter: resolved CLI arguments become one typed request."""
    resolved_kind = kind or ("infer" if getattr(args, "infer", False) else args.mode)
    if resolved_kind == "sim":
        resolved_kind = "simulate"
    return ExecutionSpec(
        kind=resolved_kind,
        executor=getattr(args, "executor", "legacy"),
        bundle=Path(args.bundle) if getattr(args, "bundle", None) else None,
        seed=int(getattr(args, "seed", 0) or 0),
        device=resolve_device(getattr(args, "device", "auto")),
        diagnostics=getattr(args, "diagnostics", True),
        checkpoint=(
            Path(args.load_weights) if getattr(args, "load_weights", None) else None
        ),
        epochs=int(getattr(args, "epochs", 0) or 0),
        batch_size=getattr(args, "batch_size", None),
        shuffle=bool(getattr(args, "input_shuffle", False)),
        updates=getattr(args, "updates", None),
        save_final_checkpoint=getattr(args, "save_final_checkpoint", None),
        save_selected_checkpoint=getattr(args, "save_selected_checkpoint", None),
        interventions=tuple(
            parse_intervention(value) for value in getattr(args, "intervention", ())
        ),
        checkpoint_selection=(
            CheckpointSelection.from_dict(json.loads(args.checkpoint_selection))
            if getattr(args, "checkpoint_selection", None)
            else None
        ),
        options={
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "bundle",
                "executor",
                "epochs",
                "batch_size",
                "shuffle",
                "updates",
                "save_final_checkpoint",
                "save_selected_checkpoint",
                "checkpoint_selection",
                "intervention",
            }
        },
    )


def resolve_device(requested: str | torch.device = "auto") -> str:
    """Resolve an explicit device or select the fastest available accelerator."""
    name = str(requested).lower()
    if name == "auto":
        forced = os.environ.get("PINGLAB_DEVICE")
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


def execute_request(
    spec: ExecutionSpec,
    *,
    legacy: Callable[[], ExecutionResult] | None = None,
) -> ExecutionResult:
    """Dispatch one typed request; the CLI supplies its unchanged legacy body."""
    if spec.executor == "legacy":
        if legacy is None:
            raise ValueError(
                "legacy execution requires the registered legacy request body"
            )
        return legacy()
    handlers = {"build": build, "simulate": simulate, "train": train, "infer": infer}
    return handlers[spec.kind](spec)
