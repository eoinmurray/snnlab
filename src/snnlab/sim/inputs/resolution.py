"""File loading and resolution of declared graph inputs and training targets."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch

from snnlab import extensions as E
from snnlab.sim.artifacts.integrity import _file_digest
from snnlab.sim.diagnostics.tracing import current_trace
from snnlab.sim.graph import extensions as X
from snnlab.sim.inputs.bindings import (
    DatasetSnapshotBinding,
    DenseArrayBinding,
    EventStreamBinding,
    InputBinding,
    PoissonInputBinding,
    ResolvedDenseInputs,
    TargetArrayBinding,
    dataset_indices,
)
from snnlab.sim.inputs.encoders import (
    accepts_device,
    encoder_device_policy,
    poisson_device_policy,
    transfer_compact,
)

DENSE_ARRAY_BINDING_SCHEMA = "tools/snnsim.dense-array-binding/v1"


EVENT_STREAM_BINDING_SCHEMA = "tools/snnsim.event-stream-binding/v1"


MIXED_INPUT_BINDING_SCHEMA = "tools/snnsim.mixed-input-bindings/v1"


POISSON_INPUT_BINDING_SCHEMA = "tools/snnsim.poisson-input-binding/v1"


DATASET_SNAPSHOT_BINDING_SCHEMA = "tools/snnsim.dataset-snapshot-binding/v1"


EXECUTION_PROTOCOL_SCHEMA = "tools/snnsim.execution-protocol/v1"


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

        def key(field: str, *, input_id=input_id) -> str:
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


def _validate_dense_values(*, spec, value, input_id, masks, _validated):
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
    if not _validated and value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"input {input_id} contains non-finite values")
    if (
        not _validated
        and signal_type == "spikes"
        and not torch.all((value == 0) | (value == 1))
    ):
        raise ValueError(f"input {input_id} spike values must be boolean or zero/one")
    if signal_type != "mask":
        value = value.float()
    return value, signal_type


def resolve_dense_array_bindings(
    graph: Mapping[str, Any],
    *,
    bindings: Sequence[DenseArrayBinding] = (),
    inputs: Mapping[str, torch.Tensor] | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    protocol: Mapping[str, Any] | None = None,
    _validated: bool = False,
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
        value, signal_type = _validate_dense_values(
            spec=spec,
            value=value,
            input_id=input_id,
            masks=masks,
            _validated=_validated,
        )

        trace = current_trace()
        with trace.region("transfer") if trace is not None else nullcontext():
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


def _validate_event_coordinates(*, input_id, binding, declared, integer_dtypes):
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
        raise ValueError(f"event input {input_id} coordinates must be one-dimensional")
    lengths = {int(value.numel()) for value in coordinates}
    if len(lengths) != 1:
        raise ValueError(f"event input {input_id} coordinate lengths must match")
    if any(value.dtype not in integer_dtypes for value in coordinates):
        raise ValueError(f"event input {input_id} coordinates must use integer dtypes")
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
            raise ValueError(f"event input {input_id} contains duplicate coordinates")
    return steps_count, batch_size, channels_count, steps, batches, channels, flat


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
        steps_count, batch_size, channels_count, steps, batches, channels, flat = (
            _validate_event_coordinates(
                input_id=input_id,
                binding=binding,
                declared=declared,
                integer_dtypes=integer_dtypes,
            )
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
        probability = transfer_compact(
            realized.reshape(1, -1, 1) * dt_ms / 1000.0, device
        )
        # Keep categorical selection's CPU stream, then draw spikes locally.
        if torch.device(device).type != "cpu":
            generator = torch.Generator(device=device).manual_seed(int(binding.seed))
        value = (
            (
                torch.rand(
                    binding.steps_count,
                    binding.batch_size,
                    int(declared[2]),
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
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
                **poisson_device_policy(device),
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


def _validate_dataset_encoder_options(encoder):
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


def _encode_poisson_features(
    *,
    features,
    sample_count,
    channels,
    _validated,
    encoder,
    dt_ms,
    device,
    selected,
    _spike_seed,
    _rate_seed,
    cap,
):
    if features.shape != (sample_count, channels):
        raise ValueError(
            f"rate-Poisson dataset features expected shape {(sample_count, channels)}, got {features.shape}"
        )
    if not np.issubdtype(features.dtype, np.floating):
        raise ValueError("rate-Poisson dataset features must use a floating dtype")
    if not _validated and (
        not np.isfinite(features).all() or np.any(features < 0) or np.any(features > 1)
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
    rates = transfer_compact(
        torch.as_tensor(features[selected], dtype=torch.float32), device
    )
    generator = torch.Generator(device=device).manual_seed(
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
            rates.unsqueeze(0)
            * transfer_compact(selected_rates, device).reshape(1, cap, 1)
            * dt_ms
            / 1000.0
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
            device=device,
        )
        < probability
    ).float()
    encoder_row = {
        "kind": encoder.kind,
        "duration_ms": duration_ms,
        "max_rate_hz": max_rate_hz,
        "seed": int(encoder.seed),
        **poisson_device_policy(device),
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
    return spikes, encoder_row


def _select_dataset_samples(*, binding, _snapshot, _indices, _validated):
    source_path = Path(binding.path)
    if source_path.suffix.lower() != ".npz":
        raise ValueError("dataset snapshot binding requires an NPZ file")
    if _snapshot is None:
        source_digest = _file_digest(source_path)
        with np.load(source_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
    else:
        arrays, source_digest = _snapshot
    if binding.snapshot_digest is not None and source_digest != binding.snapshot_digest:
        raise ValueError("dataset snapshot digest does not match saved binding")
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
        selected = dataset_indices(binding, sample_count)
        cap = len(selected)
    else:
        selected = np.asarray(_indices, dtype=np.int64)
        cap = len(selected)
    selected_labels = labels[selected].astype(np.int64, copy=False)
    sample_ids = None
    if binding.sample_ids_key is not None:
        if binding.sample_ids_key not in arrays:
            raise ValueError("dataset snapshot is missing sample identities")
        identities = np.asarray(arrays[binding.sample_ids_key])
        if (
            identities.ndim != 1
            or len(identities) != sample_count
            or identities.dtype.kind not in {"i", "u", "U"}
            or (not _validated and len(np.unique(identities)) != sample_count)
        ):
            raise ValueError("dataset sample identities are invalid or duplicated")
        sample_ids = identities[selected].tolist()
    return (
        source_path,
        source_digest,
        arrays,
        labels,
        sample_count,
        cap,
        selected,
        selected_labels,
        sample_ids,
    )


def _encode_custom_dataset(
    *,
    encoder,
    arrays,
    binding,
    labels,
    selected,
    dt_ms,
    channels,
    _spike_seed,
    device,
    cap,
):
    spec_row = {"definition": encoder.definition, "config": dict(encoder.config)}
    definition = E.resolve("encoder", spec_row)
    device_aware = accepts_device(definition.function)
    trace = current_trace()
    with trace.region("encoding") if trace is not None else nullcontext():
        callback_arrays = arrays
        if binding.sample_ids_key is not None:
            callback_arrays = {
                **arrays,
                "features": arrays[binding.feature_key],
                "labels": labels,
                "sample_ids": arrays[binding.sample_ids_key],
            }
        spikes = definition.function(
            callback_arrays,
            selected,
            dt_ms=dt_ms,
            channels=channels,
            config=encoder.config,
            seed=encoder.seed if _spike_seed is None else _spike_seed,
            **({"device": torch.device(device)} if device_aware else {}),
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
        **(encoder_device_policy(device) if device_aware else {}),
    }
    return spikes, encoder_row


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
    _event_index=None,
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
    (
        source_path,
        source_digest,
        arrays,
        labels,
        sample_count,
        cap,
        selected,
        selected_labels,
        sample_ids,
    ) = _select_dataset_samples(
        binding=binding, _snapshot=_snapshot, _indices=_indices, _validated=_validated
    )

    encoder = binding.encoder
    dt_ms = float(graph["timebase"]["dt"]["value"])
    channels = int(declared[2])
    encoder_row: dict[str, Any]
    _validate_dataset_encoder_options(encoder=encoder)

    if encoder.kind in {"rate_poisson", "prebinned_spikes"}:
        if binding.feature_key not in arrays:
            raise ValueError(
                f"dataset snapshot is missing feature key {binding.feature_key!r}"
            )
        features = np.asarray(arrays[binding.feature_key])
    if encoder.kind == "rate_poisson":
        spikes, encoder_row = _encode_poisson_features(
            features=features,
            sample_count=sample_count,
            channels=channels,
            _validated=_validated,
            encoder=encoder,
            dt_ms=dt_ms,
            device=device,
            selected=selected,
            _spike_seed=_spike_seed,
            _rate_seed=_rate_seed,
            cap=cap,
        )

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
        from snnlab.sim.inputs.events import EventIndex

        duration_ms = float(encoder.duration_ms or 0)
        index = _event_index or EventIndex(
            arrays,
            sample_count=sample_count,
            channels=channels,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
        )
        spikes, retained, collisions = index.encode(selected, device=device)
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
        spikes, encoder_row = _encode_custom_dataset(
            encoder=encoder,
            arrays=arrays,
            binding=binding,
            labels=labels,
            selected=selected,
            dt_ms=dt_ms,
            channels=channels,
            _spike_seed=_spike_seed,
            device=device,
            cap=cap,
        )

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
        _validated=encoder.kind != "custom",
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
    if sample_ids is not None:
        protocol["dataset_binding"]["sample_ids"] = sample_ids
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
