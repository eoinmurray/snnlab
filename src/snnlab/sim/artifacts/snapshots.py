"""Opt-in, inspectable network/run snapshots (not resumable checkpoints)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import torch

from snnlab import __version__
from snnlab.sim.artifacts.integrity import _json_digest

SNAPSHOT_SCHEMA = "snnlab.run-snapshot/v1"


@dataclass(frozen=True)
class SnapshotSpec:
    path: str | Path
    parameters: bool = True
    stage: Literal["start", "end", "both"] = "start"

    def __post_init__(self):
        if not isinstance(self.path, (str, Path)) or not str(self.path):
            raise ValueError("snapshot path must be a nonempty string or Path")
        if not isinstance(self.parameters, bool):
            raise TypeError("snapshot parameters must be boolean")
        if self.stage not in {"start", "end", "both"}:
            raise ValueError("snapshot stage must be start, end or both")


@dataclass(frozen=True)
class RunSnapshot:
    """Captured JSON metadata and independent CPU parameter copies."""

    metadata: dict[str, Any]
    parameters: dict[str, torch.Tensor]


def _tensor_bytes(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()


def _json_value(value):
    """Describe input arrays by content, without embedding potentially huge data."""
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "sha256": hashlib.sha256(_tensor_bytes(value).tobytes()).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("snapshot cannot describe object arrays")
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("snapshot mapping keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", type(value).__qualname__)
        descriptor = {"callable": f"{module}:{name}"}
        # Built-in recording sinks are callable objects, not plain functions.
        if hasattr(value, "identity"):
            descriptor["identity"] = _json_value(value.identity)
        if hasattr(value, "directory"):
            descriptor["directory"] = _json_value(value.directory)
        return descriptor
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported snapshot value: {type(value).__name__}")


def write_run_snapshot(path: str | Path, snapshot: RunSnapshot) -> Path:
    """Write snapshot.json and optional parameters.npz into a new directory.

    Existing snapshot files are never overwritten. JSON is published last so
    its presence marks a completed export. Input data and dynamic/optimizer/RNG
    states are not persisted; use checkpoints for continuation.
    """
    if not isinstance(snapshot, RunSnapshot):
        raise TypeError("snapshot must be a RunSnapshot")
    metadata = _json_value(snapshot.metadata)
    arrays = {}
    inventory = {}
    for index, (name, value) in enumerate(sorted(snapshot.parameters.items())):
        tensor = value.detach().cpu().contiguous()
        key = f"parameter_{index:04d}"
        # NumPy has no native bfloat16; store the original bits losslessly.
        array = (
            tensor.view(torch.uint16).numpy()
            if tensor.dtype == torch.bfloat16
            else tensor.numpy()
        )
        arrays[key] = array
        inventory[name] = {
            "file": "parameters.npz",
            "key": key,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "storage_dtype": str(array.dtype),
        }
    metadata["parameters"] = inventory
    encoded = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".snapshot.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    published_parameters = False
    try:
        if any((root / name).exists() for name in ("snapshot.json", "parameters.npz")):
            raise FileExistsError(f"snapshot already exists in {root}")
        with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=root) as temporary:
            pending = Path(temporary)
            if arrays:
                np.savez_compressed(pending / "parameters.npz", **arrays)
                metadata["parameter_file_sha256"] = hashlib.sha256(
                    (pending / "parameters.npz").read_bytes()
                ).hexdigest()
                encoded = (
                    json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)
                    + "\n"
                )
            (pending / "snapshot.json").write_text(encoded)
            if arrays:
                os.link(pending / "parameters.npz", root / "parameters.npz")
                published_parameters = True
            os.link(pending / "snapshot.json", root / "snapshot.json")
    except BaseException:
        if published_parameters:
            (root / "parameters.npz").unlink()
        raise
    finally:
        lock.unlink()
    return root


class SnapshotSession:
    """Internal lifecycle shared by execution handlers."""

    def __init__(self, spec, model, *, kind, protocol, training=None, **resolved):
        option = spec.snapshot
        self.option = (
            SnapshotSpec(option) if isinstance(option, (str, Path)) else option
        )
        self.snapshots = {}
        self.model = model
        self.metadata = None
        if self.option is None:
            return
        destinations = [Path(self.option.path)]
        if self.option.stage == "both":
            destinations = [destinations[0] / stage for stage in ("start", "end")]
        for destination in destinations:
            if any(
                (destination / name).exists()
                for name in ("snapshot.json", "parameters.npz", ".snapshot.lock")
            ):
                raise FileExistsError(f"snapshot already exists in {destination}")
        excluded = {
            "graph",
            "training",
            "snapshot",
            "runtime_state",
            "on_epoch_end",
            "tracing",
        }
        self.metadata = _json_value(
            {
                "schema": SNAPSHOT_SCHEMA,
                "snnlab_version": __version__,
                "kind": kind,
                "network": model.plan.graph,
                "graph_digest": _json_digest(model.plan.graph),
                "training": training,
                "request": {
                    f.name: getattr(spec, f.name)
                    for f in fields(spec)
                    if f.name not in excluded
                },
                "execution": {
                    "device_requested": str(spec.device),
                    "device_resolved": resolved.pop("device_resolved"),
                    "timestep_ms": model.plan.dt_ms,
                    "duration_ms": protocol.get("timing", {}).get("duration_ms"),
                    "seed": spec.seed,
                    "protocol": protocol,
                    "projection_mode": model.projection_mode,
                    "compile_mode": model.compile_mode,
                    **resolved,
                },
            }
        )

    def capture(self, stage, **resolved):
        if self.option is None or self.option.stage not in {stage, "both"}:
            return
        metadata = {
            **self.metadata,
            "stage": stage,
            "parameters_included": self.option.parameters,
        }
        metadata["execution"] = {**metadata["execution"], **_json_value(resolved)}
        parameters = (
            {
                name: value.detach().cpu().clone()
                for name, value in self.model.parameter_map().items()
            }
            if self.option.parameters
            else {}
        )
        snapshot = RunSnapshot(metadata, parameters)
        destination = Path(self.option.path)
        if self.option.stage == "both":
            destination /= stage
        write_run_snapshot(destination, snapshot)
        self.snapshots[stage] = snapshot
