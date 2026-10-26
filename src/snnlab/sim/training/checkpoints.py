"""Training checkpoints, authenticated persistence and RNG restoration."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from snnlab.sim.artifacts.integrity import _file_digest, _json_digest
from snnlab.sim.artifacts.tensors import numpy_payload, tensor_payload

TRAINING_CHECKPOINT_SCHEMA = "tools/snnsim.training-checkpoint/v1"


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
    reporting_state: Mapping[str, Any] | None = None


def _validate_checkpoint_rng(checkpoint):
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


def save_training_checkpoint(path: str | Path, checkpoint: TrainingCheckpoint) -> Path:
    """Atomically write a named, authenticated graph-training checkpoint."""
    _validate_checkpoint_rng(checkpoint=checkpoint)

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    tensors: list[dict[str, Any]] = []

    def append(group: str, name: str, value: torch.Tensor, state: str | None = None):
        key = f"tensor_{len(arrays):04d}"
        tensor = value.detach().cpu().contiguous()
        arrays[key], storage = numpy_payload(tensor)
        row = {
            "group": group,
            "name": name,
            "key": key,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            **storage,
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
        "schema_version": 5
        if checkpoint.reporting_state is not None
        else (
            4
            if checkpoint.selection_contract is not None
            and checkpoint.selection_contract["policy"]["mode"] == "epoch_metrics"
            else (3 if checkpoint.observation_state is not None else 2)
        ),
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
        "reporting_state": checkpoint.reporting_state,
        "reporting_state_digest": _json_digest(checkpoint.reporting_state),
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


def _validate_restored_rng(
    *, schema_version, manifest, rng_backend, declared_devices, accelerator_rng_states
):
    if schema_version in {2, 3, 4, 5}:
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


def _read_checkpoint_metadata(manifest, schema_version):
    reporting_state = manifest.get("reporting_state")
    if schema_version == 5 and reporting_state is None:
        raise ValueError("training checkpoint is missing reporting state")
    if "reporting_state" in manifest and manifest.get(
        "reporting_state_digest"
    ) != _json_digest(reporting_state):
        raise ValueError("training checkpoint reporting state digest mismatch")
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
    return reporting_state, observation_state, selection_metadata


def load_training_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> TrainingCheckpoint:
    """Load and authenticate a portable named graph-training checkpoint."""
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != TRAINING_CHECKPOINT_SCHEMA or manifest.get(
        "schema_version"
    ) not in {1, 2, 3, 4, 5}:
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
            value = tensor_payload(
                array,
                row,
                label=f"training-checkpoint tensor {row['group']}.{row['name']}",
            )
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
    _validate_restored_rng(
        schema_version=schema_version,
        manifest=manifest,
        rng_backend=rng_backend,
        declared_devices=declared_devices,
        accelerator_rng_states=accelerator_rng_states,
    )

    reporting_state, observation_state, selection_metadata = _read_checkpoint_metadata(
        manifest=manifest, schema_version=schema_version
    )

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
        reporting_state=reporting_state,
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
