"""Authenticated inference products, derived arrays and retention identity."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import EllipsisType
from typing import Any

import numpy as np
import torch

from snnlab.sim.artifacts.integrity import _file_digest, _json_digest
from snnlab.sim.graph.plan import _signal_axes
from snnlab.sim.requests.result import ExecutionResult
from snnlab.sim.runtime.recording import MeasurementWindow, recording_identity

INFERENCE_ARTIFACT_SCHEMA = "tools/snnsim.inference-artifacts/v1"


DERIVED_INFERENCE_SCHEMA = "tools/snnsim.derived-inference/v1"


def retention_identity(
    *,
    graph,
    start_step,
    steps_count,
    batch_size,
    diagnostics=True,
    retain_outputs=True,
    retain_parameters=True,
    retain_final_state=True,
    retain_runtime_state=True,
    projection_mode="fast",
    precision=None,
    measurement=None,
    recording=None,
    batch_offset=0,
):
    """Resolve recording/window identity before checking an inference cache."""
    for name, value in (
        ("start_step", start_step),
        ("steps_count", steps_count),
        ("batch_size", batch_size),
        ("batch_offset", batch_offset),
    ):
        if (
            type(value) is not int
            or value < 0
            or (name in {"steps_count", "batch_size"} and value == 0)
        ):
            raise ValueError(f"{name} must be a valid execution coordinate")
    if projection_mode not in {"fast", "strict"}:
        raise ValueError("projection_mode must be fast or strict")
    if any(
        not isinstance(value, bool)
        for value in (
            diagnostics,
            retain_outputs,
            retain_parameters,
            retain_final_state,
            retain_runtime_state,
        )
    ):
        raise TypeError("diagnostics and retention flags must be boolean")
    if measurement is not None:
        if not isinstance(measurement, MeasurementWindow):
            raise TypeError("measurement must be MeasurementWindow")
        measurement.validate()
    return {
        **({"precision": dict(precision)} if precision is not None else {}),
        "measurement": {
            "start_step": measurement.start_step,
            "end_step": measurement.end_step,
        }
        if measurement
        else None,
        "recording": recording_identity(
            recording, axes=_signal_axes(graph) if recording is not None else {}
        ),
        "retain_outputs": retain_outputs,
        "window": {
            "start_step": start_step,
            "steps_count": steps_count,
            "batch_size": batch_size,
            "batch_offset": batch_offset,
        },
        "diagnostics": diagnostics,
        **({"projection_mode": "fast"} if projection_mode == "fast" else {}),
        **(
            {
                "result_snapshots": {
                    "parameters": retain_parameters,
                    "final_state": retain_final_state,
                    "runtime_state": retain_runtime_state,
                }
            }
            if not all((retain_parameters, retain_final_state, retain_runtime_state))
            else {}
        ),
    }


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
    if result.recorded_signals:
        payloads["recordings.npz"] = result.recorded_signals
    files = []
    expanded_bfloat16 = {}
    for filename, tensors in payloads.items():
        arrays = {
            name: (
                value.detach().cpu().float()
                if value.dtype == torch.bfloat16
                else value.detach().cpu()
            ).numpy()
            for name, value in sorted(tensors.items())
        }
        names = [
            name for name, value in tensors.items() if value.dtype == torch.bfloat16
        ]
        if names:
            expanded_bfloat16[filename] = sorted(names)
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
    if result.metrics.get("segment_policy") is not None:
        request["segment_policy"] = result.metrics["segment_policy"]
        request["segment_window"] = result.metrics["segment_window"]
    if result.metrics.get("inference_retention") is not None:
        request["inference_retention"] = result.metrics["inference_retention"]
    manifest = {
        "schema": INFERENCE_ARTIFACT_SCHEMA,
        "schema_version": 2 if result.recorded_signals else 1,
        "graph_digest": _json_digest(graph),
        "request_seed": int(seed),
        "request_digest": _json_digest(request),
        "files": files,
    }
    if expanded_bfloat16:
        # Analysis caches use NumPy floating arithmetic. Widen bfloat16 values
        # exactly; resumable checkpoints/runtime state retain their raw bits.
        manifest["tensor_storage"] = {
            "schema": "snnlab.inference-bfloat16/v1",
            "logical_dtype": "bfloat16",
            "stored_dtype": "float32",
            "arrays": expanded_bfloat16,
        }
    manifest["artifact_digest"] = _json_digest(manifest)
    (root / "inference-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _validate_artifact_files(root, manifest):
    expected_files = {
        "recording.npz",
        "outputs.npz",
        "parameters.npz",
        "metrics.json",
    }
    if manifest["schema_version"] == 2:
        expected_files.add("recordings.npz")
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


def validate_inference_artifacts(
    path: str | Path,
    *,
    graph: Mapping[str, Any] | None = None,
    seed: int | None = None,
    expected_interventions: Mapping[str, Any] | None | EllipsisType = ...,
    expected_segments: Mapping[str, Any] | None | EllipsisType = ...,
    expected_retention: Mapping[str, Any] | None | EllipsisType = ...,
) -> Mapping[str, Any]:
    """Authenticate a persisted graph inference artifact set before reuse."""
    root = Path(path)
    manifest_path = root / "inference-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != INFERENCE_ARTIFACT_SCHEMA or manifest.get(
        "schema_version"
    ) not in {1, 2}:
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
    _validate_artifact_files(root=root, manifest=manifest)

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
    if metrics.get("segment_policy") is not None:
        request["segment_policy"] = metrics["segment_policy"]
        request["segment_window"] = metrics["segment_window"]
    if (
        expected_segments is not Ellipsis
        and metrics.get("segment_policy") != expected_segments
    ):
        raise ValueError(
            "inference artifact reset/decision policy does not match expected identity"
        )
    if metrics.get("inference_retention") is not None:
        request["inference_retention"] = metrics["inference_retention"]
    if (
        expected_retention is not Ellipsis
        and metrics.get("inference_retention") != expected_retention
    ):
        raise ValueError(
            "inference artifact retention does not match expected identity"
        )
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
