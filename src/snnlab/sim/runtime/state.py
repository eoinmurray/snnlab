"""Dynamic trajectory state, compatibility and portable persistence."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from snnlab.sim.artifacts.integrity import _file_digest, _json_digest
from snnlab.sim.graph.plan import GraphPlan

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
    segment_state: dict[str, Any] = field(default_factory=dict)
    segment_counts: dict[str, torch.Tensor] = field(default_factory=dict)
    reduction_state: dict[str, Any] = field(default_factory=dict)
    reduction_tensors: dict[str, torch.Tensor] = field(default_factory=dict)

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
            segment_state=copy.deepcopy(self.segment_state),
            segment_counts=moved(self.segment_counts),
            reduction_state=copy.deepcopy(self.reduction_state),
            reduction_tensors=moved(self.reduction_tensors),
        )


def runtime_state_compatibility(
    plan: GraphPlan, *, projection_mode="fast", threshold_inclusive=True
) -> dict[str, Any]:
    """Describe state-layout and dynamical semantics, excluding parameter values."""
    if not isinstance(threshold_inclusive, bool):
        raise ValueError("threshold_inclusive must be boolean")
    if projection_mode not in {"fast", "strict"}:
        raise ValueError("projection_mode must be fast or strict")
    parameters = {row["id"]: row for row in plan.graph.get("parameters", [])}
    builtin = {
        row["id"]
        for row in plan.graph.get("projections", [])
        if row["synapse"]["kind"] != "custom_synapse"
    }
    return {
        "schema": RUNTIME_STATE_SCHEMA,
        **(
            {"builtin_spike_threshold_inclusive": False}
            if not threshold_inclusive
            else {}
        ),
        **(
            {"zero_decay_projection_mode": "fast"}
            if projection_mode == "fast"
            and any(
                p.enabled and p.decay == 0 and p.id in builtin for p in plan.projections
            )
            else {}
        ),
        "dt_ms": plan.dt_ms,
        **(
            {"autapses": dict(plan.autapses)}
            if any("allow_autapses" in row for row in plan.graph.get("projections", []))
            else {}
        ),
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


def runtime_state_signature(
    plan: GraphPlan, *, projection_mode="fast", threshold_inclusive=True
) -> str:
    payload = runtime_state_compatibility(
        plan, projection_mode=projection_mode, threshold_inclusive=threshold_inclusive
    )
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
        "segment_counts": state.segment_counts,
        "reduction_tensors": state.reduction_tensors,
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
        "schema_version": 4
        if state.reduction_state
        else 3
        if state.segment_state
        else (2 if state.custom_state or state.currents else 1),
        "signature": state.signature,
        "compatibility": state.compatibility,
        "completed_steps": state.completed_steps,
        "tensors_file": "tensors.npz",
        "tensors_digest": tensors_digest,
        "tensors": tensors,
    }
    if state.segment_state:
        manifest["segment_state"] = state.segment_state
        manifest["segment_state_digest"] = _json_digest(state.segment_state)
    if state.reduction_state:
        manifest["reduction_state"] = state.reduction_state
        manifest["reduction_state_digest"] = _json_digest(state.reduction_state)
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
    ) not in {1, 2, 3, 4}:
        raise ValueError(f"unsupported runtime-state schema: {manifest.get('schema')}")
    if manifest.get("schema_version") == 4 and _json_digest(
        manifest.get("reduction_state", {})
    ) != manifest.get("reduction_state_digest"):
        raise ValueError("runtime measurement metadata digest mismatch")
    if manifest.get("segment_state") is not None and _json_digest(
        manifest.get("segment_state", {})
    ) != manifest.get("segment_state_digest"):
        raise ValueError("runtime decision metadata digest mismatch")
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
        "segment_counts": {},
        "reduction_tensors": {},
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
        segment_state=manifest.get("segment_state", {}),
        segment_counts=groups["segment_counts"],
        reduction_state=manifest.get("reduction_state", {}),
        reduction_tensors=groups["reduction_tensors"],
    )
