"""Structured access to inspectable run snapshots, not resumable checkpoints."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from snnlab.sim.artifacts.evidence import member_path, sha256_file
from snnlab.sim.artifacts.integrity import _json_digest
from snnlab.sim.artifacts.snapshots import SNAPSHOT_SCHEMA, RunSnapshot
from snnlab.sim.artifacts.tensors import BFLOAT16_STORAGE, tensor_payload
from snnlab.sim.training.policies import FrozenValues


@dataclass(frozen=True)
class PhysicalParameter:
    value: object
    unit: str | None


def _physical(spec):
    suffix_units = {
        "mv": "mV",
        "ms": "ms",
        "nf": "nF",
        "us": "uS",
        "na": "nA",
        "hz": "Hz",
    }
    result = {}
    for name, value in spec.items():
        if name == "config" and isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, dict) and set(item) == {"value", "unit"}:
                    result[f"config.{key}"] = PhysicalParameter(
                        copy.deepcopy(item["value"]), item["unit"]
                    )
                elif isinstance(item, (int, float)) and not isinstance(item, bool):
                    result[f"config.{key}"] = PhysicalParameter(item, None)
        if isinstance(value, dict) and set(value) == {"value", "unit"}:
            result[name] = PhysicalParameter(
                copy.deepcopy(value["value"]), value["unit"]
            )
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[name] = PhysicalParameter(
                value, suffix_units.get(name.rpartition("_")[2])
            )
    return result


class NetworkSnapshot:
    """Metadata is available without opening parameter archives or extensions."""

    def __init__(self, metadata, *, root=None, parameters=None, parameter_source=None):
        self._metadata = copy.deepcopy(metadata)
        self._root = root
        self._parameters = parameters
        self._parameter_source = parameter_source
        self.validate()

    @classmethod
    def read(cls, path, *, parameters=False):
        path = Path(path)
        root = path.parent if path.name == "snapshot.json" else path
        snapshot = cls(json.loads((root / "snapshot.json").read_text()), root=root)
        if parameters:
            snapshot._parameters = snapshot.load_parameters()
        return snapshot

    @classmethod
    def from_snapshot(cls, snapshot, *, parameters=False):
        if not isinstance(snapshot, RunSnapshot):
            raise TypeError("snapshot must be a RunSnapshot")
        metadata = copy.deepcopy(snapshot.metadata)
        source = dict(snapshot.parameters)
        metadata["parameters"] = {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
            }
            for name, value in source.items()
        }
        result = cls(metadata, parameter_source=source)
        if parameters:
            result._parameters = result.load_parameters()
        return result

    @property
    def schema(self):
        return self._metadata["schema"]

    @property
    def snnlab_version(self):
        return self._metadata.get("snnlab_version")

    @property
    def stage(self):
        return self._metadata["stage"]

    @property
    def graph_identity(self):
        return self._metadata["graph_digest"]

    @property
    def execution(self):
        return FrozenValues(
            {"kind": self._metadata["kind"], **self._metadata["execution"]}
        )

    @property
    def training(self):
        value = self._metadata.get("training")
        return FrozenValues(value) if value is not None else None

    @property
    def parameter_info(self):
        execution = self._metadata["execution"]
        initialization = execution.get("initialization", {})
        resolved = execution.get("parameter_membership") or (
            self._metadata.get("training") or {}
        ).get("resolved_parameters", {})
        inventory = self._metadata.get("parameters", {})
        return {
            row["id"]: FrozenValues(
                {
                    "id": row["id"],
                    "shape": row["shape"],
                    "unit": row.get("unit"),
                    "initializer": row["initializer"],
                    "scaling": initialization.get(row["id"], {}).get(
                        "scaling", row.get("initialization_scaling")
                    ),
                    "requested_scaling": row.get("initialization_scaling"),
                    "constraint": row.get("constraint"),
                    "trainable": row["id"] in resolved.get("trainable", ())
                    if resolved
                    else None,
                    "frozen": row["id"] in resolved.get("frozen", ())
                    if resolved
                    else None,
                    "initialization": initialization.get(row["id"]),
                    "runtime_shape": initialization.get(row["id"], {}).get(
                        "runtime_shape"
                    ),
                    "stored": inventory.get(row["id"]),
                }
            )
            for row in self._metadata["network"].get("parameters", ())
        }

    @property
    def populations(self):
        return {
            row["id"]: FrozenValues({**row, "parameters": _physical(row["neuron"])})
            for row in self._metadata["network"].get("populations", ())
        }

    @property
    def projections(self):
        parameters = self.parameter_info
        result = {}
        for row in self._metadata["network"].get("projections", ()):
            ids = row.get("parameters", ())
            weight = parameters[ids[0]] if ids else None
            result[row["id"]] = FrozenValues(
                {
                    **row,
                    "source_id": row["source"].rpartition(".")[0],
                    "target_id": row["target"].rpartition(".")[0],
                    "synapse_parameters": _physical(row["synapse"]),
                    "parameter_info": {name: parameters[name] for name in ids},
                    "matrix_shape": weight.shape if weight else None,
                    "runtime_matrix_shape": weight.runtime_shape if weight else None,
                    "initializer": weight.initializer if weight else None,
                    "scaling": weight.scaling if weight else None,
                    "constraints": {name: parameters[name].constraint for name in ids},
                    "trainable_parameters": tuple(
                        name for name in ids if parameters[name].trainable
                    )
                    if all(parameters[name].trainable is not None for name in ids)
                    else None,
                    "frozen_parameters": tuple(
                        name for name in ids if parameters[name].frozen
                    )
                    if all(parameters[name].frozen is not None for name in ids)
                    else None,
                }
            )
        return result

    def validate(self, *, verify_parameters=False):
        metadata = self._metadata
        if not isinstance(metadata, dict) or metadata.get("schema") != SNAPSHOT_SCHEMA:
            raise ValueError("unsupported network snapshot schema")
        json.dumps(metadata, allow_nan=False)
        if metadata.get("stage") not in {"start", "end"}:
            raise ValueError("snapshot stage must be start or end")
        graph = metadata.get("network")
        if not isinstance(graph, dict) or graph.get("schema") != "snnlang.graph/v1":
            raise ValueError("snapshot requires a supported graph")
        if _json_digest(graph) != metadata.get("graph_digest"):
            raise ValueError("snapshot graph digest mismatch")
        if not isinstance(metadata.get("execution"), dict) or metadata.get(
            "kind"
        ) not in {"simulate", "infer", "train"}:
            raise ValueError("snapshot execution metadata is missing or invalid")
        declared = {row["id"] for row in graph.get("parameters", ())}
        inventory = metadata.get("parameters", {})
        if not isinstance(inventory, dict) or set(inventory) - declared:
            raise ValueError(
                "snapshot parameter inventory references unknown parameters"
            )
        for name, row in inventory.items():
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("shape"), list)
                or not isinstance(row.get("dtype"), str)
            ):
                raise ValueError(f"{name}: invalid snapshot parameter metadata")
            runtime_shape = (
                metadata["execution"]
                .get("initialization", {})
                .get(name, {})
                .get("runtime_shape")
            )
            if runtime_shape is not None and row["shape"] != runtime_shape:
                raise ValueError(
                    f"{name}: snapshot shape disagrees with resolved initialization"
                )
            if self._root is not None:
                member_path(self._root, row.get("file"))
                if row.get("file") != "parameters.npz":
                    raise ValueError("unsupported snapshot parameter file")
        if verify_parameters:
            self.load_parameters()

    def load_parameters(self, *, names=None):
        inventory = self._metadata.get("parameters", {})
        names = tuple(inventory) if names is None else tuple(names)
        if set(names) - set(inventory):
            raise ValueError("requested snapshot parameters are unavailable")
        if self._parameters is not None:
            return {name: self._parameters[name].clone() for name in names}
        if self._parameter_source is not None:
            result = {}
            for name in names:
                value = self._parameter_source[name]
                row = inventory[name]
                if (
                    list(value.shape) != row["shape"]
                    or str(value.dtype).removeprefix("torch.") != row["dtype"]
                ):
                    raise ValueError(f"{name}: captured parameter metadata mismatch")
                result[name] = value.detach().cpu().clone()
            return result
        if not names:
            return {}
        root = self._root
        if root is None:
            raise ValueError("snapshot parameter arrays are unavailable")
        path = member_path(root, "parameters.npz")
        if sha256_file(path) != self._metadata.get("parameter_file_sha256"):
            raise ValueError("snapshot parameter file digest mismatch")
        result = {}
        with np.load(path, allow_pickle=False) as archive:
            for name in names:
                row = dict(inventory[name])
                if row["dtype"] == "bfloat16" and row.get("storage_dtype") == "uint16":
                    row["tensor_storage"] = BFLOAT16_STORAGE
                if row.get("key") not in archive:
                    raise ValueError(f"{name}: parameter array is missing")
                result[name] = tensor_payload(archive[row["key"]], row, label=name)
        return result
