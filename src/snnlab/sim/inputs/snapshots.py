"""Portable feature/label partitions with explicit original sample identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from snnlab.sim.artifacts.evidence import json_value, publish_file, sha256_file
from snnlab.sim.inputs.bindings import DatasetEncoder, DatasetSnapshotBinding

DATASET_SCHEMA = "snnlab.dataset-snapshot/v1"
MANIFEST_KEY = "__snnlab_manifest__"


@dataclass(frozen=True)
class DatasetPartition:
    features: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    provenance: Mapping = field(default_factory=dict)
    feature_sample_axis: int = 0

    def __post_init__(self):
        for name in ("features", "labels", "sample_ids"):
            value = np.asarray(getattr(self, name))
            if value.dtype.hasobject:
                raise ValueError(f"{name}: object arrays are not portable")
            value = value.copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if self.features.ndim < 1 or self.labels.ndim < 1 or self.sample_ids.ndim != 1:
            raise ValueError(
                "partitions require a leading sample axis and one-dimensional identities"
            )
        count = len(self.sample_ids)
        axis = self.feature_sample_axis
        if type(axis) is not int or not 0 <= axis < self.features.ndim:
            raise ValueError(
                "feature_sample_axis must identify an existing feature dimension"
            )
        if self.features.shape[axis] != count or len(self.labels) != count:
            raise ValueError("feature, label and identity counts must agree")
        if self.sample_ids.dtype.kind not in {"i", "u", "U"}:
            raise ValueError("sample IDs must be integers or Unicode strings")
        if len(np.unique(self.sample_ids)) != count:
            raise ValueError("duplicate sample identities within a partition")


class DatasetSnapshot:
    def __init__(
        self,
        *,
        dataset_id,
        partitions,
        disjoint=(),
        preprocessing=None,
        provenance=None,
        path=None,
        digest=None,
        keys=None,
    ):
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("dataset_id must be a nonempty string")
        if not partitions or any(
            not isinstance(name, str)
            or not name
            or not isinstance(partition, DatasetPartition)
            for name, partition in partitions.items()
        ):
            raise ValueError("partitions require named DatasetPartition objects")
        self.dataset_id = dataset_id
        self.partitions = dict(partitions)
        self.disjoint = tuple(disjoint)
        self.preprocessing = json_value(preprocessing)
        self.provenance = json_value(provenance or {})
        self.path, self._digest, self._keys = path, digest, keys
        self.validate()

    @classmethod
    def from_partitions(
        cls, *, dataset_id, partitions, disjoint=(), preprocessing=None, provenance=None
    ):
        return cls(
            dataset_id=dataset_id,
            partitions=partitions,
            disjoint=disjoint,
            preprocessing=preprocessing,
            provenance=provenance,
        )

    @property
    def counts(self):
        return {
            name: len(partition.sample_ids)
            for name, partition in self.partitions.items()
        }

    def validate(self):
        if len(set(self.disjoint)) != len(self.disjoint) or set(self.disjoint) - set(
            self.partitions
        ):
            raise ValueError("disjoint partitions must name unique existing splits")
        seen = set()
        for name in self.disjoint:
            identities = set(self.partitions[name].sample_ids.tolist())
            if seen & identities:
                raise ValueError(
                    "sample identities overlap in partitions declared disjoint"
                )
            seen.update(identities)
        json.dumps(
            {"preprocessing": self.preprocessing, "provenance": self.provenance},
            allow_nan=False,
        )

    def write(self, path, *, overwrite=False):
        if Path(path).suffix.lower() != ".npz":
            raise ValueError("dataset snapshots require an .npz destination")
        self.validate()
        arrays, partitions = {}, {}
        for index, (name, partition) in enumerate(self.partitions.items()):
            fields = {}
            for field_name in ("features", "labels", "sample_ids"):
                key = f"partition_{index:04d}_{field_name}"
                array = getattr(partition, field_name)
                arrays[key] = array
                fields[field_name] = {
                    "key": key,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                }
            partitions[name] = {
                "count": len(partition.sample_ids),
                "arrays": fields,
                "provenance": json_value(partition.provenance),
                "feature_sample_axis": partition.feature_sample_axis,
            }
        manifest = {
            "schema": DATASET_SCHEMA,
            "dataset_id": self.dataset_id,
            "partitions": partitions,
            "disjoint": list(self.disjoint),
            "preprocessing": self.preprocessing,
            "provenance": self.provenance,
        }
        arrays[MANIFEST_KEY] = np.frombuffer(
            json.dumps(manifest, sort_keys=True, allow_nan=False).encode(),
            dtype=np.uint8,
        )

        def write(pending):
            with pending.open("wb") as handle:
                np.savez_compressed(handle, **arrays)

        return publish_file(path, write, overwrite=overwrite)

    @classmethod
    def read(cls, path):
        path = Path(path)
        digest = "sha256:" + sha256_file(path)
        partitions, keys = {}, {}
        with np.load(path, allow_pickle=False) as archive:
            if (
                len(archive.files) != len(set(archive.files))
                or MANIFEST_KEY not in archive
            ):
                raise ValueError("dataset archive lacks a unique native manifest")
            raw = archive[MANIFEST_KEY]
            if raw.dtype != np.uint8 or raw.ndim != 1:
                raise ValueError("invalid dataset manifest storage")
            metadata = json.loads(raw.tobytes().decode())
            if metadata.get("schema") != DATASET_SCHEMA:
                raise ValueError("unsupported dataset snapshot schema")
            declared = {MANIFEST_KEY}
            for name, row in metadata["partitions"].items():
                fields = {}
                for field_name in ("features", "labels", "sample_ids"):
                    descriptor = row["arrays"][field_name]
                    key = descriptor["key"]
                    if key in declared or key not in archive:
                        raise ValueError("dataset array key is missing or duplicated")
                    declared.add(key)
                    value = archive[key]
                    if (
                        list(value.shape) != descriptor["shape"]
                        or str(value.dtype) != descriptor["dtype"]
                        or hashlib.sha256(value.tobytes()).hexdigest()
                        != descriptor["sha256"]
                    ):
                        raise ValueError("dataset array integrity mismatch")
                    fields[field_name] = value
                partition = DatasetPartition(
                    **fields,
                    provenance=row.get("provenance", {}),
                    feature_sample_axis=row.get("feature_sample_axis", 0),
                )
                if len(partition.sample_ids) != row["count"]:
                    raise ValueError("dataset partition count mismatch")
                partitions[name] = partition
                keys[name] = {
                    name: descriptor["key"]
                    for name, descriptor in row["arrays"].items()
                }
            if set(archive.files) != declared:
                raise ValueError("dataset archive contains undeclared arrays")
        if "sha256:" + sha256_file(path) != digest:
            raise ValueError("dataset changed while reading")
        return cls(
            dataset_id=metadata["dataset_id"],
            partitions=partitions,
            disjoint=metadata["disjoint"],
            preprocessing=metadata["preprocessing"],
            provenance=metadata["provenance"],
            path=path.resolve(),
            digest=digest,
            keys=keys,
        )

    def binding(
        self,
        split,
        *,
        input_id,
        encoder: DatasetEncoder,
        target_id=None,
        sample_ids=None,
    ):
        if self.path is None:
            raise ValueError(
                "bindings require a saved snapshot; write it and use DatasetSnapshot.read(path)"
            )
        if split not in self.partitions:
            raise ValueError(f"unknown dataset split {split!r}")
        if not isinstance(encoder, DatasetEncoder):
            raise TypeError("encoder must be a DatasetEncoder")
        if encoder.kind not in {"rate_poisson", "prebinned_spikes", "custom"}:
            raise ValueError(
                "feature/label snapshots support rate_poisson, prebinned_spikes or custom encoders"
            )
        indices = None
        partition = self.partitions[split]
        if encoder.kind == "rate_poisson" and (
            partition.feature_sample_axis != 0 or partition.features.ndim != 2
        ):
            raise ValueError(
                "rate_poisson requires sample-leading (samples, channels) features"
            )
        if encoder.kind == "prebinned_spikes" and (
            partition.feature_sample_axis != 1 or partition.features.ndim != 3
        ):
            raise ValueError(
                "prebinned_spikes requires (time, samples, channels) features and feature_sample_axis=1"
            )
        if partition.labels.ndim != 1 or partition.labels.dtype.kind not in {"i", "u"}:
            raise ValueError("native bindings require one-dimensional integer labels")
        if sample_ids is not None:
            requested = tuple(sample_ids)
            if len(set(requested)) != len(requested):
                raise ValueError("diagnostic sample identities must be unique")
            lookup = {
                identity: index
                for index, identity in enumerate(partition.sample_ids.tolist())
            }
            if not requested or set(requested) - set(lookup):
                raise ValueError(
                    "requested sample identities are absent from the partition"
                )
            indices = tuple(lookup[identity] for identity in requested)
        keys = self._keys[split]
        return DatasetSnapshotBinding(
            path=self.path,
            input_id=input_id,
            target_id=target_id,
            dataset_id=self.dataset_id,
            split=split,
            encoder=encoder,
            feature_key=keys["features"],
            label_key=keys["labels"],
            sample_ids_key=keys["sample_ids"],
            sample_indices=indices,
            snapshot_digest=self._digest,
        )


def stratified_split(*, features, labels, sample_ids, fractions, seed, policy):
    """Per-class largest-remainder allocation; output rows retain source order.

    Ties follow the supplied split order. Small classes may leave a split empty;
    no samples are duplicated to force class coverage.
    """
    if policy != "stratified-v1":
        raise ValueError("unsupported stratified split policy")
    if type(seed) is not int or seed < 0:
        raise ValueError("stratified seed must be a nonnegative integer")
    source = DatasetPartition(features, labels, sample_ids)
    if source.labels.ndim != 1:
        raise ValueError("stratification requires one label per sample")
    if source.labels.dtype.kind not in {"b", "i", "u", "f", "U", "S"} or (
        source.labels.dtype.kind in {"f"} and not np.isfinite(source.labels).all()
    ):
        raise ValueError(
            "stratification requires finite numeric or string class labels"
        )
    if not fractions or any(
        not isinstance(name, str) or not name for name in fractions
    ):
        raise ValueError("fractions require named splits")
    values = np.asarray(list(fractions.values()), dtype=float)
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-12)
    ):
        raise ValueError("split fractions must be finite, nonnegative and sum to one")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = [[] for _ in fractions]
    for label in np.unique(source.labels):
        members = rng.permutation(np.flatnonzero(source.labels == label))
        exact = len(members) * values
        counts = np.floor(exact).astype(int)
        for index in np.argsort(-(exact - counts), kind="stable")[
            : len(members) - counts.sum()
        ]:
            counts[index] += 1
        offset = 0
        for index, count in enumerate(counts):
            indices[index].extend(members[offset : offset + count])
            offset += count
    provenance = {
        "policy": policy,
        "seed": seed,
        "fractions": dict(fractions),
        "allocation": "per_class_largest_remainder",
        "tie_break": "split_insertion_order",
        "row_order": "source",
        "rng": "numpy.PCG64",
    }
    return {
        name: DatasetPartition(
            source.features[selected],
            source.labels[selected],
            source.sample_ids[selected],
            provenance,
        )
        for name, rows in zip(fractions, indices, strict=True)
        for selected in [np.asarray(sorted(rows), dtype=np.int64)]
    }
