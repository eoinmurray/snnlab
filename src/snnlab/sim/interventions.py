"""Typed emitted-spike interventions and authenticated replay snapshots."""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Mapping, Sequence, TypeAlias

import numpy as np
import torch

SCHEMA = "tools/snnsim.inference-interventions/v1"
REPLAY_SCHEMA = "snnlab.spike-replay/v1"


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"replay {name} must be an integer >= {minimum}")
    return int(value)


def _header(representation, start_step, steps_count, batch_size, cells_count, dt_ms):
    if isinstance(dt_ms, bool) or not math.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("replay dt_ms must be finite and positive")
    return {
        "schema": REPLAY_SCHEMA,
        "representation": representation,
        "start_step": _integer("start_step", start_step),
        "steps_count": _integer("steps_count", steps_count, 1),
        "batch_size": _integer("batch_size", batch_size, 1),
        "cells_count": _integer("cells_count", cells_count, 1),
        "dt_ms": float(dt_ms),
    }


def _digest(header, arrays):
    digest = hashlib.sha256(
        (json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    for array in arrays:
        digest.update(array.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class DenseSpikeReplay:
    value: torch.Tensor
    start_step: int
    dt_ms: float
    sha256: str

    def _snapshot(self):
        if not isinstance(self.value, torch.Tensor) or self.value.ndim != 3:
            raise ValueError(
                "dense replay requires a binary [time, batch, cells] tensor"
            )
        if self.value.is_complex():
            raise ValueError("dense replay values must use a real binary dtype")
        value = self.value.detach().cpu().clone().contiguous()
        if torch.any((value != 0) & (value != 1)):
            raise ValueError("dense replay values must be binary")
        header = _header("dense", self.start_step, *value.shape, self.dt_ms)
        array = value.to(torch.uint8).numpy()
        return header, (torch.from_numpy(array.copy()),), _digest(header, (array,))

    @classmethod
    def from_tensor(cls, value, *, start_step=0, dt_ms):
        provisional = cls(value, start_step, dt_ms, "")
        header, tensors, digest = provisional._snapshot()
        return cls(tensors[0], header["start_step"], header["dt_ms"], digest)


@dataclass(frozen=True)
class SparseSpikeReplay:
    steps: torch.Tensor
    batches: torch.Tensor
    cells: torch.Tensor
    start_step: int
    steps_count: int
    batch_size: int
    cells_count: int
    dt_ms: float
    sha256: str

    def _snapshot(self):
        header = _header(
            "sparse",
            self.start_step,
            self.steps_count,
            self.batch_size,
            self.cells_count,
            self.dt_ms,
        )
        tensors = []
        for name in ("steps", "batches", "cells"):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 1
                or value.dtype
                not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
            ):
                raise ValueError(
                    f"sparse replay {name} must be a one-dimensional integer tensor"
                )
            tensors.append(value.detach().cpu().to(torch.int64).clone())
        steps, batches, cells = tensors
        if len({value.numel() for value in tensors}) != 1:
            raise ValueError("sparse replay coordinate lengths must agree")
        relative = steps - header["start_step"]
        for name, value, upper in (
            ("steps", relative, header["steps_count"]),
            ("batches", batches, header["batch_size"]),
            ("cells", cells, header["cells_count"]),
        ):
            if torch.any(value < 0) or torch.any(value >= upper):
                raise ValueError(f"sparse replay {name} coordinates are out of range")
        coordinates = list(
            zip(steps.tolist(), batches.tolist(), cells.tolist(), strict=True)
        )
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("sparse replay contains duplicate coordinates")
        if coordinates != sorted(coordinates):
            raise ValueError(
                "sparse replay coordinates must be ordered by absolute step, batch, cell"
            )
        arrays = tuple(value.numpy().astype("<i8", copy=False) for value in tensors)
        return header, tuple(tensors), _digest(header, arrays)

    @classmethod
    def from_events(
        cls,
        *,
        steps,
        batches,
        cells,
        start_step=0,
        steps_count,
        batch_size,
        cells_count,
        dt_ms,
    ):
        provisional = cls(
            steps,
            batches,
            cells,
            start_step,
            steps_count,
            batch_size,
            cells_count,
            dt_ms,
            "",
        )
        header, tensors, digest = provisional._snapshot()
        return cls(
            *tensors,
            header["start_step"],
            header["steps_count"],
            header["batch_size"],
            header["cells_count"],
            header["dt_ms"],
            digest,
        )


SpikeReplay: TypeAlias = DenseSpikeReplay | SparseSpikeReplay


@dataclass(frozen=True)
class ReplaySpikes:
    population_id: str
    replay: SpikeReplay


@dataclass(frozen=True)
class DropSpikes:
    population_id: str
    probability: float
    seed: int | None = None


@dataclass(frozen=True)
class AddPoissonSpikes:
    population_id: str
    rate_hz: float
    seed: int | None = None


Intervention: TypeAlias = ReplaySpikes | DropSpikes | AddPoissonSpikes


@dataclass
class PreparedIntervention:
    metadata: dict
    tensors: tuple[torch.Tensor, ...] = ()

    def apply(self, spikes, absolute_step, index):
        row = self.metadata
        if row["kind"] == "replay_spikes":
            replay = row["replay"]
            if replay["representation"] == "dense":
                return self.tensors[0][absolute_step - replay["start_step"]].to(
                    device=spikes.device, dtype=spikes.dtype
                )
            steps, batches, cells = self.tensors
            begin = int(torch.searchsorted(steps, absolute_step))
            end = int(torch.searchsorted(steps, absolute_step, right=True))
            replacement = torch.zeros_like(spikes)
            replacement[
                batches[begin:end].to(spikes.device), cells[begin:end].to(spikes.device)
            ] = 1
            return replacement
        material = f"{row['seed']}:{index}:{row['kind']}:{row['population_id']}:{absolute_step}".encode()
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (
            2**63 - 1
        )
        generator = torch.Generator(device=spikes.device).manual_seed(seed)
        sample = torch.rand(spikes.shape, device=spikes.device, generator=generator)
        if row["kind"] == "drop_spikes":
            return spikes * (sample >= row["probability"])
        return torch.maximum(
            spikes, (sample < row["probability_per_step"]).to(spikes.dtype)
        )


def prepare_interventions(
    interventions: Sequence[Intervention],
    *,
    graph: Mapping,
    seed: int,
    start_step: int,
    steps_count: int,
    batch_size: int,
):
    start_step = _integer("execution start_step", start_step)
    steps_count = _integer("execution steps_count", steps_count, 1)
    batch_size = _integer("execution batch_size", batch_size, 1)
    populations = {row["id"]: row for row in graph["populations"]}
    dt_ms = float(graph["timebase"]["dt"]["value"])
    requested, prepared, keys = [], [], set()
    for intervention in interventions:
        if not isinstance(intervention, (ReplaySpikes, DropSpikes, AddPoissonSpikes)):
            raise TypeError(
                "interventions must contain ReplaySpikes, DropSpikes or AddPoissonSpikes objects"
            )
        name = intervention.population_id
        if name not in populations:
            raise ValueError(f"intervention targets unknown population {name!r}")
        if not populations[name].get("spiking"):
            raise ValueError(f"intervention population {name!r} does not emit spikes")
        tensors = ()
        if isinstance(intervention, ReplaySpikes):
            if not isinstance(
                intervention.replay, (DenseSpikeReplay, SparseSpikeReplay)
            ):
                raise TypeError(
                    "ReplaySpikes requires DenseSpikeReplay or SparseSpikeReplay"
                )
            header, tensors, digest = intervention.replay._snapshot()
            if intervention.replay.sha256 != digest:
                raise ValueError("spike replay digest mismatch")
            if (
                header["batch_size"] != batch_size
                or header["cells_count"] != populations[name]["size"]
            ):
                raise ValueError(
                    "spike replay batch/cell coverage does not match target"
                )
            if not math.isclose(header["dt_ms"], dt_ms, rel_tol=0, abs_tol=1e-12):
                raise ValueError("spike replay dt_ms does not match execution timestep")
            if (
                start_step < header["start_step"]
                or start_step + steps_count
                > header["start_step"] + header["steps_count"]
            ):
                raise ValueError(
                    "spike replay does not cover the absolute execution window"
                )
            row = {
                "kind": "replay_spikes",
                "population_id": name,
                "replay": {**header, "sha256": digest},
            }
            requested.append(dict(row))
        else:
            resolved_seed = seed if intervention.seed is None else intervention.seed
            resolved_seed = _integer("intervention seed", resolved_seed)
            if isinstance(intervention, DropSpikes):
                probability = float(intervention.probability)
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError(
                        "drop probability must be finite and between zero and one"
                    )
                row = {
                    "kind": "drop_spikes",
                    "population_id": name,
                    "probability": probability,
                }
            else:
                rate = float(intervention.rate_hz)
                probability = rate * dt_ms / 1000
                if not math.isfinite(rate) or rate < 0 or probability > 1:
                    raise ValueError(
                        "Poisson rate must be finite, non-negative, and satisfy rate times dt <= 1"
                    )
                row = {
                    "kind": "add_poisson_spikes",
                    "population_id": name,
                    "rate_hz": rate,
                }
            requested.append(
                {
                    **row,
                    **(
                        {"seed": intervention.seed}
                        if intervention.seed is not None
                        else {}
                    ),
                }
            )
            row = {**row, "seed": resolved_seed}
            if isinstance(intervention, AddPoissonSpikes):
                row["probability_per_step"] = probability
        key = (row["kind"], name)
        if key in keys:
            raise ValueError(
                f"intervention repeats {row['kind']} for population {name}"
            )
        keys.add(key)
        prepared.append(PreparedIntervention(row, tensors))
    identity = {
        "schema": SCHEMA,
        "requested": requested,
        "resolved": [item.metadata for item in prepared],
        "window": {
            "start_step": start_step,
            "steps_count": steps_count,
            "batch_size": batch_size,
            "dt_ms": dt_ms,
        },
    }
    return prepared, identity


def intervention_identity(
    interventions, *, graph, seed=0, start_step=0, steps_count, batch_size
):
    """Resolve and authenticate an expected intervention identity before cache reuse."""
    return (
        prepare_interventions(
            interventions,
            graph=graph,
            seed=seed,
            start_step=start_step,
            steps_count=steps_count,
            batch_size=batch_size,
        )[1]
        if interventions
        else None
    )


def save_spike_replay(path: str | Path, replay: SpikeReplay):
    """Save a replay payload; returns the file-byte digest required by the CLI loader."""
    header, tensors, digest = replay._snapshot()
    if digest != replay.sha256:
        raise ValueError("spike replay digest mismatch")
    fields = {
        "metadata": np.array(json.dumps({**header, "sha256": digest}, sort_keys=True))
    }
    names = (
        ("value",)
        if isinstance(replay, DenseSpikeReplay)
        else ("steps", "batches", "cells")
    )
    fields.update(
        {name: tensor.numpy() for name, tensor in zip(names, tensors, strict=True)}
    )
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **fields)
    payload = buffer.getvalue()
    Path(path).write_bytes(payload)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def load_spike_replay(path: str | Path, *, sha256: str):
    """Authenticate file bytes before decoding the snapshot, with pickle disabled."""
    payload = Path(path).read_bytes()
    if "sha256:" + hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError("spike replay file digest mismatch")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        header = json.loads(str(archive["metadata"].item()))
        if header.get("schema") != REPLAY_SCHEMA:
            raise ValueError("unsupported spike replay schema")
        shared = {name: header[name] for name in ("start_step", "dt_ms", "sha256")}
        if header["representation"] == "dense":
            if set(archive.files) != {"metadata", "value"}:
                raise ValueError("dense replay file fields do not match schema")
            replay = DenseSpikeReplay(
                torch.from_numpy(archive["value"].copy()), **shared
            )
        elif header["representation"] == "sparse":
            if set(archive.files) != {"metadata", "steps", "batches", "cells"}:
                raise ValueError("sparse replay file fields do not match schema")
            replay = SparseSpikeReplay(
                **{
                    name: torch.from_numpy(archive[name].copy())
                    for name in ("steps", "batches", "cells")
                },
                **shared,
                **{
                    name: header[name]
                    for name in ("steps_count", "batch_size", "cells_count")
                },
            )
        else:
            raise ValueError("unsupported spike replay representation")
    checked, _, digest = replay._snapshot()
    if (
        checked != {key: value for key, value in header.items() if key != "sha256"}
        or digest != replay.sha256
    ):
        raise ValueError("spike replay payload metadata or digest mismatch")
    return replay


def parse_intervention(value: str):
    """CLI drop:ID=p, add:ID=Hz, or replay:ID=PATH@sha256:FILE_DIGEST."""
    target, separator, raw = value.partition("=")
    kind, colon, name = target.partition(":")
    if not separator or not colon or not name or not raw:
        raise ValueError(
            "intervention expects drop:ID=p, add:ID=Hz or replay:ID=PATH@sha256:DIGEST"
        )
    if kind == "drop":
        return DropSpikes(name, float(raw))
    if kind == "add":
        return AddPoissonSpikes(name, float(raw))
    if kind == "replay":
        path, marker, digest = raw.rpartition("@sha256:")
        if not marker or not path or len(digest) != 64:
            raise ValueError("replay intervention requires PATH@sha256:FILE_DIGEST")
        return ReplaySpikes(name, load_spike_replay(path, sha256="sha256:" + digest))
    raise ValueError(f"unsupported intervention kind {kind!r}")
