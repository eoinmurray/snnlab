"""Batch-local spike encoding over immutable, temporary memory-mapped snapshots."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import zipfile
from contextlib import nullcontext
from dataclasses import replace
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from snnlab.sim.epoch_observations import evaluation_rng

SCHEMA = "snnlab.batch-encoding/v1"


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"dataset {name} must be an integer >= {minimum}")
    return int(value)


def stream_seed(root, domain, phase, epoch, indices, draw):
    payload = json.dumps(
        [int(root), domain, phase, int(epoch), list(map(int, indices)), int(draw)],
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


class DatasetProvider:
    """Resolve identity once; provide encoded batches without retained spike history."""

    def __init__(
        self,
        graph,
        binding,
        *,
        device="cpu",
        execution_seed=0,
        protocol=None,
        encoding_seeds=(),
    ):
        from snnlab.sim.execution import (
            TargetArrayBinding,
            _file_digest,
            resolve_dataset_snapshot_binding,
        )

        binding = copy.deepcopy(binding)
        self.graph, self.binding, self.device = graph, binding, device
        self.execution_seed = _integer("execution seed", execution_seed)
        _integer("encoder seed", binding.encoder.seed)
        _integer("order seed", binding.order_seed)
        if not isinstance(binding.shuffle, bool):
            raise ValueError("dataset shuffle must be boolean")
        self.draw_seeds = tuple(
            _integer("validation encoding seed", seed) for seed in encoding_seeds
        ) or (binding.encoder.seed,)
        if len(set(self.draw_seeds)) != len(self.draw_seeds):
            raise ValueError("validation encoding seeds must be distinct")
        if encoding_seeds and binding.encoder.kind not in {"rate_poisson", "custom"}:
            raise ValueError(
                "validation encoding seeds require a stochastic dataset encoder"
            )
        self._temporary = tempfile.TemporaryDirectory(prefix="snnlab-dataset-")
        source = Path(binding.path)
        before = _file_digest(source)
        arrays = {}
        with zipfile.ZipFile(source) as archive:
            for index, member in enumerate(archive.infolist()):
                if not member.filename.endswith(".npy") or member.is_dir():
                    raise ValueError("dataset NPZ must contain only named NPY arrays")
                key = member.filename[:-4]
                if key in arrays:
                    raise ValueError("dataset NPZ contains duplicate array names")
                destination = Path(self._temporary.name) / f"array-{index}.npy"
                with (
                    archive.open(member) as incoming,
                    destination.open("wb") as outgoing,
                ):
                    shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                arrays[key] = np.load(destination, mmap_mode="r", allow_pickle=False)
        if _file_digest(source) != before:
            raise ValueError("dataset snapshot changed while resolving its content")
        self.snapshot = arrays, before
        if binding.label_key not in arrays:
            raise ValueError("dataset snapshot is missing labels")
        labels = arrays[binding.label_key]
        if (
            labels.ndim != 1
            or not np.issubdtype(labels.dtype, np.integer)
            or len(labels) == 0
        ):
            raise ValueError(
                "dataset snapshot labels must be a non-empty one-dimensional integer array"
            )
        total = len(labels)
        cap = (
            total
            if binding.sample_cap is None
            else _integer("sample_cap", binding.sample_cap, 1)
        )
        if cap > total:
            raise ValueError("dataset sample cap exceeds available samples")
        order = (
            torch.randperm(
                total, generator=torch.Generator().manual_seed(binding.order_seed)
            )
            if binding.shuffle
            else torch.arange(total)
        )
        self.selected = order[:cap].numpy()
        self.sample_count = cap
        self.labels = torch.from_numpy(np.array(labels[self.selected], dtype=np.int64))
        # Validate stored feature values in bounded sample slices; exact replay remains exact.
        if binding.encoder.kind in {"rate_poisson", "prebinned_spikes"}:
            if binding.feature_key not in arrays:
                raise ValueError("dataset snapshot is missing features")
            features = arrays[binding.feature_key]
            for start in range(0, total, 32):
                chunk = (
                    features[start : start + 32]
                    if binding.encoder.kind == "rate_poisson"
                    else features[:, start : start + 32]
                )
                if binding.encoder.kind == "rate_poisson":
                    if (
                        not np.isfinite(chunk).all()
                        or np.any(chunk < 0)
                        or np.any(chunk > 1)
                    ):
                        raise ValueError(
                            "rate-Poisson dataset features must be finite in [0, 1]"
                        )
                elif not np.all((chunk == 0) | (chunk == 1)):
                    raise ValueError("prebinned dataset spikes must be binary")
        first, _ = resolve_dataset_snapshot_binding(
            graph,
            binding,
            device="cpu",
            execution_seed=execution_seed,
            _snapshot=self.snapshot,
            _indices=self.selected[:1],
            _validated=True,
        )
        self.steps_count = next(iter(first.tensors.values())).shape[0]
        self.protocol = copy.deepcopy(first.protocol)
        self.protocol["dataset"].update(sample_cap=cap, batch_size=min(32, cap))
        self.protocol["inputs"][0]["shape"][1] = cap
        self.protocol["dataset_binding"]["selected_indices"] = self.selected.tolist()
        encoder_row = self.protocol["dataset_binding"]["encoder"]
        encoder_row.pop("selected_rates_hz", None)
        encoder_row.pop("retained_events", None)
        encoder_row.pop("binary_collisions", None)
        self.protocol["batch_encoding"] = {
            "schema": SCHEMA,
            "backend": "registered_encoder"
            if binding.encoder.kind == "custom"
            else "cpu",
            "random_dtype": "float32"
            if binding.encoder.kind == "rate_poisson"
            else None,
            "training": "fresh_each_presentation"
            if binding.encoder.kind in {"rate_poisson", "custom"}
            else "exact_replay",
            "evaluation": "fixed_draw_set",
            "draw_seeds": list(self.draw_seeds),
            "streams": {
                "dataset_order": binding.order_seed,
                "epoch_order": execution_seed,
                "rate_selection": binding.encoder.seed,
                "spike_encoding": binding.encoder.seed,
            },
            "epoch_order_derivation": "execution_seed + epoch",
            "seed_derivation": "sha256(root,domain,phase,epoch,absolute_sample_indices,draw_seed)",
        }
        supplied = dict(protocol or {})
        dataset = dict(supplied.pop("dataset", {}))
        for key, value in dataset.items():
            if key not in self.protocol["dataset"] or (
                value is not None and value != self.protocol["dataset"][key]
            ):
                raise ValueError(
                    "dataset execution protocol metadata does not match binding"
                )
        if set(supplied) & set(self.protocol):
            raise ValueError(
                "dataset execution protocol cannot override reserved fields"
            )
        self.protocol.update(supplied)
        self.targets = (
            (
                TargetArrayBinding(
                    binding.target_id,
                    self.labels,
                    source={
                        **self.protocol["dataset_binding"]["source"],
                        "array": binding.label_key,
                    },
                ),
            )
            if binding.target_id
            else ()
        )

    def batch(self, indices, *, phase="evaluation", epoch=0, draw_seed=None):
        from snnlab.sim.execution import resolve_dataset_snapshot_binding

        local = np.asarray(indices, dtype=np.int64)
        absolute = self.selected[local]
        draw = self.draw_seeds[0] if draw_seed is None else draw_seed
        epoch = epoch if phase == "train" else 0
        spike_seed = stream_seed(
            self.binding.encoder.seed, "spike_encoding", phase, epoch, absolute, draw
        )
        rate_seed = stream_seed(
            self.binding.encoder.seed, "rate_selection", phase, epoch, absolute, draw
        )
        with (
            evaluation_rng(spike_seed)
            if self.binding.encoder.kind == "custom"
            else nullcontext()
        ):
            resolved, _ = resolve_dataset_snapshot_binding(
                self.graph,
                self.binding,
                device=self.device,
                execution_seed=self.execution_seed,
                _snapshot=self.snapshot,
                _indices=absolute,
                _spike_seed=spike_seed,
                _rate_seed=rate_seed,
                _validated=True,
            )
        if next(iter(resolved.tensors.values())).shape[0] != self.steps_count:
            raise ValueError(
                "dataset encoder must preserve presentation duration across batches"
            )
        self.last_batch = {
            "phase": phase,
            "epoch": epoch,
            "sample_indices": absolute.tolist(),
            "draw_seed": int(draw),
            "rate_selection_seed": rate_seed,
            "spike_encoding_seed": spike_seed,
            "encoder": resolved.protocol["dataset_binding"]["encoder"],
        }
        return resolved

    def with_protocol(self, protocol):
        self.protocol = protocol
        return self


def sample_count(inputs):
    return (
        inputs.sample_count
        if isinstance(inputs, DatasetProvider)
        else next(iter(inputs.tensors.values())).shape[1]
    )


def steps_count(inputs):
    return (
        inputs.steps_count
        if isinstance(inputs, DatasetProvider)
        else next(iter(inputs.tensors.values())).shape[0]
    )


def batch_tensors(inputs, indices, *, phase="evaluation", epoch=0, draw_seed=None):
    if isinstance(inputs, DatasetProvider):
        return inputs.batch(
            indices, phase=phase, epoch=epoch, draw_seed=draw_seed
        ).tensors
    return {
        name: value.index_select(1, torch.as_tensor(indices, device=value.device))
        for name, value in inputs.tensors.items()
    }


def simulate_dataset(
    model, provider, *, batch_size, diagnostics, interventions, runtime_state
):
    from snnlab.sim.execution import DenseSpikeReplay, ReplaySpikes, SparseSpikeReplay
    from snnlab.sim.interventions import prepare_interventions

    batch_size = _integer("batch_size", batch_size, 1)
    if runtime_state is not None and provider.sample_count > batch_size:
        raise ValueError(
            "dataset inference runtime continuation requires a single complete batch"
        )
    start_step = runtime_state.completed_steps if runtime_state is not None else 0
    _, identity = prepare_interventions(
        interventions,
        graph=model.plan.graph,
        seed=model.seed,
        start_step=start_step,
        steps_count=provider.steps_count,
        batch_size=provider.sample_count,
    )
    collected_outputs, collected_diagnostics, collected_states = {}, {}, {}
    batches = []
    result = None
    for start in range(0, provider.sample_count, batch_size):
        end = min(start + batch_size, provider.sample_count)
        active = []
        for intervention in interventions:
            if isinstance(intervention, ReplaySpikes):
                replay = intervention.replay
                header, values, digest = replay._snapshot()
                if digest != replay.sha256:
                    raise ValueError("spike replay digest mismatch")
                if isinstance(replay, DenseSpikeReplay):
                    replay = DenseSpikeReplay.from_tensor(
                        values[0][:, start:end],
                        start_step=header["start_step"],
                        dt_ms=header["dt_ms"],
                    )
                else:
                    steps, batches_index, cells = values
                    mask = (batches_index >= start) & (batches_index < end)
                    replay = SparseSpikeReplay.from_events(
                        steps=steps[mask],
                        batches=batches_index[mask] - start,
                        cells=cells[mask],
                        start_step=header["start_step"],
                        steps_count=header["steps_count"],
                        batch_size=end - start,
                        cells_count=header["cells_count"],
                        dt_ms=header["dt_ms"],
                    )
                active.append(replace(intervention, replay=replay))
            else:
                active.append(intervention)
        encoded = provider.batch(torch.arange(start, end), phase="inference")
        result = model(
            encoded.tensors,
            diagnostics=diagnostics,
            interventions=tuple(active),
            runtime_state=runtime_state,
        )
        batches.append(copy.deepcopy(provider.last_batch))
        for target, values in (
            (collected_outputs, result.outputs),
            (collected_diagnostics, result.diagnostics),
            (collected_states, result.final_state),
        ):
            for name, value in values.items():
                target.setdefault(name, []).append(value)
    for field, collected, axes in (
        ("outputs", collected_outputs, result._output_axes),
        ("diagnostics", collected_diagnostics, result._diagnostic_axes),
    ):
        setattr(
            result,
            field,
            {
                name: torch.cat(values, dim=axes[name].index("batch"))
                for name, values in collected.items()
            },
        )
    result.final_state = {
        name: torch.cat(values, dim=0) for name, values in collected_states.items()
    }
    if provider.sample_count > batch_size:
        result.runtime_state = None
    result.metrics["inference_interventions"] = identity if interventions else None
    result.metrics["encoding_batches"] = batches
    provider.protocol["dataset"]["batch_size"] = batch_size
    return result
