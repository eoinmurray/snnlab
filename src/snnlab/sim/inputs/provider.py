"""Batch-local spike encoding over immutable, temporary memory-mapped snapshots."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from snnlab.sim.inputs.encoders import (
    accepts_device,
    encoder_device_policy,
    poisson_device_policy,
)
from snnlab.sim.training.observations import evaluation_rng

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
        from snnlab.sim.artifacts.integrity import _file_digest
        from snnlab.sim.inputs.bindings import TargetArrayBinding, dataset_indices
        from snnlab.sim.inputs.resolution import resolve_dataset_snapshot_binding

        binding = copy.deepcopy(binding)
        self.graph, self.binding, self.device = graph, binding, device
        self.device_aware_encoder = False
        if binding.encoder.kind == "custom":
            from snnlab import extensions as E

            definition = E.resolve(
                "encoder",
                {
                    "definition": binding.encoder.definition,
                    "config": binding.encoder.config,
                },
            )
            self.device_aware_encoder = accepts_device(definition.function)
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
        arrays, before = self._load_dataset_arrays(
            binding=binding, _file_digest=_file_digest
        )

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
        self.selected = dataset_indices(binding, total)
        cap = len(self.selected)
        self.sample_ids = self.selected.tolist()
        if binding.sample_ids_key is not None:
            if binding.sample_ids_key not in arrays:
                raise ValueError("dataset snapshot is missing sample identities")
            identities = arrays[binding.sample_ids_key]
            if (
                identities.ndim != 1
                or len(identities) != total
                or identities.dtype.kind not in {"i", "u", "U"}
                or len(np.unique(identities)) != total
            ):
                raise ValueError("dataset sample identities are invalid or duplicated")
            self.sample_ids = identities[self.selected].tolist()
        self.sample_count = cap
        self.labels = torch.from_numpy(np.array(labels[self.selected], dtype=np.int64))
        # Validate stored feature values in bounded sample slices; exact replay remains exact.
        self._validate_dataset_encoder(
            binding=binding, arrays=arrays, graph=graph, total=total
        )

        first, _ = resolve_dataset_snapshot_binding(
            graph,
            binding,
            device="cpu",
            execution_seed=execution_seed,
            _snapshot=self.snapshot,
            _indices=self.selected[:1],
            _validated=True,
            _event_index=self.event_index,
        )
        self.steps_count = next(iter(first.tensors.values())).shape[0]
        self.protocol = copy.deepcopy(first.protocol)
        self.protocol["dataset"].update(sample_cap=cap, batch_size=min(32, cap))
        self.protocol["inputs"][0]["shape"][1] = cap
        self.protocol["dataset_binding"]["selected_indices"] = self.selected.tolist()
        if binding.sample_ids_key is not None:
            self.protocol["dataset_binding"]["sample_ids"] = self.sample_ids
        encoder_row = self.protocol["dataset_binding"]["encoder"]
        encoder_row.pop("selected_rates_hz", None)
        encoder_row.pop("retained_events", None)
        encoder_row.pop("binary_collisions", None)
        if binding.encoder.kind == "rate_poisson":
            encoder_row.update(poisson_device_policy(device))
        if self.device_aware_encoder:
            encoder_row.update(encoder_device_policy(device))
        self.protocol["batch_encoding"] = {
            "schema": SCHEMA,
            "backend": "registered_encoder"
            if binding.encoder.kind == "custom"
            else torch.device(device).type
            if binding.encoder.kind in {"rate_poisson", "event_bin"}
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
        self._merge_dataset_protocol(protocol=protocol)

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
        result, metadata = self._batch(
            indices, phase=phase, epoch=epoch, draw_seed=draw_seed
        )
        self.last_batch = metadata
        return result

    def _batch(
        self, indices, *, phase="evaluation", epoch=0, draw_seed=None, device=None
    ):
        from snnlab.sim.inputs.resolution import resolve_dataset_snapshot_binding

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
                device=self.device if device is None else device,
                execution_seed=self.execution_seed,
                _snapshot=self.snapshot,
                _indices=absolute,
                _spike_seed=spike_seed,
                _rate_seed=rate_seed,
                _validated=True,
                _event_index=self.event_index,
            )
        if next(iter(resolved.tensors.values())).shape[0] != self.steps_count:
            raise ValueError(
                "dataset encoder must preserve presentation duration across batches"
            )
        metadata = {
            "phase": phase,
            "epoch": epoch,
            "sample_indices": absolute.tolist(),
            "draw_seed": int(draw),
            "rate_selection_seed": rate_seed,
            "spike_encoding_seed": spike_seed,
            "encoder": resolved.protocol["dataset_binding"]["encoder"],
        }
        if self.binding.sample_ids_key is not None:
            metadata["sample_ids"] = self.snapshot[0][self.binding.sample_ids_key][
                absolute
            ].tolist()
        return resolved, metadata

    def with_protocol(self, protocol):
        self.protocol = protocol
        return self

    def _load_dataset_arrays(self, binding, _file_digest):
        source = Path(binding.path)
        before = _file_digest(source)
        if binding.snapshot_digest is not None and before != binding.snapshot_digest:
            raise ValueError("dataset snapshot digest does not match saved binding")
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
        return arrays, before

    def _validate_dataset_encoder(self, *, binding, arrays, graph, total):
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
        self.event_index = None
        if binding.encoder.kind == "event_bin":
            from snnlab.sim.inputs.events import EventIndex

            input_spec = next(
                row for row in graph["inputs"] if row["id"] == binding.input_id
            )
            self.event_index = EventIndex(
                arrays,
                sample_count=total,
                channels=int(input_spec["shape"][2]),
                duration_ms=float(binding.encoder.duration_ms or 0),
                dt_ms=float(graph["timebase"]["dt"]["value"]),
            )

    def _merge_dataset_protocol(self, protocol):
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


def _batch_preparer(*, inputs, device, device_encoding, stream):
    def prepare(request):
        indices, phase, epoch, draw_seed = request
        resolved, metadata = inputs._batch(
            indices,
            phase=phase,
            epoch=epoch,
            draw_seed=draw_seed,
            device=device if device_encoding else "cpu",
        )
        tensors = resolved.tensors
        if stream is not None and not device_encoding:
            tensors = {name: value.pin_memory() for name, value in tensors.items()}
        return tensors, metadata

    return prepare


def _batch_stager(*, pending, staged, stream, device, submit):
    def stage(pool):
        request, future = pending.popleft()
        host, metadata = future.result()
        if stream is not None:
            with torch.cuda.stream(stream):
                tensors = {
                    name: value.to(device, non_blocking=True)
                    for name, value in host.items()
                }
                event = torch.cuda.Event()
                event.record(stream)
        else:
            tensors = {name: value.to(device) for name, value in host.items()}
            event = None
        staged.append((request, tensors, metadata, host, event))
        submit(pool)

    return stage


def iter_batches(inputs, requests, *, prefetch=0):
    """Yield (request, tensors, encoding metadata), with bounded CPU/CUDA lookahead.

    Requests are (indices, phase, epoch, draw_seed). Custom encoders and device
    generation stay on the caller thread. Only isolated CPU preparation uses
    the worker; callbacks may consume process-global RNG state.
    """
    _integer("prefetch", prefetch)
    if not isinstance(inputs, DatasetProvider) or not prefetch:
        for request in requests:
            indices, phase, epoch, draw_seed = request
            tensors = batch_tensors(
                inputs, indices, phase=phase, epoch=epoch, draw_seed=draw_seed
            )
            yield (
                request,
                tensors,
                copy.deepcopy(inputs.last_batch)
                if isinstance(inputs, DatasetProvider)
                else None,
            )
        return

    device = torch.device(inputs.device)
    stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    pending = deque()
    staged = deque()
    requests = iter(requests)
    device_encoding = device.type != "cpu" and (
        inputs.binding.encoder.kind in {"rate_poisson", "event_bin"}
        or inputs.device_aware_encoder
    )
    caller_preparation = device_encoding or inputs.binding.encoder.kind == "custom"

    prepare = _batch_preparer(
        inputs=inputs, device=device, device_encoding=device_encoding, stream=stream
    )

    def submit(pool):
        try:
            request = next(requests)
        except StopIteration:
            return False
        if caller_preparation:
            future = Future()
            # Preparation happens only on the caller. evaluation_rng in _batch
            # restores callback RNG state before numerical execution resumes.
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                future.set_result(prepare(request))
        else:
            future = pool.submit(prepare, request)
        pending.append((request, future))
        return True

    stage = _batch_stager(
        pending=pending, staged=staged, stream=stream, device=device, submit=submit
    )

    # One worker preserves deterministic preparation order; queue length bounds
    # the number of encoded CPU and device batches owned by this iterator.
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="snnlab-encoding"
    ) as pool:
        try:
            for _ in range(prefetch):
                if not submit(pool):
                    break
            while pending or staged:
                while pending and len(staged) < prefetch:
                    stage(pool)
                request, tensors, metadata, host, event = staged.popleft()
                if event is not None:
                    current = torch.cuda.current_stream(device)
                    current.wait_event(event)
                    for tensor in tensors.values():
                        tensor.record_stream(current)
                yield request, tensors, metadata
                # Pinned source lifetime must extend through the asynchronous copy.
                if event is not None:
                    event.synchronize()
        finally:
            for _, future in pending:
                future.cancel()
            if stream is not None:
                stream.synchronize()


def simulate_dataset(
    model,
    provider,
    *,
    batch_size,
    diagnostics,
    interventions,
    runtime_state,
    measurement=None,
    recording=None,
    retain_outputs=True,
    retain_parameters=True,
    retain_final_state=True,
    retain_runtime_state=True,
    prefetch=0,
):
    from snnlab.sim.runtime.interventions import (
        DenseSpikeReplay,
        ReplaySpikes,
        SparseSpikeReplay,
        prepare_interventions,
    )

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
    collected_outputs, collected_diagnostics, collected_states, collected_recordings = (
        {},
        {},
        {},
        {},
    )
    batches = []
    result = None
    requests = (
        (
            torch.arange(start, min(start + batch_size, provider.sample_count)),
            "inference",
            0,
            None,
        )
        for start in range(0, provider.sample_count, batch_size)
    )
    for request, encoded, metadata in iter_batches(
        provider, requests, prefetch=prefetch
    ):
        indices = request[0]
        start, end = int(indices[0]), int(indices[-1]) + 1
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
        active_recording = recording
        if recording is not None and recording.sink is not None:

            def dataset_sink(block, *, start=start, end=end):
                recording.sink(
                    replace(block, sample_ids=tuple(provider.sample_ids[start:end]))
                )

            active_recording = replace(recording, sink=dataset_sink)
        result = model(
            encoded,
            diagnostics=diagnostics,
            interventions=tuple(active),
            runtime_state=runtime_state,
            measurement=measurement,
            recording=active_recording,
            retain_outputs=retain_outputs,
            retain_parameters=retain_parameters,
            retain_final_state=retain_final_state,
            retain_runtime_state=retain_runtime_state,
            batch_offset=start,
        )
        batches.append(metadata)
        for target, values in (
            (collected_outputs, result.outputs),
            (collected_diagnostics, result.diagnostics),
            (collected_states, result.final_state if retain_outputs else {}),
            (collected_recordings, result.recorded_signals),
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
    kinds = {row.signal: row.kind for row in recording.signals} if recording else {}
    result.recorded_signals = {
        name: torch.cat(values, dim=0 if kinds[name] == "spike_events" else 1)
        for name, values in collected_recordings.items()
    }
    result._batch_size = provider.sample_count
    result._recording_info = {
        name: replace(
            info,
            batch_size=provider.sample_count,
            sample_ids=tuple(provider.sample_ids),
        )
        for name, info in result._recording_info.items()
    }
    result.final_state = {
        name: torch.cat(values, dim=0) for name, values in collected_states.items()
    }
    if provider.sample_count > batch_size:
        result.runtime_state = None
    result.metrics["inference_interventions"] = identity if interventions else None
    if result.metrics.get("inference_retention"):
        result.metrics["inference_retention"]["window"].update(
            batch_size=provider.sample_count, batch_offset=0
        )
    result.metrics["encoding_batches"] = batches
    provider.protocol["dataset"]["batch_size"] = batch_size
    return result
