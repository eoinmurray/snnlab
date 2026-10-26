"""Train the calibrated SHD COBA workload through SNNLab's native trainer."""

import argparse
import csv
import gzip
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

import snnlab
from snnlab import extensions, lang
from snnlab.lang import training
from snnlab.sim import (
    Batching,
    CheckpointSelection,
    DatasetEncoder,
    DatasetSnapshotBinding,
    EpochObservations,
    Performance,
    TraceSpec,
    TrainingMetrics,
    ValidationSpec,
    resolve_device,
    train,
)
from snnlab.sim.inputs.encoders import transfer_compact

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    import benchmark

    benchmark.cli()
    raise SystemExit


def reset_results(output):
    """Replace a run directory so stale reports/checkpoints cannot survive a rerun."""
    output = Path(output).absolute()
    source = Path(__file__).resolve()
    if output.is_symlink() or output.resolve() in source.parents:
        raise ValueError("output must be a dedicated results directory")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    return output


TRAINABLE = (
    "input_to_E1.weight",
    "E1_to_E2.weight",
    "readout_projection.weight",
)


@dataclass(frozen=True)
class Config:
    epochs: int = 10
    batch_size: int = 256
    seed: int = 42
    steps: int = 1000
    dt_ms: float = 1.0
    channels: int = 700
    hidden: int = 128
    classes: int = 20
    validation_fraction: float = 0.1
    input_noise_hz: float = 1.6
    spike_deletion_probability: float = 0.0008

    def __post_init__(self):
        if (
            min(
                self.epochs,
                self.batch_size,
                self.steps,
                self.channels,
                self.hidden,
                self.classes,
            )
            < 1
        ):
            raise ValueError("counts must be positive")
        if self.dt_ms <= 0 or not 0 < self.validation_fraction < 1:
            raise ValueError("invalid timestep or validation fraction")
        if (
            not 0 <= self.spike_deletion_probability <= 1
            or not 0 <= self.input_noise_hz * self.dt_ms / 1000 <= 1
        ):
            raise ValueError("invalid corruption probabilities")


_DEFAULT_CONFIG = Config()


def equal_class_cross_entropy(prediction, target, config):
    losses = torch.nn.functional.cross_entropy(
        prediction, target.long(), reduction="none"
    )
    counts = torch.zeros(config["classes"], device=losses.device, dtype=losses.dtype)
    counts.scatter_add_(0, target, torch.ones_like(target, dtype=losses.dtype))
    return (losses / counts[target]).sum() / (counts > 0).sum()


def rate_normalized_scores(sources, parameters, config):
    scores, counts = sources
    rate = counts.mean(dim=-1, keepdim=True) / config["steps"]
    return scores * config["reference_rate"] / rate.clamp_min(config["minimum_rate"])


def centered_readout(shape, config, *, device, dtype):
    value = torch.randn(shape, device=device, dtype=dtype) * config["std"]
    return value - value.mean(dim=0, keepdim=True)


def identity_gain(shape, config, *, device, dtype):
    return torch.eye(*shape, device=device, dtype=dtype) * config["gain"]


def adam_optimizer(groups, config):
    return torch.optim.Adam(groups, **config)


extensions.register_objective(
    "benchmarks.shd.equal_class_ce/v1",
    equal_class_cross_entropy,
    classification=True,
    capture_safe=True,
)
extensions.register_operation(
    "benchmarks.shd.rate_normalize/v1", rate_normalized_scores, capture_safe=True
)
extensions.register_initializer("benchmarks.shd.centered_readout/v1", centered_readout)
extensions.register_initializer("benchmarks.shd.identity_gain/v1", identity_gain)
extensions.register_optimizer("benchmarks.shd.adam/v1", adam_optimizer)


def author_network(cfg=_DEFAULT_CONFIG):
    net = lang.Network("shd_coba", dt=cfg.dt_ms * lang.ms)
    source = net.input(
        "drive",
        shape=("time", "batch", cfg.channels),
        signal_type="spikes",
        unit="spike",
    )
    learned, populations = [], []
    for index, zero_fraction in ((1, 0.95), (2, 0.9)):
        population = net.population(
            f"E{index}",
            size=cfg.hidden,
            neuron=lang.COBA_LIF(
                tau_mem=20 * lang.ms,
                capacitance_nf=1.0,
                leak_us=0.05,
                resting_mv=-65.0,
                reset_mv=-65.0,
                threshold_mv=-50.0,
                refractory=0 * lang.ms,
                voltage_grad_dampen=1.0,
            ),
        )
        projection = net.connect(
            source,
            population.excitatory,
            name="input_to_E1" if index == 1 else "E1_to_E2",
            synapse=lang.AMPA(tau=10 * lang.ms),
            weight=lang.LowerClampedNormal(
                mean=0.09, std=0.009, initial_zero_fraction=zero_fraction
            ),
            constraint=lang.NonNegative(),
            initialization_scaling="fan_in_normalized",
        )
        learned.append(projection.weight)
        populations.append(population)
        source = population.spikes
    readout = net.population(
        "readout",
        size=cfg.classes,
        spiking=False,
        neuron=lang.LeakyIntegrator(tau=20 * lang.ms),
    )
    projection = net.connect(
        source,
        readout.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=10 * lang.ms),
        weight=lang.CustomInitializer("benchmarks.shd.centered_readout/v1", std=0.1),
        initialization_scaling="direct",
    )
    scores = lang.ops.reduce(
        readout.voltage, operation="mean", over="time", name="scores"
    )
    gain = net.parameter(
        "score_gain.weight",
        shape=(cfg.classes, cfg.classes),
        initializer=lang.CustomInitializer(
            "benchmarks.shd.identity_gain/v1", gain=20.0
        ),
    )
    scores = net.operation(
        "linear",
        scores,
        name="scaled_scores",
        shape=scores.shape,
        unit=scores.unit,
        signal_type="continuous",
        parameters=(gain,),
        size=cfg.classes,
        trainable=False,
    )
    counts = lang.ops.reduce(
        populations[-1].spikes,
        operation="sum",
        over="time",
        name="normalization_counts",
    )
    scores = lang.ops.custom(
        "benchmarks.shd.rate_normalize/v1",
        [scores, counts],
        name="normalized_scores",
        shape=scores.shape,
        unit="1",
        steps=cfg.steps,
        reference_rate=0.02 * cfg.dt_ms,
        minimum_rate=0.001 * cfg.dt_ms,
    )
    net.output("class_scores", scores)
    for index, population in enumerate(populations, 1):
        net.output(
            f"e{index}_counts",
            lang.ops.reduce(
                population.spikes,
                operation="sum",
                over="time",
                name=f"e{index}_count_reduction",
            ),
        )
        if index == 2:
            net.output(
                "e_counts",
                lang.ops.reduce(
                    population.spikes,
                    operation="sum",
                    over="time",
                    name="e_count_reduction",
                ),
            )
    recipe = lang.TrainSpec(
        objectives=(
            training.CustomObjective(
                "benchmarks.shd.equal_class_ce/v1",
                prediction=scores,
                target="class",
                classes=cfg.classes,
            ),
        ),
        parameter_groups=(
            training.ParameterGroup((learned[0],), name="input", lr=0.000005),
            training.ParameterGroup((learned[1],), name="interlayer", lr=0.00002),
            training.ParameterGroup((projection.weight,), name="readout", lr=0.01),
            training.ParameterGroup((gain,), name="fixed_loop", lr=0.0, frozen=True),
        ),
        optimizer=training.CustomOptimizer(
            "benchmarks.shd.adam/v1", lr=0.001, betas=[0.9, 0.999], weight_decay=0.0
        ),
        surrogate=training.FastSigmoid(
            slope=100 / 15, gradient_scale=0.01, threshold_inclusive=False
        ),
        gradient_clip=None,
        epochs=cfg.epochs,
        presentation_duration=cfg.steps * cfg.dt_ms * lang.ms,
    )
    return lang.compile(net, training=recipe)


URL = "https://zenkelab.org/datasets/shd/shd_train.h5.gz"
# Published by the dataset authors in /datasets/md5sums.txt.
ARCHIVE_MD5 = "d47c9825dee33347913e8ce0f2be08b0"


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def acquire(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "shd_train.h5.gz"
    if not archive.exists():
        temporary = cache / "shd_train.download"
        with (
            urllib.request.urlopen(URL, timeout=60) as source,
            temporary.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)
        if digest(temporary, "md5") != ARCHIVE_MD5:
            temporary.unlink()
            raise ValueError("SHD download checksum mismatch")
        temporary.replace(archive)
    if digest(archive, "md5") != ARCHIVE_MD5:
        raise ValueError("cached SHD checksum mismatch")
    path = cache / "shd_train.h5"
    temporary = cache / "shd_train.unpack"
    with gzip.open(archive, "rb") as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target)
    temporary.replace(path)
    return path, {
        "url": URL,
        "archive_md5": ARCHIVE_MD5,
        "archive_sha256": digest(archive),
        "hdf5_sha256": digest(path),
    }


def select_indices(labels, cfg):
    """Match the source's seed-42 per-class 10% holdout, retaining every item."""
    labels = np.asarray(labels)
    if (
        labels.ndim != 1
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any((labels < 0) | (labels >= cfg.classes))
    ):
        raise ValueError("invalid SHD labels")
    rng = np.random.default_rng(cfg.seed)
    train, validation = [], []
    for label in range(cfg.classes):
        available = np.flatnonzero(labels == label)
        selected = rng.permutation(available)
        held_out = max(1, round(len(available) * cfg.validation_fraction))
        if held_out >= len(available):
            raise ValueError("each class needs training and validation examples")
        train.extend(selected[:-held_out])
        validation.extend(selected[-held_out:])
    return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def prepare(path, output, cfg):
    """Save ragged events; dense minibatches are created only by the trainer."""
    with h5py.File(path, "r") as handle:
        labels = np.asarray(handle["labels"], dtype=np.int64)
        train, validation = select_indices(labels, cfg)
        indices = np.concatenate((train, validation))
        times, channels, offsets = [], [], [0]
        dropped = 0
        for index in indices:
            t = np.asarray(handle["spikes/times"][index], dtype=np.float64) * 1000
            ch = np.asarray(handle["spikes/units"][index], dtype=np.int64)
            if (
                t.shape != ch.shape
                or not np.isfinite(t).all()
                or np.any(t < 0)
                or np.any((ch < 0) | (ch >= cfg.channels))
            ):
                raise ValueError("invalid SHD events")
            keep = t < cfg.steps * cfg.dt_ms
            dropped += int((~keep).sum())
            times.append(t[keep])
            channels.append(ch[keep])
            offsets.append(offsets[-1] + int(keep.sum()))
    np.savez_compressed(
        output,
        event_offsets=np.asarray(offsets, dtype=np.int64),
        event_time_ms=np.concatenate(times),
        event_channel=np.concatenate(channels),
        train_labels=labels[train],
        validation_labels=labels[validation],
        official_train_indices=indices,
    )
    return {
        "kind": "SHD",
        "train_count": len(train),
        "validation_count": len(validation),
        "train_class_counts": np.bincount(
            labels[train], minlength=cfg.classes
        ).tolist(),
        "validation_class_counts": np.bincount(
            labels[validation], minlength=cfg.classes
        ).tolist(),
        "selection_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "snapshot_sha256": digest(output),
        "dropped_events": dropped,
        "official_test_accessed": False,
    }


def synthetic_snapshot(output, cfg):
    """Small download-free plumbing check, explicitly not SHD performance data."""
    rng = np.random.default_rng(cfg.seed)
    train_labels = np.tile(np.arange(cfg.classes), 2)
    validation_labels = np.arange(cfg.classes)
    labels = np.concatenate((train_labels, validation_labels))
    times, channels, offsets = [], [], [0]
    for label in labels:
        count = max(1, round(cfg.channels * cfg.steps * 0.15))
        t = rng.integers(cfg.steps, size=count) * cfg.dt_ms
        ch = rng.integers(cfg.channels, size=count)
        ch[: count // 4] = label * (cfg.channels // cfg.classes)
        times.append(t)
        channels.append(ch)
        offsets.append(offsets[-1] + count)
    np.savez_compressed(
        output,
        event_offsets=np.asarray(offsets, dtype=np.int64),
        event_time_ms=np.concatenate(times),
        event_channel=np.concatenate(channels),
        train_labels=train_labels,
        validation_labels=validation_labels,
    )
    return {
        "kind": "synthetic-smoke",
        "train_count": len(train_labels),
        "validation_count": len(validation_labels),
        "snapshot_sha256": digest(output),
        "official_test_accessed": False,
    }


def encode_events(arrays, selected, *, dt_ms, channels, config, seed, device="cpu"):
    """Preserve CPU NumPy draws; device preparation uses a local Torch stream.

    Accelerator calls record snnlab.encoder-device/v1 in execution provenance.
    The device stream is repeatable on the same backend; it does not reproduce
    NumPy's CPU draws. Occupancy is sampled before insertion/deletion on both.
    """
    device = torch.device(device)
    if device.type != "cpu":
        selected = np.asarray(selected, dtype=np.int64) + config["sample_offset"]
        starts = np.asarray(arrays["event_offsets"])[selected]
        lengths = np.asarray(arrays["event_offsets"])[selected + 1] - starts
        columns = np.repeat(np.arange(len(selected)), lengths)
        positions = np.repeat(
            starts - (np.cumsum(lengths) - lengths), lengths
        ) + np.arange(int(lengths.sum()))
        steps = np.floor(arrays["event_time_ms"][positions] / dt_ms).astype(np.int64)
        keep = (steps >= 0) & (steps < config["steps"])
        linear = (steps[keep] * len(selected) + columns[keep]) * channels + arrays[
            "event_channel"
        ][positions][keep]
        spikes = torch.zeros(
            (config["steps"], len(selected), channels),
            device=device,
            dtype=torch.float32,
        )
        spikes.view(-1)[
            transfer_compact(torch.from_numpy(np.unique(linear)), device)
        ] = 1
        noise = torch.rand(
            spikes.shape,
            device=device,
            dtype=torch.float32,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        return torch.where(
            spikes.bool(),
            (noise >= config["spike_deletion_probability"]).float(),
            (noise < config["input_noise_hz"] * dt_ms / 1000).float(),
        )
    spikes = np.zeros((config["steps"], len(selected), channels), dtype=np.float32)
    for column, local_index in enumerate(selected):
        index = int(local_index) + config["sample_offset"]
        start, end = arrays["event_offsets"][index : index + 2]
        steps = np.floor(arrays["event_time_ms"][start:end] / dt_ms).astype(np.int64)
        keep = (steps >= 0) & (steps < len(spikes))
        spikes[steps[keep], column, arrays["event_channel"][start:end][keep]] = 1
    noise = np.random.default_rng(seed).random(spikes.shape, dtype=np.float32)
    occupied = spikes != 0
    spikes[occupied & (noise < config["spike_deletion_probability"])] = 0
    spikes[~occupied & (noise < config["input_noise_hz"] * dt_ms / 1000)] = 1
    return torch.from_numpy(spikes)


extensions.register_encoder("benchmarks.shd.events/v1", encode_events)


ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def progress(report):
    print(
        f"epoch {report.epoch}/{report.total_epochs}: train={report.train.accuracy:.1%}; "
        f"validation={report.validation.accuracy:.1%}; updates={report.updates}; "
        f"timing={dict(report.timing)}",
        flush=True,
    )


def execution_kwargs(cfg, bundle, snapshot, dataset, output, device, compile_mode):
    def binding(split):
        return DatasetSnapshotBinding(
            path=Path(snapshot),
            input_id="drive",
            dataset_id=dataset["kind"],
            split=split,
            target_id="class",
            label_key=f"{split}_labels",
            encoder=DatasetEncoder(
                "custom",
                definition="benchmarks.shd.events/v1",
                seed=cfg.seed + (2 if split == "train" else 4),
                config={
                    "steps": cfg.steps,
                    "sample_offset": 0 if split == "train" else dataset["train_count"],
                    "input_noise_hz": cfg.input_noise_hz,
                    "spike_deletion_probability": cfg.spike_deletion_probability,
                },
            ),
        )

    return dict(
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(binding("train"),),
        validation=ValidationSpec(input_bindings=(binding("validation"),)),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        device=device,
        metrics=TrainingMetrics(
            train="online",
            validation="epoch",
            initial=("validation",),
            update_history=False,
        ),
        batching=Batching(
            order="stratified",
            stratify_by="class",
            remainder="merge_small",
            min_batch_size=min(cfg.classes, cfg.batch_size),
        ),
        observations=EpochObservations(parameter_norms=TRAINABLE),
        checkpoint_selection=CheckpointSelection.disabled(),
        save_final_checkpoint=Path(output) / "final.checkpoint",
        diagnostics=False,
        performance=Performance(compile_mode=compile_mode, dataset_prefetch=1),
        extension_validation="structural",
        training_timing=True,
        timing_mode="synchronized",
        on_epoch_end=progress,
    )


def run(
    cfg,
    output,
    cache,
    device="auto",
    compile_mode="auto",
    smoke=False,
    cpu_threads=4,
    trace_mode="off",
    trace_skip_updates=1,
    trace_updates=3,
    trace_detailed_labels=True,
):
    if trace_mode not in {"off", "monitor", "profile", "both"}:
        raise ValueError("invalid trace mode")
    trace_spec = TraceSpec(
        Path(output) / "tracing",
        mode=trace_mode,
        skip_updates=trace_skip_updates,
        active_updates=trace_updates,
        detailed_labels=trace_detailed_labels,
    )
    output = Path(output)
    output = reset_results(output)
    torch.set_num_threads(cpu_threads)
    device = torch.device(resolve_device(device))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    mode = (
        ("default" if device.type == "cuda" else None)
        if compile_mode == "auto"
        else (None if compile_mode == "eager" else compile_mode)
    )
    started = time.perf_counter()
    snapshot = output / "spikes.npz"
    if smoke:
        dataset = synthetic_snapshot(snapshot, cfg)
    else:
        path, acquisition = acquire(cache)
        dataset = {**prepare(path, snapshot, cfg), **acquisition}
    bundle = author_network(cfg)
    bundle.write(output / "network.bundle")
    sources = sorted((ROOT / "src/snnlab").rglob("*.py")) + sorted(
        Path(__file__).parent.glob("*.py")
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    metadata = {
        "configuration": asdict(cfg),
        "dataset": dataset,
        "engine": "snnlab.sim.train",
        "compile_mode": mode,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "cpu_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "snnlab": snnlab.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "git_commit": commit,
        "source_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in sources
        },
        "smoke": smoke,
        "diagnostic_tracing": {
            "mode": trace_mode,
            "skip_updates": trace_skip_updates,
            "active_updates": trace_updates,
            "detailed_labels": trace_detailed_labels,
        },
    }
    write_json(output / "configuration.json", metadata)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    result = train(
        **execution_kwargs(cfg, bundle, snapshot, dataset, output, str(device), mode)
    )
    if any(
        not torch.isfinite(value).all().item()
        for value in result.training_checkpoint.parameters.values()
    ):
        raise ValueError("nonfinite final parameters")
    write_json(output / "metrics.json", result.metrics)
    write_json(output / "timing.json", result.metrics["training_timing"])
    summary = {
        "kind": dataset["kind"],
        "completed_updates": result.training_checkpoint.completed_updates,
        "wall_s": time.perf_counter() - started,
        "wall_scope": "data preparation, graph authoring, native train call and result verification; excludes process/Modal startup",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
        "memory_scope": "native training including validation/checkpointing and final parameter verification; allocator reserved pool may predate reset",
    }
    write_json(output / "summary.json", summary)
    if trace_mode != "off":
        # Release baseline model/graphs before building an independent diagnostic.
        del result
        diagnostic = dict(
            execution_kwargs(cfg, bundle, snapshot, dataset, output, str(device), mode),
            tracing=trace_spec,
            updates=trace_skip_updates + trace_updates,
            validation=None,
            metrics=TrainingMetrics(
                train="online", validation="none", initial=(), update_history=False
            ),
            observations=None,
            save_final_checkpoint=None,
            on_epoch_end=None,
        )
        traced = train(**diagnostic)
        write_json(output / "diagnostic_metrics.json", traced.metrics)
        # Baseline timings and peaks above exclude this separate diagnostic call.
        summary["diagnostic_trace"] = {
            "directory": "tracing",
            "status": traced.metrics["tracing"]["status"],
        }
        write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results",
        help="results directory, replaced on each run",
    )
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/shd-cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--compile-mode", choices=("auto", "default", "eager"), default="auto"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="synthetic, 64-step, two-epoch CPU-sized check; not performance evidence",
    )
    parser.add_argument(
        "--trace-mode",
        choices=("off", "monitor", "profile", "both"),
        default="off",
        help="separate bounded diagnostic after baseline training",
    )
    parser.add_argument("--trace-skip-updates", type=int, default=1)
    parser.add_argument("--trace-updates", type=int, default=3)
    parser.add_argument(
        "--trace-detailed-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="per-projection and neuron labels in profile mode",
    )
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu threads must be positive")
    cfg = Config(epochs=args.epochs)
    if args.smoke:
        cfg = replace(cfg, epochs=2, steps=64)
    print(
        json.dumps(
            run(
                cfg,
                args.output,
                args.cache,
                args.device,
                args.compile_mode,
                args.smoke,
                args.cpu_threads,
                args.trace_mode,
                args.trace_skip_updates,
                args.trace_updates,
                args.trace_detailed_labels,
            ),
            indent=2,
        )
    )


RUN_ID = "shd-59e3371bb384"
FIGURES = ("learning", "epoch-timing", "sparsity", "diagnostics")


def trace_events(path):
    """Stream the top-level Kineto traceEvents array with bounded buffering."""
    decoder = json.JSONDecoder()
    with Path(path).open() as source:
        buffer = ""
        while '"traceEvents"' not in buffer:
            chunk = source.read(1024 * 1024)
            if not chunk:
                raise ValueError("missing traceEvents array")
            buffer += chunk
        marker = buffer.index('"traceEvents"')
        position = buffer.index("[", marker) + 1
        while True:
            while position < len(buffer) and buffer[position] in " \t\r\n,":
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                event, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                buffer = buffer[position:]
                position = 0
                chunk = source.read(1024 * 1024)
                if not chunk:
                    raise ValueError("incomplete traceEvents array") from None
                buffer += chunk
                continue
            position = end
            yield event
            if position > 1024 * 1024:
                buffer, position = buffer[position:], 0


def extract_trace(trace, destination):
    """Fixed selection: first 100 ms of the first captured host forward span."""
    kernels = defaultdict(lambda: {"calls": 0, "duration_us": 0.0})
    first_forward = None
    candidates = []
    for event in trace_events(trace):
        if event.get("ph") != "X":
            continue
        cat, name = event.get("cat"), event.get("name", "")
        ts, duration = event.get("ts", 0), event.get("dur", 0)
        if cat == "kernel":
            kernels[name]["calls"] += 1
            kernels[name]["duration_us"] += duration
        if cat == "user_annotation" and name == "snnlab::forward":
            first_forward = ts if first_forward is None else min(first_forward, ts)
    if first_forward is None:
        raise ValueError("trace has no host forward annotation")
    # A second streaming pass avoids depending on event ordering in the export.
    for event in trace_events(trace):
        if event.get("ph") != "X":
            continue
        cat, name = event.get("cat"), event.get("name", "")
        ts, duration = event.get("ts", 0), event.get("dur", 0)
        if ts >= first_forward + 100_000 or ts + duration <= first_forward:
            continue
        lane = None
        if cat == "user_annotation" and name.startswith("snnlab::projection/"):
            lane = "Host: projection calls"
        elif cat == "user_annotation" and name.startswith("snnlab::neuron/"):
            lane = "Host: neuron calls"
        elif cat == "kernel":
            lane = "CUDA kernels"
        elif cat == "gpu_memcpy":
            lane = "CUDA copies"
        if lane:
            candidates.append(
                {"lane": lane, "name": name, "ts_us": ts, "duration_us": duration}
            )
    events = [
        {
            "lane": row["lane"],
            "name": row["name"],
            "start_ms": max(0, (row["ts_us"] - first_forward) / 1000),
            "duration_ms": (
                min(row["ts_us"] + row["duration_us"], first_forward + 100_000)
                - max(row["ts_us"], first_forward)
            )
            / 1000,
        }
        for row in candidates
    ]
    if not any(row["lane"] == "CUDA kernels" for row in events):
        raise ValueError("selected excerpt has no CUDA kernel events")
    result = {
        "run_id": RUN_ID,
        "raw_sha256": digest(trace),
        "raw_bytes": Path(trace).stat().st_size,
        "selection": "first 100 ms after the first captured CPU snnlab::forward start; all kernels aggregated over complete capture",
        "duration_ms": 100,
        "events": events,
        "kernels": [
            {"name": name, **row}
            for name, row in sorted(
                kernels.items(), key=lambda item: item[1]["duration_us"], reverse=True
            )
        ],
    }
    write_json(destination, result)
    return result


def analyze(root):
    root = Path(root)

    def load(name):
        return json.loads((root / name).read_text())

    cloud, cfg, metrics = (
        load("cloud.json"),
        load("configuration.json"),
        load("metrics.json"),
    )
    summary, timing, tracing = (
        load("summary.json"),
        load("timing.json"),
        load("tracing/summary.json"),
    )
    excerpt = load("timeline-excerpt.json")
    if cloud["run_id"] != RUN_ID or excerpt["run_id"] != RUN_ID:
        raise ValueError("report must use the explicitly pinned SHD run")
    if (
        cloud["status"] != "complete"
        or summary["completed_updates"] != 290
        or cfg["smoke"]
    ):
        raise ValueError("expected completed, full-data, 290-update run")
    expected = {
        "epochs": 10,
        "batch_size": 256,
        "seed": 42,
        "steps": 1000,
        "dt_ms": 1.0,
        "channels": 700,
        "hidden": 128,
        "classes": 20,
    }
    if any(cfg["configuration"][name] != value for name, value in expected.items()):
        raise ValueError("workload configuration differs from the report protocol")
    if (cfg["dataset"]["train_count"], cfg["dataset"]["validation_count"]) != (
        7340,
        816,
    ) or cfg["dataset"]["official_test_accessed"]:
        raise ValueError("dataset split differs from the report protocol")
    if tracing["status"] != "complete" or len(tracing["captured_updates"]) != 3:
        raise ValueError("expected three complete diagnostic batches")
    if [row["epoch"] for row in metrics["epochs"]] != list(range(11)):
        raise ValueError("missing initial validation or training epochs")
    if (
        sum(row["calls"] for row in excerpt["kernels"])
        != tracing["timeline"]["kernel_count"]
    ):
        raise ValueError("raw kernel events do not match retained tracing summary")
    cpu = defaultdict(lambda: {"calls": 0, "self_cpu_us": 0.0})
    with (root / "tracing/operators.csv").open() as handle:
        for row in csv.DictReader(handle):
            name = row["operator"]
            if (
                name.startswith(("snnlab::", "[memory]"))
                or float(row["self_cpu_us"]) <= 0
            ):
                continue
            if name.startswith("## Call CompiledFxGraph"):
                name = "Compiled graph calls"
            cpu[name]["calls"] += int(row["calls"])
            cpu[name]["self_cpu_us"] += float(row["self_cpu_us"])
    phase = defaultdict(list)
    for row in tracing["phases"]:
        phase[row["region"]].append(row["host_enqueue_s"])
    host_copies = [
        row
        for row in tracing["timeline"]["copies"]
        if "HtoD" in str(row["name"]) and row["bytes"] is not None
    ]
    examples = cfg["dataset"]["train_count"] * cfg["configuration"]["epochs"]
    warm = [row for row in timing["epochs"] if row["epoch"] >= 2]
    result = {
        "run_id": RUN_ID,
        "baseline": {
            "training_s": timing["training_s"],
            "validation_s": timing["validation_s"],
            "native_total_s": timing["total_s"],
            "workflow_wall_s": summary["wall_s"],
            "examples_per_s": examples / timing["training_s"],
            "update_mean_s": timing["training_s"] / 290,
            "warm_training_epoch_mean_s": float(
                np.mean([row["training_s"] for row in warm])
            ),
            "peak_allocated_gib": summary["peak_cuda_allocated_bytes"] / 2**30,
            "peak_reserved_gib": summary["peak_cuda_reserved_bytes"] / 2**30,
            "final_train_accuracy": metrics["epochs"][-1]["train_accuracy"],
            "final_validation_accuracy": metrics["epochs"][-1]["validation_accuracy"],
            "best_validation_accuracy": max(
                row["validation_accuracy"] for row in metrics["epochs"]
            ),
        },
        "diagnostic": {
            "phase_host_medians_s": {
                name: float(np.median(values)) for name, values in phase.items()
            },
            "window_wall_s": tracing["window_wall_s"],
            "cuda_peak_allocated_gib": max(
                row["cuda_peak_allocated_bytes"] for row in tracing["memory"]
            )
            / 2**30,
            "cuda_peak_reserved_gib": max(
                row["cuda_peak_reserved_bytes"] for row in tracing["memory"]
            )
            / 2**30,
            "host_to_device_bytes": sum(row["bytes"] for row in host_copies),
            "host_to_device_copy_s": sum(row["duration_us"] for row in host_copies)
            / 1e6,
            "host_to_device_aggregate_GB_per_s": sum(
                row["bytes"] for row in host_copies
            )
            / (sum(row["duration_us"] for row in host_copies) * 1000),
            "cpu_operators": [
                {"name": name, **row}
                for name, row in sorted(
                    cpu.items(), key=lambda item: item[1]["self_cpu_us"], reverse=True
                )[:5]
            ],
            "cuda_kernels": excerpt["kernels"][:5],
        },
        "source_sha256": {
            str(path.relative_to(root)): digest(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.suffix in {".json", ".csv"}
            and path.name not in {"report-data.json", "presentation.json"}
        },
        "plot_script_sha256": digest(__file__),
    }
    write_json(root / "report-data.json", result)
    return cfg, metrics, timing, tracing, excerpt, result


def render(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    from snnlab.viz import Theme

    root = Path(root)
    cfg, metrics, timing, tracing, excerpt, analysis = analyze(root)
    theme = Theme()
    theme.apply()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "savefig.dpi": 240,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": RUN_ID,
        }
    )

    def setup(fig, axes, title, footer):
        fig.suptitle(title, x=0.075, ha="left", fontsize=13, fontweight="bold")
        for index, ax in enumerate(np.asarray(axes).flat):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(False)
            ax.set_title(
                f"{'ABCD'[index]}  {ax.get_title(loc='left') or ax.get_title()}",
                loc="left",
            )
        fig.text(0.075, 0.015, footer, fontsize=8, color=theme.muted)

    def save(fig, name):
        for suffix in ("png", "svg", "pdf"):
            metadata = (
                {"Date": None}
                if suffix == "svg"
                else {"CreationDate": None, "ModDate": None}
                if suffix == "pdf"
                else None
            )
            fig.savefig(
                root / f"{name}.{suffix}", bbox_inches="tight", metadata=metadata
            )
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"wspace": 0.32})
    fig.subplots_adjust(top=0.76, bottom=0.20, left=0.075, right=0.97)
    for key, label, color, style, rows in (
        ("train", "Train: online", theme.ink, "-", metrics["epochs"][1:]),
        (
            "validation",
            "Validation: fixed weights",
            theme.accent,
            "--",
            metrics["epochs"],
        ),
    ):
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(
            epochs,
            [row[f"{key}_cross_entropies"]["objective[0]"] for row in rows],
            color=color,
            linestyle=style,
            marker="o",
            ms=3,
            label=label,
        )
        axes[1].plot(
            epochs,
            [row[f"{key}_accuracy"] for row in rows],
            color=color,
            linestyle=style,
            marker="o",
            ms=3,
        )
    axes[0].set(
        title="Cross-entropy",
        xlabel="Completed epoch",
        ylabel="Example-mean cross-entropy",
        ylim=(0, 3.3),
    )
    axes[1].set(
        title="Accuracy",
        xlabel="Completed epoch",
        ylabel="Classification accuracy",
        ylim=(0, 0.55),
    )
    for ax in axes:
        ax.set(xticks=range(0, 11, 2), xlim=(0, 10))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.07, 0.91),
        ncol=2,
        frameon=False,
    )
    setup(
        fig,
        axes,
        "SHD / learning over ten epochs",
        "7,340 train / 816 validation  |  seed 42  |  one run; unsmoothed measurements",
    )
    save(fig, "learning")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.09, right=0.97)
    rows = timing["epochs"]
    epochs = np.array([row["epoch"] for row in rows])
    training = np.array([row["training_s"] for row in rows])
    validation = np.array([row["validation_s"] for row in rows])
    ax.bar(epochs, training, color=theme.ink, label="Training updates")
    ax.bar(epochs, validation, bottom=training, color=theme.accent, label="Validation")
    ax.set(
        title="Training and validation phases",
        xlabel="Epoch (0 = initial validation)",
        ylabel="Time (s)",
        xticks=epochs,
        ylim=(0, 90),
    )
    ax.legend(loc="upper right", frameon=False)
    setup(
        fig,
        [ax],
        "SHD / baseline epoch costs",
        "Tracing disabled  |  setup, checkpoint and reporting time excluded from bars",
    )
    save(fig, "epoch-timing")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.7))
    fig.subplots_adjust(
        top=0.84, bottom=0.10, left=0.09, right=0.97, hspace=0.60, wspace=0.34
    )
    activity = {row["signal"]: row for row in tracing["activity"]}
    labels = ["Input", "E1", "E2"]
    signal_ids = ["input/drive", "population/E1", "population/E2"]
    density = [activity[name]["density"] * 100 for name in signal_ids]
    axes[0, 0].bar(labels, density, color=[theme.mid_grey, theme.ink, theme.accent])
    for index, value in enumerate(density):
        axes[0, 0].text(index, value + 0.25, f"{value:.2f}%", ha="center", fontsize=9)
    axes[0, 0].set(
        title="Spike density",
        ylabel="Nonzero spike bins (%)",
        ylim=(0, 11),
    )
    rates = [activity[name]["per_neuron_rate_hz"] for name in signal_ids[1:]]
    bins = np.linspace(0, np.ceil(max(map(max, rates)) / 20) * 20, 21)
    for values, label, color in zip(
        rates, labels[1:], [theme.ink, theme.accent], strict=True
    ):
        axes[0, 1].hist(
            values, bins=bins, histtype="step", linewidth=1.8, label=label, color=color
        )
    axes[0, 1].set(
        title="Per-neuron rates",
        xlabel="Rate over 768 utterances (Hz)",
        ylabel="Neurons per bin",
    )
    axes[0, 1].legend(frameon=False)
    names = ["input_to_E1.weight", "E1_to_E2.weight", "readout_projection.weight"]
    for phase, offset, color, label in (
        ("before_window", -0.19, theme.ink, "After 5 updates"),
        ("after_window", 0.19, theme.accent, "After 8 updates"),
    ):
        rows = {
            row["parameter"]: row for row in tracing["weights"] if row["phase"] == phase
        }
        axes[1, 0].bar(
            np.arange(3) + offset,
            [rows[name]["zero_fraction"] * 100 for name in names],
            width=0.36,
            color=color,
            label=label,
        )
    axes[1, 0].set(
        title="Numerical weight zeros",
        ylabel="Exactly zero weights (%)",
        xticks=range(3),
        xticklabels=["Input→E1", "E1→E2", "E2→out"],
        ylim=(0, 55),
    )
    axes[1, 0].legend(frameon=False)
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Block and gradient structure", loc="left")
    table_rows = []
    rows = {
        row["parameter"]: row
        for row in tracing["weights"]
        if row["phase"] == "after_window"
    }
    for label, name in zip(["Input→E1", "E1→E2", "E2→out"], names, strict=True):
        row = rows[name]
        table_rows.append(
            [label, f"{row['blocks'] - row['nonzero_blocks']}/{row['blocks']}"]
        )
    table = axes[1, 1].table(
        cellText=table_rows,
        colLabels=["Projection", "Zero 16×16 tiles"],
        loc="upper center",
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0.37, 1, 0.48],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for cell in table.get_celld().values():
        cell.set_edgecolor(theme.rule)
    axes[1, 1].text(
        0,
        0.10,
        "100% of sampled gradient entries\nnonzero in all three learned matrices\n(three batches; before clipping).",
        fontsize=9,
        va="bottom",
    )
    setup(
        fig,
        axes,
        "SHD / sparse spikes, dense sampled gradients",
        "Diagnostic batches 6–8 of a fresh run  |  early training, not the final checkpoint",
    )
    save(fig, "sparsity")

    fig, axes = plt.subplots(
        2, 1, figsize=(9, 6.8), gridspec_kw={"height_ratios": [1.1, 1]}
    )
    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.25, right=0.97, hspace=0.62)
    phase_names = [
        "batch_prepare_wait",
        "forward",
        "backward",
        "loss",
        "optimizer_constraints",
    ]
    phase_labels = [
        "Batch preparation",
        "Forward",
        "Backward",
        "Loss",
        "Adam + constraints",
    ]
    for index, name in enumerate(phase_names):
        values = [
            row["host_enqueue_s"] * 1000
            for row in tracing["phases"]
            if row["region"] == name
        ]
        axes[0].scatter(
            values, np.array([-0.13, 0, 0.13]) + index, s=24, color=theme.ink
        )
        axes[0].scatter(
            [np.median(values)],
            [index],
            s=180,
            color=theme.accent,
            marker="|",
            zorder=3,
        )
    axes[0].set(
        title="Instrumented phase times",
        yticks=range(5),
        yticklabels=phase_labels,
        xlabel="Host enqueue duration (ms; log scale)",
        xscale="log",
        xlim=(0.3, 3000),
    )
    axes[0].invert_yaxis()
    lanes = [
        "Host: projection calls",
        "Host: neuron calls",
        "CUDA kernels",
        "CUDA copies",
    ]
    colors = [theme.accent, theme.mid_grey, theme.ink, theme.amber]
    for index, (lane, color) in enumerate(zip(lanes, colors, strict=True)):
        spans = [
            (event["start_ms"], event["duration_ms"])
            for event in excerpt["events"]
            if event["lane"] == lane
        ]
        axes[1].broken_barh(
            spans, (index - 0.25, 0.5), facecolors=color, edgecolors="none"
        )
    if not any(event["lane"] == "CUDA copies" for event in excerpt["events"]):
        axes[1].text(
            50,
            3,
            "No copy events in this interval",
            color=theme.muted,
            fontsize=8,
            ha="center",
            va="center",
        )
    axes[1].set(
        title="First 100 ms of the first captured forward",
        yticks=range(4),
        yticklabels=lanes,
        xlabel="Time from host forward start (ms)",
        xlim=(0, 100),
        ylim=(-0.7, 3.7),
    )
    axes[1].invert_yaxis()
    setup(
        fig,
        axes,
        "SHD / diagnostic execution, not baseline timing",
        "Dots: three batches; red ticks: medians  |  activity monitoring and detailed profiler labels enabled",
    )
    save(fig, "diagnostics")
    outputs = {
        f"{name}.{suffix}": digest(root / f"{name}.{suffix}")
        for name in FIGURES
        for suffix in ("png", "pdf", "svg")
    }
    write_json(
        root / "presentation.json",
        {
            "run_id": RUN_ID,
            "source_sha256": analysis["source_sha256"],
            "renderer_sha256": digest(__file__),
            "figures": outputs,
        },
    )
    return analysis


def _validate_modal_options(*, epochs, trace_mode, trace_skip_updates, trace_updates):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if trace_mode not in {"off", "monitor", "profile", "both"}:
        raise ValueError("trace mode must be off, monitor, profile or both")
    if trace_skip_updates < 0 or trace_updates < 1:
        raise ValueError("invalid trace update window")


def _collect_modal_reports(*, run_id, cloud_paths, volume, LOCAL_ROOT):
    """Read persisted reports directly from the Volume; no worker invocation."""
    cloud = cloud_paths(run_id)
    remote = f"/{cloud['directory']}"
    # A completion manifest is written only after all requested work succeeds.
    # Missing manifests propagate rather than collecting an unfinished run.
    try:
        completed = json.loads(b"".join(volume.read_file(f"{remote}/cloud.json")))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"No completion manifest for {run_id}; check the Modal dashboard and collect after the run succeeds"
        ) from exc
    if completed.get("status") != "complete" or completed.get("run_id") != run_id:
        raise ValueError("remote manifest does not describe this completed run")
    names = completed["report_files"]
    for name in names:
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.suffix not in {".json", ".csv"}
        ):
            raise ValueError("invalid report path in remote manifest")
    destination = LOCAL_ROOT / "benchmarks/shd/results"
    ticket_path = destination / "cloud.json"
    ticket = json.loads(ticket_path.read_text()) if ticket_path.exists() else {}
    if ticket.get("run_id") != run_id:
        ticket = {}
    reset_results(destination)
    for name in names:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(volume.read_file(f"{remote}/{name}")))
    ticket_path.write_text(json.dumps({**ticket, **completed}, indent=2) + "\n")
    print(f"Collected reports: {destination}")
    return destination


def modal_launcher(local_root=None, data_root=None):
    import argparse
    import json
    import re
    import uuid
    from pathlib import Path

    import modal

    LOCAL_ROOT = local_root or (
        Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
    )
    app = modal.App("snnlab-shd")
    VOLUME = "snnlab-shd"
    DATA_ROOT = data_root or Path("/data")
    volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
    image = None
    if modal.is_local():
        image = (
            modal.Image.debian_slim(python_version="3.12")
            .pip_install(
                "torch==2.10.0",
                "torchvision==0.25.0",
                "numpy>=2.2.6",
                "scipy>=1.15.3",
                "matplotlib>=3.10.8",
                "scikit-learn>=1.7.2",
                "h5py>=3.16.0",
            )
            .env(
                {
                    "PYTHONPATH": "/workspace/src:/workspace/benchmarks/shd",
                    "OMP_NUM_THREADS": "4",
                }
            )
            .add_local_dir(
                LOCAL_ROOT / "src", "/workspace/src", ignore=["**/__pycache__/**"]
            )
            .add_local_dir(
                LOCAL_ROOT / "benchmarks/shd",
                "/workspace/benchmarks/shd",
                ignore=["results/**", "**/__pycache__/**"],
            )
        )

    @app.function(
        serialized=True,
        image=image,
        gpu="H100",
        cpu=4,
        memory=16384,
        timeout=7200,
        volumes={"/data": volume},
        max_containers=1,
    )
    def worker(
        run_id: str,
        epochs: int,
        smoke: bool,
        trace_mode: str = "off",
        trace_skip_updates: int = 1,
        trace_updates: int = 3,
        trace_detailed_labels: bool = True,
    ):
        from dataclasses import replace

        from benchmark import Config, run

        validate_run_id(run_id)
        validate_options(epochs, trace_mode, trace_skip_updates, trace_updates)
        cfg = Config(epochs=epochs)
        if smoke:
            cfg = replace(cfg, epochs=2, steps=64)
        volume.reload()
        output = DATA_ROOT / "runs" / run_id
        try:
            run(
                cfg,
                output,
                DATA_ROOT / "cache",
                device="cuda:0",
                smoke=smoke,
                trace_mode=trace_mode,
                trace_skip_updates=trace_skip_updates,
                trace_updates=trace_updates,
                trace_detailed_labels=trace_detailed_labels,
            )
            names = [
                "configuration.json",
                "metrics.json",
                "timing.json",
                "summary.json",
            ]
            if trace_mode != "off":
                names.append("diagnostic_metrics.json")
                names.extend(
                    str(path.relative_to(output))
                    for path in (output / "tracing").iterdir()
                    if path.name != "trace.json"
                )
            cloud = cloud_paths(run_id)
            cloud.update(
                status="complete",
                report_files=names,
                trace=f"runs/{run_id}/tracing/trace.json"
                if (output / "tracing/trace.json").exists()
                else None,
            )
            # This manifest and all results persist without any local result handler.
            (output / "cloud.json").write_text(json.dumps(cloud, indent=2) + "\n")
            return {"run_id": run_id, "status": "complete"}
        finally:
            volume.commit()

    def validate_run_id(run_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
            raise ValueError(
                "run ID must contain 1–80 letters, digits, underscores or hyphens"
            )
        return run_id

    def cloud_paths(run_id):
        validate_run_id(run_id)
        return {
            "volume": VOLUME,
            "run_id": run_id,
            "directory": f"runs/{run_id}",
            "checkpoint": f"runs/{run_id}/final.checkpoint",
            "snapshot": f"runs/{run_id}/spikes.npz",
        }

    def validate_options(epochs, trace_mode, trace_skip_updates, trace_updates):
        _validate_modal_options(
            epochs=epochs,
            trace_mode=trace_mode,
            trace_skip_updates=trace_skip_updates,
            trace_updates=trace_updates,
        )

    def submit(
        epochs=10,
        smoke=False,
        trace_mode="off",
        trace_skip_updates=1,
        trace_updates=3,
        trace_detailed_labels=True,
    ):
        """Submit inside a detached App context, then return without waiting."""
        validate_options(epochs, trace_mode, trace_skip_updates, trace_updates)
        run_id = ("smoke-" if smoke else "shd-") + uuid.uuid4().hex[:12]
        print(f"Submitting {run_id}", flush=True)
        call = worker.spawn(
            run_id,
            epochs,
            smoke,
            trace_mode,
            trace_skip_updates,
            trace_updates,
            trace_detailed_labels,
        )
        ticket = {
            **cloud_paths(run_id),
            "call_id": call.object_id,
            "status": "submitted",
        }
        destination = LOCAL_ROOT / "benchmarks/shd/results"
        reset_results(destination)
        (destination / "cloud.json").write_text(json.dumps(ticket, indent=2) + "\n")
        print(
            f"Submitted {run_id}; call={call.object_id}; ticket={destination / 'cloud.json'}",
            flush=True,
        )
        print(
            f"Collect later: python benchmarks/shd/benchmark.py --modal --mode collect --run-id {run_id}",
            flush=True,
        )
        return ticket

    def collect(run_id):
        return _collect_modal_reports(
            run_id=run_id, cloud_paths=cloud_paths, volume=volume, LOCAL_ROOT=LOCAL_ROOT
        )

    def cli():
        """Direct Python CLI always launches with detach=True; collection is read-only."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--mode", choices=("launch", "collect"), default="launch")
        parser.add_argument("--run-id", default="")
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--smoke", action="store_true")
        parser.add_argument(
            "--trace-mode", choices=("off", "monitor", "profile", "both"), default="off"
        )
        parser.add_argument("--trace-skip-updates", type=int, default=1)
        parser.add_argument("--trace-updates", type=int, default=3)
        parser.add_argument(
            "--trace-detailed-labels",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        args = parser.parse_args()
        if args.mode == "collect":
            return collect(args.run_id)
        validate_options(
            args.epochs, args.trace_mode, args.trace_skip_updates, args.trace_updates
        )
        with modal.enable_output(), app.run(detach=True):
            return submit(
                args.epochs,
                args.smoke,
                args.trace_mode,
                args.trace_skip_updates,
                args.trace_updates,
                args.trace_detailed_labels,
            )

    return SimpleNamespace(**locals())


def cli():
    if "--modal" in sys.argv:
        sys.argv.remove("--modal")
        modal_launcher().cli()
    elif "--plot" in sys.argv:
        parser = argparse.ArgumentParser(
            description="Render existing benchmark results"
        )
        parser.add_argument("--plot", action="store_true")
        parser.add_argument(
            "--output", type=Path, default=Path(__file__).parent / "results"
        )
        parser.add_argument("--trace", type=Path)
        args = parser.parse_args()
        if args.trace:
            extract_trace(args.trace, args.output / "timeline-excerpt.json")
        render(args.output)
    else:
        main()
