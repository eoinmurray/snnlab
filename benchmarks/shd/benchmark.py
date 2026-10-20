"""Train the calibrated SHD COBA workload through SNNLab's native trainer."""

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from shd_data import acquire, digest, prepare, synthetic_snapshot
from shd_recipe import TRAINABLE, Config, author_network

import snnlab
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
    # Never mix or overwrite completed runs, checkpoints or dataset snapshots.
    output.mkdir(parents=True, exist_ok=False)
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
        "--output", type=Path, required=True, help="new run directory (must not exist)"
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


if __name__ == "__main__":
    main()
