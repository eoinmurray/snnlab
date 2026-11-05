"""Opt-in complete native updates, observations and validation on CPU or CUDA."""

import argparse
import importlib.util
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from snnlab import extensions, sim

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "native_finetuning_example", ROOT / "examples/native-finetuning/example.py"
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def run_case(path, *, diagonal, microbatch, device, compile_mode, epochs, cuda_graphs):
    bundle = example.circuit(diagonal=diagonal)
    reference = sim.build(
        bundle=bundle, device="cpu", performance=sim.Performance(compile_mode=None)
    )
    parameters = reference.parameters
    if not diagonal:
        parameters["private.weight"] = torch.eye(2) * 0.02
    signature = sim.runtime_state_signature(reference.model.plan)
    encoder = sim.DatasetEncoder(
        "custom",
        definition="examples.private_inhibition/v1",
        config={"signature": signature},
        seed=41,
        rng_policy="sample_v1",
    )
    binding = sim.DatasetSnapshotBinding(
        path,
        "drive",
        "synthetic",
        "train",
        encoder,
        target_id="label",
        sample_ids_key="ids",
        sample_indices=tuple(range(7)),
    )
    validation = replace(binding, split="validation", sample_indices=(7, 8, 9))
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    boundary = started
    intervals = []

    def observe_epoch(report):
        nonlocal boundary
        synchronize(device)
        now = time.perf_counter()
        intervals.append(
            {
                "epoch": report.epoch,
                "scope": "cold setup plus initial evaluation and first complete epoch"
                if report.epoch == 1
                else "warmed complete epoch including preparation, transfers and evaluation",
                "seconds": now - boundary,
                "training_examples_per_s": 7 / report.timing["training_s"],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                if device.startswith("cuda")
                else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                if device.startswith("cuda")
                else None,
            }
        )
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        boundary = time.perf_counter()

    result = sim.train(
        bundle=bundle,
        weights=sim.WeightInitialization(parameters),
        input_bindings=[binding],
        validation=sim.ValidationSpec(
            input_bindings=[validation], encoding_seeds=(101, 102, 103)
        ),
        epochs=epochs,
        batch_size=4,
        batching=sim.Batching(microbatch_size=microbatch),
        metrics=sim.TrainingMetrics(train="evaluate"),
        observations=sim.EpochObservations(population_rates=("E",)),
        seed=7,
        device=device,
        performance=sim.Performance(
            compile_mode=compile_mode,
            compile_block_steps=4,
            precision="highest",
            cuda_graphs=cuda_graphs,
        ),
        training_timing=True,
        timing_mode="synchronized",
        on_epoch_end=observe_epoch,
    )
    synchronize(device)
    seconds = time.perf_counter() - started
    return {
        "connectivity": "one_to_one" if diagonal else "dense_identity",
        "batch_counts": result.metrics["batch_counts"],
        "input_shape": [20, 7, 2],
        "effective_batch_sizes": [4, 3],
        "whole_train_call_s": seconds,
        "epoch_intervals": intervals,
        "cuda_graphs": result.metrics["cuda_graphs"],
        "timing": result.metrics["training_timing"],
        "performance": result.metrics["performance"],
        "epochs": result.metrics["epochs"],
        "parameter_bytes": sum(
            value.numel() * value.element_size() for value in result.parameters.values()
        ),
        "peak_allocated_bytes": max(
            [row["peak_allocated_bytes"] for row in intervals]
            + [torch.cuda.max_memory_allocated(device)]
        )
        if device.startswith("cuda")
        else None,
        "peak_reserved_bytes": max(
            [row["peak_reserved_bytes"] for row in intervals]
            + [torch.cuda.max_memory_reserved(device)]
        )
        if device.startswith("cuda")
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--cuda-graphs", choices=("auto", "off"), default="auto")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    args = parser.parse_args()
    extensions.register_encoder("examples.private_inhibition/v1", example.encode)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "synthetic.npz"
    np.savez(
        path,
        features=np.asarray([[0.8, 0.1], [0.1, 0.8]] * 5, dtype=np.float32),
        labels=np.asarray([0, 1] * 5),
        ids=np.arange(10),
    )
    report = {
        "schema": "snnlab.native-finetuning-benchmark/v2",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": args.device,
            "device_name": torch.cuda.get_device_name(args.device)
            if args.device.startswith("cuda")
            else platform.processor(),
            "precision": "float32/highest",
        },
        "scope": "Complete train call: preparation, transfers, forward/backward, clipping, optimizer/constraints, initial and epoch train/validation measurements. No persistence or downstream calibration. First epoch includes compiler/capture startup; later epochs are warmed. Process/device/library caches remain shared between cases.",
        "cases": [
            run_case(
                path,
                diagonal=diagonal,
                microbatch=microbatch,
                device=args.device,
                compile_mode="default" if args.compile else None,
                epochs=args.epochs,
                cuda_graphs=args.cuda_graphs,
            )
            for diagonal in (False, True)
            for microbatch in (None, 2)
        ],
    }
    output = args.output / "report.json"
    output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(output)


if __name__ == "__main__":
    main()
