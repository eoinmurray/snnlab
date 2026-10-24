"""Quick, offline scikit-learn digits training with built-in conductance LIF cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import snnlab
from snnlab import lang
from snnlab.sim import (
    Batching,
    CheckpointSelection,
    DenseArrayBinding,
    Performance,
    TrainingMetrics,
    ValidationSpec,
    encode_images_poisson,
    train,
)


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


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    seed: int = 17
    epochs: int = 20
    batch_size: int = 128
    hidden: int = 64
    steps: int = 16
    input_lr: float = 0.0003
    readout_lr: float = 0.01

    def __post_init__(self):
        for name in ("epochs", "batch_size", "hidden", "steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("input_lr", "readout_lr"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32 - 2:
            raise ValueError("seed must be an integer in [0, 2**32 - 2)")


def prepare_data(cfg, *, smoke=False):
    """One fixed stratified split and fixed binary spike replay for every epoch."""
    digits = load_digits()
    train_indices, test_indices = train_test_split(
        np.arange(len(digits.target)),
        train_size=100 if smoke else None,
        test_size=40 if smoke else 0.2,
        random_state=cfg.seed,
        stratify=digits.target,
    )
    pixels = torch.tensor(digits.data / 16.0, dtype=torch.float32)
    labels = torch.tensor(digits.target, dtype=torch.long)

    def encode(indices, seed):
        return encode_images_poisson(
            pixels[indices],
            cfg.steps,
            dt=1.0,
            max_rate_hz=1000.0,
            generator=torch.Generator().manual_seed(seed),
        ), labels[indices]

    metadata = {
        "name": "sklearn.datasets.load_digits",
        "smoke": smoke,
        "samples": len(labels),
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "sha256": hashlib.sha256(
            digits.data.astype("<f8").tobytes() + digits.target.astype("<i8").tobytes()
        ).hexdigest(),
        "encoding": {
            "kind": "fixed Bernoulli rate spikes",
            "pixel_divisor": 16.0,
            "max_rate_hz": 1000.0,
            "dt_ms": 1.0,
            "train_seed": cfg.seed + 1,
            "test_seed": cfg.seed + 2,
        },
    }
    return (
        encode(train_indices, cfg.seed + 1),
        encode(test_indices, cfg.seed + 2),
        metadata,
    )


def author_network(cfg):
    net = lang.Network("scikit_digits_coba", dt=1 * lang.ms)
    pixels = net.input(
        "pixels", shape=("time", "batch", 64), signal_type="spikes", unit="spike"
    )
    hidden = net.population(
        "hidden",
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
        pixels,
        hidden.excitatory,
        name="pixels_to_hidden",
        synapse=lang.AMPA(tau=2 * lang.ms),
        weight=lang.Uniform(0.0, 0.02),
        constraint=lang.NonNegative(),
        initialization_scaling="direct",
    )
    counts = lang.reduce(hidden.spikes, operation="sum", over="time", name="counts")
    scores = lang.ops.linear(counts, size=10, name="classifier")
    net.output("class_scores", scores)
    recipe = lang.TrainSpec(
        objectives=(lang.CrossEntropy(prediction=scores, target="class"),),
        parameter_groups=(
            lang.ParameterGroup(
                (projection.weight,), name="conductance", lr=cfg.input_lr
            ),
            lang.ParameterGroup(
                ("classifier.weight",), name="readout", lr=cfg.readout_lr
            ),
        ),
        optimizer=lang.AdamW(weight_decay=0.0),
        surrogate=lang.FastSigmoid(slope=1.0),
        gradient_clip=1.0,
        presentation_duration=cfg.steps * lang.ms,
    )
    return lang.compile(net, training=recipe)


def progress(report):
    print(
        f"epoch {report.epoch:02d}/{report.total_epochs}: "
        f"online train={report.train.accuracy:.1%}; "
        f"test={report.validation.accuracy:.1%}; "
        f"{report.timing.total_s:.2f}s",
        flush=True,
    )


def run(cfg, output, *, smoke=False, cpu_threads=1, device="cpu"):
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    output = Path(output)
    output = reset_results(output)
    torch.set_num_threads(cpu_threads)
    device = torch.device(device)
    compile_mode = "default" if device.type == "cuda" else None
    started = time.perf_counter()
    (train_spikes, train_labels), (test_spikes, test_labels), dataset = prepare_data(
        cfg, smoke=smoke
    )
    bundle = author_network(cfg)
    bundle.write(output / "network.bundle")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    result = train(
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(DenseArrayBinding(input_id="pixels", value=train_spikes),),
        targets={"class": train_labels},
        validation=ValidationSpec(
            input_bindings=(DenseArrayBinding(input_id="pixels", value=test_spikes),),
            targets={"class": test_labels},
        ),
        device=str(device),
        seed=cfg.seed,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        batching=Batching(order="shuffle", remainder="keep"),
        metrics=TrainingMetrics(
            train="online", validation="epoch", initial=("validation",)
        ),
        checkpoint_selection=CheckpointSelection.disabled(),
        save_final_checkpoint=output / "final.checkpoint",
        diagnostics=False,
        performance=Performance(compile_mode=compile_mode),
        training_timing=True,
        on_epoch_end=progress,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_s = time.perf_counter() - training_started
    if any(
        not torch.isfinite(p).all()
        for p in result.training_checkpoint.parameters.values()
    ):
        raise ValueError("nonfinite final parameters")
    final = result.metrics["epochs"][-1]
    report = {
        "configuration": asdict(cfg),
        "dataset": dataset,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "snnlab": snnlab.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "cpu_threads": torch.get_num_threads(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "compile_mode": compile_mode,
        },
        "summary": {
            "test_accuracy": final["validation_accuracy"],
            "completed_updates": result.training_checkpoint.completed_updates,
            "native_train_s": training_s,
            "workflow_s": time.perf_counter() - started,
            "timing_scope": "workflow includes data preparation, encoding, authoring, bundle/checkpoint writes and training; excludes imports and results.json write",
        },
        "metrics": result.metrics,
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"Final test accuracy: {report['summary']['test_accuracy']:.1%}; "
        f"native train: {training_s:.2f}s; saved {output / 'results.json'}",
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results",
    )
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--hidden", type=int, default=Config.hidden)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--device", default="cpu", help="Torch device, e.g. cpu or cuda:0"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="100 train / 40 test digits, 2 epochs, 8 steps",
    )
    args = parser.parse_args()
    cfg = Config(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        steps=args.steps,
    )
    if args.smoke:
        cfg = replace(cfg, epochs=2, steps=8)
    run(
        cfg,
        args.output,
        smoke=args.smoke,
        cpu_threads=args.cpu_threads,
        device=args.device,
    )


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
