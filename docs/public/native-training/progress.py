"""Native online metrics, unequal-class batching and resumable epoch progress.

Synthetic SHD-shaped inputs exercise the API; this is not an SHD accuracy study.
Run: python examples/native-training/progress.py --output /tmp/native-training
Full shape: add --samples 8156 --batch-size 256 --steps 1000 (substantial runtime).
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from progress_objectives import equal_class_loss

from snnlab import extensions, lang, viz
from snnlab.sim import (
    Batching,
    CheckpointSelection,
    DatasetEncoder,
    DatasetSnapshotBinding,
    EpochReport,
    ExecutionSpec,
    TrainingMetrics,
    ValidationSpec,
    save_training_checkpoint,
    train,
)


def report_epoch(report: EpochReport):
    seconds = f"; {report.timing.total_s:.2f}s" if report.timing is not None else ""
    print(
        f"epoch {report.epoch}/{report.total_epochs}: "
        f"online train={report.train.accuracy:.1%}; "
        f"validation={report.validation.accuracy:.1%}; "
        f"updates={report.updates}; examples={report.examples}{seconds}",
        flush=True,
    )


def snapshot(path, count, seed):
    rng = np.random.default_rng(seed)
    probabilities = np.linspace(1, 3, 20)
    probabilities /= probabilities.sum()
    labels = rng.choice(20, size=count, p=probabilities)
    # 700 channels with class-dependent rates, generated independently by split.
    features = np.full((count, 700), 0.02, dtype=np.float32)
    for label in range(20):
        features[labels == label, label * 35 : (label + 1) * 35] = 0.6
    np.savez_compressed(path, features=features, labels=labels)
    return np.bincount(labels, minlength=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "artifacts"
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--equal-class-loss", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.batch_size, args.steps, args.epochs) <= 0:
        parser.error("counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    counts = snapshot(args.output / "train.npz", args.samples, 43)
    snapshot(args.output / "validation.npz", 80, 44)
    print(f"training class counts: {counts.tolist()}", flush=True)

    net = lang.Network("native_progress", dt=1 * lang.ms)
    inputs = net.input(
        "events", shape=("time", "batch", 700), signal_type="spikes", unit="spike"
    )
    scores = lang.SpikeCount(source=inputs, classes=20, name="scores")
    net.output("class_scores", scores)
    if args.equal_class_loss:
        extensions.register_objective(
            "examples.equal-class-progress/v1", equal_class_loss, classification=True
        )
        weights = args.samples / (
            max(1, int((counts > 0).sum())) * np.maximum(counts, 1)
        )
        objective = lang.CustomObjective(
            "examples.equal-class-progress/v1",
            prediction=scores,
            target="class",
            weights=weights.tolist(),
        )
    else:
        objective = lang.CrossEntropy(prediction=scores, target="class")
    recipe = lang.TrainSpec(
        objectives=(objective,),
        parameter_groups=(
            lang.ParameterGroup(
                ("scores_projection.weight",), name="readout", lr=0.01
            ),
        ),
        optimizer=lang.AdamW(weight_decay=0),
        presentation_duration=args.steps * lang.ms,
    )
    bundle = lang.compile(net, training=recipe)
    bundle.write(args.output / "network.bundle")
    viz.render_diagram(lang.diagram(bundle, view="expanded"), args.output / "network.png", scale=2)
    encoder = DatasetEncoder(
        "rate_poisson", duration_ms=args.steps, max_rate_hz=500, seed=45
    )
    binding = DatasetSnapshotBinding(
        args.output / "train.npz",
        "events",
        "synthetic-shd-shape",
        "train",
        encoder,
        target_id="class",
    )
    validation = ValidationSpec(
        input_bindings=(
            replace(binding, path=args.output / "validation.npz", split="validation"),
        )
    )
    spec = ExecutionSpec(
        kind="train",
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(binding,),
        validation=validation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=42,
        device=args.device,
        batching=Batching(
            order="stratified",
            stratify_by="class",
            remainder="merge_small",
            min_batch_size=min(20, args.batch_size),
        ),
        metrics=TrainingMetrics(
            train="online",
            validation="epoch",
            initial=("validation",),
            update_history=False,
        ),
        checkpoint_selection=CheckpointSelection(),
        training_timing=True,
        on_epoch_end=report_epoch,
        diagnostics=False,
    )
    # Persist a partial epoch, then resume the same schedule without replaying reports.
    partial_path = args.output / "partial.checkpoint"
    partial = train(replace(spec, updates=1, save_final_checkpoint=partial_path))
    if partial.training_checkpoint.data_state["epoch"] == args.epochs:
        result = partial
        save_training_checkpoint(
            args.output / "final.checkpoint", result.training_checkpoint
        )
        save_training_checkpoint(
            args.output / "selected.checkpoint", result.selected_checkpoint
        )
    else:
        result = train(
            replace(
                spec,
                checkpoint=partial_path,
                save_final_checkpoint=args.output / "final.checkpoint",
                save_selected_checkpoint=args.output / "selected.checkpoint",
            )
        )
    payload = {
        "epochs": result.metrics["epochs"],
        "timing_this_call": result.metrics["training_timing"],
        "selection": result.selected_checkpoint.selection_record,
        "schedule": result.metrics["execution_protocol"]["training_iteration"],
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved final/selected checkpoints and results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
