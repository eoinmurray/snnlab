import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from snnlab import lang, viz
from snnlab.sim import (
    Batching,
    CheckpointSelection,
    DatasetEncoder,
    DatasetSnapshotBinding,
    TrainingMetrics,
    ValidationSpec,
    train,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def prepare_results():
    if OUTPUT_DIR.is_symlink():
        OUTPUT_DIR.unlink()
    elif OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)


def snapshot(path, count, seed):
    rng = np.random.default_rng(seed)
    probabilities = np.linspace(1, 3, 20)
    probabilities /= probabilities.sum()
    labels = rng.choice(20, size=count, p=probabilities)

    features = np.full((count, 700), 0.02, dtype=np.float32)
    for label in range(20):
        features[labels == label, label * 35 : (label + 1) * 35] = 0.6
    np.savez_compressed(path, features=features, labels=labels)


def main():
    prepare_results()
    snapshot(OUTPUT_DIR / "train.npz", 200, 43)
    snapshot(OUTPUT_DIR / "validation.npz", 80, 44)

    net = lang.Network("native_progress", dt=1 * lang.ms)
    inputs = net.input(
        "events", shape=("time", "batch", 700), signal_type="spikes", unit="spike"
    )
    scores = lang.SpikeCount(source=inputs, classes=20, name="scores")
    net.output("class_scores", scores)
    objective = lang.CrossEntropy(prediction=scores, target="class")
    recipe = lang.TrainSpec(
        objectives=(objective,),
        parameter_groups=(
            lang.ParameterGroup(("scores_projection.weight",), name="readout", lr=0.01),
        ),
        optimizer=lang.AdamW(weight_decay=0),
        presentation_duration=16 * lang.ms,
    )
    bundle = lang.compile(net, training=recipe)
    bundle.write(OUTPUT_DIR / "network.bundle")
    viz.render_diagram(
        lang.diagram(bundle, view="expanded"), OUTPUT_DIR / "network.png", scale=2
    )
    encoder = DatasetEncoder("rate_poisson", duration_ms=16, max_rate_hz=500, seed=45)
    binding = DatasetSnapshotBinding(
        OUTPUT_DIR / "train.npz",
        "events",
        "synthetic-shd-shape",
        "train",
        encoder,
        target_id="class",
    )
    validation = ValidationSpec(
        input_bindings=(
            replace(binding, path=OUTPUT_DIR / "validation.npz", split="validation"),
        )
    )
    training_kwargs = dict(
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(binding,),
        validation=validation,
        epochs=2,
        batch_size=64,
        seed=42,
        device="cpu",
        batching=Batching(
            order="stratified",
            stratify_by="class",
            remainder="merge_small",
            min_batch_size=20,
        ),
        metrics=TrainingMetrics(
            train="online",
            validation="epoch",
            initial=("validation",),
            update_history=False,
        ),
        checkpoint_selection=CheckpointSelection(),
        training_timing=True,
        diagnostics=False,
    )

    partial_path = OUTPUT_DIR / "partial.checkpoint"
    train(**training_kwargs, updates=1, save_final_checkpoint=partial_path)
    result = train(
        **training_kwargs,
        checkpoint=partial_path,
        save_final_checkpoint=OUTPUT_DIR / "final.checkpoint",
        save_selected_checkpoint=OUTPUT_DIR / "selected.checkpoint",
    )
    payload = {
        "epochs": result.metrics["epochs"],
        "timing_this_call": result.metrics["training_timing"],
        "selection": result.selected_checkpoint.selection_record,
        "schedule": result.metrics["execution_protocol"]["training_iteration"],
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
