"""Offline digits experiment with explicit execution and native evidence utilities.

Run: uv run python examples/native-training/digits.py --smoke --output /tmp/digits-evidence
Use a fresh output directory for each run. Dataset splitting is deliberately
declared here; this example does not reproduce another experiment's split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits

from snnlab import lang, sim, viz


def author_network():
    net = lang.Network("digits_evidence", dt=1 * lang.ms)
    pixels = net.input(
        "pixels", shape=("time", "batch", 64), signal_type="spikes", unit="spike"
    )
    hidden = net.population(
        "hidden",
        size=32,
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
    connection = net.connect(
        pixels,
        hidden.excitatory,
        name="pixels_to_hidden",
        synapse=lang.AMPA(tau=2 * lang.ms),
        weight=lang.Uniform(0.0, 0.02),
        initialization_scaling="direct",
        constraint=lang.NonNegative(),
    )
    counts = lang.reduce(hidden.spikes, operation="sum", over="time", name="counts")
    scores = lang.ops.linear(counts, size=10, name="classifier")
    net.output("class_scores", scores)
    recipe = lang.TrainSpec(
        objectives=(lang.CrossEntropy(prediction=scores, target="class"),),
        parameter_groups=(
            lang.ParameterGroup((connection.weight,), name="conductance", lr=0.0003),
            lang.ParameterGroup(("classifier.weight",), name="classifier", lr=0.01),
        ),
        optimizer=lang.AdamW(weight_decay=0.0),
        surrogate=lang.FastSigmoid(slope=1.0),
        gradient_clip=1.0,
        presentation_duration=12 * lang.ms,
    )
    return lang.compile(net, training=recipe)


def run(output, *, epochs=3, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    seed = 17
    digits = load_digits()
    ids = np.arange(200 if smoke else len(digits.target))
    # Scaling, subset choice and partition policy are scientific choices here.
    features = (digits.data[ids] / 16.0).astype(np.float32)
    labels = digits.target[ids].astype(np.int64)
    partitions = sim.stratified_split(
        features=features,
        labels=labels,
        sample_ids=ids,
        fractions={"train": 0.6, "validation": 0.2, "test": 0.2},
        seed=seed,
        policy="stratified-v1",
    )
    dataset = sim.DatasetSnapshot.from_partitions(
        dataset_id="sklearn-digits-evidence",
        partitions=partitions,
        disjoint=("train", "validation", "test"),
        preprocessing={"operation": "divide", "divisor": 16.0, "dtype": "float32"},
        provenance={
            "loader": "sklearn.datasets.load_digits",
            "source_rows": ids.tolist(),
        },
    )
    dataset.write(output / "dataset.npz")
    dataset = sim.DatasetSnapshot.read(output / "dataset.npz")
    encoder = sim.DatasetEncoder(
        "rate_poisson", duration_ms=12.0, max_rate_hz=1000.0, seed=seed
    )
    train_binding = dataset.binding(
        "train", input_id="pixels", target_id="class", encoder=encoder
    )
    validation_binding = dataset.binding(
        "validation", input_id="pixels", target_id="class", encoder=encoder
    )
    test_binding = dataset.binding(
        "test", input_id="pixels", target_id="class", encoder=encoder
    )
    bundle = author_network()
    bundle.write(output / "network.bundle", visualise=False)
    performance = sim.Performance(
        compile_mode=None, compile_block_steps=1, cuda_graphs="off", precision="highest"
    )
    result = sim.train(
        bundle=bundle,
        input_bindings=(train_binding,),
        validation=sim.ValidationSpec(input_bindings=(validation_binding,)),
        test=sim.TestSpec(input_bindings=(test_binding,)),
        epochs=epochs,
        batch_size=32,
        batching=sim.Batching(order="shuffle", remainder="keep"),
        seed=seed,
        device="cpu",
        performance=performance,
        checkpoint_selection=sim.CheckpointSelection(
            metric=sim.SelectionMetric("loss"), include_initial=True
        ),
        save_selected_checkpoint=output / "selected.checkpoint",
        on_epoch_end=sim.EpochLogger(),
        diagnostics=False,
        snapshot=sim.SnapshotSpec(
            path=output / "training", stage="both", parameters=True
        ),
    )
    result.write_metrics(output / "metrics.json")
    history = sim.TrainingHistory.read(output / "metrics.json")
    history.validate(
        expected_epochs=epochs,
        require_complete=True,
        expected_examples={"test": dataset.counts["test"]},
        expected_encoding_draws={"test": (seed,)},
    )
    snapshot = sim.NetworkSnapshot.read(output / "training" / "end")
    print(f"snapshot: {snapshot.stage}, dt={snapshot.execution.timestep_ms} ms")
    # Diagnostic selection refers to an existing test member, not a new split.
    diagnostic_id = int(dataset.partitions["test"].sample_ids[0])
    diagnostic = dataset.binding(
        "test", sample_ids=(diagnostic_id,), input_id="pixels", encoder=encoder
    )
    recorded = sim.simulate(
        bundle=bundle,
        checkpoint=output / "selected.checkpoint",
        input_bindings=(diagnostic,),
        batch_size=1,
        seed=seed,
        device="cpu",
        performance=performance,
        diagnostics=False,
        recording=sim.RecordingSpec((sim.SignalRecording("hidden.spikes"),)),
        snapshot=sim.SnapshotSpec(
            path=output / "diagnostic", stage="start", parameters=False
        ),
    )
    recorded.write_recording(output / "recording", signals=("hidden.spikes",), batch=0)
    recording = sim.RecordingData.read(output / "recording")
    figure, axes = plt.subplots(1, 3, figsize=(12, 3))
    viz.plot_learning_curves(
        {"loss": axes[0], "accuracy": axes[1]}, history, include_test=True
    )
    viz.plot_raster(axes[2], recording, signal="hidden.spikes", color="tab:blue", s=4)
    for axis, title in zip(
        axes, ("Loss", "Accuracy", f"Test member {diagnostic_id}"), strict=True
    ):
        axis.set_title(title)
    axes[0].legend(fontsize=7)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Cell")
    figure.tight_layout()
    figure.savefig(output / "evidence.png", dpi=120)
    plt.close(figure)
    print(f"held-out test accuracy: {history.test.accuracy:.1%}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.output, epochs=args.epochs, smoke=args.smoke)
