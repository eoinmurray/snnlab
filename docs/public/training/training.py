"""Train a spike-pattern classifier and save weights and epoch curves beside it."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from snnlab import lang, viz
from snnlab.lang import training
from snnlab.sim.execution import (
    DenseArrayBinding,
    ExecutionSpec,
    ValidationSpec,
    save_training_checkpoint,
    train,
)

OUTPUT_DIR = Path(__file__).resolve().parent
DT_MS = 0.5
DURATION_MS = 200
HIGH_RATE_HZ = 100
LOW_RATE_HZ = 10
TRAIN_SAMPLES = 128
VALIDATION_SAMPLES = 64
BATCH_SIZE = 32
EPOCHS = 20
INPUT_LEARNING_RATE = 0.001
READOUT_LEARNING_RATE = 1.0
SEED = 17


def make_dataset(samples, seed):
    """Balanced classes with independent rate variation and Poisson spike draws."""
    rng = np.random.default_rng(seed)
    labels = np.arange(samples) % 2
    rng.shuffle(labels)
    patterns = np.array(
        [
            [HIGH_RATE_HZ, HIGH_RATE_HZ, LOW_RATE_HZ, LOW_RATE_HZ],
            [LOW_RATE_HZ, LOW_RATE_HZ, HIGH_RATE_HZ, HIGH_RATE_HZ],
        ]
    )
    rates = patterns[labels] * rng.uniform(0.85, 1.15, size=(samples, 4))
    probability = rates * DT_MS / 1000
    steps = round(DURATION_MS / DT_MS)
    spikes = rng.random((steps, samples, 4)) < probability
    return torch.tensor(spikes, dtype=torch.float32), torch.tensor(
        labels, dtype=torch.long
    )


def main():
    # 1. Create the network.
    net = lang.Network("spike_pattern_classifier", dt=DT_MS * lang.ms)
    # 2. Define inputs, labelled datasets and bindings.
    inputs = net.input(
        "inputs", shape=("time", "batch", 4), signal_type="spikes", unit="spike"
    )
    train_spikes, train_labels = make_dataset(TRAIN_SAMPLES, seed=SEED + 1)
    validation_spikes, validation_labels = make_dataset(
        VALIDATION_SAMPLES, seed=SEED + 2
    )
    train_input = DenseArrayBinding(input_id="inputs", value=train_spikes)
    validation_input = DenseArrayBinding(input_id="inputs", value=validation_spikes)

    # 3. Define the excitatory layer and two-class readout.
    cells = net.population("E", size=16, neuron=lang.COBA_LIF(tau_mem=20 * lang.ms))
    input_projection = net.connect(
        inputs,
        cells.excitatory,
        name="input_to_E",
        synapse=lang.AMPA(tau=2 * lang.ms),
        weight=lang.Uniform(0.0, 0.8),
        constraint=lang.NonNegative(),
    )
    w_in = input_projection.weight
    # Two non-spiking readout cells, one per class.
    readout = net.population(
        "readout",
        size=2,
        neuron=lang.LeakyIntegrator(tau=20 * lang.ms, initial_voltage=0.0),
        spiking=False,
    )
    # Explicit output weights: 2 readout cells receive spikes from 16 E cells.
    w_out = net.parameter(
        "w_out", shape=(2, 16), unit="uS", initializer=lang.Normal(0.0, 0.1)
    )
    net.connect(
        cells.spikes,
        readout.excitatory,
        name="E_to_readout",
        synapse=lang.LeakyIntegrator(tau=20 * lang.ms),
        weight=w_out,
    )
    scores = lang.ops.reduce(
        readout.voltage, operation="mean", over="time", name="mean_readout_voltage"
    )
    # 4. Declare the official output and optional diagnostics.
    net.output("class_scores", scores)
    net.expose(inputs, name="input_spikes")
    net.expose(cells.spikes, name="e_spikes")
    net.expose(readout.voltage, name="readout_voltage")

    # 5. Define the loss, trainable parameters and optimizer.
    objective = training.CrossEntropy(prediction=scores, target="class")
    readout_parameters = training.ParameterGroup(
        (w_out,), name="readout", lr=READOUT_LEARNING_RATE
    )
    input_parameters = training.ParameterGroup(
        (w_in,), name="input_to_E", lr=INPUT_LEARNING_RATE
    )
    recipe = lang.TrainSpec(
        objectives=(objective,),
        parameter_groups=(input_parameters, readout_parameters),
        optimizer=training.AdamW(weight_decay=0.0),
        surrogate=training.FastSigmoid(slope=1.0),
        gradient_clip=1.0,
        presentation_duration=DURATION_MS * lang.ms,
    )
    # 6. Compile and save the bundle for training and later inference.
    bundle = lang.compile(net, training=recipe, target="tools/snnsim")
    diagram = lang.diagram(bundle, view="expanded")
    viz.render_diagram(
        diagram,
        OUTPUT_DIR / "network.png",
        scale=2,
        height_to_width_ratio=None,
        canvas_size=(1920, 900),
    )
    bundle_path = bundle.write(OUTPUT_DIR / "network.bundle")
    checkpoint_path = OUTPUT_DIR / "trained.checkpoint"

    # 7. Train and measure each epoch in one call.
    validation = ValidationSpec(
        input_bindings=(validation_input,),
        targets={"class": validation_labels},
    )
    execution = ExecutionSpec(
        kind="train",
        bundle=bundle_path,
        input_bindings=(train_input,),
        targets={"class": train_labels},
        validation=validation,
        seed=SEED,
        device="cpu",
        diagnostics=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    result = train(execution)
    assert result.training_checkpoint is not None
    save_training_checkpoint(checkpoint_path, result.training_checkpoint)
    history = result.metrics["epochs"]
    for row in history:
        print(
            f"Epoch {row['epoch']:02d}: train loss={row['train_loss']:.3f}, "
            f"validation loss={row['validation_loss']:.3f}, "
            f"train accuracy={row['train_accuracy']:.1%}, "
            f"validation accuracy={row['validation_accuracy']:.1%}",
        )

    # 8. Save epoch metrics and plot training curves.
    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(history, indent=2) + "\n")
    # Plot the epoch curves.
    grid = viz.FigureGrid(
        rows=2, columns=1, bounds=(0.13, 0.1, 0.83, 0.82), row_gap=0.09
    )
    grid.place("loss", row=0, column=0)
    grid.place("accuracy", row=1, column=0)
    figure = grid.figure(figsize=(8, 6.5), dpi=150)
    loss_axis = grid.add_axes(figure, "loss")
    accuracy_axis = grid.add_axes(figure, "accuracy", sharex=loss_axis)
    epochs = [row["epoch"] for row in history]
    for split, color in (("train", "#1a1a1a"), ("validation", "#a74727")):
        loss_axis.plot(
            epochs,
            [row[f"{split}_loss"] for row in history],
            label=split.capitalize(),
            color=color,
        )
        accuracy_axis.plot(
            epochs,
            [100 * row[f"{split}_accuracy"] for row in history],
            label=split.capitalize(),
            color=color,
        )
    loss_axis.set_title("Classification loss", pad=10)
    loss_axis.set_ylabel("Cross-entropy")
    loss_axis.tick_params(labelbottom=False)
    loss_axis.legend(frameon=False)
    accuracy_axis.axhline(50, color="#888888", linestyle=":", label="Chance (50%)")
    accuracy_axis.set_title("Classification accuracy", pad=10)
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylim(0, 105)
    accuracy_axis.set_xlim(0, EPOCHS)
    accuracy_axis.set_xticks(range(0, EPOCHS + 1, 5))
    accuracy_axis.legend(frameon=False, loc="lower right")
    figure_path = OUTPUT_DIR / "training.png"
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {figure_path}")
    print(f"Saved {metrics_path}")
    print(f"Inference can load {bundle_path} and {checkpoint_path}")


if __name__ == "__main__":
    main()
