"""Train a weighted spike-count readout on a tiny two-class fixture."""

from pathlib import Path

import torch

from snnlab import lang as snn
from snnlab.lang import training
from snnlab.sim.execution import ExecutionSpec, infer, train


def main(out=Path("artifacts/examples/train_readout")):
    net = snn.Network("two_class_readout", dt=1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    scores = snn.readouts.SpikeCount(source=events, classes=2, name="classifier")
    net.output("scores", scores)
    recipe = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            training.ParameterGroup(
                [row["id"] for row in net.parameters],
                name="readout",
                lr=0.1,
            )
        ],
        optimizer=training.AdamW(weight_decay=0.0),
        presentation_duration=3 * snn.ms,
    )
    bundle = snn.compile(net, training=recipe, target="tools/snnsim")
    labels = torch.tensor([0, 1, 0, 1])
    inputs = torch.zeros(3, 4, 2)
    inputs[:, torch.arange(4), labels] = 1
    checkpoint = Path(out) / "checkpoint"
    result = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            inputs={"events": inputs},
            targets={"label": labels},
            seed=17,
            device="cpu",
            options={"updates": 10, "save_final_checkpoint": checkpoint},
        )
    )
    losses = [row["loss"] for row in result.metrics["updates"]]
    assert losses[-1] < losses[0]
    prediction = (
        infer(
            ExecutionSpec(
                kind="infer",
                executor="graph",
                graph=bundle.graph,
                inputs={"events": inputs},
                checkpoint=checkpoint,
                seed=17,
                device="cpu",
            )
        )
        .outputs["scores"]
        .argmax(dim=-1)
    )
    assert torch.equal(prediction, labels)
    print(f"Cross-entropy: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print("Predictions:", prediction.tolist())
    print("Labels:     ", labels.tolist())


if __name__ == "__main__":
    main()
