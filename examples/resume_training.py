"""Compare uninterrupted CPU training with a saved and resumed run."""

from pathlib import Path

import torch

from snnlab import lang as snn
from snnlab.lang import training
from snnlab.sim.execution import ExecutionSpec, load_training_checkpoint, train


def main(out=Path("artifacts/examples/resume_training")):
    net = snn.Network("resume_readout", dt=1 * snn.ms)
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
                name="all",
                lr=0.1,
            )
        ],
        optimizer=training.AdamW(weight_decay=0.0),
        presentation_duration=3 * snn.ms,
    )
    bundle = snn.compile(net, training=recipe, target="tools/snnsim")
    inputs = torch.zeros(3, 2, 2)
    inputs[:, 0, 0] = 1
    inputs[:, 1, 1] = 1
    common = dict(
        kind="train",
        executor="graph",
        graph=bundle.graph,
        training=bundle.training,
        inputs={"events": inputs},
        targets={"label": torch.tensor([0, 1])},
        device="cpu",
        seed=17,
    )
    whole = train(ExecutionSpec(**common, options={"updates": 4}))
    checkpoint = Path(out) / "checkpoint"
    train(
        ExecutionSpec(
            **common,
            options={
                "updates": 2,
                "save_final_checkpoint": checkpoint,
            },
        )
    )
    loaded = load_training_checkpoint(checkpoint)
    assert loaded.completed_updates == 2
    resumed = train(
        ExecutionSpec(
            **common,
            checkpoint=checkpoint,
            options={"updates": 2},
        )
    )
    for name in whole.parameters:
        torch.testing.assert_close(
            resumed.parameters[name], whole.parameters[name], rtol=0, atol=0
        )
    for name, state in whole.optimizer_state.items():
        for key, value in state.items():
            torch.testing.assert_close(
                resumed.optimizer_state[name][key], value, rtol=0, atol=0
            )
    assert resumed.metrics["updates"] == whole.metrics["updates"][2:]
    print("Resumed from update:", resumed.metrics["resumed_from_update"])
    print("Final parameters, optimizer state and resumed losses match exactly")


if __name__ == "__main__":
    main()
