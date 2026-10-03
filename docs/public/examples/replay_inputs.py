"""Represent the same four spikes as a dense array and sparse events."""

from pathlib import Path

import torch

from snnlab import lang as snn
from snnlab.sim.execution import EventStreamBinding, ExecutionSpec, simulate


def main(out=Path("artifacts/examples/replay_inputs")):
    net = snn.Network("raw_counts", dt=1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    counts = snn.ops.reduce(events, operation="sum", over="time", name="channel_totals")
    net.output("counts", counts)
    graph = snn.compile(net, target="tools/snnsim").graph

    dense = torch.zeros(3, 2, 2)  # time, batch, channel
    steps = torch.tensor([0, 1, 2, 2])
    batches = torch.tensor([0, 0, 0, 1])
    channels = torch.tensor([0, 1, 0, 1])
    dense[steps, batches, channels] = 1
    sparse = EventStreamBinding(
        "events",
        steps=steps,
        batches=batches,
        channels=channels,
        steps_count=3,
        batch_size=2,
    )
    dense_result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs={"events": dense},
            device="cpu",
            seed=17,
        )
    )
    event_result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            event_bindings=(sparse,),
            device="cpu",
            seed=17,
        )
    )
    expected = torch.tensor([[2.0, 1.0], [0.0, 1.0]])
    torch.testing.assert_close(dense_result.outputs["counts"], expected, rtol=0, atol=0)
    torch.testing.assert_close(event_result.outputs["counts"], expected, rtol=0, atol=0)
    print("Counts per presentation and channel:", expected.tolist())
    print("Dense and sparse outputs match exactly")


if __name__ == "__main__":
    main()
