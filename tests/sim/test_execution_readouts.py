"""Focused acceptance tests for the typed seam and graph executor."""

from __future__ import annotations

import torch

from snnlab import lang as snn
from snnlab.sim.execution import (
    ExecutionSpec,
    simulate,
)


def _standard_readout_graph(
    readout: str, *, duration: float | None = None, mask: bool = False
):
    net = snn.Network("readout_fixture", dt=100 * snn.ms)
    source = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    valid = (
        net.input("valid", shape=("time", "batch"), signal_type="mask")
        if mask
        else None
    )
    if readout == "final":
        value = snn.readouts.FinalVoltage(source=source, classes=2, name="scores")
    elif readout == "count":
        value = snn.readouts.SpikeCount(source=source, classes=2, name="scores")
    elif readout == "rate":
        value = snn.readouts.SpikeRate(
            source=source,
            classes=2,
            name="scores",
            duration=duration,
            mask=valid,
        )
    elif readout == "cumulative":
        value = snn.readouts.CumulativePotential(
            source=source, classes=2, name="scores"
        )
    else:
        raise ValueError(readout)
    net.output("class_scores", value)
    graph = snn.compile(net, target="tools/snnsim").graph
    for parameter in graph["parameters"]:
        parameter["initializer"] = {"kind": "constant", "value": 1.0}
    return graph


def test_graph_standard_readouts_match_hand_calculated_fixtures():
    events = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ]
    )
    projected = events.sum(dim=2, keepdim=True).expand(-1, -1, 2)

    final = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=_standard_readout_graph("final"),
            inputs={"events": events},
        )
    )
    torch.testing.assert_close(
        final.outputs["class_scores"], projected[-1], rtol=0, atol=0
    )

    count = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=_standard_readout_graph("count"),
            inputs={"events": events},
        )
    )
    torch.testing.assert_close(
        count.outputs["class_scores"], projected.sum(dim=0), rtol=0, atol=0
    )

    rate = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=_standard_readout_graph("rate", duration=0.3),
            inputs={"events": events},
        )
    )
    torch.testing.assert_close(
        rate.outputs["class_scores"], projected.sum(dim=0) / 0.3, rtol=0, atol=0
    )

    cumulative = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=_standard_readout_graph("cumulative"),
            inputs={"events": events},
        )
    )
    torch.testing.assert_close(
        cumulative.outputs["class_scores"], projected.cumsum(dim=0), rtol=0, atol=0
    )


def test_graph_masked_spike_rate_uses_valid_duration_in_spikes_per_second():
    events = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ]
    )
    valid = torch.tensor(
        [
            [True, True],
            [False, True],
            [True, False],
        ]
    )
    projected = events.sum(dim=2, keepdim=True).expand(-1, -1, 2)
    expected_count = (projected * valid[:, :, None]).sum(dim=0)
    expected_seconds = valid.sum(dim=0).to(projected.dtype)[:, None] * 0.1
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=_standard_readout_graph("rate", mask=True),
            inputs={"events": events, "valid": valid},
        )
    )
    torch.testing.assert_close(
        result.outputs["class_scores"],
        expected_count / expected_seconds,
        rtol=0,
        atol=0,
    )
