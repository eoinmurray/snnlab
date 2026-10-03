"""Focused acceptance tests for the typed seam and graph executor."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from snnlab import lang as snn
from snnlab.sim.execution import (
    DenseArrayBinding,
    ExecutionSpec,
    load_dense_array_bindings,
    resolve_dense_array_bindings,
    simulate,
)
from snnlab.sim.tool import main


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


def test_dense_array_bindings_match_hand_calculation_and_record_protocol():
    graph = _standard_readout_graph("rate", mask=True)
    events = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ]
    )
    valid = torch.tensor([[True, True], [False, True], [True, False]])
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            seed=17,
            input_bindings=(
                DenseArrayBinding(
                    "events", events, {"kind": "fixture", "id": "events-v1"}
                ),
                DenseArrayBinding("valid", valid, {"kind": "fixture", "id": "mask-v1"}),
            ),
            protocol={
                "dataset": {
                    "identity": "hand-calculated-v1",
                    "split": "test",
                    "sample_cap": 2,
                    "shuffle": False,
                }
            },
        )
    )
    projected = events.sum(dim=2, keepdim=True).expand(-1, -1, 2)
    expected_count = (projected * valid[:, :, None]).sum(dim=0)
    expected_seconds = valid.sum(dim=0).to(projected.dtype)[:, None] * 0.1
    torch.testing.assert_close(
        result.outputs["class_scores"],
        expected_count / expected_seconds,
        rtol=0,
        atol=0,
    )
    protocol = result.metrics["execution_protocol"]
    assert protocol["schema"] == "tools/snnsim.execution-protocol/v1"
    assert protocol["binding_schema"] == "tools/snnsim.dense-array-binding/v1"
    assert protocol["dataset"] == {
        "identity": "hand-calculated-v1",
        "split": "test",
        "sample_cap": 2,
        "shuffle": False,
        "batch_size": 2,
    }
    assert protocol["timing"] == {"dt_ms": 100.0, "steps": 3, "duration_ms": 300.0}
    assert protocol["masks"] == ["valid"]
    assert protocol["seeds"] == {"execution": 17}


@pytest.mark.parametrize(
    ("bindings", "message"),
    [
        ((DenseArrayBinding("events", torch.zeros(3, 1, 2)),), "missing=['valid']"),
        (
            (
                DenseArrayBinding("events", torch.zeros(3, 1, 2)),
                DenseArrayBinding("valid", torch.ones(3, 2, dtype=torch.bool)),
            ),
            "leading shape expected (3, 1)",
        ),
        (
            (
                DenseArrayBinding("events", torch.zeros(3, 1, 3)),
                DenseArrayBinding("valid", torch.ones(3, 1, dtype=torch.bool)),
            ),
            "trailing shape expected (2,)",
        ),
        (
            (
                DenseArrayBinding("events", torch.full((3, 1, 2), 0.5)),
                DenseArrayBinding("valid", torch.ones(3, 1, dtype=torch.bool)),
            ),
            "spike values must be boolean or zero/one",
        ),
    ],
)
def test_dense_array_bindings_fail_closed_on_contract_mismatch(bindings, message):
    with pytest.raises(ValueError) as exc:
        resolve_dense_array_bindings(
            _standard_readout_graph("rate", mask=True), bindings=bindings
        )
    assert message in str(exc.value)


def test_dense_array_file_loader_and_cli_emit_replayable_protocol(tmp_path):
    graph = _standard_readout_graph("count")
    bundle = snn.compiler.Bundle(
        graph=graph,
        training=None,
        manifest={
            "schema": "snnlang.bundle/v1",
            "graph_digest": snn.compiler.digest(graph),
            "files": [{"path": "graph.json", "digest": snn.compiler.digest(graph)}],
            "assets": [],
            "compiler": {"name": "test", "version": "1"},
            "target": None,
        },
        diagnostics=[],
        asset_sources={},
    ).write(tmp_path / "graph.bundle")
    input_file = tmp_path / "dense.npz"
    events = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
    np.savez(input_file, events=events)
    bindings = load_dense_array_bindings(input_file, graph)
    assert len(bindings) == 1 and bindings[0].input_id == "events"
    assert bindings[0].source["digest"].startswith("sha256:")
    out = tmp_path / "out"
    assert (
        main(
            [
                "sim",
                "--executor",
                "graph",
                "--bundle",
                str(bundle),
                "--input-file",
                str(input_file),
                "--input-dataset-id",
                "fixture-snapshot-v1",
                "--input-split",
                "test",
                "--no-input-shuffle",
                "--seed",
                "23",
                "--out-dir",
                str(out),
            ]
        )
        == 0
    )
    metrics = json.loads((out / "metrics.json").read_text())
    protocol = metrics["execution_protocol"]
    assert protocol["dataset"] == {
        "identity": "fixture-snapshot-v1",
        "split": "test",
        "shuffle": False,
        "sample_cap": 1,
        "batch_size": 1,
    }
    assert protocol["timing"] == {"dt_ms": 100.0, "steps": 2, "duration_ms": 200.0}
    assert protocol["seeds"] == {"execution": 23}
    assert protocol["inputs"][0]["source"]["digest"].startswith("sha256:")
