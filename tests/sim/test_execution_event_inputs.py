"""Focused acceptance tests for the typed seam and graph executor."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from snnlab import lang as snn
from snnlab.sim.execution import (
    DenseArrayBinding,
    EventStreamBinding,
    ExecutionSpec,
    load_event_stream_bindings,
    resolve_event_stream_bindings,
    resolve_input_bindings,
    simulate,
)
from snnlab.sim.tool import main
from tests.sim._execution_builders import (
    coupled_graph as _coupled_graph,
)
from tests.sim._execution_builders import (
    standard_readout_graph as _standard_readout_graph,
)


def test_event_stream_binding_matches_hand_calculated_spike_count():
    graph = _standard_readout_graph("count")
    binding = EventStreamBinding(
        "events",
        steps=torch.tensor([0, 1, 2, 2]),
        batches=torch.tensor([0, 0, 0, 1]),
        channels=torch.tensor([0, 1, 0, 1]),
        steps_count=3,
        batch_size=2,
        source={"kind": "fixture", "id": "events-v1"},
    )
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            event_bindings=(binding,),
            seed=31,
            protocol={
                "dataset": {
                    "identity": "event-fixture-v1",
                    "split": "test",
                    "shuffle": False,
                }
            },
        )
    )
    torch.testing.assert_close(
        result.outputs["class_scores"],
        torch.tensor([[3.0, 3.0], [1.0, 1.0]]),
        rtol=0,
        atol=0,
    )
    protocol = result.metrics["execution_protocol"]
    assert protocol["binding_schema"] == "tools/snnsim.event-stream-binding/v1"
    assert protocol["representation"] == "event_stream"
    assert protocol["inputs"][0]["event_count"] == 4
    assert protocol["dataset"] == {
        "identity": "event-fixture-v1",
        "split": "test",
        "shuffle": False,
        "sample_cap": 2,
        "batch_size": 2,
    }
    assert protocol["timing"] == {
        "dt_ms": 100.0,
        "steps": 3,
        "duration_ms": 300.0,
    }
    assert protocol["resolution"] == {
        "coordinates": "zero_based_integer_steps",
        "ordering": "step,batch,channel",
        "duplicates": "reject",
        "materialization": "binary_dense",
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"steps": torch.tensor([3])}, "step coordinates must be in [0, 3)"),
        (
            {
                "steps": torch.tensor([1, 0]),
                "batches": torch.tensor([0, 0]),
                "channels": torch.tensor([0, 1]),
            },
            "coordinates must be ordered",
        ),
        (
            {
                "steps": torch.tensor([0, 0]),
                "batches": torch.tensor([0, 0]),
                "channels": torch.tensor([1, 1]),
            },
            "contains duplicate coordinates",
        ),
        ({"channels": torch.tensor([2])}, "channel coordinates must be in [0, 2)"),
    ],
)
def test_event_stream_bindings_fail_closed(updates, message):
    values = {
        "steps": torch.tensor([0]),
        "batches": torch.tensor([0]),
        "channels": torch.tensor([0]),
    }
    values.update(updates)
    binding = EventStreamBinding("events", **values, steps_count=3, batch_size=1)
    with pytest.raises(ValueError) as exc:
        resolve_event_stream_bindings(
            _standard_readout_graph("count"), bindings=(binding,)
        )
    assert message in str(exc.value)


def test_event_stream_binding_rejects_non_spike_graph_input():
    graph = _standard_readout_graph("count")
    graph["inputs"][0]["signal_type"] = "continuous"
    binding = EventStreamBinding(
        "events",
        steps=torch.tensor([0]),
        batches=torch.tensor([0]),
        channels=torch.tensor([0]),
        steps_count=1,
        batch_size=1,
    )
    with pytest.raises(ValueError, match="requires signal_type spikes"):
        resolve_event_stream_bindings(graph, bindings=(binding,))


def test_event_stream_spikes_can_share_a_request_with_dense_valid_time_mask():
    graph = _standard_readout_graph("rate", mask=True)
    events = EventStreamBinding(
        "events",
        steps=torch.tensor([0, 0, 1]),
        batches=torch.tensor([0, 1, 0]),
        channels=torch.tensor([0, 1, 1]),
        steps_count=3,
        batch_size=2,
    )
    valid = DenseArrayBinding(
        "valid",
        torch.tensor([[True, True], [True, False], [False, False]]),
    )
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            event_bindings=(events,),
            input_bindings=(valid,),
        )
    )
    torch.testing.assert_close(
        result.outputs["class_scores"],
        torch.tensor([[10.0, 10.0], [10.0, 10.0]]),
        rtol=0,
        atol=0,
    )
    protocol = result.metrics["execution_protocol"]
    assert protocol["representation"] == "mixed"
    assert protocol["binding_schema"] == "tools/snnsim.mixed-input-bindings/v1"
    assert protocol["masks"] == ["valid"]
    resolved = resolve_input_bindings(
        graph, dense_bindings=(valid,), event_bindings=(events,)
    )
    assert set(resolved.tensors) == {"events", "valid"}


def test_event_stream_file_loader_and_cli_emit_replayable_protocol(tmp_path):
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
    event_file = tmp_path / "events.npz"
    np.savez(
        event_file,
        steps=np.array([0, 1], dtype=np.int64),
        batches=np.array([0, 0], dtype=np.int64),
        channels=np.array([0, 1], dtype=np.int64),
        steps_count=np.array(2, dtype=np.int64),
        batch_size=np.array(1, dtype=np.int64),
    )
    bindings = load_event_stream_bindings(event_file, graph)
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
                "--event-file",
                str(event_file),
                "--input-dataset-id",
                "event-snapshot-v1",
                "--input-split",
                "test",
                "--no-input-shuffle",
                "--seed",
                "41",
                "--out-dir",
                str(out),
            ]
        )
        == 0
    )
    metrics = json.loads((out / "metrics.json").read_text())
    protocol = metrics["execution_protocol"]
    assert protocol["representation"] == "event_stream"
    assert protocol["inputs"][0]["event_count"] == 2
    assert protocol["dataset"]["identity"] == "event-snapshot-v1"
    assert protocol["dataset"]["split"] == "test"
    assert protocol["dataset"]["shuffle"] is False
    assert protocol["seeds"] == {"execution": 41}
    assert protocol["inputs"][0]["source"]["digest"].startswith("sha256:")
    with pytest.raises(SystemExit, match="--event-file requires --executor graph"):
        main(
            [
                "sim",
                "--event-file",
                str(event_file),
                "--out-dir",
                str(tmp_path / "legacy-out"),
            ]
        )


def test_named_multi_input_event_file_resolves_each_graph_port(tmp_path):
    graph = _coupled_graph(direction="uncoupled")
    event_file = tmp_path / "multi-events.npz"
    arrays = {
        "drive_a.steps": np.array([0, 1], dtype=np.int64),
        "drive_a.batches": np.array([0, 0], dtype=np.int64),
        "drive_a.channels": np.array([0, 2], dtype=np.int64),
        "drive_a.steps_count": np.array(2, dtype=np.int64),
        "drive_a.batch_size": np.array(1, dtype=np.int64),
        "drive_b.steps": np.array([], dtype=np.int64),
        "drive_b.batches": np.array([], dtype=np.int64),
        "drive_b.channels": np.array([], dtype=np.int64),
        "drive_b.steps_count": np.array(2, dtype=np.int64),
        "drive_b.batch_size": np.array(1, dtype=np.int64),
    }
    np.savez(event_file, **arrays)
    bindings = load_event_stream_bindings(event_file, graph)
    resolved = resolve_event_stream_bindings(graph, bindings=bindings)
    assert set(resolved.tensors) == {"drive_a", "drive_b"}
    assert resolved.tensors["drive_a"].shape == (2, 1, 3)
    assert resolved.tensors["drive_a"].sum() == 2
    assert resolved.tensors["drive_b"].shape == (2, 1, 2)
    assert resolved.tensors["drive_b"].sum() == 0
