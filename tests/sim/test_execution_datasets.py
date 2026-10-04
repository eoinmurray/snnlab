"""Focused acceptance tests for the typed seam and graph executor."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from snnlab import lang as snn
from snnlab.sim.execution import (
    DatasetEncoder,
    DatasetSnapshotBinding,
    ExecutionSpec,
    PoissonInputBinding,
    resolve_dataset_snapshot_binding,
    resolve_poisson_input_bindings,
    train,
)
from snnlab.sim.tool import main
from tests.sim._execution_builders import (
    direct_train_bundle as _direct_train_bundle,
)
from tests.sim._execution_builders import (
    standard_readout_graph as _standard_readout_graph,
)


def test_categorical_poisson_samples_one_reproducible_rate_per_presentation():
    graph = _standard_readout_graph("count")
    binding = PoissonInputBinding(
        input_id="events",
        steps_count=4,
        batch_size=5,
        rates_hz=(0.0, 1.0, 5.0),
        seed=41,
        categorical=True,
    )
    first = resolve_poisson_input_bindings(graph, bindings=(binding,))
    second = resolve_poisson_input_bindings(graph, bindings=(binding,))
    torch.testing.assert_close(first.tensors["events"], second.tensors["events"])
    row = first.protocol["inputs"][0]
    assert len(row["realized_rates_hz"]) == 5
    assert set(row["realized_rates_hz"]) <= {0.0, 1.0, 5.0}
    assert row["selection"] == "uniform_independent_per_presentation"


def test_dense_dataset_snapshot_rate_poisson_is_selected_and_seeded(tmp_path):
    graph = _standard_readout_graph("count")
    snapshot = tmp_path / "mnist.npz"
    np.savez(
        snapshot,
        features=np.asarray(
            [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
            dtype=np.float32,
        ),
        labels=np.asarray([0, 1, 0, 1], dtype=np.int64),
    )
    binding = DatasetSnapshotBinding(
        path=snapshot,
        input_id="events",
        target_id="label",
        dataset_id="mnist-fixture-sha256",
        split="train",
        encoder=DatasetEncoder(
            "rate_poisson", duration_ms=300.0, max_rate_hz=10.0, seed=17
        ),
        sample_cap=3,
        shuffle=True,
        order_seed=23,
    )
    first, targets = resolve_dataset_snapshot_binding(graph, binding)
    second, _ = resolve_dataset_snapshot_binding(graph, binding)
    torch.testing.assert_close(first.tensors["events"], second.tensors["events"])
    assert first.tensors["events"].shape == (3, 3, 2)
    assert targets[0].target_id == "label"
    protocol = first.protocol
    assert protocol["binding_schema"] == "tools/snnsim.dataset-snapshot-binding/v1"
    assert protocol["dataset"] == {
        "identity": "mnist-fixture-sha256",
        "split": "train",
        "sample_cap": 3,
        "batch_size": 3,
        "shuffle": True,
    }
    assert protocol["dataset_binding"]["selected_indices"] == [3, 0, 2]
    assert protocol["dataset_binding"]["source"]["digest"].startswith("sha256:")


def test_prebinned_dataset_snapshot_trains_with_bound_labels(tmp_path):
    bundle = _direct_train_bundle()
    snapshot = tmp_path / "prebinned.npz"
    features = np.zeros((3, 2, 2), dtype=np.uint8)
    features[:, 0, 0] = 1
    features[:, 1, 1] = 1
    np.savez(snapshot, features=features, labels=np.asarray([0, 1]))
    result = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(
                DatasetSnapshotBinding(
                    path=snapshot,
                    input_id="events",
                    target_id="label",
                    dataset_id="prebinned-fixture",
                    split="train",
                    encoder=DatasetEncoder("prebinned_spikes"),
                ),
            ),
            seed=13,
        )
    )
    assert result.metrics["execution_protocol"]["representation"] == "dataset_snapshot"
    assert result.metrics["execution_protocol"]["targets"][0]["source"][
        "digest"
    ].startswith("sha256:")
    assert result.metrics["updates"][0]["loss"] == pytest.approx(0.6931471824645996)


def test_event_dataset_snapshot_bins_selected_samples_and_records_collisions(tmp_path):
    graph = _standard_readout_graph("count")
    snapshot = tmp_path / "shd.npz"
    np.savez(
        snapshot,
        labels=np.asarray([4, 7], dtype=np.int64),
        event_sample=np.asarray([0, 0, 0, 1], dtype=np.int64),
        event_time_ms=np.asarray([0.0, 99.9, 99.9, 200.0], dtype=np.float32),
        event_channel=np.asarray([0, 1, 1, 0], dtype=np.int64),
    )
    resolved, targets = resolve_dataset_snapshot_binding(
        graph,
        DatasetSnapshotBinding(
            path=snapshot,
            input_id="events",
            target_id="digit",
            dataset_id="shd-fixture",
            split="train",
            encoder=DatasetEncoder("event_bin", duration_ms=300.0),
        ),
    )
    assert resolved.tensors["events"].shape == (3, 2, 2)
    assert resolved.tensors["events"].sum() == 3
    encoder = resolved.protocol["dataset_binding"]["encoder"]
    assert encoder["retained_events"] == 4
    assert encoder["binary_collisions"] == 1
    assert targets[0].value.tolist() == [4, 7]


def test_dataset_snapshot_rejects_ambiguous_encoder_fields_and_bad_events(tmp_path):
    graph = _standard_readout_graph("count")
    dense = tmp_path / "dense.npz"
    np.savez(
        dense,
        features=np.zeros((1, 2), dtype=np.uint8),
        labels=np.asarray([0]),
    )
    with pytest.raises(ValueError, match="does not accept duration"):
        resolve_dataset_snapshot_binding(
            graph,
            DatasetSnapshotBinding(
                path=dense,
                input_id="events",
                dataset_id="dense",
                split="train",
                encoder=DatasetEncoder("prebinned_spikes", duration_ms=100.0),
            ),
        )
    events = tmp_path / "events.npz"
    np.savez(
        events,
        labels=np.asarray([0]),
        event_sample=np.asarray([0]),
        event_time_ms=np.asarray([300.0], dtype=np.float32),
        event_channel=np.asarray([0]),
    )
    with pytest.raises(ValueError, match="out of snapshot bounds"):
        resolve_dataset_snapshot_binding(
            graph,
            DatasetSnapshotBinding(
                path=events,
                input_id="events",
                dataset_id="events",
                split="train",
                encoder=DatasetEncoder("event_bin", duration_ms=300.0),
            ),
        )


def test_poisson_binding_rejects_invalid_rate_probability():
    graph = _standard_readout_graph("count")
    with pytest.raises(ValueError, match="rate times dt exceeds probability one"):
        resolve_poisson_input_bindings(
            graph,
            bindings=(
                PoissonInputBinding(
                    input_id="events",
                    steps_count=1,
                    batch_size=1,
                    rates_hz=(10001.0,),
                    seed=0,
                ),
            ),
        )


def test_graph_cli_generates_fixed_rate_poisson_protocol(tmp_path):
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
    out = tmp_path / "out"
    assert (
        main(
            [
                "sim",
                "--executor",
                "graph",
                "--bundle",
                str(bundle),
                "--poisson-protocol",
                "fixed-rate",
                "--input-rate",
                "0",
                "--n-batch",
                "1",
                "--t-ms",
                "200",
                "--seed",
                "13",
                "--out-dir",
                str(out),
            ]
        )
        == 0
    )
    protocol = json.loads((out / "metrics.json").read_text())["execution_protocol"]
    assert protocol["representation"] == "poisson"
    assert protocol["inputs"][0]["realized_rates_hz"] == [0.0]
    assert protocol["seeds"] == {"execution": 13, "poisson": {"events": 13}}
