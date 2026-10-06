"""Focused acceptance tests for the typed seam and graph executor."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from snnlab.sim.execution import (
    TargetArrayBinding,
    load_target_array_bindings,
    load_training_checkpoint,
    resolve_target_array_bindings,
)
from snnlab.sim.tool import main
from tests.sim._execution_builders import (
    direct_train_bundle as _direct_train_bundle,
)


def test_target_array_binding_is_named_and_digest_bearing(tmp_path):
    bundle = _direct_train_bundle()
    path = tmp_path / "targets.npy"
    np.save(path, np.array([0, 1, 1], dtype=np.int64))
    bindings = load_target_array_bindings(path, bundle.training)
    assert bindings[0].target_id == "label"
    assert bindings[0].source["digest"].startswith("sha256:")
    resolved, rows = resolve_target_array_bindings(
        bundle.training, bindings=bindings, sample_count=3
    )
    assert resolved["label"].tolist() == [0, 1, 1]
    assert rows[0]["source"]["path"] == str(path)
    with pytest.raises(ValueError, match="target ids do not match recipe"):
        resolve_target_array_bindings(
            bundle.training,
            bindings=(TargetArrayBinding("wrong", torch.tensor([0, 1, 1])),),
            sample_count=3,
        )


def test_graph_training_cli_loads_targets_and_writes_resume_checkpoint(tmp_path):
    bundle = _direct_train_bundle().write(tmp_path / "train.bundle")
    inputs = np.zeros((3, 4, 2), dtype=np.float32)
    inputs[:, ::2, 0] = 1
    inputs[:, 1::2, 1] = 1
    input_path = tmp_path / "inputs.npy"
    target_path = tmp_path / "targets.npy"
    np.save(input_path, inputs)
    np.save(target_path, np.array([0, 1, 0, 1], dtype=np.int64))
    checkpoint = tmp_path / "checkpoint"
    out_dir = tmp_path / "run"
    assert (
        main(
            [
                "train",
                "--executor",
                "graph",
                "--bundle",
                str(bundle),
                "--input-file",
                str(input_path),
                "--target-file",
                str(target_path),
                "--batch-size",
                "2",
                "--input-shuffle",
                "--save-final-checkpoint",
                str(checkpoint),
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert len(metrics["updates"]) == 2
    assert metrics["execution_protocol"]["targets"][0]["source"]["digest"].startswith(
        "sha256:"
    )
    assert load_training_checkpoint(checkpoint).data_state == {
        "epoch": 1,
        "batch": 0,
    }


def test_graph_training_cli_resolves_portable_dataset_snapshot(tmp_path):
    bundle = _direct_train_bundle().write(tmp_path / "train.bundle")
    snapshot = tmp_path / "dataset.npz"
    features = np.zeros((3, 4, 2), dtype=np.uint8)
    features[:, ::2, 0] = 1
    features[:, 1::2, 1] = 1
    np.savez(snapshot, features=features, labels=np.asarray([0, 1, 0, 1]))
    out_dir = tmp_path / "run"
    assert (
        main(
            [
                "train",
                "--executor",
                "graph",
                "--bundle",
                str(bundle),
                "--dataset-file",
                str(snapshot),
                "--dataset-encoder",
                "prebinned-spikes",
                "--dataset-target-id",
                "label",
                "--input-dataset-id",
                "fixture-v1",
                "--input-split",
                "train",
                "--max-samples",
                "3",
                "--input-shuffle",
                "--batch-size",
                "2",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    protocol = json.loads((out_dir / "metrics.json").read_text())["execution_protocol"]
    assert protocol["representation"] == "dataset_snapshot"
    assert protocol["dataset"] == {
        "identity": "fixture-v1",
        "split": "train",
        "sample_cap": 3,
        "batch_size": 2,
        "shuffle": True,
    }
    assert protocol["dataset_binding"]["encoder"]["kind"] == "prebinned_spikes"
