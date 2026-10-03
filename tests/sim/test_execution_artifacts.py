"""Focused acceptance tests for the typed seam and graph executor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from snnlab.lang.examples.build_examples import ping_classifier
from snnlab.sim.execution import (
    ExecutionSpec,
    derive_inference_products,
    simulate,
    validate_derived_inference_products,
    validate_inference_artifacts,
    write_inference_artifacts,
)
from tests.sim._execution_builders import (
    standard_readout_graph as _standard_readout_graph,
)


def test_inference_artifact_manifest_authenticates_cache_identity(tmp_path):
    graph = _standard_readout_graph("count")
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs={"events": torch.ones(2, 1, 2)},
            seed=23,
        )
    )
    root = tmp_path / "inference"
    manifest = write_inference_artifacts(root, result, graph=graph, seed=23)
    assert manifest["schema"] == "tools/snnsim.inference-artifacts/v1"
    assert manifest["request_seed"] == 23
    assert {row["path"] for row in manifest["files"]} == {
        "recording.npz",
        "outputs.npz",
        "parameters.npz",
        "metrics.json",
    }
    outputs = next(row for row in manifest["files"] if row["path"] == "outputs.npz")
    assert outputs["arrays"] == [
        {"name": "class_scores", "shape": [1, 2], "dtype": "float32"}
    ]
    assert validate_inference_artifacts(root, graph=graph, seed=23) == manifest
    with pytest.raises(ValueError, match="request seed"):
        validate_inference_artifacts(root, seed=24)


def test_inference_artifact_validation_rejects_payload_corruption(tmp_path):
    graph = _standard_readout_graph("count")
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs={"events": torch.zeros(1, 1, 2)},
        )
    )
    root = tmp_path / "inference"
    write_inference_artifacts(root, result, graph=graph, seed=0)
    with (root / "outputs.npz").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="outputs.npz digest"):
        validate_inference_artifacts(root)


def test_derived_inference_products_use_named_public_tensors(tmp_path):
    bundle = ping_classifier()
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            inputs={"image": torch.zeros(2, 2, 784)},
            seed=3,
        )
    )
    source = tmp_path / "source"
    source_manifest = write_inference_artifacts(
        source, result, graph=bundle.graph, seed=3
    )
    derived = tmp_path / "derived"
    summary = derive_inference_products(
        source,
        derived,
        logits_id="class_logits",
        labels=np.asarray([0, 1], dtype=np.int64),
        spike_recordings=("sensory_ping_E.spikes",),
    )
    assert summary["schema"] == "tools/snnsim.derived-inference/v1"
    assert summary["source_artifact_digest"] == source_manifest["artifact_digest"]
    assert summary["accuracy"] == 0.5
    rates = np.load(derived / "rates.npz", allow_pickle=False)
    rasters = np.load(derived / "rasters.npz", allow_pickle=False)
    try:
        assert rates["sensory_ping_E.spikes"].shape == (2, 256)
        assert rasters["sensory_ping_E.spikes.shape"].tolist() == [2, 2, 256]
        assert rasters["sensory_ping_E.spikes.steps"].dtype == np.int64
    finally:
        rates.close()
        rasters.close()
    validate_derived_inference_products(
        derived, source_artifact_digest=source_manifest["artifact_digest"]
    )


def test_derived_inference_products_fail_closed_on_names_and_corruption(tmp_path):
    bundle = ping_classifier()
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            inputs={"image": torch.zeros(1, 1, 784)},
        )
    )
    source = tmp_path / "source"
    write_inference_artifacts(source, result, graph=bundle.graph, seed=0)
    with pytest.raises(ValueError, match="do not contain logits"):
        derive_inference_products(
            source,
            tmp_path / "missing",
            logits_id="missing",
            labels=np.asarray([0]),
        )
    derived = tmp_path / "derived"
    derive_inference_products(
        source,
        derived,
        logits_id="class_logits",
        labels=np.asarray([0]),
    )
    with (derived / "predictions.npy").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="predictions.npy digest"):
        validate_derived_inference_products(derived)
