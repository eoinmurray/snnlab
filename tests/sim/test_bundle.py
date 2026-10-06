"""Authenticated graph and training data survive CLI backend retirement."""

import json

import pytest

from snnlab import lang
from snnlab.sim.bundle import (
    BundleCompatibilityError,
    load_graph_bundle,
    load_training_recipe,
)
from snnlab.sim.execution import ExecutionSpec, simulate
from snnlab.sim.tool import main
from tests.sim._bundle_builders import ping_classifier


def test_graph_and_training_bundle_digests_are_authenticated(tmp_path):
    bundle = ping_classifier()
    path = bundle.write(tmp_path / "network.bundle")
    manifest, graph = load_graph_bundle(path)
    assert graph == bundle.graph
    assert load_training_recipe(path, manifest, graph) == bundle.training
    value = json.loads((path / "training.json").read_text())
    value["epochs"] += 1
    (path / "training.json").write_text(json.dumps(value))
    with pytest.raises(BundleCompatibilityError, match="digest"):
        load_training_recipe(path, manifest, graph)


def test_graph_bundle_tampering_is_rejected(tmp_path):
    path = ping_classifier().write(tmp_path / "network.bundle")
    graph = json.loads((path / "graph.json").read_text())
    graph["name"] = "tampered"
    (path / "graph.json").write_text(json.dumps(graph))
    with pytest.raises(BundleCompatibilityError, match="digest"):
        load_graph_bundle(path)


def test_training_requires_an_authenticated_recipe(tmp_path):
    path = ping_classifier().write(tmp_path / "network.bundle")
    manifest, graph = load_graph_bundle(path)
    (path / "training.json").unlink()
    with pytest.raises(BundleCompatibilityError, match="requires training.json"):
        load_training_recipe(path, manifest, graph)


def test_missing_bundle_is_an_explicit_error(tmp_path):
    missing = tmp_path / "missing.bundle"
    for load in (lang.load_bundle, load_graph_bundle):
        with pytest.raises(FileNotFoundError, match="bundle not found"):
            load(missing)
    with pytest.raises(FileNotFoundError, match="bundle not found"):
        simulate(ExecutionSpec(kind="simulate", bundle=missing, device="cpu"))
    with pytest.raises(SystemExit, match="bundle not found"):
        main(
            [
                "sim",
                "--bundle",
                str(missing),
                "--out-dir",
                str(tmp_path / "run"),
                "--poisson-protocol",
                "fixed-rate",
                "--input-rate",
                "20",
                "--t-ms",
                "1",
            ]
        )
    assert not (tmp_path / "run").exists()
