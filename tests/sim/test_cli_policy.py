"""The CLI executes graph bundles and rejects retired network controls."""

import json

import numpy as np
import pytest

from snnlab.lang.compiler import digest
from snnlab.sim.tool import main, parse_args
from tests.sim._bundle_builders import ping_classifier
from tests.sim._execution_builders import direct_train_bundle


@pytest.mark.parametrize(
    "flag",
    [
        "--model",
        "--dt",
        "--infer",
        "--load-config",
        "--dataset",
        "--lr",
        "--recording-mode",
        "--adaptive-threshold",
    ],
)
def test_legacy_switches_are_rejected(flag):
    with pytest.raises(SystemExit) as error:
        parse_args(["sim", flag, "1"])
    assert error.value.code == 2


def test_no_execution_defaults_to_an_implicit_network(tmp_path):
    with pytest.raises(SystemExit, match="explicit --bundle"):
        main(["sim", "--out-dir", str(tmp_path)])
    assert not list(tmp_path.iterdir())


def test_graph_cli_writes_named_artifacts_without_legacy_bookkeeping(tmp_path):
    bundle = ping_classifier().write(tmp_path / "network.bundle")
    output = tmp_path / "run"
    assert (
        main(
            [
                "sim",
                "--bundle",
                str(bundle),
                "--poisson-protocol",
                "fixed-rate",
                "--t-ms",
                "0.2",
                "--n-batch",
                "1",
                "--device",
                "cpu",
                "--out-dir",
                str(output),
            ]
        )
        == 0
    )
    assert {p.name for p in output.iterdir()} == {
        "outputs.npz",
        "recording.npz",
        "parameters.npz",
        "metrics.json",
        "inference-manifest.json",
    }
    with np.load(output / "outputs.npz", allow_pickle=False) as payload:
        assert payload["class_logits"].shape == (1, 10)
    assert json.loads((output / "metrics.json").read_text())["device"] == "cpu"


def test_retired_commands_and_abbreviated_switches_are_rejected():
    for argv in (["dump-weights"], ["sim", "--diag"], ["sim", "--executor", "legacy"]):
        with pytest.raises(SystemExit):
            parse_args(argv)


@pytest.mark.parametrize("epochs", [None, 0, 1])
def test_training_uses_recipe_epochs_and_honors_explicit_zero(tmp_path, epochs):
    bundle = direct_train_bundle()
    bundle.training["epochs"] = 2
    for entry in bundle.manifest["files"]:
        if entry["path"] == "training.json":
            entry["digest"] = digest(bundle.training)
    network = bundle.write(tmp_path / "network.bundle")
    inputs, targets = tmp_path / "inputs.npz", tmp_path / "targets.npz"
    np.savez(inputs, events=np.ones((3, 2, 2), dtype=np.float32))
    np.savez(targets, label=np.array([0, 1], dtype=np.int64))
    output = tmp_path / "trained"
    args = [
        "train",
        "--bundle",
        str(network),
        "--input-file",
        str(inputs),
        "--target-file",
        str(targets),
        "--device",
        "cpu",
        "--out-dir",
        str(output),
    ]
    if epochs is not None:
        args += ["--epochs", str(epochs)]
    assert main(args) == 0
    metrics = json.loads((output / "metrics.json").read_text())
    expected_epochs = bundle.training["epochs"] if epochs is None else epochs
    assert len(metrics["updates"]) == max(1, expected_epochs)
