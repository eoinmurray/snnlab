"""Exercise real offline digits learning and portable native artifacts."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from snnlab.sim import DenseArrayBinding, load_training_checkpoint
from snnlab.sim.requests.handlers import build, simulate
from snnlab.sim.requests.spec import ExecutionSpec

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/scikit-digits/benchmark.py"
spec = importlib.util.spec_from_file_location("scikit_digits_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


def test_split_and_spike_replay_are_reproducible_and_disjoint():
    cfg = benchmark.Config()
    train, test, metadata = benchmark.prepare_data(cfg)
    again = benchmark.prepare_data(cfg)
    assert train[0].shape == (16, 1437, 64)
    assert test[0].shape == (16, 360, 64)
    train_indices, test_indices = metadata["train_indices"], metadata["test_indices"]
    assert not set(train_indices) & set(test_indices)
    assert sorted(train_indices + test_indices) == list(range(1797))
    assert metadata == again[2]
    for split, repeated in zip((train, test), again[:2], strict=True):
        torch.testing.assert_close(split[0], repeated[0], rtol=0, atol=0)
        torch.testing.assert_close(split[1], repeated[1], rtol=0, atol=0)
        assert set(split[0].unique().tolist()) == {0.0, 1.0}
        assert set(split[1].tolist()) == set(range(10))


@pytest.mark.integration
def test_default_learns_and_checkpoint_reproduces_final_accuracy(tmp_path):
    cfg = benchmark.Config()
    output = tmp_path / "digits"
    report = benchmark.run(cfg, output)
    assert report["summary"]["test_accuracy"] > 0.85
    assert report["summary"]["completed_updates"] == cfg.epochs * 12
    rows = report["metrics"]["epochs"]
    assert rows[-1]["validation_loss"] < rows[0]["validation_loss"]
    assert all(row["train_examples"] == 1437 for row in rows[1:])
    assert all(row["validation_examples"] == 360 for row in rows)
    assert report == json.loads((output / "results.json").read_text())

    checkpoint = load_training_checkpoint(output / "final.checkpoint")
    initial = build(
        ExecutionSpec(kind="build", bundle=output / "network.bundle", seed=cfg.seed)
    )
    for name, value in checkpoint.parameters.items():
        assert torch.isfinite(value).all()
        assert not torch.equal(value, initial.parameters[name])
    assert (checkpoint.parameters["pixels_to_hidden.weight"] >= 0).all()
    _, (spikes, labels), _ = benchmark.prepare_data(cfg)
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            bundle=output / "network.bundle",
            checkpoint=output / "final.checkpoint",
            device="cpu",
            input_bindings=(DenseArrayBinding(input_id="pixels", value=spikes),),
            diagnostics=False,
        )
    )
    accuracy = (
        (result.outputs["class_scores"].argmax(-1) == labels).float().mean().item()
    )
    assert accuracy == pytest.approx(report["summary"]["test_accuracy"])
    stale = output / "old-report.json"
    stale.write_text("{}")
    repeated = benchmark.run(cfg, output)
    assert not stale.exists()
    assert repeated["summary"]["test_accuracy"] == report["summary"]["test_accuracy"]


@pytest.mark.integration
def test_smoke_cli_uses_real_digits_and_writes_finite_metrics(tmp_path):
    output = tmp_path / "smoke"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--smoke", "--output", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((output / "results.json").read_text())
    assert report["dataset"]["smoke"] is True
    assert len(report["dataset"]["train_indices"]) == 100
    assert len(report["dataset"]["test_indices"]) == 40
    assert report["summary"]["completed_updates"] == 2
    assert report["configuration"]["steps"] == 8
    assert np.isfinite(report["metrics"]["epochs"][-1]["validation_loss"])
