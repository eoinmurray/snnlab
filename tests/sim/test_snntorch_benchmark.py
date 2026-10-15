"""Acceptance checks for the matched MNIST/Modal benchmark, without downloads."""

import importlib
import json
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("snntorch")
BENCHMARK = Path(__file__).resolve().parents[2] / "benchmarks" / "snntorch-comparison"
sys.path.insert(0, str(BENCHMARK))
comparison = runner = importlib.import_module("benchmark")
plotter = importlib.import_module("plot")
pytestmark = [pytest.mark.integration, pytest.mark.snntorch]


@pytest.mark.parametrize("alpha", [0.0, 0.9])
def test_tutorial_current_models_match_forward_backward_and_adam(alpha):
    config = comparison.Config(hidden=16, steps=8, alpha=alpha)
    result = comparison.verify_pair(config, comparison.initial_weights(config))
    assert result["passed"]
    assert set(result["gradient_errors"]) == {
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
        "inputs",
    }


@pytest.mark.parametrize("alpha", [0.0, 0.9])
def test_fast_projection_mode_preserves_strict_trajectory_gradients_and_update(alpha):
    config = comparison.Config(hidden=16, steps=8, alpha=alpha)
    weights = comparison.initial_weights(config)
    fast = comparison.LabNetwork(config, weights)
    strict = comparison.LabNetwork(config, weights)
    strict.executor = comparison.GraphExecutor(
        comparison.plan_graph(strict.graph),
        trainable_parameters=tuple(strict.executor.parameter_map()),
        surrogate=comparison.lang.CustomSurrogate("comparison.atan/v1").json(),
        projection_mode="strict",
        extension_validation="structural",
    )
    strict.executor.load_state_dict(fast.executor.state_dict())
    images = torch.rand(8, 784, generator=torch.Generator().manual_seed(31))
    labels = torch.arange(8)
    outputs = []
    for model in (fast, strict):
        spikes, voltage = model(images)
        outputs.append((spikes, voltage))
        comparison.loss_fn(voltage, labels).backward()
    for actual, expected in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for key, gradient in fast.canonical_gradients().items():
        torch.testing.assert_close(
            gradient, strict.canonical_gradients()[key], atol=0, rtol=0
        )
    for model in (fast, strict):
        torch.optim.Adam(model.parameters(), lr=config.lr).step()
    for key, parameter in fast.canonical_parameters().items():
        torch.testing.assert_close(
            parameter, strict.canonical_parameters()[key], atol=0, rtol=0
        )


def test_benchmark_forward_omits_unused_snapshots_and_diagnostics(monkeypatch):
    config = comparison.Config(hidden=16, steps=8)
    model = comparison.LabNetwork(config, comparison.initial_weights(config))
    original = model.executor.forward
    captured = []

    def forward(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(model.executor, "forward", forward)
    spikes, voltage = model(torch.zeros(2, 784))
    result = captured[0]
    assert spikes.shape == voltage.shape == (8, 2, 10)
    assert not result.parameters and not result.final_state and not result.diagnostics
    assert result.runtime_state is None
    assert result.metrics["projection_mode"] == "fast"


def test_subset_and_minibatch_order_are_reproducible_and_exhaustive():
    labels = torch.repeat_interleave(torch.arange(10), torch.arange(10) + 10)
    indices = comparison.stratified_indices(labels, 29, 17)
    assert len(indices) == len(indices.unique()) == 29
    assert torch.equal(indices, comparison.stratified_indices(labels, 29, 17))
    assert not torch.equal(indices, comparison.stratified_indices(labels, 29, 18))
    expected = torch.bincount(labels).double() * 29 / len(labels)
    assert torch.all(
        (torch.bincount(labels[indices], minlength=10) - expected).abs() <= 1
    )
    order = comparison.epoch_order(29, 17, 0)
    assert torch.equal(order.sort().values, torch.arange(29))
    assert not torch.equal(order, comparison.epoch_order(29, 17, 1))


def test_two_epoch_training_and_plot_artifacts_keep_partial_batches(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    config = comparison.Config(
        train_size=18, test_size=11, hidden=12, steps=6, batch_size=8, epochs=2
    )
    packet = runner.synthetic_packet(config)
    native_train = comparison.train
    calls = []

    def capture_train(execution):
        result = native_train(execution)
        calls.append((execution, result))
        return result

    monkeypatch.setattr(comparison, "train", capture_train)
    results = [
        runner.run_training(backend, config, packet, tmp_path / f"{backend}.pt")
        for backend in ("snntorch", "snnlab")
    ]
    assert len(calls) == 1
    execution, native_result = calls[0]
    assert isinstance(execution, comparison.ExecutionSpec)
    assert execution.epochs == 2 and execution.batch_size == 8 and execution.shuffle
    assert execution.validation is not None
    assert execution.training_timing
    assert execution.training["optimizer"]["kind"] == "custom_optimizer"
    assert native_result.training_checkpoint.completed_updates == 6
    assert [row["epoch"] for row in native_result.metrics["epochs"]] == [0, 1, 2]
    assert results[1]["training_api"] == "snnlab.sim.train"
    assert results[1]["train_call_seconds"] > 0
    assert (tmp_path / "snnlab.native/manifest.json").exists()
    left_checkpoint = torch.load(tmp_path / "snntorch.pt", weights_only=True)
    right_checkpoint = torch.load(tmp_path / "snnlab.pt", weights_only=True)
    for name, expected in left_checkpoint["parameters"].items():
        torch.testing.assert_close(
            right_checkpoint["parameters"][name], expected, atol=2e-6, rtol=2e-5
        )
    for left, right in zip(results[0]["updates"], results[1]["updates"]):
        assert left["train_loss"] == pytest.approx(right["train_loss"], abs=2e-4)
    for result in results:
        assert result["complete"] and result["synthetic"]
        timing = result["training_timing"]
        phases = (
            "setup",
            "training",
            "train_evaluation",
            "validation",
            "checkpoint",
            "reporting",
        )
        assert timing["total_s"] == pytest.approx(
            sum(timing[f"{phase}_s"] for phase in phases)
        )
        assert all(timing[f"{phase}_s"] >= 0 for phase in phases)
        assert len(timing["epochs"]) == 3
        assert timing["training_s"] == pytest.approx(
            sum(row["train_seconds"] for row in result["epochs"])
        )
        assert all(
            row["train_seconds"] > 0 and row["test_seconds"] > 0
            for row in result["epochs"]
        )
        assert len(result["updates"]) == 6
        assert result["source_identity"]["commit"]
        assert result["source_identity"]["snnlab_version"] == comparison.snnlab_version
        assert result["snnlab_execution"]["projection_mode"] == "fast"
        checkpoint = torch.load(tmp_path / f"{result['backend']}.pt", weights_only=True)
        assert checkpoint["source_identity"] == result["source_identity"]
        assert all(
            row["train_examples"] == 18 and row["test_examples"] == 11
            for row in result["epochs"]
        )
    for left, right in zip(results[0]["epochs"], results[1]["epochs"]):
        assert left["train_loss"] == pytest.approx(right["train_loss"], abs=2e-4)
        assert left["test_loss"] == pytest.approx(right["test_loss"], abs=2e-4)
        assert left["test_accuracy"] == right["test_accuracy"]
    report = runner.combined_results(dict(zip(("snntorch", "snnlab"), results)))
    runner.write_json(tmp_path / "results.json", report)
    png = plotter.make_plots(tmp_path)
    assert png.read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "training-curves.pdf").read_bytes().startswith(b"%PDF")
    mismatched = results[1]
    mismatched["config"]["seed"] += 1
    runner.write_json(tmp_path / "results.json", report)
    with pytest.raises(ValueError, match="config"):
        plotter.make_plots(tmp_path)


def test_minimal_layout_and_retained_results():
    assert sorted(path.name for path in BENCHMARK.glob("*.py")) == [
        "benchmark.py",
        "modal_app.py",
        "plot.py",
    ]
    assert sorted(path.name for path in (BENCHMARK / "results").iterdir()) == [
        "results.json",
        "training-curves.pdf",
        "training-curves.png",
    ]
    report = json.loads((BENCHMARK / "results/results.json").read_text())
    for data in report["training"].values():
        assert len(data["epochs"]) == 20 and len(data["updates"]) == 940
    if report["profiling"] is not None:
        assert (
            sum(
                len(row["samples_ms"])
                for state in report["profiling"]["states"].values()
                for rows in state["timings"].values()
                for row in rows.values()
            )
            == 900
        )
    else:
        assert (
            report["training"]["snnlab"]["training_api"] == "snnlab.sim.train"
        )
        assert report["cloud"]["native_training_checkpoint"].endswith("snnlab.native")


def test_runtime_gate_accepts_newer_release_and_checks_timing_api(monkeypatch):
    monkeypatch.setattr(comparison, "snnlab_version", "0.13.0")
    comparison.require_optimized_runtime()
    monkeypatch.setattr(comparison, "snnlab_version", "0.12.0")
    with pytest.raises(RuntimeError, match="requires SNNLab"):
        comparison.require_optimized_runtime()
    monkeypatch.setattr(comparison, "snnlab_version", "0.13.0")
    monkeypatch.setattr(comparison, "ExecutionSpec", lambda: None)
    with pytest.raises(RuntimeError, match="training_timing"):
        comparison.require_optimized_runtime()
