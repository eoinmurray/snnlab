"""SHD benchmark acceptance checks without downloading or launching GPU work."""

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from snnlab.sim import load_training_checkpoint
from snnlab.sim.requests.handlers import build
from snnlab.sim.requests.spec import ExecutionSpec

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = ROOT / "benchmarks/shd"
sys.path.insert(0, str(HERE))

runner = importlib.import_module("benchmarks.shd.benchmark")
data = recipe = runner


def test_default_recipe_and_full_data_tail():
    cfg = recipe.Config()
    bundle = recipe.author_network(cfg)
    assert (cfg.epochs, cfg.batch_size, cfg.steps) == (10, 256, 1000)
    assert [(p["id"], p["size"]) for p in bundle.graph["populations"]] == [
        ("E1", 128),
        ("E2", 128),
        ("readout", 20),
    ]
    assert all(p["connection"] == "feedforward" for p in bundle.graph["projections"])
    assert bundle.training["resolved_parameters"]["frozen"] == ["score_gain.weight"]
    spec = runner.execution_kwargs(
        cfg,
        bundle,
        Path("spikes.npz"),
        {"kind": "SHD", "train_count": 7340},
        Path("out"),
        "cpu",
        None,
    )
    assert spec["metrics"].train == "online"
    assert spec["batching"].remainder == "merge_small"
    assert spec["batching"].min_batch_size == 20
    assert divmod(7340, cfg.batch_size) == (28, 172)
    assert spec["input_bindings"][0].encoder.config["sample_offset"] == 0
    assert spec["validation"].input_bindings[0].encoder.config["sample_offset"] == 7340


def test_split_retains_uneven_classes_without_overlap():
    cfg = recipe.Config()
    labels = np.repeat(np.arange(20), np.arange(20) + 21)
    train, validation = data.select_indices(labels, cfg)
    assert not set(train) & set(validation)
    np.testing.assert_array_equal(
        np.sort(np.concatenate((train, validation))), np.arange(len(labels))
    )
    again = data.select_indices(labels, cfg)
    np.testing.assert_array_equal(train, again[0])
    for label in range(20):
        assert (labels[validation] == label).sum() == round(
            (labels == label).sum() * 0.1
        )


def test_ragged_preparation_and_encoder_binary_or_and_split_offset(tmp_path):
    cfg = recipe.Config(steps=3, classes=2, channels=4)
    path = tmp_path / "train.h5"
    with h5py.File(path, "w") as handle:
        handle["labels"] = np.repeat(np.arange(2), 10)
        times = handle.create_dataset(
            "spikes/times", (20,), dtype=h5py.vlen_dtype(np.dtype("float64"))
        )
        channels = handle.create_dataset(
            "spikes/units", (20,), dtype=h5py.vlen_dtype(np.dtype("int64"))
        )
        for index in range(20):
            times[index] = [0.0001, 0.0009, 0.002, 0.003]
            channels[index] = [index % 4, index % 4, 1, 2]
    output = tmp_path / "spikes.npz"
    metadata = data.prepare(path, output, cfg)
    assert (metadata["train_count"], metadata["validation_count"]) == (18, 2)
    assert metadata["dropped_events"] == 20
    with np.load(output) as loaded:
        arrays = dict(loaded)
    config = {
        "steps": 3,
        "sample_offset": 18,
        "input_noise_hz": 0,
        "spike_deletion_probability": 0,
    }
    encoded = data.encode_events(
        arrays, [0, 1], dt_ms=1, channels=4, config=config, seed=46
    )
    assert encoded.shape == (3, 2, 4)
    assert encoded.sum().item() == 4
    for column in range(2):
        index = arrays["official_train_indices"][18 + column]
        assert encoded[0, column, index % 4] == 1
        assert encoded[2, column, 1] == 1


def test_encoder_noise_is_seeded_and_uses_original_occupancy():
    arrays = {
        "event_offsets": np.array([0, 1]),
        "event_time_ms": np.array([0.0]),
        "event_channel": np.array([0]),
    }
    config = {
        "steps": 4,
        "sample_offset": 0,
        "input_noise_hz": 1000,
        "spike_deletion_probability": 1,
    }
    value = data.encode_events(arrays, [0], dt_ms=1, channels=2, config=config, seed=44)
    assert value[0, 0, 0] == 0  # Deleted occupied bins are not reinserted.
    assert value.sum() == 7
    torch.testing.assert_close(
        value,
        data.encode_events(arrays, [0], dt_ms=1, channels=2, config=config, seed=44),
    )


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_device_encoder_preserves_binning_offsets_and_original_occupancy(device):
    arrays = {
        "event_offsets": np.array([0, 1, 3, 3, 6]),
        "event_time_ms": np.array([0.0, 1.0, 1.9, 0.0, 0.9, 3.0]),
        "event_channel": np.array([0, 1, 1, 0, 0, 1]),
    }
    config = {
        "steps": 3,
        "sample_offset": 1,
        "input_noise_hz": 0,
        "spike_deletion_probability": 0,
    }
    selected = [2, 0, 1]
    expected = data.encode_events(
        arrays, selected, dt_ms=1, channels=2, config=config, seed=44
    )
    actual = data.encode_events(
        arrays, selected, dt_ms=1, channels=2, config=config, seed=44, device=device
    )
    assert actual.device.type == device
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    noisy = {**config, "input_noise_hz": 1000, "spike_deletion_probability": 1}
    value = data.encode_events(
        arrays, selected, dt_ms=1, channels=2, config=noisy, seed=44, device=device
    )
    torch.testing.assert_close(value.cpu(), 1 - expected, atol=0, rtol=0)
    noisy = {**config, "input_noise_hz": 100, "spike_deletion_probability": 0.1}
    a = data.encode_events(
        arrays, selected, dt_ms=1, channels=2, config=noisy, seed=44, device=device
    )
    b = data.encode_events(
        arrays, selected, dt_ms=1, channels=2, config=noisy, seed=44, device=device
    )
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_bad_cached_archive_is_rejected_without_network(tmp_path):
    (tmp_path / "shd_train.h5.gz").write_bytes(b"not SHD")
    with pytest.raises(ValueError, match="cached SHD checksum mismatch"):
        data.acquire(tmp_path)


def test_native_smoke_updates_all_learned_weights_and_exports_portable_bundle(tmp_path):
    cfg = replace(recipe.Config(), epochs=2, steps=64)
    output = tmp_path / "run"
    summary = runner.run(
        cfg, output, tmp_path / "unused-cache", device="cpu", smoke=True
    )
    assert summary["completed_updates"] == 2
    assert summary["kind"] == "synthetic-smoke"
    checkpoint = load_training_checkpoint(output / "final.checkpoint")
    bundle = recipe.author_network(cfg)
    initial = build(
        ExecutionSpec(
            kind="build",
            graph=bundle.graph,
            training=bundle.training,
            seed=cfg.seed,
            device="cpu",
        )
    )
    for key in recipe.TRAINABLE:
        assert torch.isfinite(checkpoint.parameters[key]).all()
        assert not torch.equal(initial.parameters[key], checkpoint.parameters[key])
    torch.testing.assert_close(
        initial.parameters["score_gain.weight"],
        checkpoint.parameters["score_gain.weight"],
        rtol=0,
        atol=0,
    )
    timing = json.loads((output / "timing.json").read_text())
    assert timing["training_s"] > 0
    assert timing["validation_s"] > 0
    assert timing["train_evaluation_s"] == 0
    code = "from snnlab.sim import build; import sys; r=build(\n    bundle=sys.argv[1],\n    device='cpu',\n); assert len(r.parameters)==4"
    subprocess.run(
        [sys.executable, "-c", code, str(output / "network.bundle")],
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(HERE))),
        },
    )
    stale = output / "old-report.json"
    stale.write_text("{}")
    repeated = runner.run(cfg, output, tmp_path, device="cpu", smoke=True)
    assert not stale.exists()
    assert repeated["completed_updates"] == summary["completed_updates"]
    assert (output / "final.checkpoint").is_dir()


def test_benchmark_tracing_is_a_separate_native_call_and_preserves_baseline(tmp_path):
    cfg = replace(recipe.Config(), epochs=2, steps=16)
    baseline_dir, traced_dir = tmp_path / "baseline", tmp_path / "traced"
    baseline = runner.run(
        cfg, baseline_dir, tmp_path / "cache", device="cpu", smoke=True
    )
    traced = runner.run(
        cfg,
        traced_dir,
        tmp_path / "cache",
        device="cpu",
        smoke=True,
        trace_mode="both",
        trace_skip_updates=0,
        trace_updates=1,
    )
    assert baseline["completed_updates"] == traced["completed_updates"] == 2
    assert "diagnostic_trace" not in baseline
    assert traced["diagnostic_trace"]["status"] == "complete"
    first = load_training_checkpoint(baseline_dir / "final.checkpoint")
    second = load_training_checkpoint(traced_dir / "final.checkpoint")
    for name in first.parameters:
        torch.testing.assert_close(
            first.parameters[name], second.parameters[name], rtol=0, atol=0
        )
    baseline_metrics = json.loads((traced_dir / "metrics.json").read_text())
    diagnostic = json.loads((traced_dir / "diagnostic_metrics.json").read_text())
    assert "tracing" not in baseline_metrics
    assert len(diagnostic["tracing"]["captured_updates"]) == 1
    assert diagnostic["training_timing"]["validation_s"] == 0
    trace = json.loads((traced_dir / "tracing/trace.json").read_text())
    labels = {event.get("name") for event in trace["traceEvents"]}
    assert "snnlab::neuron/E1" in labels
    assert "snnlab::projection/input_to_E1" in labels
    assert "snnlab::encoding" in labels
    assert "snnlab::transfer" in labels
