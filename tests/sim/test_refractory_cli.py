"""Physical timing survives public CLI dispatch and saved configurations."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
import torch

from snnlab.sim import infer as infer_module
from snnlab.sim import models as M
from snnlab.sim import tool
from snnlab.sim import train as train_module


def test_refractory_load_precedence_and_historical_missing_fields(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"dt": 0.1}))
    old = tool.parse_args(["sim", "--load-config", str(path)])
    assert (old.refractory_e_ms, old.refractory_i_ms) == (3.0, 1.5)
    assert old.refractory_policy == "nearest"
    path.write_text(json.dumps({
        "dt": 0.1, "refractory_e_ms": 1.2, "refractory_i_ms": 0.6,
        "refractory_policy": "exact",
    }))
    saved = tool.parse_args(["sim", "--load-config", str(path)])
    assert (saved.refractory_e_ms, saved.refractory_i_ms) == (1.2, 0.6)
    assert saved.refractory_policy == "exact"
    override = tool.parse_args([
        "sim", "--load-config", str(path), "--refractory-e-ms=2.4",
        "--refractory-policy", "nearest",
    ])
    assert (override.refractory_e_ms, override.refractory_i_ms) == (2.4, 0.6)
    assert override.refractory_policy == "nearest"


@pytest.mark.parametrize("handler,target,extra", [
    (tool._run_train, "train", []),
    (tool._emit_infer, "infer", []),
    (tool._emit_infer, "infer_and_snapshot", [True]),
    (tool._emit_probe, "probe", []),
    (tool._run_dump_weights, "dump_weights", []),
])
def test_dispatch_passes_physical_refractory(handler, target, extra, monkeypatch, tmp_path):
    captured = {}

    def receive(**kwargs):
        captured.update(kwargs)
        return {"acc": 0.0}

    monkeypatch.setattr(tool, target, receive)
    args = tool.parse_args([
        "train" if target == "train" else "sim",
        "--dt", "0.3", "--refractory-e-ms", "1.2",
        "--refractory-i-ms", "0.6", "--refractory-policy", "exact",
    ])
    handler(args, None, tmp_path, logging.getLogger(), *extra)
    assert captured["refractory_e_ms"] == 1.2
    assert captured["refractory_i_ms"] == 0.6
    assert captured["refractory_policy"] == "exact"


def test_tiny_training_saves_resolved_timing_without_dataset_download(monkeypatch, tmp_path):
    x = np.zeros((10, 784), dtype=np.float32)
    y = np.arange(10, dtype=np.int64)
    monkeypatch.setattr(train_module, "load_dataset", lambda *a, **k: (x, x, y, y))
    monkeypatch.setattr(train_module, "_auto_device", lambda: torch.device("cpu"))
    train_module.train(
        dt=0.3, t_ms=2.0, epochs=0, hidden_sizes=[4], out_dir=tmp_path,
        snapshot_init=False, snapshot_end=False, refractory_e_ms=1.2,
        refractory_i_ms=0.6, refractory_policy="exact", seed=42,
    )
    saved = json.loads((tmp_path / "config.json").read_text())
    metrics = json.loads((tmp_path / "metrics.json").read_text())["config"]
    for data in (saved, metrics):
        assert data["t_ms"] == data["nominal_duration_ms"] == 2.0
        assert data["duration_steps"] == 6
        assert data["realized_duration_ms"] == pytest.approx(1.8)
        assert data["refractory_e_steps"] == 4
        assert data["refractory_i_steps"] == 2
        assert data["refractory_policy"] == "exact"
    loaded = tool.parse_args(["sim", "--load-config", str(tmp_path / "config.json")])
    assert (loaded.refractory_e_ms, loaded.refractory_i_ms) == (1.2, 0.6)


@pytest.mark.parametrize("input_steps", [None, 4])
def test_probe_records_actual_input_length_and_refractory(monkeypatch, tmp_path, input_steps):
    monkeypatch.setattr(infer_module, "_auto_device", lambda: torch.device("cpu"))
    build_net = infer_module.build_net
    built = []

    def capture_net(*args, **kwargs):
        net = build_net(*args, **kwargs)
        built.append(net)
        return net

    monkeypatch.setattr(infer_module, "build_net", capture_net)
    input_file = None
    if input_steps is not None:
        input_file = tmp_path / "input.npz"
        np.savez(input_file, input_spikes=np.zeros((input_steps, 4), dtype=np.float32))
    infer_module.probe(
        dt=0.3, t_ms=2.0, hidden_sizes=[4], n_in=4, n_batch=1,
        out_dir=tmp_path, refractory_e_ms=1.2, refractory_i_ms=0.6,
        refractory_policy="exact", seed=42, input_file=input_file,
    )
    expected_steps = input_steps if input_steps is not None else 6
    for metadata in (
        json.loads((tmp_path / "metrics.json").read_text())["config"],
        built[0].timing_metadata,
    ):
        assert metadata["duration_steps"] == M.T_steps == expected_steps
        assert metadata["realized_duration_ms"] == pytest.approx(expected_steps * 0.3)
        assert metadata["nominal_duration_ms"] == M.T_ms == 2.0
        assert (metadata["refractory_e_steps"], metadata["refractory_i_steps"]) == (4, 2)


def test_cli_records_supplied_input_duration_separately_from_request(monkeypatch, tmp_path):
    monkeypatch.setenv("PINGLAB_DEVICE", "cpu")
    monkeypatch.setenv("PINGLAB_NO_COMPILE", "1")
    source = tmp_path / "input.npz"
    np.savez(source, input_spikes=np.zeros((4, 4), dtype=np.float32))
    out = tmp_path / "result"
    assert tool.main([
        "sim", "--input-file", str(source), "--out-dir", str(out),
        "--dt", "0.3", "--t-ms", "2", "--n-hidden", "4", "--n-in", "4",
        "--refractory-e-ms", "1.2", "--refractory-i-ms", "0.6",
        "--refractory-policy", "exact",
    ]) == 0
    for metadata in (
        json.loads((out / "config.json").read_text()),
        json.loads((out / "metrics.json").read_text())["config"],
    ):
        assert metadata["nominal_duration_ms"] == metadata["t_ms"] == 2.0
        assert metadata["duration_steps"] == 4
        assert metadata["realized_duration_ms"] == pytest.approx(1.2)
        assert metadata["refractory_e_steps"] == 4
        assert metadata["refractory_i_steps"] == 2
