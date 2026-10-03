"""Mode-drift: "same inputs → same outputs across modes".

In-process invariants are fast (default). CLI propagation tests spawn
subprocesses and are marked `slow`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

# ── In-process invariants (fast) ─────────────────────────────────────────


def _equal_state_dicts(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_build_net_deterministic_ping():
    from snnlab.sim.config import build_net

    torch.manual_seed(0)
    a = build_net(
        "ping",
        w_in=(0.3, 0.03),
        w_in_initial_zero_fraction=0.95,
        ei_strength=0.5,
        ei_ratio=2.0,
        recurrent_initial_zero_fraction=0.2,
    )
    torch.manual_seed(0)
    b = build_net(
        "ping",
        w_in=(0.3, 0.03),
        w_in_initial_zero_fraction=0.95,
        ei_strength=0.5,
        ei_ratio=2.0,
        recurrent_initial_zero_fraction=0.2,
    )
    assert _equal_state_dicts(a, b)


def test_encode_images_poisson_deterministic():
    from snnlab.sim.tool import encode_images_poisson

    images = torch.rand(4, 64)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = encode_images_poisson(
        images, T_steps=200, dt=0.25, max_rate_hz=10.0, generator=g1
    )
    b = encode_images_poisson(
        images, T_steps=200, dt=0.25, max_rate_hz=10.0, generator=g2
    )
    assert torch.equal(a, b)


def test_load_dataset_deterministic_mnist():
    from snnlab.sim.tool import load_dataset

    a_tr, a_te, ay_tr, ay_te = load_dataset("mnist", max_samples=200, split=True)
    b_tr, b_te, by_tr, by_te = load_dataset("mnist", max_samples=200, split=True)
    assert np.array_equal(a_tr, b_tr)
    assert np.array_equal(a_te, b_te)
    assert np.array_equal(ay_tr, by_tr)
    assert np.array_equal(ay_te, by_te)


def test_validation_and_official_test_are_distinct_mnist():
    from snnlab.sim.tool import load_dataset

    _, validation_x, _, validation_y = load_dataset(
        "mnist", max_samples=500, split=True, evaluation_split="validation"
    )
    _, test_x, _, test_y = load_dataset(
        "mnist", max_samples=500, split=True, evaluation_split="test"
    )
    assert len(validation_y) == 50
    assert len(test_y) == 10_000
    assert not np.array_equal(validation_x, test_x)


# ── CLI propagation (slow) ───────────────────────────────────────────────


def _run_cli(*args, timeout=180):
    cmd = ["uv", "run", "python", "-m", "snnlab.sim", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr


def _train_probe(tmp_dir, **extra):
    args = [
        "train",
        "--model",
        "ping",
        "--dataset",
        "mnist",
        "--max-samples",
        "100",
        "--epochs",
        "0",
        "--dt",
        "0.25",
        "--w-in",
        "10",
        "--w-in-initial-zero-fraction",
        "0",
        "--out-dir",
        str(tmp_dir),
        "--wipe-dir",
    ]
    for k, v in extra.items():
        args.extend([f"--{k.replace('_', '-')}", str(v)])
    rc, _, _ = _run_cli(*args)
    assert rc == 0
    metrics_path = Path(tmp_dir) / "metrics.json"
    assert metrics_path.exists()
    return json.loads(metrics_path.read_text())


def _read_config(tmp_dir):
    cfg_path = Path(tmp_dir) / "config.json"
    assert cfg_path.exists(), f"no config.json in {tmp_dir}"
    return json.loads(cfg_path.read_text())


@pytest.mark.slow
def test_input_rate_propagates_train(tmp_path):
    expected = 33.0
    metrics = _train_probe(tmp_path / "ir-train", **{"input-rate": expected})
    assert float(metrics["config"]["input_rate"]) == expected


@pytest.mark.slow
def test_t_ms_propagates_train(tmp_path):
    expected = 150.0
    metrics = _train_probe(tmp_path / "tms-train", **{"t-ms": expected})
    assert float(metrics["config"]["t_ms"]) == expected


@pytest.mark.slow
def test_train_selects_on_validation_then_infer_uses_official_test(tmp_path):
    """Training and inference use validation and official-test data respectively."""
    train_dir = tmp_path / "match-train"
    rc, _, _ = _run_cli(
        "train",
        "--model",
        "ping",
        "--dataset",
        "mnist",
        "--max-samples",
        "200",
        "--epochs",
        "2",
        "--dt",
        "0.25",
        "--w-in",
        "10",
        "--w-in-initial-zero-fraction",
        "0",
        "--ei-strength",
        "0",
        "--out-dir",
        str(train_dir),
        "--wipe-dir",
    )
    assert rc == 0
    train_metrics = json.loads((train_dir / "metrics.json").read_text())
    assert train_metrics["best_acc"] is not None
    split = train_metrics["config"]["dataset_split"]
    assert split["optimizer_train_samples"] == 180
    assert split["validation_samples"] == 20
    assert split["official_test_used_during_training"] is False

    infer_dir = tmp_path / "match-infer"
    rc, _, _ = _run_cli(
        "sim",
        "--infer",
        "--model",
        "ping",
        "--dataset",
        "mnist",
        "--max-samples",
        "200",
        "--dt",
        "0.25",
        "--w-in",
        "10",
        "--w-in-initial-zero-fraction",
        "0",
        "--ei-strength",
        "0",
        "--load-weights",
        str(train_dir / "weights.pth"),
        "--out-dir",
        str(infer_dir),
        "--wipe-dir",
    )
    assert rc == 0
    infer_metrics = json.loads((infer_dir / "metrics.json").read_text())
    assert infer_metrics["n_total"] == 200
    assert infer_metrics["config"]["evaluation_partition"] == "official_mnist_test"
