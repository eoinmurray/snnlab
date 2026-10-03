"""End-to-end smoke tests — spawn cli.py subprocesses in each mode.

Marked `slow` because each test launches a fresh `uv run` process.
Run with: `uv run pytest -m slow`
"""

from __future__ import annotations

import subprocess

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

OSC = "uv run python -m snnlab.sim"


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("smoke")


def _run(cmd, timeout=120):
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )
    assert result.returncode == 0, (
        f"cmd failed (exit {result.returncode}):\n  {cmd}\n"
        f"stderr: {result.stderr[-500:]}"
    )


# ── pinglab-cli modes ───────────────────────────────────────────────────


def test_sim_no_output(out_dir):
    _run(f"{OSC} sim --out-dir {out_dir}")


# ── Training modes ───────────────────────────────────────────────────────


@pytest.mark.parametrize("dataset", ["mnist"])
def test_train_one_epoch(out_dir, dataset):
    _run(
        f"{OSC} train --epochs 1 --dataset {dataset} --n-hidden 64 "
        f"--max-samples 50 --out-dir {out_dir}/train-{dataset}"
    )


def test_train_epochs_zero_probe(out_dir):
    _run(f"{OSC} train --epochs 0 --n-hidden 64 --out-dir {out_dir}/train-init")


def test_train_ping(out_dir):
    _run(
        f"{OSC} train --epochs 2 --ei-strength 0.5 --n-hidden 64 "
        f"--max-samples 50 --v-grad-dampen 1000 --out-dir {out_dir}/train-ping"
    )
