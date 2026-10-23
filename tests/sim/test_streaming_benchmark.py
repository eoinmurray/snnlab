"""Timing bar charts and numerical checks for the streaming benchmark."""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/streaming/benchmark.py"
spec = importlib.util.spec_from_file_location("streaming_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize(
    "changes",
    [
        {"dt_ms": 0.3},
        {"windows": 0},
        {"windows": 1.5},
        {"sparsity": 1.0},
        {"rate_hz": float("nan")},
        {"hidden": 0},
        {"window_ms": float("inf")},
        {"block_steps": 65},
    ],
)
def test_invalid_conditions_fail_before_execution(changes):
    with pytest.raises(ValueError):
        replace(benchmark.Config(), **changes).validate()


@pytest.mark.parametrize("mode", ["eager", "default"])
def test_streaming_timing_bars_and_numerical_gate(tmp_path, monkeypatch, mode):
    original = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kw: original(fn, backend="eager", fullgraph=True, dynamic=False),
    )
    cfg = benchmark.Config(hidden=32, windows=3, compile_mode=mode)
    previous_threads = torch.get_num_threads()
    previous_precision = torch.get_float32_matmul_precision()
    try:
        report = benchmark.run(cfg, tmp_path / "samples")
    finally:
        torch.set_num_threads(previous_threads)
        torch.set_float32_matmul_precision(previous_precision)
    assert report["validation"]["passed"]
    assert report["validation"]["probe_spikes"]["hidden_spikes"] > 0
    expected = "compiled_blocks" if mode == "default" else "kernel_loop"
    assert report["validation"]["initial_execution_path"]["selected"] == expected
    phases = report["advance_breakdown"]
    assert phases["windows"] == 3
    assert phases["numerical_gate_passed"]
    assert phases["execution_path"]["selected"] == expected
    names = {row["name"] for row in phases["stages"]}
    assert "State restore / init" in names
    assert "Runtime-state snapshot" in names
    assert ("Compiled numerical block" in names) == (mode == "default")
    for data, key in ((phases, "phase_ns"), (report["breakdown"], "stage_ns")):
        with np.load(tmp_path / data["samples_file"]) as raw:
            ns = raw[key]
            assert ns.shape[0] == 3
            assert (ns >= 0).all()
            np.testing.assert_array_equal(ns[:, 0], ns[:, 1:].sum(axis=1))
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"cases": [report]}, allow_nan=False))
    for internal in (False, True):
        image = benchmark.plot_breakdown(report_path, internal=internal)
        assert image.read_bytes().startswith(b"\x89PNG")
    assert not list(tmp_path.glob("*.pstats"))
    assert not list(tmp_path.glob("*-trace.json"))
    # Redrawing uses the same single-file entry point and never executes a model.
    monkeypatch.setattr(
        benchmark, "run", lambda *a: pytest.fail("redraw executed model")
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--plot-only", str(tmp_path)])
    benchmark.main()
