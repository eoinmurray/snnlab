"""Elevate's custom state must survive compiled streaming chunk boundaries."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = (
    Path(__file__).resolve().parents[2] / "benchmarks/streaming-elevate/benchmark.py"
)
spec = importlib.util.spec_from_file_location("elevate_benchmark_test", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize("mode", ["eager", "default"])
def test_elevate_streaming_state_and_instrumentation(monkeypatch, mode):
    original = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kw: original(fn, backend="eager", fullgraph=True, dynamic=False),
    )
    cfg = benchmark.Config(
        hidden=8,
        inputs=4,
        readouts=2,
        windows=3,
        warmup_windows=100,
        compile_mode=mode,
    )
    threads = torch.get_num_threads()
    precision = torch.get_float32_matmul_precision()
    try:
        result = benchmark.run(cfg)
    finally:
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision(precision)
    assert result["validation"]["passed"]
    assert result["validation"]["probe_spikes"]["hidden_spikes"] > 0
    assert result["validation"]["probe_spikes"]["readout_spikes"] > 0
    assert result["advance_breakdown"]["numerical_gate_passed"]
    path = result["validation"]["continuation_execution_path"]["selected"]
    # Custom neurons currently select the kernel loop even with compilation on.
    assert path == "kernel_loop"
    if mode == "default":
        assert (
            "registered neuron, synapse or surrogate"
            in result["validation"]["continuation_execution_path"]["fallback_reasons"]
        )
    assert result["deadlines"]["completed_steps"] == cfg.steps * cfg.windows
    graph = benchmark.circuit(cfg)
    for population in graph["populations"]:
        assert population["neuron"]["kind"] == "custom_neuron"
