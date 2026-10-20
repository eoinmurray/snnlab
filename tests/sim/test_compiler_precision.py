"""Actual Inductor must preserve intermediate reduced-dtype rounding."""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def execution_benchmark():
    path = Path(__file__).resolve().parents[2] / "benchmarks/execution/benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "execution_precision_regression", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("block_steps", [1, 4])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
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
def test_bfloat16_compilation_preserves_recurrent_gradients_and_adam(
    execution_benchmark, block_steps, device
):
    # With default Inductor cast elision this 128-step case changes every
    # drive-gradient element, with relative errors above 30% on CPU/CUDA.
    graph, trainable = execution_benchmark.circuit(32, 32, True)
    result = execution_benchmark.numerical_gate(
        graph, trainable, torch.device(device), block_steps, torch.bfloat16, 128, 32
    )
    assert result["passed"]
    assert result["compiler_precision_policy"] == "snnlab.eager-precision-casts/v1"


@pytest.mark.parametrize("block_steps", [1, 4])
def test_minimum_cpu_precision_fallback(execution_benchmark, block_steps, monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.10.0")
    graph, trainable = execution_benchmark.circuit(32, 32, True)
    result = execution_benchmark.numerical_gate(
        graph, trainable, torch.device("cpu"), block_steps, torch.bfloat16, 128, 32
    )
    assert result["passed"]
    assert result["compiler_fallback"] == (
        "PyTorch 2.10 reduced-precision CPU uses eager tensor graphs"
    )
