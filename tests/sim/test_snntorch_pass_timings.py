"""Pass-timing smoke validation without Modal or MNIST downloads."""

import importlib
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("snntorch")
BENCHMARK = Path(__file__).resolve().parents[2] / "benchmarks" / "snntorch-comparison"
sys.path.insert(0, str(BENCHMARK))
comparison = timing = importlib.import_module("benchmark")
pytestmark = [pytest.mark.integration, pytest.mark.snntorch]


def test_isolated_pass_sampler_uses_fresh_graphs_without_changing_weights():
    config = comparison.Config(hidden=8, steps=3)
    weights = comparison.initial_weights(config)
    models = {
        "snntorch": comparison.TorchNetwork(config, weights),
        "snnlab": comparison.LabNetwork(config, weights),
    }
    images = torch.rand(2, 784, generator=torch.Generator().manual_seed(3))
    labels = torch.tensor([0, 1])
    assert timing.compare_outputs(models, images, labels)[
        "finite_outputs_and_gradients"
    ]
    result = timing.measure_passes(models, images, labels, warmups=1, repetitions=3)
    for backend, model in models.items():
        assert set(result[backend]) == {
            "training_forward",
            "backward",
            "inference_forward",
        }
        for row in result[backend].values():
            assert len(row["samples_ms"]) == 3
            assert 0 < row["q1_ms"] <= row["median_ms"] <= row["q3_ms"]
        for key, value in model.canonical_parameters().items():
            torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="unknown timing"):
        timing.timed_pass(models["snnlab"], images, labels, "optimizer")
