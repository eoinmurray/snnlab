"""Sustained float32 training acceptance, distinct from checkpoint equivalence."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from snnlab.sim import extensions as X

snn = pytest.importorskip("snntorch")
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "zero_decay_training_regression", ROOT / "benchmarks" / "zero_decay.py"
)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)
pytestmark = [pytest.mark.integration, pytest.mark.snntorch]


class MatchedReference(benchmark.TorchModel):
    """Match bias projection backward as well as the forward equations."""

    def forward(self, events):
        synaptic = events.new_zeros(events.shape[1], 256)
        voltage = torch.zeros_like(synaptic)
        out_synaptic = events.new_zeros(events.shape[1], 10)
        out_voltage = torch.zeros_like(out_synaptic)
        ones = events.new_ones(events.shape[1], 1)
        spikes = []
        for frame in events:
            hidden, synaptic, voltage = self.hidden(
                frame @ self.values["input_weight"] + ones @ self.values["input_bias"],
                synaptic,
                voltage,
            )
            output, out_synaptic, out_voltage = self.output(
                hidden @ self.values["output_weight"]
                + ones @ self.values["output_bias"],
                out_synaptic,
                out_voltage,
            )
            spikes.append(output)
        return torch.stack(spikes).sum(0)


@pytest.mark.parametrize("mode", ["strict", "fast"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.accelerator,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA unavailable"
                ),
            ],
        ),
    ],
)
def test_float32_spiking_training_matches_for_forty_clipped_adam_updates(
    mode, device, monkeypatch
):
    """Compare gradients, moments and weights at every independently evolved step.

    The diagnostic surrogate intervention matches snnTorch's division instead
    of reciprocal multiplication. The real registered-surrogate benchmark is
    separately required to pass the independent-training acceptance gate.
    """
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.random.fork_rng(devices=[] if device == "cpu" else [0]):
            weights = {k: v.to(device) for k, v in benchmark.initial_weights().items()}
            reference = MatchedReference(weights).to(device)
            with monkeypatch.context() as patch:
                patch.setattr(
                    X, "make_spike", lambda **kwargs: snn.surrogate.fast_sigmoid(25)
                )
                candidate = benchmark.GraphModel(weights, mode, False).to(device)
            models = [reference, candidate]
            optimizers = [torch.optim.Adam(m.parameters(), lr=0.001) for m in models]
            generator = torch.Generator().manual_seed(42)
            features = torch.rand(40, 8, 784, generator=generator).to(device)
            labels = torch.randint(10, (40, 8), generator=generator).to(device)
            saw_spikes = saw_clipping = False
            for step in range(40):
                events = benchmark.encode(features[step], 42 + step)
                scores, losses, gradients = [], [], []
                for model, optimizer in zip(models, optimizers):
                    optimizer.zero_grad(set_to_none=True)
                    score = model(events)
                    loss = torch.nn.functional.cross_entropy(score, labels[step])
                    loss.backward()
                    scores.append(score.detach())
                    losses.append(loss.detach())
                    gradients.append(
                        {
                            k: v.grad.clone()
                            for k, v in model.parameters_by_name().items()
                        }
                    )
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    saw_clipping |= bool(norm > 1)
                    optimizer.step()
                saw_spikes |= bool(scores[0].sum() > 0)
                torch.testing.assert_close(scores[0], scores[1], atol=0, rtol=0)
                torch.testing.assert_close(losses[0], losses[1], atol=0, rtol=0)
                for name, ref in reference.parameters_by_name().items():
                    actual = candidate.parameters_by_name()[name]
                    torch.testing.assert_close(
                        gradients[0][name],
                        gradients[1][name],
                        atol=0,
                        rtol=0,
                        msg=f"gradient at update {step}: {name}",
                    )
                    torch.testing.assert_close(ref, actual, atol=0, rtol=0)
                    for key in ("step", "exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(
                            optimizers[0].state[ref][key],
                            optimizers[1].state[actual][key],
                            atol=0,
                            rtol=0,
                        )
            assert saw_spikes and saw_clipping
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cuda.matmul.allow_tf32 = tf32


def test_original_accuracy_gap_fails_despite_shared_checkpoint_agreement():
    """Retained full-run evidence must fail a training-equivalence assertion."""
    report = json.loads((ROOT / "benchmarks/zero-decay/results.json").read_text())
    candidate = report["candidate"]
    assert candidate["agreement"]["batch_128"]["initial"]["class_scores_exact"]
    assert candidate["agreement"]["batch_128"]["trained"]["class_scores_exact"]
    with pytest.raises(AssertionError, match="independent training agreement"):
        benchmark.assert_training_agreement(candidate["pipelines"])
    for name in ("strict_full", "fast_full", "fast_outputs"):
        row = candidate["pipelines"][name]
        benchmark.assert_training_agreement({"snntorch": row, name: row})


@pytest.mark.parametrize("field", ["epoch_losses", "test_loss", "test_accuracy"])
def test_training_gate_checks_each_metric_independently_of_timing(field):
    report = json.loads((ROOT / "benchmarks/zero-decay/results.json").read_text())
    pipelines = report["candidate"]["pipelines"]
    reference = pipelines["snntorch"]
    actual = {**reference, "train_s": 0.001, "test_s": 9999}
    benchmark.assert_training_agreement({"snntorch": reference, "candidate": actual})
    actual[field] = pipelines["strict_full"][field]
    with pytest.raises(AssertionError, match=f"candidate/{field}"):
        benchmark.assert_training_agreement({"snntorch": reference, "candidate": actual})
