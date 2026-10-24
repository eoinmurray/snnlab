"""Validate the extension example against an independent adaptive ODE solver."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from snnlab import extensions

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/hodgkin-huxley"


@pytest.fixture(scope="module")
def example():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(EXAMPLE))
        patch.setattr(extensions, "_REGISTRY", {})
        spec = importlib.util.spec_from_file_location(
            "hh_example", EXAMPLE / "example.py"
        )
        module = importlib.util.module_from_spec(spec)
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        result = module.run_example()
        yield module, result


def test_trajectory_and_spikes_match_independent_solver(example):
    module, result = example
    times = (np.arange(len(result.outputs["voltage"])) + 1) * module.DT_MS
    comparison = module
    _, _, reference, metrics = comparison.compare(result)
    event_times = metrics["scipy_spike_times_ms"]

    np.testing.assert_allclose(
        result.outputs["voltage"].ravel(), reference[0], atol=0.08, rtol=0
    )
    for index, gate in enumerate(("m", "h", "n"), start=1):
        actual = result.outputs[gate].ravel()
        assert np.all((actual >= 0) & (actual <= 1))
        np.testing.assert_allclose(actual, reference[index], atol=0.001, rtol=0)
    spikes = result.outputs["spikes"].ravel()
    spike_times = times[spikes > 0]
    assert len(spike_times) >= 3
    np.testing.assert_allclose(
        spike_times, event_times, atol=module.DT_MS + 0.002, rtol=0
    )
    assert result.outputs["voltage"].max() > 30
    assert result.outputs["voltage"][: round(10 / module.DT_MS)].max() < -64
    # No reset: voltage continues upward after the detected crossing.
    indices = np.flatnonzero(spikes)
    voltage = result.outputs["voltage"].ravel()
    assert np.all(voltage[indices + 1] > voltage[indices])


def test_rate_singularities_have_finite_values_and_gradients(example):
    module, _ = example
    v = torch.tensor([-65.0, -55.0, -40.0, 0.0, 40.0], requires_grad=True)
    rates = module.rates(v)
    for alpha, beta in rates:
        assert torch.isfinite(alpha).all() and torch.isfinite(beta).all()
    sum((alpha + beta).sum() for alpha, beta in rates).backward()
    assert torch.isfinite(v.grad).all()
    assert rates[0][0][2].item() == pytest.approx(1)
    assert rates[2][0][1].item() == pytest.approx(0.1)
