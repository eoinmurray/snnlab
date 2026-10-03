"""Contract tests for the sample-wise hidden-E activity regulariser."""

from __future__ import annotations

import pytest
import torch

from snnlab.sim.train import _firing_rate_penalty


def penalty(counts, *, target_hz=10.0, strength=0.5, duration_s=0.2):
    return _firing_rate_penalty(
        counts,
        target_hz=target_hz,
        strength=strength,
        presentation_s=duration_s,
    )


def test_penalty_is_applied_per_sample_before_batch_average():
    # Sample rates are 20 Hz and 0 Hz.  The first contributes
    # 0.5 * (20 - 10)^2 and the second contributes zero; their mean is 25.
    counts = [torch.tensor([[4.0, 4.0], [0.0, 0.0]])]
    assert penalty(counts).item() == pytest.approx(25.0)


def test_penalty_is_invariant_to_batch_duplication():
    counts = torch.tensor([[4.0, 4.0], [0.0, 0.0]])
    assert penalty([counts]).item() == pytest.approx(
        penalty([counts.repeat(3, 1)]).item()
    )


def test_penalty_is_invariant_to_population_width():
    counts = torch.tensor([[4.0, 2.0], [0.0, 0.0]])
    assert penalty([counts]).item() == pytest.approx(
        penalty([counts.repeat(1, 4)]).item()
    )


def test_penalty_uses_rate_not_raw_count():
    short = [torch.tensor([[2.0, 2.0]])]
    long = [torch.tensor([[4.0, 4.0]])]
    assert penalty(short, duration_s=0.1).item() == pytest.approx(
        penalty(long, duration_s=0.2).item()
    )


def test_penalty_is_zero_at_or_below_ceiling():
    counts = [torch.tensor([[2.0, 2.0], [1.0, 2.0]])]
    assert penalty(counts).item() == 0.0


@pytest.mark.parametrize(
    ("target_hz", "strength", "duration_s"),
    [(-1.0, 1.0, 0.2), (1.0, -1.0, 0.2), (1.0, 1.0, 0.0)],
)
def test_penalty_rejects_invalid_contract(target_hz, strength, duration_s):
    with pytest.raises(ValueError):
        penalty(
            [torch.ones(1, 1)],
            target_hz=target_hz,
            strength=strength,
            duration_s=duration_s,
        )
