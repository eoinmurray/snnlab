"""Actual spike accounting must not change the intervention's random stream."""

import pytest
import torch

from snnlab.sim.infer import _make_perturb_fn


@pytest.mark.parametrize(
    "mode,level",
    [("drop", 0), ("drop", 0.5), ("drop", 1), ("add", 0), ("add", 500), ("add", 1000)],
)
def test_counts_match_changed_slots_without_changing_draws(mode, level):
    counts = {}
    measured = _make_perturb_fn(
        mode, level, 1, torch.Generator().manual_seed(42), counts
    )
    plain = _make_perturb_fn(mode, level, 1, torch.Generator().manual_seed(42))
    expected = {"e1": [0, 0, 0, 0], "i1": [0, 0, 0, 0]}
    for _ in range(3):
        e = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
        i = torch.tensor([[1.0, 0.0]])
        after = measured(e, i, 1)
        reference = plain(e, i, 1)
        for key, before, result, wanted in zip(expected, (e, i), after, reference):
            assert torch.equal(result, wanted)
            values = [
                int(before.sum()),
                int(result.sum()),
                int(((before == 0) & (result == 1)).sum()),
                int(((before == 1) & (result == 0)).sum()),
            ]
            expected[key] = [a + b for a, b in zip(expected[key], values)]
    for key, values in expected.items():
        assert counts[key]["counts"].tolist() == values
        assert values[1] == values[0] + values[2] - values[3]
    assert counts["e1"]["slots"] == 12
    assert counts["i1"]["slots"] == 6
    if mode == "add" and level == 1000:
        assert expected["e1"][2] == 6  # occupied slots do not receive another spike


@pytest.mark.parametrize(
    "mode,level", [("add", -1), ("add", 1001), ("drop", 1.1), ("add", float("nan"))]
)
def test_invalid_probability_is_rejected(mode, level):
    with pytest.raises(ValueError, match="probability"):
        _make_perturb_fn(mode, level, 1, torch.Generator())
