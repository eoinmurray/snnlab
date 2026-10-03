"""init_lif_state: randomised voltage init underpins PING's symmetry
breaking. If the distribution drifts, every PING run starts in lockstep
and produces spurious oscillations.
"""

import pytest
import torch

from snnlab.sim import models as M
from snnlab.sim.models import init_lif_state


class TestInitLifState:
    def test_default_is_uniform_rest(self):
        v, ref = init_lif_state(B=4, N=100, device=torch.device("cpu"))
        assert torch.all(v == M.E_L)
        assert torch.all(ref == 0)
        assert ref.dtype == torch.long

    def test_randomize_distributes_in_rest_to_threshold_band(self):
        torch.manual_seed(0)
        v, _ = init_lif_state(B=1, N=10_000, device=torch.device("cpu"), randomize=True)
        # Every sample strictly within [E_L, V_th]
        assert v.min().item() >= M.E_L
        assert v.max().item() < M.V_th
        # Uniform distribution ⇒ mean ≈ (E_L + V_th) / 2, span covers nearly full band
        expected_mean = (M.E_L + M.V_th) / 2
        assert v.mean().item() == pytest.approx(expected_mean, abs=0.3)
        assert v.min().item() < M.E_L + 1.0  # left tail reaches near E_L
        assert v.max().item() > M.V_th - 1.0  # right tail reaches near V_th

    def test_ref_std_produces_nonzero_refractory(self):
        torch.manual_seed(0)
        _, ref = init_lif_state(
            B=1, N=1000, device=torch.device("cpu"), ref_mean=5.0, ref_std=2.0
        )
        # Clamped to [0, ∞), so mean is >= 0. With mean=5 std=2, mean should
        # be near 5 and all values non-negative integers.
        assert ref.dtype == torch.long
        assert ref.min().item() >= 0
        assert ref.float().mean().item() == pytest.approx(5.0, abs=0.5)
