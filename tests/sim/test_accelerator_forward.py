from __future__ import annotations

import pytest
import torch

from snnlab.sim.accelerator_forward import run_forward_accelerator_check

pytestmark = pytest.mark.accelerator


def test_forward_accelerator_check_rejects_cpu():
    with pytest.raises(ValueError, match="expected mps or cuda"):
        run_forward_accelerator_check("cpu")


@pytest.mark.slow
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_forward_ping_is_reproducible_and_matches_legacy_on_mps():
    result = run_forward_accelerator_check("mps")
    result.require_passed()
    assert result.device == "mps"
    assert result.snnlang_reproducibility.passed
    assert result.legacy_graph_parity.passed
