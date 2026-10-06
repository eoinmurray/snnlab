"""Coverage for datasets.py error branches.

SHD loader tests were removed — SHD is not a supported path yet. What remains
covers the unknown-dataset guards in load_dataset / _load_dataset_image.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from snnlab.sim.datasets import load_dataset


def test_load_dataset_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("not_a_dataset")


def test_evaluation_only_never_loads_training_and_preserves_test_pixels(monkeypatch):
    from types import SimpleNamespace

    from torchvision import datasets

    pixels = torch.arange(12 * 784).remainder(256).to(torch.uint8).reshape(12, 28, 28)
    labels = torch.arange(12).remainder(10)
    calls = []

    def mnist(*, train, **kwargs):
        calls.append(train)
        return SimpleNamespace(data=pixels + int(train), targets=labels)

    monkeypatch.setattr(datasets, "MNIST", mnist)
    old = load_dataset("mnist", split=True, evaluation_split="test")
    calls.clear()
    new = load_dataset(
        "mnist", split=True, evaluation_split="test", evaluation_only=True
    )
    assert calls == [False]
    assert new[0] is None and new[2] is None
    assert np.array_equal(old[1], new[1])
    assert np.array_equal(old[3], new[3])


@pytest.mark.parametrize("kwargs", [{}, {"split": True}, {"evaluation_split": "test"}])
def test_evaluation_only_rejects_training_or_validation_requests(kwargs):
    with pytest.raises(ValueError, match="evaluation_only requires"):
        load_dataset("mnist", evaluation_only=True, **kwargs)
