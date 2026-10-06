"""Mode-drift: "same inputs → same outputs across modes".

In-process invariants are fast (default). CLI propagation tests spawn
subprocesses and are marked `slow`.
"""

from __future__ import annotations

import numpy as np
import torch

# ── In-process invariants (fast) ─────────────────────────────────────────


def test_encode_images_poisson_deterministic():
    from snnlab.sim.encoders import encode_images_poisson

    images = torch.rand(4, 64)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = encode_images_poisson(
        images, T_steps=200, dt=0.25, max_rate_hz=10.0, generator=g1
    )
    b = encode_images_poisson(
        images, T_steps=200, dt=0.25, max_rate_hz=10.0, generator=g2
    )
    assert torch.equal(a, b)


def test_load_dataset_deterministic_mnist():
    from snnlab.sim.datasets import load_dataset

    a_tr, a_te, ay_tr, ay_te = load_dataset("mnist", max_samples=200, split=True)
    b_tr, b_te, by_tr, by_te = load_dataset("mnist", max_samples=200, split=True)
    assert np.array_equal(a_tr, b_tr)
    assert np.array_equal(a_te, b_te)
    assert np.array_equal(ay_tr, by_tr)
    assert np.array_equal(ay_te, by_te)


def test_validation_and_official_test_are_distinct_mnist():
    from snnlab.sim.datasets import load_dataset

    _, validation_x, _, validation_y = load_dataset(
        "mnist", max_samples=500, split=True, evaluation_split="validation"
    )
    _, test_x, _, test_y = load_dataset(
        "mnist", max_samples=500, split=True, evaluation_split="test"
    )
    assert len(validation_y) == 50
    assert len(test_y) == 10_000
    assert not np.array_equal(validation_x, test_x)


# ── CLI propagation (slow) ───────────────────────────────────────────────
