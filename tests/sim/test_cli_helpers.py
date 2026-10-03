"""Unit tests for assorted CLI helpers that were uncovered.

Targets pure-ish functions that don't need a GPU or a full training loop:
encoders, seed plumbing, key helpers, scan-var dispatch, CLI parsing.
"""


import numpy as np
import torch

from snnlab.sim.tool import (
    _auto_device,
    encode_batch,
    encode_images_poisson,
    parse_args,
    primary_hid_key,
    primary_inh_key,
    seed_everything,
)


class TestEncodeImagesPoisson:
    def test_shape_and_binary(self):
        images = torch.full((4, 16), 0.5)
        out = encode_images_poisson(images, T_steps=50, dt=1.0, max_rate_hz=100.0)
        assert out.shape == (50, 4, 16)
        assert set(out.unique().tolist()) <= {0.0, 1.0}

    def test_rate_proportional_to_intensity(self):
        # Mean firing rate over many steps ≈ pixel × max_rate × dt/1000.
        torch.manual_seed(0)
        images = torch.tensor([[0.0, 0.5, 1.0]])
        out = encode_images_poisson(images, T_steps=20000, dt=1.0, max_rate_hz=200.0)
        rates = out.mean(dim=0).squeeze()  # per-pixel mean spike prob
        # p = pixel * max_rate * dt/1000 = pixel * 0.2
        assert rates[0].item() == 0.0
        assert abs(rates[1].item() - 0.1) < 0.01
        assert abs(rates[2].item() - 0.2) < 0.01

    def test_generator_determinism(self):
        images = torch.full((2, 8), 0.7)
        g1 = torch.Generator().manual_seed(123)
        g2 = torch.Generator().manual_seed(123)
        a = encode_images_poisson(
            images, T_steps=30, dt=1.0, max_rate_hz=50.0, generator=g1
        )
        b = encode_images_poisson(
            images, T_steps=30, dt=1.0, max_rate_hz=50.0, generator=g2
        )
        assert torch.equal(a, b)

    def test_per_presentation_rates(self):
        images = torch.ones((2, 4))
        rates = torch.tensor([0.0, 1000.0])
        out = encode_images_poisson(images, T_steps=3, dt=1.0, max_rate_hz=rates)
        assert out[:, 0].sum().item() == 0
        assert out[:, 1].sum().item() == 12

    def test_clamps_above_one(self):
        # Pixels > 1 should be clamped, not throw or produce p > 1 weirdness.
        images = torch.full((1, 4), 5.0)
        out = encode_images_poisson(images, T_steps=20, dt=1.0, max_rate_hz=100.0)
        assert out.shape == (20, 1, 4)
        assert set(out.unique().tolist()) <= {0.0, 1.0}


class TestEncodeBatch:
    def test_passthrough_for_3d(self):
        # (B, T, N_in) → (T, B, N_in)
        x = torch.randn(2, 30, 700)
        out = encode_batch(x, dt=1.0)
        assert out.shape == (30, 2, 700)
        assert torch.equal(out, x.permute(1, 0, 2))


class TestSeedEverything:
    def test_none_is_noop(self):
        # Should not raise; should not change torch state in a sticky way.
        before = torch.rand(1).item()
        seed_everything(None)
        after = torch.rand(1).item()
        # Just that it didn't crash; values may differ.
        assert isinstance(before, float) and isinstance(after, float)

    def test_round_trip_determinism(self):
        seed_everything(2024)
        a_torch = torch.rand(3).tolist()
        a_np = np.random.rand(3).tolist()
        seed_everything(2024)
        b_torch = torch.rand(3).tolist()
        b_np = np.random.rand(3).tolist()
        assert a_torch == b_torch
        assert a_np == b_np


class TestPrimaryKeys:
    def test_single_layer_hid(self):
        rec = {"hid": 1, "input": 2}
        assert primary_hid_key(rec) == "hid"

    def test_multi_layer_picks_deepest(self):
        rec = {"hid_1": 1, "hid_2": 2, "hid_3": 3}
        assert primary_hid_key(rec) == "hid_3"

    def test_no_hid_falls_back(self):
        # Function returns 'hid' even if absent — caller's responsibility to
        # check; this documents the behavior.
        rec = {"input": 1}
        assert primary_hid_key(rec) == "hid"

    def test_inh_returns_none_when_absent(self):
        assert primary_inh_key({"hid": 1}) is None

    def test_inh_picks_deepest(self):
        rec = {"inh_1": 1, "inh_2": 2}
        assert primary_inh_key(rec) == "inh_2"


class TestAutoDevice:
    def test_returns_torch_device(self):
        d = _auto_device()
        assert isinstance(d, torch.device)
        assert d.type in {"cpu", "cuda", "mps"}


class TestParseArgs:
    def test_train_subparser(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv", ["cli", "train", "--epochs", "3", "--lr", "0.05"]
        )
        args = parse_args()
        assert args.epochs == 3
        assert args.lr == 0.05

    def test_train_default_lr(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["cli", "train"])
        args = parse_args()
        assert args.lr == 0.01
        assert args.epochs == 0
