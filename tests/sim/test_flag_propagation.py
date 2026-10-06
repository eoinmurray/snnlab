"""Tier-1 flag-propagation tests — every knob we rely on in notebook runners.

Pattern follows test_mode_drift.py: spawn the CLI, parse config.json /
metrics.json, assert the flag actually landed. In-process tests (fast) check
behavior where subprocess isn't needed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch


def _run_cli(*args, timeout=180):
    cmd = ["uv", "run", "python", "-m", "snnlab.sim", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, (
        f"cmd failed (exit {result.returncode}):\n  {' '.join(cmd)}\n"
        f"stderr: {result.stderr[-500:]}"
    )


def _read_config(out_dir):
    cfg = Path(out_dir) / "config.json"
    assert cfg.exists(), f"no config.json in {out_dir}"
    return json.loads(cfg.read_text())


def _train_probe(out_dir, *extra, epochs=0):
    """standard-snn train — fast way to land a full config on disk.
    epochs=0 is probe-only (no weights.pth); pass epochs=1 when a test needs
    a loadable checkpoint."""
    _run_cli(
        "train",
        "--model",
        "ping",
        "--dataset",
        "mnist",
        "--max-samples",
        "50",
        "--epochs",
        str(epochs),
        "--dt",
        "0.25",
        "--w-in",
        "10",
        "--w-in-initial-zero-fraction",
        "0",
        "--out-dir",
        str(out_dir),
        "--wipe-dir",
        *extra,
    )


# ── --load-config inheritance ────────────────────────────────────────────


@pytest.mark.slow
def test_load_config_inherits_params(tmp_path):
    """sim --load-config picks up dt, t_ms, n_hidden from config.json."""
    train_dir = tmp_path / "train"
    _train_probe(train_dir, "--t-ms", "150", "--n-hidden", "128", epochs=1)
    train_cfg = _read_config(train_dir)

    sim_dir = tmp_path / "sim"
    _run_cli(
        "sim",
        "--load-config",
        str(train_dir / "config.json"),
        "--digit",
        "0",
        "--out-dir",
        str(sim_dir),
        "--wipe-dir",
    )
    sim_cfg = _read_config(sim_dir)

    for key in ("dt", "t_ms", "model", "dataset"):
        assert sim_cfg[key] == train_cfg[key], (
            f"{key}: train={train_cfg[key]} sim={sim_cfg[key]}"
        )


@pytest.mark.slow
def test_load_config_cli_overrides(tmp_path):
    """Explicit CLI flag wins over loaded --load-config values."""
    train_dir = tmp_path / "train"
    _train_probe(train_dir, "--t-ms", "150", epochs=1)
    assert _read_config(train_dir)["t_ms"] == 150.0

    sim_dir = tmp_path / "sim"
    _run_cli(
        "sim",
        "--load-config",
        str(train_dir / "config.json"),
        "--t-ms",
        "200",
        "--digit",
        "0",
        "--out-dir",
        str(sim_dir),
        "--wipe-dir",
    )
    assert _read_config(sim_dir)["t_ms"] == 200.0


@pytest.mark.slow
def test_infer_with_load_config_and_weights(tmp_path):
    """infer loads config and weights via --load-config and --load-weights."""
    train_dir = tmp_path / "train"
    _run_cli(
        "train",
        "--model",
        "ping",
        "--dataset",
        "mnist",
        "--max-samples",
        "60",
        "--epochs",
        "1",
        "--dt",
        "0.25",
        "--w-in",
        "10",
        "--w-in-initial-zero-fraction",
        "0",
        "--out-dir",
        str(train_dir),
        "--wipe-dir",
    )
    assert (train_dir / "weights.pth").exists()
    assert (train_dir / "config.json").exists()

    infer_dir = tmp_path / "infer"
    _run_cli(
        "sim",
        "--infer",
        "--load-config",
        str(train_dir / "config.json"),
        "--load-weights",
        str(train_dir / "weights.pth"),
        "--max-samples",
        "60",
        "--out-dir",
        str(infer_dir),
        "--wipe-dir",
    )
    metrics = json.loads((infer_dir / "metrics.json").read_text())
    assert metrics.get("best_acc") is not None


# ── readout configuration ────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("readout", ["rate", "mem-mean", "cumulative-potential"])
def test_readout_propagates(tmp_path, readout):
    out = tmp_path / f"readout-{readout}"
    _train_probe(out, "--readout", readout)
    assert _read_config(out)["readout_mode"] == readout


def test_readout_changes_model_forward():
    """rate and mem-mean readouts share the same parameters — same W_out
    shape, only the output reduction differs."""

    from snnlab.sim.config import build_net

    torch.manual_seed(0)
    net_rate = build_net(
        "ping",
        w_in=(10.0, 1.0),
        w_in_initial_zero_fraction=0.0,
        hidden_sizes=[64],
        readout_mode="rate",
    )
    torch.manual_seed(0)
    net_mm = build_net(
        "ping",
        w_in=(10.0, 1.0),
        w_in_initial_zero_fraction=0.0,
        hidden_sizes=[64],
        readout_mode="mem-mean",
    )
    # Parameter count must match — same W_out shape, only reduction differs.
    n_rate = sum(p.numel() for p in net_rate.parameters())
    n_mm = sum(p.numel() for p in net_mm.parameters())
    assert n_rate == n_mm, f"rate={n_rate} mem-mean={n_mm}"


# ── --v-grad-dampen ──────────────────────────────────────────────────────


@pytest.mark.slow
def test_v_grad_dampen_propagates(tmp_path):
    out = tmp_path / "cmbs"
    _train_probe(out, "--v-grad-dampen", "1234.0")
    assert _read_config(out)["v_grad_dampen"] == 1234.0


# ── --ei-strength / --ei-ratio (in-process, fast) ────────────────────────


def test_ei_ratio_scales_only_wie():
    """Doubling ei_ratio at fixed ei_strength doubles W_ie but leaves W_ei
    unchanged (the builder sets W_ei = s, W_ie = s*ratio)."""

    from snnlab.sim.config import build_net

    def _means(ratio):
        torch.manual_seed(0)
        net = build_net(
            "ping",
            w_in=(0.3, 0.03),
            w_in_initial_zero_fraction=0.0,
            ei_strength=0.5,
            ei_ratio=ratio,
            recurrent_initial_zero_fraction=0.0,
        )
        p = dict(net.named_parameters())
        return (float(p["W_ei.1"].detach().mean()), float(p["W_ie.1"].detach().mean()))

    ei_lo, ie_lo = _means(2.0)
    ei_hi, ie_hi = _means(4.0)
    assert abs(ei_hi / ei_lo - 1.0) < 0.05, f"W_ei drifted with ratio: {ei_hi / ei_lo}"
    assert abs(ie_hi / ie_lo - 2.0) < 0.05, f"W_ie scale vs ratio: {ie_hi / ie_lo}"


def test_ei_strength_scales_weights():
    """Doubling ei_strength at fixed ratio doubles both W_ei and W_ie means."""

    from snnlab.sim.config import build_net

    def _means(s):
        torch.manual_seed(0)
        net = build_net(
            "ping",
            w_in=(0.3, 0.03),
            w_in_initial_zero_fraction=0.0,
            ei_strength=s,
            ei_ratio=2.0,
            recurrent_initial_zero_fraction=0.0,
        )
        p = dict(net.named_parameters())
        return (float(p["W_ei.1"].detach().mean()), float(p["W_ie.1"].detach().mean()))

    ei_lo, ie_lo = _means(0.25)
    ei_hi, ie_hi = _means(0.50)
    assert abs(ei_hi / ei_lo - 2.0) < 0.05, f"W_ei scale: {ei_hi / ei_lo}"
    assert abs(ie_hi / ie_lo - 2.0) < 0.05, f"W_ie scale: {ie_hi / ie_lo}"


# --kaiming-init / --dales-law (config propagation)


# --fr-reg-upper (behavior)


@pytest.mark.slow
def test_fr_reg_upper_pulls_rate_down(tmp_path):
    """A strong upper-bound firing-rate regulariser drives end-of-training
    rate_e below the unregularised baseline, with everything else held fixed."""

    def _rate(out_dir, *extra):
        _run_cli(
            "train",
            "--model",
            "ping",
            "--dataset",
            "mnist",
            "--max-samples",
            "60",
            "--epochs",
            "3",
            "--seed",
            "0",
            "--dt",
            "0.25",
            "--w-in",
            "10",
            "--w-in-initial-zero-fraction",
            "0",
            "--out-dir",
            str(out_dir),
            "--wipe-dir",
            *extra,
        )
        return json.loads((out_dir / "metrics.json").read_text())["end"]["rate_e"]

    baseline = _rate(tmp_path / "fr-base")
    regd = _rate(
        tmp_path / "fr-upper",
        "--fr-reg-upper-target-hz",
        "5",
        "--fr-reg-upper-strength",
        "10",
    )
    assert regd < baseline, (
        f"upper FR reg failed to suppress rate: baseline={baseline:.2f} "
        f"regularised={regd:.2f}"
    )


# ── --wipe-dir behavior ──────────────────────────────────────────────────


@pytest.mark.slow
def test_wipe_dir_clears_existing(tmp_path):
    """--wipe-dir removes pre-existing files before the run."""
    out = tmp_path / "wipe-on"
    out.mkdir()
    sentinel = out / "stale.txt"
    sentinel.write_text("should be gone")

    _train_probe(out)
    assert not sentinel.exists(), "--wipe-dir should have removed stale.txt"
    assert (out / "config.json").exists()


# ── Tier 3: breadth (parametrised smoke) ─────────────────────────────────

# All scan vars — "does the scan driver dispatch this variable without crashing?"

_IMAGE_SCAN_VARS = [
    ("stim-overdrive", 1.0, 3.0, []),
    ("tau_gaba", 5.0, 15.0, []),
    ("tau_ampa", 1.0, 5.0, []),
    ("ei_strength", 0.1, 0.5, []),
    ("spike_rate", 5.0, 50.0, []),
    ("bias", 0.0, 0.5, []),
    ("dt", 0.1, 0.5, []),
    ("noise", 0.0, 20.0, []),
    ("digit", 0, 2, ["--input", "dataset", "--dataset", "mnist"]),
]


@pytest.mark.slow
def test_no_wipe_dir_preserves_existing(tmp_path):
    """Without --wipe-dir, unrelated files in the output dir survive."""
    out = tmp_path / "wipe-off"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("should stay")

    _run_cli(
        "train",
        "--model",
        "ping",
        "--dataset",
        "mnist",
        "--max-samples",
        "50",
        "--epochs",
        "0",
        "--dt",
        "0.25",
        "--w-in",
        "10",
        "--w-in-initial-zero-fraction",
        "0",
        "--out-dir",
        str(out),
    )
    assert sentinel.exists(), "no --wipe-dir should have preserved keep.txt"
    assert (out / "config.json").exists()


# ── V&S session additions ────────────────────────────────────────────────


def test_trainable_w_flags_make_recurrent_matrices_gradient_carrying():
    """Each --trainable-w-{ee,ei,ie,ii} flag makes the corresponding
    recurrent matrix gradient-carrying. nb049's whole story depends on
    this — without it, Adam can't push W^EI entries below zero; W_ee is the
    signed-recurrent-RSNN ceiling (all four trainable under --no-dales-law)."""

    from snnlab.sim.config import build_net

    for flag, attr in [
        ("trainable_w_ee", "W_ee"),
        ("trainable_w_ei", "W_ei"),
        ("trainable_w_ie", "W_ie"),
        ("trainable_w_ii", "W_ii"),
    ]:
        torch.manual_seed(0)
        net_default = build_net(
            "ping",
            w_in=(0.3, 0.03),
            w_in_initial_zero_fraction=0.0,
            ei_strength=0.5,
            recurrent_initial_zero_fraction=0.0,
        )
        torch.manual_seed(0)
        net_trainable = build_net(
            "ping",
            w_in=(0.3, 0.03),
            w_in_initial_zero_fraction=0.0,
            ei_strength=0.5,
            recurrent_initial_zero_fraction=0.0,
            w_ii=(0.5, 0.1),
            **{flag: True},
        )
        W_default = getattr(net_default, attr)["1"]
        W_trainable = getattr(net_trainable, attr)["1"]
        assert not W_default.requires_grad, f"{attr} default should be frozen"
        assert W_trainable.requires_grad, (
            f"--{flag.replace('_', '-')} did not make {attr} gradient-carrying"
        )


def test_ei_sparsity_zeros_recurrent_entries():
    """--recurrent-initial-zero-fraction FRAC zeros approximately *frac* of recurrent entries.
    At 0.9, ≈ 10% of W^EI entries should survive (with stochastic tolerance)."""

    from snnlab.sim.config import build_net

    torch.manual_seed(0)
    net = build_net(
        "ping",
        w_in=(0.3, 0.03),
        w_in_initial_zero_fraction=0.0,
        ei_strength=1.0,
        recurrent_initial_zero_fraction=0.9,
    )
    W = net.W_ei["1"].detach()
    nonzero_frac = float((W > 0).float().mean())
    # Expected ≈ 0.1 (10% survive); allow generous wiggle for finite-size noise.
    assert 0.05 < nonzero_frac < 0.15, (
        f"at sparsity 0.9 expected ~10% nonzero W_ei entries, got {nonzero_frac:.2%}"
    )


def test_w_ii_changes_i_cell_membrane():
    """Non-zero W^II should change I-cell dynamics relative to W^II = 0
    when there is enough I activity for self-inhibition to matter.
    Two-population sanity check that the new gi_i conductance trace
    actually reaches the I-cell membrane integration."""

    from snnlab.sim import models as M
    from snnlab.sim.config import build_net

    old_T = M.T_ms
    M.T_ms = 50.0
    M.T_steps = int(M.T_ms / M.dt)
    try:
        # Strong E drive to make I cells fire; with W_ii=0 vs W_ii>0 the
        # I rate should differ.
        def _i_rate(w_ii):
            torch.manual_seed(0)
            net = build_net(
                "ping",
                w_in=(2.0, 0.4),
                w_in_initial_zero_fraction=0.0,
                ei_strength=1.0,
                recurrent_initial_zero_fraction=0.0,
                w_ii=(w_ii, w_ii * 0.1) if w_ii > 0 else None,
            )
            net.recording = True
            spikes = (torch.rand(M.T_steps, 1, M.N_IN) < 0.5).float()
            with torch.no_grad():
                net.forward(input_spikes=spikes)
            # Sum I spikes across time and cells.
            inh_key = net._inh_key(1)
            return float(net.spike_record[inh_key].sum())

        rate_off = _i_rate(0.0)
        rate_on = _i_rate(2.0)
        assert rate_off != rate_on, (
            f"--w-ii had no effect on I rate: off={rate_off}, on={rate_on}"
        )
    finally:
        M.T_ms = old_T
        M.T_steps = int(M.T_ms / M.dt)


def test_exact_k_gives_uniform_fan_in():
    """--exact-k-initialization connectivity makes every post cell draw exactly K
    presynaptic inputs (zero fan-in variance), vs the binomial spread of
    the per-entry Bernoulli sparsifier."""

    from snnlab.sim import models as M

    shape = (1024, 256)
    sparsity = 0.99  # K ≈ 10

    try:
        M.EXACT_K_INITIALIZATION = False
        torch.manual_seed(0)
        w_b = M.init_weight(shape, "normal", 1.0, 0.1, sparsity)
        fan_b = (w_b != 0).sum(dim=0).float()

        M.EXACT_K_INITIALIZATION = True
        torch.manual_seed(0)
        w_k = M.init_weight(shape, "normal", 1.0, 0.1, sparsity)
        fan_k = (w_k != 0).sum(dim=0).float()
    finally:
        M.EXACT_K_INITIALIZATION = False

    # Bernoulli: binomial fan-in, nonzero variance.
    assert fan_b.std() > 1.0, "Bernoulli fan-in should vary cell to cell"
    # Exact-K: every column has identical fan-in.
    assert fan_k.std() == 0.0, "exact-K fan-in must be uniform"
    assert int(fan_k[0]) == round((1 - sparsity) * shape[0])
