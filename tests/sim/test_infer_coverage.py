"""Coverage-focused tests for src/cli/infer.py.

These exercise the pure helper _hidden_sizes_from_state_dict, the perturbation
path, the recording-backed output emitters (per_cell_rates / pop_traces /
rasters), infer_and_snapshot() extras (tau_gaba / scale / skip_load), probe()
and dump_weights().

All configs are deliberately tiny (dt=0.1, t_ms=100, mnist, hidden [32]) so the
whole file stays fast. One module-scoped checkpoint is trained and reused.

The autouse conftest fixture resets models.py globals around every test; we do
NOT add our own globals reset here.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from snnlab.sim.infer import (
    _hidden_sizes_from_state_dict,
    dump_weights,
    infer,
    infer_and_snapshot,
    probe,
)
from snnlab.sim.train import train

# Mark integration classes individually so the pure shape helpers run fast.


@pytest.fixture(scope="module")
def trained_ckpt():
    """Train ONE tiny checkpoint and reuse it across every test in this module.

    Module-scoped so training (the slow part) runs exactly once, like the
    `trained_weights` fixture in test_infer_integration.py.
    """
    with TemporaryDirectory() as tmpdir:
        train_dir = Path(tmpdir) / "train"
        train_dir.mkdir()
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=2,
            dataset="mnist",
            max_samples=100,
            lr=0.01,
            hidden_sizes=[32],
            seed=42,
            out_dir=train_dir,
        )
        weights_path = train_dir / "weights.pth"
        assert weights_path.exists(), "training should produce weights.pth"
        yield weights_path


@pytest.fixture
def tmp_out():
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ─────────────────────────────────────────────────────────────────────────────
# 1. _hidden_sizes_from_state_dict — pure, cheap, thorough
# ─────────────────────────────────────────────────────────────────────────────


class TestHiddenSizesFromStateDict:
    def test_single_hidden_layer(self):
        # W_ff.0: (N_IN, hid), W_ff.1: (hid, N_OUT). Only W_ff.0 is a hidden
        # block; its output dim (hid) is the recovered hidden size.
        state = {
            "W_ff.0": torch.zeros(784, 32),
            "W_ff.1": torch.zeros(32, 10),
        }
        assert _hidden_sizes_from_state_dict(state) == [32]

    def test_two_hidden_layers(self):
        # blocks: IN→h1, h1→h2, h2→OUT ⇒ hidden = [h1, h2]
        state = {
            "W_ff.0": torch.zeros(784, 64),
            "W_ff.1": torch.zeros(64, 48),
            "W_ff.2": torch.zeros(48, 10),
        }
        assert _hidden_sizes_from_state_dict(state) == [64, 48]

    def test_no_wff_keys_returns_none(self):
        state = {"W_ei.1": torch.zeros(8, 8), "W_ie.1": torch.zeros(32, 8)}
        assert _hidden_sizes_from_state_dict(state) is None

    def test_single_wff_block_returns_none(self):
        # Only one W_ff block ⇒ len(idxs) < 2 ⇒ None (degenerate).
        state = {"W_ff.0": torch.zeros(784, 32)}
        assert _hidden_sizes_from_state_dict(state) is None

    def test_ignores_non_wff_keys(self):
        # Non-W_ff keys and a bogus non-digit W_ff key must be ignored.
        state = {
            "W_ff.0": torch.zeros(784, 32),
            "W_ff.1": torch.zeros(32, 10),
            "W_ei.1": torch.zeros(8, 8),
            "some.other.param": torch.zeros(3),
            "W_ff.bias": torch.zeros(5),  # not a digit index → ignored
        }
        assert _hidden_sizes_from_state_dict(state) == [32]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Perturbation path through infer()
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestInferPerturb:
    @pytest.mark.parametrize(
        "mode,level",
        [
            ("drop", 0.2),
            ("add", 10.0),
        ],
    )
    def test_perturb_modes_return_acc(self, trained_ckpt, mode, level):
        result = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_ckpt,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
            perturb_mode=mode,
            perturb_level=level,
        )
        assert isinstance(result, dict)
        assert "acc" in result
        assert 0.0 <= result["acc"] <= 100.0

    def test_unknown_perturb_mode_raises(self, trained_ckpt):
        with pytest.raises(ValueError, match="unknown perturbation mode"):
            infer(
                model_name="ping",
                dt=0.1,
                t_ms=100.0,
                load_weights=trained_ckpt,
                dataset="mnist",
                max_samples=50,
                hidden_sizes=[32],
                perturb_mode="nonsense",
                perturb_level=0.1,
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. infer() output emitters
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestInferEmitters:
    def test_all_outputs_written(self, trained_ckpt, tmp_out):
        result = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_ckpt,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
            out_dir=tmp_out,
            outputs={"per_cell_rates", "pop_traces", "rasters"},
        )
        assert "acc" in result

        # metrics.json always written
        assert (tmp_out / "metrics.json").exists()

        # per_cell_rates.npz
        pcr = tmp_out / "per_cell_rates.npz"
        assert pcr.exists()
        d = np.load(pcr)
        assert "rate_e_per_cell" in d.files
        assert d["rate_e_per_cell"].shape[0] == 32  # N_E
        assert "rate_e_per_sample" in d.files

        # pop_traces.npz. The stratified test split keeps only a fraction of
        # max_samples, so assert on structure (>0 trials, correct T) not the
        # exact sample count.
        pt = tmp_out / "pop_traces.npz"
        assert pt.exists()
        d = np.load(pt)
        assert "pop_e" in d.files
        n_trials = d["pop_e"].shape[0]
        assert n_trials > 0
        # T = round(100/0.1) = 1000 (round, not int, to dodge float error)
        assert d["pop_e"].shape[1] == round(100.0 / 0.1)

        # rasters.npz
        rz = tmp_out / "rasters.npz"
        assert rz.exists()
        d = np.load(rz)
        for k in (
            "e_trial", "e_t", "e_cell", "i_trial", "i_t", "i_cell",
            "out_trial", "out_t", "out_cell",
        ):
            assert k in d.files
        assert int(d["n_e"]) == 32
        # rasters counts the same trials the pop traces recorded
        assert int(d["n_trials"]) == n_trials


# ─────────────────────────────────────────────────────────────────────────────
# 4. infer_and_snapshot() extra paths
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestInferAndSnapshotExtras:
    def test_tau_gaba_and_scale_and_skip_load(self, trained_ckpt, tmp_out):
        # tau_gaba override + skip_load path in infer_and_snapshot. skip_load
        # drops matching keys so strict=False load survives on a fresh sub-block.
        infer_and_snapshot(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_ckpt,
            dataset="mnist",
            hidden_sizes=[32],
            out_dir=tmp_out,
            tau_gaba=5.0,
            skip_load=["W_ei"],
        )
        assert (tmp_out / "recording.npz").exists()

    def test_snapshot_with_perturb_and_sample_index(self, trained_ckpt, tmp_out):
        # sample_index path + perturb hook in snapshot.
        infer_and_snapshot(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_ckpt,
            dataset="mnist",
            hidden_sizes=[32],
            out_dir=tmp_out,
            sample_index=3,
            perturb_mode="drop",
            perturb_level=0.1,
        )
        assert (tmp_out / "recording.npz").exists()

    def test_snapshot_sample_out_of_range_warns(self, trained_ckpt, tmp_out):
        # sample far out of range for digit 0 → warns and falls back to sample 0.
        infer_and_snapshot(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_ckpt,
            dataset="mnist",
            hidden_sizes=[32],
            out_dir=tmp_out,
            digit=0,
            sample=10_000_000,
        )
        assert (tmp_out / "recording.npz").exists()


# ─────────────────────────────────────────────────────────────────────────────
# 5. infer() scale_w_* args (in-place weight scaling path)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestInferScaleWeights:
    def test_scale_all_three(self, trained_ckpt):
        result = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_ckpt,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
            scale_w_in=1.5,
            scale_w_ei=1.2,
            scale_w_ie=0.8,
        )
        assert 0.0 <= result["acc"] <= 100.0

    def test_infer_tau_gaba_override(self, trained_ckpt):
        # Exercises the tau_gaba override branch (sets M.tau_gaba).
        from snnlab.sim import models as M

        result = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_ckpt,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
            tau_gaba=6.0,
        )
        assert "acc" in result
        assert M.tau_gaba == pytest.approx(6.0)

    def test_skip_load_prefix(self, trained_ckpt):
        # Exercise the skip_load branch inside infer().
        result = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_ckpt,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
            skip_load=["W_ei"],
        )
        assert "acc" in result


# ─────────────────────────────────────────────────────────────────────────────
# 6. probe()
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestProbe:
    def test_probe_fresh_net_returns_rates(self, tmp_out):
        # Untrained net (load_weights=None), no dataset. n_in small to stay cheap.
        result = probe(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            hidden_sizes=[32],
            n_in=64,
            n_batch=8,
            input_rate_hz=25.0,
            seed=0,
            out_dir=tmp_out,
        )
        assert "rate_e_hz" in result
        assert "rate_i_hz" in result
        assert result["rate_e_hz"] >= 0.0
        assert (tmp_out / "metrics.json").exists()

    def test_probe_with_outputs(self, tmp_out):
        probe(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            hidden_sizes=[32],
            n_in=64,
            n_batch=8,
            input_rate_hz=25.0,
            seed=0,
            out_dir=tmp_out,
            outputs={"rasters", "per_cell_rates"},
        )
        assert (tmp_out / "rasters.npz").exists()
        assert (tmp_out / "per_cell_rates.npz").exists()

    def test_probe_with_checkpoint(self, trained_ckpt, tmp_out):
        # load_weights path: hidden_sizes recovered from the checkpoint, n_in
        # must match the trained W_in (mnist → 784).
        result = probe(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            hidden_sizes=None,
            n_in=784,
            n_batch=8,
            input_rate_hz=25.0,
            seed=0,
            load_weights=trained_ckpt,
            out_dir=tmp_out,
        )
        assert result["rate_e_hz"] >= 0.0

    def test_probe_explicit_wei_wie_and_ninh(self, tmp_out):
        # w_ei_mean/w_ie_mean override branch + n_inh sizing.
        result = probe(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            hidden_sizes=[32],
            n_in=64,
            n_inh=8,
            n_batch=8,
            w_ei_mean=0.5,
            w_ie_mean=1.0,
            tau_gaba=5.0,
            seed=0,
            out_dir=tmp_out,
        )
        assert result["rate_e_hz"] >= 0.0

    def test_probe_private_w_in(self, tmp_out):
        # private_w_in identity-W_in path (scale defaults to 1.0 with w_in=None).
        # n_in must equal n_e for the identity wiring.
        result = probe(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            hidden_sizes=[32],
            n_in=32,
            n_batch=8,
            private_w_in=True,
            seed=0,
            out_dir=tmp_out,
        )
        assert result["rate_e_hz"] >= 0.0

    def test_probe_input_file(self, tmp_out):
        # input_file primitive: build a tiny (T, B, N_IN) stream and run it.
        T, B, N = 100, 4, 64
        rng = np.random.default_rng(0)
        arr = (rng.random((T, B, N)) < 0.02).astype("float32")
        in_path = tmp_out / "input.npz"
        np.savez(in_path, input_spikes=arr)
        result = probe(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            hidden_sizes=[32],
            n_in=N,
            seed=0,
            input_file=str(in_path),
            out_dir=tmp_out,
        )
        assert result["rate_e_hz"] >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6c. i_override_file paths (infer + infer_and_snapshot)
# ─────────────────────────────────────────────────────────────────────────────


def _write_i_override(path, T, n_i, n_trials, seed=0):
    """Build a sparse I-spike override NPZ in the format infer() expects."""
    rng = np.random.default_rng(seed)
    trials, ts, cells = [], [], []
    for tr in range(n_trials):
        # a handful of spikes per trial
        k = 5
        trials.append(np.full(k, tr, dtype="int32"))
        ts.append(rng.integers(0, T, size=k).astype("int32"))
        cells.append(rng.integers(0, n_i, size=k).astype("int32"))
    np.savez(
        path,
        T=np.int32(T),
        n_i=np.int32(n_i),
        n_trials=np.int32(n_trials),
        i_trial=np.concatenate(trials),
        i_t=np.concatenate(ts),
        i_cell=np.concatenate(cells),
    )


@pytest.fixture
def synthetic_inference(monkeypatch, tmp_path):
    """Real simulator, synthetic images and an untrained tiny checkpoint; no downloads."""
    from snnlab.sim import infer as module
    from snnlab.sim import models as M
    from snnlab.sim.config import build_net, setup_model_globals

    M.N_IN = 784
    setup_model_globals([8])
    torch.manual_seed(42)
    weights = tmp_path / "weights.pth"
    torch.save(build_net("ping", hidden_sizes=[8], readout_mode="mem-mean").state_dict(), weights)
    images = np.random.default_rng(11).random((70, 784)).astype("float32")
    labels = np.arange(70, dtype="int64") % 10
    loaders, nets = [], []

    def dataset(*args, **kwargs):
        loaders.append(kwargs)
        return None, images, None, labels

    def build(*args, **kwargs):
        net = build_net(*args, **kwargs)
        nets.append(net)
        return net

    monkeypatch.setattr(module, "load_dataset", dataset)
    monkeypatch.setattr(module, "_auto_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(module, "build_net", build)
    kwargs = dict(model_name="ping", dt=0.1, t_ms=2.0, load_weights=weights,
                  dataset="mnist", hidden_sizes=[8], seed=42, readout_mode="mem-mean")
    return kwargs, loaders, nets


def test_override_metrics_do_not_record_and_match_full_recording(synthetic_inference, tmp_path):
    kwargs, loaders, nets = synthetic_inference
    override = tmp_path / "override.npz"
    _write_i_override(override, T=20, n_i=2, n_trials=65)
    full_dir, lean_dir = tmp_path / "full", tmp_path / "lean"
    full_dir.mkdir()
    lean_dir.mkdir()
    full = infer(**kwargs, max_samples=65, i_override_file=override,
                 outputs={"rasters"}, out_dir=full_dir)
    lean = infer(**kwargs, max_samples=65, i_override_file=override, out_dir=lean_dir)
    assert full == lean
    assert full["rates_hz"]["inh"] > 0
    assert nets[0].recording and not nets[1].recording
    assert not hasattr(nets[1], "spike_record")
    assert {p.name for p in lean_dir.iterdir()} == {"metrics.json"}
    assert all(call["evaluation_only"] for call in loaders)


def test_i_only_baseline_matches_full_i_events(synthetic_inference, tmp_path):
    kwargs, _, nets = synthetic_inference
    full_dir, lean_dir = tmp_path / "full", tmp_path / "lean"
    full_dir.mkdir()
    lean_dir.mkdir()
    full = infer(**kwargs, max_samples=65, outputs={"rasters"}, out_dir=full_dir)
    lean = infer(**kwargs, max_samples=65, outputs={"rasters"},
                 recording_mode="inhibitory", out_dir=lean_dir)
    assert full == lean
    assert set(nets[-1].spike_record) == {"inh"}
    with np.load(full_dir / "rasters.npz") as original, np.load(lean_dir / "rasters.npz") as selected:
        assert int(selected["n_trials"]) == 65
        assert int(selected["T"]) == 20
        assert not any(k.startswith(("e_", "out_")) for k in selected.files)
        for key in selected.files:
            assert np.array_equal(selected[key], original[key])


@pytest.mark.parametrize("mode", ["spikes", "inhibitory"])
@pytest.mark.parametrize("override", [False, True])
def test_selected_snapshots_preserve_spikes_without_trace_buffers(synthetic_inference, tmp_path, mode, override):
    kwargs, _, nets = synthetic_inference
    if override:
        path = tmp_path / "override.npz"
        _write_i_override(path, T=20, n_i=2, n_trials=1)
        kwargs = {**kwargs, "i_override_file": path}
    full_dir, lean_dir = tmp_path / "full", tmp_path / "lean"
    infer_and_snapshot(**kwargs, sample_index=3, out_dir=full_dir)
    infer_and_snapshot(**kwargs, sample_index=3, out_dir=lean_dir, recording_mode=mode)
    assert set(nets[-1].spike_record) == ({"hid", "inh"} if mode == "spikes" else {"inh"})
    with np.load(full_dir / "recording.npz") as full, np.load(lean_dir / "recording.npz") as lean:
        expected = {"dt", "n_e", "n_i", "label", "spk_i"}
        if mode == "spikes":
            expected.add("spk_e")
        assert set(lean.files) == expected
        for key in lean.files:
            assert np.array_equal(lean[key], full[key])


@pytest.mark.slow
class TestIOverride:
    def test_infer_i_override(self, trained_ckpt, tmp_out):
        # n_i must match the checkpoint's I-pool (32//4 = 8); T = round(100/0.1).
        ov_path = tmp_out / "iov.npz"
        _write_i_override(ov_path, T=round(100.0 / 0.1), n_i=8, n_trials=64)
        result = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_ckpt,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
            i_override_file=str(ov_path),
        )
        assert "acc" in result

    def test_snapshot_i_override(self, trained_ckpt, tmp_out):
        ov_path = tmp_out / "iov_snap.npz"
        _write_i_override(ov_path, T=round(100.0 / 0.1), n_i=8, n_trials=1)
        infer_and_snapshot(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_ckpt,
            dataset="mnist",
            hidden_sizes=[32],
            out_dir=tmp_out,
            i_override_file=str(ov_path),
        )
        assert (tmp_out / "recording.npz").exists()


# ─────────────────────────────────────────────────────────────────────────────
# 7. dump_weights()
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestDumpWeights:
    def test_dump_weights_writes_npz(self, trained_ckpt, tmp_out):
        result = dump_weights(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_ckpt,
            dataset="mnist",
            hidden_sizes=[32],
            out_dir=tmp_out,
            seed=42,
        )
        assert result["n_arrays"] > 0
        out_path = tmp_out / "weights_dump.npz"
        assert out_path.exists()
        d = np.load(out_path)
        # init + trained arrays for the E-I matrices and W_ff blocks
        names = set(d.files)
        assert any(n.startswith("W_ff_") and n.endswith("_init") for n in names)
        assert any(n.startswith("W_ff_") and n.endswith("_trained") for n in names)
        assert any(n.startswith("W_ei_") and n.endswith("_init") for n in names)

    def test_dump_weights_kaiming(self, trained_ckpt, tmp_out):
        # kaiming_init (randomize_init=False) path.
        result = dump_weights(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_ckpt,
            dataset="mnist",
            hidden_sizes=[32],
            out_dir=tmp_out,
            seed=42,
            kaiming_init=True,
        )
        assert result["n_arrays"] > 0

    def test_dump_weights_requires_load_weights(self, tmp_out):
        with pytest.raises(AssertionError):
            dump_weights(
                model_name="ping",
                dt=0.1,
                t_ms=100.0,
                load_weights=None,
                dataset="mnist",
                hidden_sizes=[32],
                out_dir=tmp_out,
            )


@pytest.mark.parametrize(
    "fields",
    [
        {
            "pop_e",
            "rate_e_per_cell",
            "e_trial",
            "e_t",
            "e_cell",
            "i_trial",
            "i_t",
            "i_cell",
        },
        {
            "pop_e",
            "rate_e_per_sample",
            "e_trial",
            "e_t",
            "e_cell",
            "i_trial",
            "i_t",
            "i_cell",
        },
    ],
)
def test_selected_inference_arrays_equal_full_pass(
    synthetic_inference, tmp_path, fields
):
    kwargs, _, nets = synthetic_inference
    full_dir, lean_dir = tmp_path / "full", tmp_path / "lean"
    full_dir.mkdir()
    lean_dir.mkdir()
    outputs = {"rasters", "pop_traces", "per_cell_rates"}
    full = infer(**kwargs, max_samples=65, outputs=outputs, out_dir=full_dir)
    lean = infer(
        **kwargs,
        max_samples=65,
        outputs=outputs,
        out_dir=lean_dir,
        recording_mode="spikes",
        output_fields=fields,
    )
    assert full == lean
    assert set(nets[-1].spike_record) == {"hid", "inh"}
    metadata = {"dt", "T", "n_trials", "n_e", "n_i"}
    for filename in ("rasters.npz", "pop_traces.npz", "per_cell_rates.npz"):
        with (
            np.load(full_dir / filename) as original,
            np.load(lean_dir / filename) as selected,
        ):
            assert set(selected.files) == set(original.files) & (fields | metadata)
            for key in selected.files:
                np.testing.assert_array_equal(selected[key], original[key])


def test_probe_postburn_events_preserve_full_window_rates(tmp_path, monkeypatch):
    from snnlab.sim import infer as module

    monkeypatch.setattr(module, "_auto_device", lambda: torch.device("cpu"))
    full_dir, lean_dir = tmp_path / "full", tmp_path / "lean"
    full_dir.mkdir()
    lean_dir.mkdir()
    kwargs = dict(
        model_name="ping",
        dt=0.1,
        t_ms=20,
        hidden_sizes=[8],
        n_in=8,
        n_inh=2,
        seed=42,
        input_rate_hz=1000,
        w_in=(5.0, 0.5),
        n_batch=2,
        outputs={"rasters"},
    )
    fields = {f"{p}_{f}" for p in ("e", "i") for f in ("trial", "t", "cell")}
    full = probe(**kwargs, out_dir=full_dir)
    lean = probe(
        **kwargs,
        out_dir=lean_dir,
        recording_mode="spikes",
        output_fields=fields,
        recording_start_step=100,
    )
    assert full == lean
    with (
        np.load(full_dir / "rasters.npz") as original,
        np.load(lean_dir / "rasters.npz") as selected,
    ):
        assert int(selected["recording_start_step"]) == 100
        assert int(selected["T"]) == int(original["T"]) == 200
        assert not any(k.startswith("out_") for k in selected.files)
        assert len(original["e_t"]) > 0
        for population in ("e", "i"):
            keep = original[f"{population}_t"] >= 100
            for field in ("trial", "t", "cell"):
                key = f"{population}_{field}"
                np.testing.assert_array_equal(selected[key], original[key][keep])


def test_selected_weight_dump_matches_full(synthetic_inference, tmp_path):
    kwargs, _, _ = synthetic_inference
    kwargs.pop("readout_mode")
    fields = {
        f"W_{kind}_1_{stage}" for kind in ("ei", "ie") for stage in ("init", "trained")
    }
    dump_weights(**kwargs, out_dir=tmp_path / "full")
    dump_weights(**kwargs, out_dir=tmp_path / "lean", output_fields=fields)
    with (
        np.load(tmp_path / "full/weights_dump.npz") as full,
        np.load(tmp_path / "lean/weights_dump.npz") as lean,
    ):
        assert set(lean.files) == fields
        for key in fields:
            np.testing.assert_array_equal(lean[key], full[key])
