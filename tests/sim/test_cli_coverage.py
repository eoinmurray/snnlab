"""In-process coverage tests for tools/snnsim/tool.py.

These drive cli.py IN-PROCESS (not via subprocess) so the lines actually earn
coverage credit — subprocess CLI runs run in a separate interpreter and are
invisible to coverage. The existing suites split the work:

  - test_cli_helpers.py   → encoders / seed / key helpers / parse_args basics
  - test_flag_propagation.py → mostly subprocess round-trips (no coverage credit)

This file fills the large uncovered blocks that only in-process calls can hit:

  * parse_args across every subcommand + the post-parse auto-flip / --load-config
    branch (lines ~975-1021)
  * _build_config_mapping introspection (79-169)
  * _apply_load_config JSON merge + CLI-override precedence (172-260)
  * configure_models model-global writes (1023-1056)
  * save_run_artifacts + _print_intro (1058-1190)
  * _resolve_w_in (1269-1274)
  * the dispatch handlers _run_sim / _run_train / _emit_infer / _emit_probe /
    _run_dump_weights via tiny real main([...]) runs (1193-1509)

Speed budget: one module-scoped 1-epoch train checkpoint is reused by every
downstream infer / dump-weights test. Runs use mnist, max_samples≈60, dt 0.5,
n-hidden 32, t-ms 40 to stay well under the ~120s target.

A shared conftest.py autouse fixture already resets models.py globals between
every test, so this file adds no globals-reset fixture of its own.
"""

from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from snnlab.sim import models as M

# The `cli` name resolves to the package (cli/__init__.py), which only
# re-exports a subset of symbols. The private helpers under test (_run_sim,
# _apply_load_config, main, …) live on the tool module, so import from
# there directly rather than the package facade.
from snnlab.sim import tool as cli
from snnlab.sim.tool import (
    _apply_load_config,
    _build_config_mapping,
    _build_parent_parser,
    _resolve_w_in,
    configure_models,
    parse_args,
    save_run_artifacts,
)

# Fast, well-behaved run knobs shared across every dispatch test.
_FAST = ["--n-hidden", "32", "--dt", "0.5", "--t-ms", "40"]


# ─────────────────────────────────────────────────────────────────────────
# parse_args — subparser coverage + post-parse logic
# ─────────────────────────────────────────────────────────────────────────


class TestParseArgsSubcommands:
    @pytest.mark.parametrize("snapshot", [False, True])
    @pytest.mark.parametrize("mode", ["full", "spikes", "inhibitory"])
    def test_recording_mode_reaches_inference(
        self, monkeypatch, tmp_path, snapshot, mode
    ):
        args = parse_args(
            [
                "sim",
                "--infer",
                "--load-weights",
                "fixture.pth",
                "--recording-mode",
                mode,
            ]
        )
        seen = []

        def capture(**kwargs):
            seen.append(kwargs)
            return {"acc": 0}

        monkeypatch.setattr(cli, "infer_and_snapshot" if snapshot else "infer", capture)
        cli._emit_infer(args, None, tmp_path, None, snapshot_mode=snapshot)
        assert seen[0]["recording_mode"] == mode

    def test_sim_defaults(self):
        args = parse_args(["sim"])
        assert args.mode == "sim"
        assert args.model == "ping"
        assert args.input == "synthetic-spikes"
        assert args.dt == pytest.approx(0.25)
        # sim-only flags exist with their defaults
        assert args.infer is False
        assert args.n_batch == 64
        assert args.n_in == 784

    def test_sim_infer_requires_weights_exits(self):
        # --infer with no --load-weights hits the sys.exit(1) guard (line ~990).
        with pytest.raises(SystemExit) as exc:
            parse_args(["sim", "--infer"])
        assert exc.value.code == 1

    def test_train_flags(self):
        args = parse_args(
            ["train", "--epochs", "5", "--lr", "0.001", "--batch-size", "16"]
        )
        assert args.mode == "train"
        assert args.epochs == 5
        assert args.lr == pytest.approx(0.001)
        assert args.batch_size == 16

    def test_dump_weights_subparser(self):
        args = parse_args(["dump-weights", "--model", "ping"])
        assert args.mode == "dump-weights"
        # dump-weights carries its own load-config / load-weights options
        assert hasattr(args, "load_config")
        assert hasattr(args, "load_weights")

    def test_no_mode_prints_help_and_exits_zero(self):
        # args.mode is None → parser.print_help() + sys.exit(0) (lines 974-976).
        with pytest.raises(SystemExit) as exc:
            parse_args([])
        assert exc.value.code == 0

    def test_readout_and_dales_law_flags(self):
        args = parse_args(["train", "--readout", "mem-mean", "--dales-law"])
        assert args.readout_mode == "mem-mean"
        assert args.dales_law is True
        args2 = parse_args(["train", "--readout", "rate"])
        assert args2.dales_law is True  # default
        args3 = parse_args(["train", "--readout", "spike-count"])
        assert args3.readout_mode == "spike-count"

    def test_nargs_weight_and_hidden_flags(self):
        args = parse_args(["train", "--n-hidden", "64", "32", "--w-in", "10", "2"])
        assert args.n_hidden == [64, 32]
        assert args.w_in == [10.0, 2.0]

    def test_drive_and_exec_flags(self):
        args = parse_args(
            [
                "sim",
                "--seed",
                "7",
            ]
        )
        assert args.seed == 7


class TestParseArgsAutoFlip:
    def test_digit_auto_flips_to_dataset(self):
        # --digit given, --input left at default → auto-flip to dataset input.
        args = parse_args(["sim", "--digit", "3"])
        assert args.input == "dataset"
        assert args._input_auto is True

    def test_explicit_input_not_flipped(self):
        # An explicit --input synthetic-spikes must be honoured even with --digit.
        args = parse_args(["sim", "--input", "synthetic-spikes", "--digit", "3"])
        assert args.input == "synthetic-spikes"
        assert args._input_auto is False

    def test_no_dataset_flag_no_flip(self):
        args = parse_args(["sim"])
        assert args.input == "synthetic-spikes"
        assert args._input_auto is False


# ─────────────────────────────────────────────────────────────────────────
# _build_config_mapping — parser introspection (single source of truth)
# ─────────────────────────────────────────────────────────────────────────


class TestBuildConfigMapping:
    def test_maps_dest_and_special_keys(self):
        parent = _build_parent_parser()
        config_to_args, dest_to_flag = _build_config_mapping(parent)

        # Special-case config-key remaps (dest ≠ config.json key).
        assert config_to_args["hidden_sizes"] == "n_hidden"
        assert config_to_args["input_rate"] == "spike_rate"
        assert config_to_args["readout"] == "readout_mode"
        # Plain pass-through.
        assert config_to_args["model"] == "model"
        assert config_to_args["dt"] == "dt"

    def test_prefers_positive_long_flag(self):
        parent = _build_parent_parser()
        _, dest_to_flag = _build_config_mapping(parent)
        # dales_law has --dales-law (positive) and --no-dales-law (negative);
        # the positive long form must win.
        assert dest_to_flag["dales_law"] == "--dales-law"
        assert dest_to_flag["model"] == "--model"

    def test_help_action_excluded(self):
        parent = _build_parent_parser()
        config_to_args, _ = _build_config_mapping(parent)
        assert "help" not in config_to_args.values()


# ─────────────────────────────────────────────────────────────────────────
# _apply_load_config — JSON merge, CLI precedence, legacy handling
# ─────────────────────────────────────────────────────────────────────────


def _mapping():
    parent = _build_parent_parser()
    return _build_config_mapping(parent)


class TestApplyLoadConfig:
    def test_inherits_and_cli_overrides(self, tmp_path, capsys):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "model": "ping",
                    "dt": 0.5,
                    "t_ms": 150.0,
                    "hidden_sizes": [128],
                    "dales_law": True,
                }
            )
        )
        config_to_args, dest_to_flag = _mapping()

        # argv explicitly sets --t-ms → must NOT be overwritten by config's 150.
        argv = ["--load-config", str(cfg_path), "--t-ms", "200"]
        args = Namespace(
            load_config=str(cfg_path),
            t_ms=200.0,
            dt=0.25,
            model="ping",
            n_hidden=None,
            dales_law=True,
        )
        _apply_load_config(args, argv, config_to_args, dest_to_flag)

        # dt was not passed on the CLI → inherited from config.
        assert args.dt == pytest.approx(0.5)
        # hidden_sizes → n_hidden inherited.
        assert args.n_hidden == [128]
        # t_ms explicitly set on CLI → NOT overwritten.
        assert args.t_ms == pytest.approx(200.0)

        out = capsys.readouterr().out
        assert "inherited" in out

    def test_missing_file_exits(self, tmp_path):
        config_to_args, dest_to_flag = _mapping()
        missing = tmp_path / "nope.json"
        args = Namespace(load_config=str(missing))
        with pytest.raises(SystemExit) as exc:
            _apply_load_config(
                args, ["--load-config", str(missing)], config_to_args, dest_to_flag
            )
        assert exc.value.code == 1

    def test_legacy_n_hidden_int_promoted(self, tmp_path):
        # Old config.json with n_hidden int and no hidden_sizes → promoted to list.
        cfg_path = tmp_path / "old.json"
        cfg_path.write_text(json.dumps({"model": "ping", "n_hidden": 256}))
        config_to_args, dest_to_flag = _mapping()
        args = Namespace(load_config=str(cfg_path), n_hidden=None, model="ping")
        _apply_load_config(
            args, ["--load-config", str(cfg_path)], config_to_args, dest_to_flag
        )
        assert args.n_hidden == [256]

    def test_training_readout_mode_key_is_inherited(self, tmp_path):
        cfg_path = tmp_path / "train-config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "model": "ping",
                    "readout_mode": "cumulative-potential",
                }
            )
        )
        config_to_args, dest_to_flag = _mapping()
        args = Namespace(load_config=str(cfg_path), model="ping", readout_mode="rate")
        _apply_load_config(
            args, ["--load-config", str(cfg_path)], config_to_args, dest_to_flag
        )
        assert args.readout_mode == "cumulative-potential"

    def test_legacy_model_alias_remapped(self, tmp_path, monkeypatch, capsys):
        # LEGACY_MODEL_ALIASES is empty in the live config, so seed one to drive
        # the alias-remap branch (config's "model" rewritten to the new name).
        from snnlab.sim import config as _config

        monkeypatch.setattr(
            _config, "LEGACY_MODEL_ALIASES", {"old_ping": "ping"}, raising=False
        )
        cfg_path = tmp_path / "legacy.json"
        cfg_path.write_text(json.dumps({"model": "old_ping", "dt": 0.5}))
        config_to_args, dest_to_flag = _mapping()
        args = Namespace(load_config=str(cfg_path), model="ping", dt=0.25)
        _apply_load_config(
            args, ["--load-config", str(cfg_path)], config_to_args, dest_to_flag
        )
        assert args.model == "ping"
        assert "legacy model name" in capsys.readouterr().out

    def test_legacy_n_hidden_list_promoted(self, tmp_path):
        # n_hidden already a list (no hidden_sizes) → copied through as list.
        cfg_path = tmp_path / "oldlist.json"
        cfg_path.write_text(json.dumps({"model": "ping", "n_hidden": [64, 32]}))
        config_to_args, dest_to_flag = _mapping()
        args = Namespace(load_config=str(cfg_path), n_hidden=None, model="ping")
        _apply_load_config(
            args, ["--load-config", str(cfg_path)], config_to_args, dest_to_flag
        )
        assert args.n_hidden == [64, 32]

    def test_missing_critical_flag_warns(self, tmp_path, capsys):
        # config.json without dales_law → the "predates these flags" warning.
        cfg_path = tmp_path / "c.json"
        cfg_path.write_text(json.dumps({"model": "ping", "dt": 0.5}))
        config_to_args, dest_to_flag = _mapping()
        args = Namespace(load_config=str(cfg_path), dt=0.25, model="ping")
        _apply_load_config(
            args, ["--load-config", str(cfg_path)], config_to_args, dest_to_flag
        )
        out = capsys.readouterr().out
        assert "WARNING" in out and "dales_law" in out


# ─────────────────────────────────────────────────────────────────────────
# configure_models — the one sanctioned models-global boundary
# ─────────────────────────────────────────────────────────────────────────


class TestConfigureModels:
    def test_parses_variable_training_rates(self):
        args = parse_args(["train", "--input-rates", "0.5", "1", "25"])
        assert args.input_rates == [0.5, 1.0, 25.0]

    def test_sets_globals_from_args(self):
        args = parse_args(
            [
                "sim",
                "--surrogate-slope",
                "40",
                "--input-rate",
                "33",
                "--t-ms",
                "77",
            ]
        )
        args.exact_k_initialization = True
        args.recurrent_initial_zero_fraction = 0.5
        configure_models(args)
        assert M.SURROGATE_SLOPE == pytest.approx(40.0)
        assert M.max_rate_hz == pytest.approx(33.0)
        assert M.T_ms == pytest.approx(77.0)
        assert M.EXACT_K_INITIALIZATION is True

    def test_none_valued_args_left_alone(self):
        # No surrogate override → the slope global keeps its module default; only
        # the always-written max_rate_hz / T_ms move.
        before_slope = M.SURROGATE_SLOPE
        args = parse_args(["sim", "--input-rate", "10", "--t-ms", "50"])
        configure_models(args)
        assert M.SURROGATE_SLOPE == before_slope
        assert M.max_rate_hz == pytest.approx(10.0)
        assert M.T_ms == pytest.approx(50.0)


# ─────────────────────────────────────────────────────────────────────────
# save_run_artifacts + _print_intro — provenance / banner / run.sh
# ─────────────────────────────────────────────────────────────────────────


class TestSaveRunArtifacts:
    def test_writes_config_and_runsh(self, tmp_path):
        out = tmp_path / "run"
        args = parse_args(
            ["train", "--epochs", "0", "--n-hidden", "32", "--dataset", "mnist"]
        )
        log = save_run_artifacts(out, args, "train")
        cfg = json.loads((out / "config.json").read_text())
        assert cfg["mode"] == "train"
        # provenance metadata was merged in.
        assert "run_id" in cfg
        # None-valued args are dropped; set ones survive.
        assert cfg["epochs"] == 0
        assert cfg["dataset"] == "mnist"
        assert (out / "run.sh").exists()
        assert (out / "run.sh").read_text().startswith("#!/bin/bash")
        # returns a configured logger
        assert log.name == "cli"

    def test_wipe_dir_clears_existing(self, tmp_path):
        out = tmp_path / "run"
        out.mkdir()
        stale = out / "stale.txt"
        stale.write_text("gone")
        args = parse_args(["sim", "--wipe-dir", "--n-hidden", "32"])
        save_run_artifacts(out, args, "sim")
        assert not stale.exists()
        assert (out / "config.json").exists()

    def test_print_intro_sim_synthetic_subtitle(self, tmp_path, capsys):
        # Drives _print_intro's synthetic-spikes branch (no dataset used).
        out = tmp_path / "run"
        args = parse_args(["sim", "--n-hidden", "64", "32", "--input-rate", "20"])
        save_run_artifacts(out, args, "sim")
        printed = capsys.readouterr().out
        assert "synthetic-spikes" in printed
        # multi-layer hidden renders with the arrow join
        assert "64→32" in printed


# ─────────────────────────────────────────────────────────────────────────
# _resolve_w_in
# ─────────────────────────────────────────────────────────────────────────


class TestResolveWIn:
    def test_default(self):
        assert _resolve_w_in(SimpleNamespace(w_in=None)) == [0.3, 0.06]

    def test_single_value_expands_to_ten_percent_std(self):
        w = _resolve_w_in(SimpleNamespace(w_in=[10.0]))
        assert w[0] == pytest.approx(10.0)
        assert w[1] == pytest.approx(1.0)

    def test_pair_passthrough(self):
        assert _resolve_w_in(SimpleNamespace(w_in=[5.0, 2.0])) == [5.0, 2.0]


# ─────────────────────────────────────────────────────────────────────────
# End-to-end dispatch via main([...]) — the expensive torch surface.
# One module-scoped 1-epoch checkpoint feeds every infer / dump-weights test.
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train a tiny 1-epoch PING checkpoint once; reuse for infer/dump-weights.

    module-scoped so the ~5s train cost is paid a single time.
    """
    out = tmp_path_factory.mktemp("trained")
    rc = cli.main(
        [
            "train",
            "--model",
            "ping",
            "--dataset",
            "mnist",
            "--max-samples",
            "60",
            "--epochs",
            "1",
            "--w-in",
            "10",
            "--w-in-initial-zero-fraction",
            "0",
            "--out-dir",
            str(out),
            "--wipe-dir",
            *_FAST,
        ]
    )
    assert rc == 0
    assert (out / "weights.pth").exists()
    assert (out / "config.json").exists()
    return out


class TestDispatchRuns:
    @pytest.mark.parametrize("tau_ms", [6.0, 9.0])
    def test_single_trial_gaba_decay_matches_requested_constant(self, tmp_path, tau_ms):
        import numpy as np

        out = tmp_path / "gaba-decay"
        assert (
            cli.main(
                [
                    "sim",
                    "--model",
                    "ping",
                    "--input",
                    "synthetic-spikes",
                    "--n-hidden",
                    "64",
                    "--n-inh",
                    "16",
                    "--n-in",
                    "128",
                    "--input-rate",
                    "45",
                    "--ei-strength",
                    "1.5",
                    "--w-in",
                    "1.5",
                    "0.3",
                    "--seed",
                    "42",
                    "--dt",
                    "0.1",
                    "--t-ms",
                    "200",
                    "--tau-gaba",
                    str(tau_ms),
                    "--output-fields",
                    "gi_e_1",
                    "--out-dir",
                    str(out),
                ]
            )
            == 0
        )
        with np.load(out / "recording.npz") as recording:
            conductance = recording["gi_e_1"]
        before, after = conductance[:-1], conductance[1:]
        # Between inhibitory arrivals, recorded conductance must follow the
        # requested exponential decay, independently of metadata or spike rates.
        decaying = (before > 1e-6) & (after < before)
        assert decaying.sum() > 10
        np.testing.assert_allclose(
            after[decaying] / before[decaying],
            np.exp(-0.1 / tau_ms),
            rtol=2e-6,
        )

    def test_artifact_writing_requires_explicit_out_dir(self):
        with pytest.raises(SystemExit, match="requires an explicit --out-dir"):
            cli.main(["sim"])

    def test_run_sim_metrics_snapshot(self, tmp_path):
        # Metrics-only synthetic-spikes sim → recording.npz (the _run_sim tail).
        out = tmp_path / "sim"
        rc = cli.main(
            [
                "sim",
                "--model",
                "ping",
                "--input",
                "synthetic-spikes",
                "--input-rate",
                "20",
                "--out-dir",
                str(out),
                "--wipe-dir",
                *_FAST,
            ]
        )
        assert rc == 0
        assert (out / "recording.npz").exists()

    def test_run_sim_cell_drive_tonic_path(self, tmp_path):
        # Historical config replay preserves the per-cell tonic drive path.
        out = tmp_path / "celldrive"
        saved = tmp_path / "config.json"
        saved.write_text(json.dumps({"quenched_drive": [0.5, 0.1]}))
        rc = cli.main(
            [
                "sim",
                "--model",
                "ping",
                "--input",
                "synthetic-spikes",
                "--load-config",
                str(saved),
                "--out-dir",
                str(out),
                "--wipe-dir",
                *_FAST,
            ]
        )
        assert rc == 0
        assert (out / "recording.npz").exists()

    def test_run_train_probe_epoch0(self, tmp_path):
        # epochs=0 probe path through _run_train (no weights.pth, but config +
        # metrics land).
        out = tmp_path / "probe0"
        rc = cli.main(
            [
                "train",
                "--model",
                "ping",
                "--dataset",
                "mnist",
                "--max-samples",
                "50",
                "--epochs",
                "0",
                "--w-in",
                "10",
                "--w-in-initial-zero-fraction",
                "0",
                "--out-dir",
                str(out),
                "--wipe-dir",
                *_FAST,
            ]
        )
        assert rc == 0
        assert (out / "config.json").exists()

    def test_emit_probe_batched(self, tmp_path):
        # --n-batch routes _run_sim into the batched-Poisson _emit_probe branch.
        out = tmp_path / "probe"
        rc = cli.main(
            [
                "sim",
                "--model",
                "ping",
                "--n-batch",
                "4",
                "--n-in",
                "32",
                "--out-dir",
                str(out),
                "--wipe-dir",
                *_FAST,
            ]
        )
        assert rc == 0
        assert (out / "recording.npz").exists()

    def test_emit_infer_test_accuracy(self, tmp_path, trained):
        # sim --infer test-set eval path through _emit_infer (non-snapshot).
        out = tmp_path / "infer"
        rc = cli.main(
            [
                "sim",
                "--infer",
                "--load-config",
                str(trained / "config.json"),
                "--load-weights",
                str(trained / "weights.pth"),
                "--max-samples",
                "60",
                "--out-dir",
                str(out),
                "--wipe-dir",
            ]
        )
        assert rc == 0
        metrics = json.loads((out / "metrics.json").read_text())
        assert metrics  # non-empty metrics dict

    def test_emit_infer_snapshot_mode(self, tmp_path, trained, monkeypatch):
        # --digit triggers snapshot_mode inside _run_sim → _emit_infer's
        # infer_and_snapshot branch. NOTE: _run_sim decides snapshot_mode by
        # scanning the *process* sys.argv (not the argv passed to main), so an
        # in-process call must set sys.argv for the snapshot branch to fire.
        out = tmp_path / "snap"
        argv = [
            "sim",
            "--infer",
            "--load-config",
            str(trained / "config.json"),
            "--load-weights",
            str(trained / "weights.pth"),
            "--digit",
            "0",
            "--sample",
            "0",
            "--out-dir",
            str(out),
            "--wipe-dir",
        ]
        monkeypatch.setattr("sys.argv", ["pinglab-cli", *argv])
        rc = cli.main(argv)
        assert rc == 0
        # snapshot mode records a single-sample spike trajectory to recording.npz.
        assert (out / "recording.npz").exists()

    def test_dump_weights(self, tmp_path, trained):
        out = tmp_path / "dump"
        rc = cli.main(
            [
                "dump-weights",
                "--load-config",
                str(trained / "config.json"),
                "--load-weights",
                str(trained / "weights.pth"),
                "--out-dir",
                str(out),
                "--wipe-dir",
            ]
        )
        assert rc == 0
        assert (out / "weights_dump.npz").exists()
