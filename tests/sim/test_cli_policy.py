"""CLI removal and warning policy, including historical checkpoint replay."""

import json
import re
import warnings

import pytest

from snnlab.sim.cli_policy import (
    DEPRECATED_LEGACY_FLAGS,
    LEGACY_CONFIG_DEFAULTS,
    REMOVED_LEGACY_FLAGS,
    LegacyCLIWarning,
)
from snnlab.sim.tool import parse_args


@pytest.mark.parametrize("flag", sorted(REMOVED_LEGACY_FLAGS))
def test_removed_flags_are_rejected(flag, capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["sim", flag])
    assert exc.value.code == 2
    assert "removed legacy CLI arguments: " + flag in capsys.readouterr().err


def test_removed_equals_form_and_abbreviations_are_rejected(capsys):
    with pytest.raises(SystemExit):
        parse_args(["sim", "--scale-w-ei=2"])
    assert "removed legacy CLI arguments" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        parse_args(["sim", "--scale-w", "2"])
    assert "unrecognized arguments" in capsys.readouterr().err


def test_warning_aggregates_explicit_legacy_flags_once():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_args(["sim", "--model=ping", "--n-hidden", "32", "--model", "ping"])
    assert len(caught) == 1
    assert caught[0].category is LegacyCLIWarning
    message = str(caught[0].message)
    assert message.count("--model") == 1
    assert "--n-hidden" in message
    assert "--dt" not in message
    assert "--executor graph --bundle" in message


def test_shared_and_graph_flags_do_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        args = parse_args(
            [
                "sim",
                "--executor",
                "graph",
                "--seed",
                "7",
                "--t-ms",
                "10",
                "--n-batch",
                "2",
                "--input-rate",
                "5",
                "--poisson-protocol",
                "fixed-rate",
            ]
        )
    assert not caught
    assert args.executor == "graph"


def test_defaults_do_not_warn_or_change():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        args = parse_args(["train"])
    assert not caught
    assert args.executor == "legacy"
    for key, value in LEGACY_CONFIG_DEFAULTS.items():
        assert getattr(args, key) == value


def test_help_marks_legacy_once_and_omits_removed_flags(capsys):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(SystemExit) as exc:
            parse_args(["sim", "--help"])
    assert exc.value.code == 0
    assert not caught
    help_text = capsys.readouterr().out
    assert "[deprecated legacy] [deprecated legacy]" not in help_text
    assert "[deprecated legacy]" in help_text
    for flag in REMOVED_LEGACY_FLAGS:
        assert not re.search(re.escape(flag) + r"(?![\w-])", help_text)


def test_historical_config_restores_removed_settings(tmp_path):
    values = {
        "signed_readout": True,
        "readout_bias": True,
        "state_clamp": True,
        "train_leak": True,
        "adaptive_threshold": True,
        "tau_m_e_bounds_ms": [6, 40],
        "recurrent_initial_zero_fraction": 0.5,
        "w_ei": [0.5, 0.1],
        "readout_w_out_scale": 2.0,
        "dales_law": False,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values))
    with pytest.warns(LegacyCLIWarning, match="--load-config"):
        args = parse_args(["sim", "--load-config", str(path)])
    for key, value in values.items():
        assert getattr(args, key) == value
    assert "--load-config" in DEPRECATED_LEGACY_FLAGS


def test_sim_only_fields_do_not_gain_new_config_replay_effects(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"scale_w_ei": 2.0, "transition_bundle": "old.bundle"}))
    with pytest.warns(LegacyCLIWarning):
        args = parse_args(["sim", "--load-config", str(path)])
    assert args.scale_w_ei == 1.0
    assert args.transition_bundle is None
