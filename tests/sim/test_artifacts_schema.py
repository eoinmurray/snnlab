"""Schema / contract tests for the machine-read run artifacts.

These files ARE the CLI↔notebook boundary — notebooks parse config.json,
run.jsonl, metrics.jsonl, metrics.json and weights.pth, so a renamed key breaks
a notebook silently at read time. These tests pin the shape so that break
happens here, loudly, instead.

Most of this exercises the writers directly (fast, no training); the on-disk
end-to-end check that a real train→infer produces the documented metrics.json is
marked slow.
"""

from __future__ import annotations

import json

import pytest
import torch

from snnlab.sim import models as M
from snnlab.sim import runlog
from snnlab.sim.config import build_net, setup_model_globals
from snnlab.sim.infer import _hidden_sizes_from_state_dict

# ── run.jsonl — the canonical event spine ─────────────────────────────────


def test_run_jsonl_is_typed_jsonl(tmp_path):
    """Every line is a JSON object carrying at least {ts, event}; events land in
    order with their fields intact. run.jsonl is the lossless machine record."""
    runlog.init_events(tmp_path)
    runlog.event("run_start", mode="train")
    runlog.event("config", model="ping", dt=0.1)
    runlog.event("epoch", epoch=1, acc=42.0)
    runlog.event("summary", best_acc=42.0)
    runlog.close_events()

    lines = (tmp_path / "run.jsonl").read_text().splitlines()
    recs = [json.loads(ln) for ln in lines]
    assert [r["event"] for r in recs] == ["run_start", "config", "epoch", "summary"]
    assert all("ts" in r for r in recs)
    assert recs[2]["epoch"] == 1 and recs[2]["acc"] == 42.0


def test_event_is_noop_when_not_initialised(tmp_path):
    """event() must be a safe no-op once the log is closed — importing/using
    runlog outside a run cannot crash or write stray files."""
    runlog.close_events()  # ensure no live sink
    runlog.event("stray", x=1)  # must not raise
    assert not (tmp_path / "run.jsonl").exists()


# ── metrics.jsonl — per-epoch timeseries sidecar ──────────────────────────


def test_metrics_jsonl_autostamps_and_preserves_fields(tmp_path):
    m = runlog.MetricsJsonl(tmp_path / "metrics.jsonl")
    m.write(epoch=1, acc=10.0, loss=2.0)
    m.write(epoch=2, acc=20.0, loss=1.0)
    m.close()

    rows = [json.loads(ln) for ln in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all("timestamp" in r for r in rows)  # auto-added if not supplied
    assert rows[0]["epoch"] == 1 and rows[1]["acc"] == 20.0


# ── weights.pth — the state_dict key convention ───────────────────────────


def test_weights_state_dict_key_convention(tmp_path):
    """A saved COBANet checkpoint uses W_ff.<i> (ParameterList: in→h1→…→out) and
    W_ee/W_ei/W_ie/W_ii.<layer> (ParameterDict). This is the contract that
    infer()'s _hidden_sizes_from_state_dict, dump-weights, and notebooks read."""
    M.N_IN = 784
    setup_model_globals([32])
    net = build_net("ping", hidden_sizes=[32], device="cpu")

    path = tmp_path / "weights.pth"
    torch.save(net.state_dict(), path)
    state = torch.load(path)

    assert {"W_ff.0", "W_ff.1", "W_ee.1", "W_ei.1", "W_ie.1", "W_ii.1"} <= set(state)
    # W_ff.0 is (N_IN, n_hidden); the readout W_ff.1 is (n_hidden, N_OUT).
    assert tuple(state["W_ff.0"].shape) == (784, 32)
    # The key convention must round-trip through the size-recovery helper.
    assert _hidden_sizes_from_state_dict(state) == [32]


# ── config.json — provenance + resolved run config ────────────────────────


def test_config_json_has_mode_and_provenance(tmp_path):
    """config.json leads with `mode` + provenance (git_sha, run_id, env hash,
    torch version), then every non-None arg. Notebooks read run params from it."""
    from snnlab.sim.tool import parse_args, save_run_artifacts

    args = parse_args(["sim", "--out-dir", str(tmp_path), "--model", "ping"])
    save_run_artifacts(tmp_path, args, "sim")
    runlog.close_events()  # release the run.jsonl handle opened by save_run_artifacts

    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["mode"] == "sim"
    assert {"git_sha", "python_env_hash", "run_id", "torch_version"} <= cfg.keys()
    assert cfg["model"] == "ping"


# ── end-to-end: the real metrics.json a train→infer writes ────────────────


@pytest.mark.slow
class TestEndToEndArtifacts:
    """A real (tiny) train→infer, then validate the on-disk artifacts' schema.
    Complements the writer tests above by proving the actual pipeline emits the
    documented contract."""

    @pytest.fixture(scope="class")
    def run_dir(self, tmp_path_factory):
        from snnlab.sim.train import train

        d = tmp_path_factory.mktemp("artifacts")
        train(
            model_name="ping", dt=0.1, t_ms=100.0, epochs=2, dataset="mnist",
            max_samples=100, lr=0.01, hidden_sizes=[32], seed=42, out_dir=d,
        )
        return d

    def test_train_writes_weights_and_config(self, run_dir):
        for name in ("weights.pth", "metrics.json", "metrics.jsonl"):
            assert (run_dir / name).exists(), f"train did not write {name}"
        # config.json now carries `mode` (like sim/infer) plus the resolved train
        # params + provenance.
        cfg = json.loads((run_dir / "config.json").read_text())
        assert cfg["mode"] == "train"
        assert {"model", "dataset", "n_hidden", "git_sha", "run_id"} <= cfg.keys()
        assert cfg["model"] == "ping" and cfg["dataset"] == "mnist"

    def test_infer_metrics_json_schema(self, run_dir, tmp_path):
        from snnlab.sim.infer import infer

        infer(
            model_name="ping", dt=0.1, t_ms=100.0,
            load_weights=run_dir / "weights.pth", dataset="mnist",
            max_samples=50, hidden_sizes=[32], out_dir=tmp_path,
        )
        m = json.loads((tmp_path / "metrics.json").read_text())
        assert m["mode"] == "infer"
        assert {"model", "config", "best_acc", "n_correct", "n_total",
                "rates_hz", "hid_rate_hz"} <= m.keys()
        assert {"dt", "t_ms", "n_hidden", "n_inh", "n_in", "dataset"} <= m["config"].keys()
        assert isinstance(m["n_correct"], int) and isinstance(m["n_total"], int)
