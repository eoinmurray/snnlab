import json
import re

import pytest

from snnlab.sim.runlog import (
    MetricsJsonl,
    WarningTracker,
    format_bytes,
    format_eta,
    list_output_files,
    provenance,
    run_id,
    write_test_predictions,
)


class TestProvenance:
    def test_required_fields_present(self):
        """Provenance dict must carry every field that reproduces a run."""
        p = provenance()
        required = {
            "git_sha",
            "torch_version",
            "device",
            "python_env_hash",
            "run_id",
            "started_at",
        }
        assert required.issubset(p.keys())
        for k in required:
            assert p[k], f"{k} should be non-empty"

    def test_run_id_format(self):
        assert re.match(r"^r-\d{8}-\d{6}$", run_id())


class TestMetricsJsonl:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        m = MetricsJsonl(path)
        m.write(epoch=0, acc=10.5, loss=2.3)
        m.write(epoch=1, acc=40.0, loss=1.1)
        m.close()

        lines = path.read_text().splitlines()
        assert len(lines) == 2
        a, b = json.loads(lines[0]), json.loads(lines[1])
        assert a["epoch"] == 0 and a["acc"] == 10.5
        assert b["epoch"] == 1 and b["loss"] == 1.1
        # Timestamp auto-injected on every write
        assert "timestamp" in a and "timestamp" in b

    def test_write_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deeper" / "metrics.jsonl"
        m = MetricsJsonl(path)
        m.write(epoch=0)
        m.close()
        assert path.exists()


class TestTestPredictions:
    def test_roundtrip(self, tmp_path):
        import numpy as np

        path = tmp_path / "preds.json"
        preds = [
            {"idx": 0, "true": 3, "pred": 3, "correct": True, "logits": [0.1, 0.9]},
            {
                "idx": 1,
                "true": 7,
                "pred": 2,
                "correct": False,
                # numpy scalars land here from argmax/max on tensors; default=float
                # coerces them.
                "confidence": np.float32(0.87),
            },
        ]
        write_test_predictions(path, preds)
        loaded = json.loads(path.read_text())
        assert loaded[0]["idx"] == 0 and loaded[0]["correct"] is True
        assert loaded[1]["pred"] == 2
        assert loaded[1]["confidence"] == pytest.approx(0.87, abs=1e-5)


class TestFormatters:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0, "0s"),
            (45, "45s"),
            (60, "1m00s"),
            (510, "8m30s"),
            (3600, "1h00m"),
            (4320, "1h12m"),
        ],
    )
    def test_format_eta(self, seconds, expected):
        assert format_eta(seconds) == expected

    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, "0 B"),
            (512, "512 B"),
            (2048, "2.0 KB"),
            (5 * 1024 * 1024, "5.0 MB"),
        ],
    )
    def test_format_bytes(self, n, expected):
        assert format_bytes(n) == expected


class TestListOutputFiles:
    def test_lists_nested_files_with_sizes(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.bin").write_bytes(b"\x00" * 32)

        files = dict(list_output_files(tmp_path))
        assert files["a.txt"] == 5
        assert files["sub/b.bin".replace("/", __import__("os").sep)] == 32

    def test_missing_dir_returns_empty(self, tmp_path):
        assert list_output_files(tmp_path / "nope") == []


class TestWarningTracker:
    """Flags require BOTH a prolonged streak (≥3) AND 'stuck' (no improvement
    for 5+ epochs). Extreme firing rates alone aren't pathological if the
    network is still learning.
    """

    def _warm_up_stuck(self, w, activity=30.0):
        """Prime the tracker: one good epoch, then 5 flat epochs so stuck=True."""
        w.tick(ep=0, acc=50.0, activity=activity)  # best=50
        for ep in range(1, 6):  # 5 flat epochs
            w.tick(ep=ep, acc=50.0, activity=activity)
        assert w.no_progress_since >= 5

    def test_dead_flag_requires_streak_and_stuck(self):
        w = WarningTracker()
        self._warm_up_stuck(w)
        # Dead streak just starting: no flag yet
        flags = w.tick(ep=6, acc=50.0, activity=0.5)
        assert not any("dead" in str(f) for f in flags)
        w.tick(ep=7, acc=50.0, activity=0.5)
        flags = w.tick(ep=8, acc=50.0, activity=0.5)  # streak reaches 3
        assert any("dead" in str(f) for f in flags)

    def test_dead_not_flagged_while_still_improving(self):
        w = WarningTracker()
        # Activity 0 every epoch but acc keeps climbing → not stuck, no flag
        for ep in range(10):
            flags = w.tick(ep=ep, acc=float(ep * 5), activity=0.0)
            assert not any("dead" in str(f) for f in flags)

    def test_saturated_flag_requires_streak_and_stuck(self):
        w = WarningTracker()
        self._warm_up_stuck(w)
        w.tick(ep=6, acc=50.0, activity=99.0)
        w.tick(ep=7, acc=50.0, activity=99.0)
        flags = w.tick(ep=8, acc=50.0, activity=99.0)
        assert any("saturated" in str(f) for f in flags)

    def test_nan_flag_immediate(self):
        w = WarningTracker()
        flags = w.tick(ep=0, acc=10.0, activity=30.0, loss=float("nan"))
        assert any("NaN" in str(f) for f in flags)

    def test_new_best_resets_stuck_counter(self):
        w = WarningTracker()
        w.tick(ep=0, acc=20.0, activity=30.0)
        assert w.best_acc == 20.0
        w.tick(ep=1, acc=30.0, activity=30.0)
        assert w.best_acc == 30.0
        assert w.no_progress_since == 0

    def test_warnings_aggregated_into_contiguous_ranges(self):
        w = WarningTracker()
        self._warm_up_stuck(w)
        # 5 more flat epochs with zero activity → 'dead' fires for all 5;
        # the ranged record should be a single span, not 5 individual rows.
        for ep in range(6, 11):
            w.tick(ep=ep, acc=50.0, activity=0.0)
        dead_lines = [l for l in w.summary_lines() if "dead" in l]
        assert len(dead_lines) == 1, f"expected single span, got {dead_lines}"
