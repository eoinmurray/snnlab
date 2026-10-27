"""Deadline scheduling, combined plots and streaming numerical checks."""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/streaming/benchmark.py"
spec = importlib.util.spec_from_file_location("streaming_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize(
    "changes",
    [
        {"dt_ms": 0.3},
        {"windows": 0},
        {"period_ms": 0},
        {"spin_us": -1},
        {"spin_us": float("nan")},
        {"period_ms": float("nan")},
        {"period_ms": 1e-8},
        {"windows": 1.5},
        {"sparsity": 1.0},
        {"rate_hz": float("nan")},
        {"hidden": 0},
        {"window_ms": float("inf")},
        {"block_steps": 65},
    ],
)
def test_invalid_conditions_fail_before_execution(changes):
    with pytest.raises(ValueError):
        replace(benchmark.Config(), **changes).validate()


@pytest.mark.parametrize("mode", ["eager", "default"])
def test_streaming_timing_bars_and_numerical_gate(tmp_path, monkeypatch, mode):
    original = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kw: original(fn, backend="eager", fullgraph=True, dynamic=False),
    )
    cfg = benchmark.Config(hidden=32, windows=3, compile_mode=mode)
    previous_threads = torch.get_num_threads()
    previous_precision = torch.get_float32_matmul_precision()
    try:
        report = benchmark.run(cfg)
    finally:
        torch.set_num_threads(previous_threads)
        torch.set_float32_matmul_precision(previous_precision)
    assert report["validation"]["passed"]
    assert report["validation"]["probe_spikes"]["hidden_spikes"] > 0
    expected = "compiled_blocks" if mode == "default" else "kernel_loop"
    assert report["validation"]["initial_execution_path"]["selected"] == expected
    phases = report["advance_breakdown"]
    assert phases["windows"] == 3
    assert phases["numerical_gate_passed"]
    assert phases["execution_path"]["selected"] == expected
    names = {row["name"] for row in phases["stages"]}
    assert ("State restore / init" in names) == (mode == "eager")
    assert ("Runtime-state snapshot" in names) == (mode == "eager")
    assert ("Compiled numerical block" in names) == (mode == "default")
    for data in (phases, report["breakdown"]):
        for key in ("wall_us", "thread_cpu_us"):
            assert data["stages"][0][key] == pytest.approx(
                sum(s[key] for s in data["stages"][1:])
            )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"cases": [report]}, allow_nan=False))
    deadlines = report["deadlines"]
    assert deadlines["windows"] == 3
    assert deadlines["completed_steps"] == cfg.steps * cfg.windows
    assert deadlines["period_ms"] == 1.0
    raw = benchmark.deadline_arrays(
        np.asarray(deadlines["samples"], dtype=np.int64), 1_000_000
    )
    np.testing.assert_array_equal(raw["release_ns"], np.arange(3) * 1_000_000)
    np.testing.assert_array_equal(raw["deadline_ns"], raw["release_ns"] + 1_000_000)
    assert (raw["start_ns"] >= raw["release_ns"]).all()
    assert (raw["finish_ns"] >= raw["start_ns"]).all()
    np.testing.assert_array_equal(
        raw["deadline_missed"], raw["finish_ns"] > raw["deadline_ns"]
    )
    assert deadlines["missed_deadlines"] == int(raw["deadline_missed"].sum())
    assert deadlines["callback_budget_overruns"] == int(raw["callback_overrun"].sum())
    image = benchmark.plot_report(report_path)
    assert image.name == "streaming.png"
    assert image.read_bytes().startswith(b"\x89PNG")
    assert not (tmp_path / "breakdown.png").exists()
    assert not (tmp_path / "advance-internals.png").exists()
    assert {p.name for p in tmp_path.iterdir()} == {"report.json", "streaming.png"}
    assert not list(tmp_path.glob("*.pstats"))
    assert not list(tmp_path.glob("*-trace.json"))
    # Redrawing uses the same single-file entry point and never executes a model.
    monkeypatch.setattr(
        benchmark, "run", lambda *a: pytest.fail("redraw executed model")
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--plot-only", str(tmp_path)])
    benchmark.main()
    assert benchmark.plot_detailed_report(report_path).name == "streaming-details.png"


@pytest.mark.parametrize(
    "durations, oversleep, starts, finishes, misses, overruns",
    [
        (
            [250, 10, 10],
            0,
            [0, 250, 260],
            [250, 260, 270],
            [True, True, False],
            [True, False, False],
        ),
        (
            [20, 30, 10],
            80,
            [0, 180, 210],
            [20, 210, 220],
            [False, True, False],
            [False, False, False],
        ),
        (
            [100, 100, 100],
            0,
            [0, 100, 200],
            [100, 200, 300],
            [False, False, False],
            [False, False, False],
        ),
    ],
)
def test_absolute_deadlines_catch_up_without_dropping_or_resetting(
    durations,
    oversleep,
    starts,
    finishes,
    misses,
    overruns,
):
    now = 0
    calls = 0

    def clock():
        return now

    def sleep(seconds):
        nonlocal now
        now += round(seconds * 1e9) + oversleep

    def callback():
        nonlocal now, calls
        now += durations[calls]
        calls += 1

    samples = benchmark.deadline_schedule(callback, 3, 100, clock=clock, sleep=sleep)
    data = benchmark.deadline_arrays(samples, 100)
    assert calls == 3
    np.testing.assert_array_equal(data["release_ns"], [0, 100, 200])
    np.testing.assert_array_equal(data["start_ns"], starts)
    np.testing.assert_array_equal(data["finish_ns"], finishes)
    np.testing.assert_array_equal(data["deadline_missed"], misses)
    np.testing.assert_array_equal(data["callback_overrun"], overruns)


def test_legacy_reports_redraw_without_inventing_deadline_data(tmp_path):
    report = {
        "cases": [
            {
                "config": {"hidden": 32, "device": "cpu"},
                "breakdown": {
                    "windows": 3,
                    "stages": [
                        {"name": "Full callback", "wall_us": 100, "thread_cpu_us": 90}
                    ],
                },
            }
        ]
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    assert benchmark.plot_report(path).read_bytes().startswith(b"\x89PNG")


def test_callback_summary_pools_raw_durations_and_separates_late_starts(tmp_path):
    # Three cheap callbacks start late; only one callback actually exceeds budget.
    runs = [
        {
            "deadlines": {
                "period_ms": 1,
                "samples": [
                    [0, 0, 2_000_000],
                    [1_000_000, 2_000_000, 2_100_000],
                ],
            },
        },
        {
            "deadlines": {
                "period_ms": 1,
                "samples": [
                    [0, 950_000, 1_050_000],
                    [1_000_000, 2_000_000, 2_100_000],
                    [2_000_000, 2_100_000, 2_200_000],
                ],
            },
        },
    ]
    summary = benchmark.callback_summary(runs)
    assert summary["windows"] == 5
    assert summary["seeds"] == 2
    assert summary["mean_ms"] == pytest.approx(0.48)
    # Linear sample percentiles of [0.1, 0.1, 0.1, 0.1, 2.0].
    assert summary["p95_ms"] == pytest.approx(1.62)
    assert summary["p99_ms"] == pytest.approx(1.924)
    assert summary["missed_deadlines"] == 4
    assert summary["callback_overruns"] == 1
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"cases": [{"config": {"hidden": 32}, "runs": runs}]}))
    assert benchmark.plot_report(path).read_bytes().startswith(b"\x89PNG")


def test_callback_summary_rejects_missing_samples_and_mixed_budgets():
    with pytest.raises(ValueError, match="requires paced samples"):
        benchmark.callback_summary([{}])
    runs = [
        {"deadlines": {"period_ms": period, "samples": [[0, 0, 100_000]]}}
        for period in (1, 2)
    ]
    with pytest.raises(ValueError, match="same deadline budget"):
        benchmark.callback_summary(runs)


def test_interval_rates_reset_and_use_actual_partial_bin_counts():
    rates = benchmark.deadline_rates(
        np.array([0, 99, 100, 199, 200]) * 1_000_000,
        np.array([True, False, False, False, True]),
        np.array([False, False, True, False, False]),
    )
    np.testing.assert_array_equal(rates["edges_ms"], [0, 100, 200, 300])
    np.testing.assert_array_equal(rates["windows"], [2, 2, 1])
    np.testing.assert_array_equal(rates["miss_pct"], [50, 0, 100])
    np.testing.assert_array_equal(rates["overrun_pct"], [0, 50, 0])
    empty = benchmark.deadline_rates(
        np.array([0, 200_000_000]),
        [False, True],
        [False, False],
    )
    assert np.isnan(empty["miss_pct"][1])
    assert empty["miss_pct"][2] == 100


def test_new_run_overwrites_results_and_removes_stale_cases(tmp_path, monkeypatch):
    (tmp_path / "notes.txt").write_text("preserve unrelated files")
    (tmp_path / "streaming.png").write_text("old plot")
    (tmp_path / "case-999-hidden-64-deadlines.npz").write_bytes(b"legacy samples")

    def run(cfg):
        return {"config": {"hidden": cfg.hidden, "seed": cfg.seed}}

    def plot(path):
        output = path.with_name("streaming.png")
        output.write_text("new plot")
        return output

    monkeypatch.setattr(benchmark, "run", run)
    monkeypatch.setattr(benchmark, "plot_report", plot)
    for sizes in ([32, 64], [32]):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(SCRIPT),
                "--output",
                str(tmp_path),
                "--hidden-sizes",
                *map(str, sizes),
            ],
        )
        benchmark.main()
        report = json.loads((tmp_path / "report.json").read_text())
        assert [c["config"]["hidden"] for c in report["cases"]] == sizes
        assert not list(tmp_path.glob("*.npz"))
        assert report["seeds"] == [7, 8, 9, 10, 11]
        assert report["complete"]
        assert report["wall_time_seconds"] > 0
        for case in report["cases"]:
            assert [r["config"]["seed"] for r in case["runs"]] == report["seeds"]
    assert (tmp_path / "streaming.png").read_text() == "new plot"
    assert (tmp_path / "notes.txt").read_text() == "preserve unrelated files"
    assert all(p.is_file() for p in tmp_path.iterdir())


def test_spin_guard_sleeps_early_but_never_releases_early():
    now = -1
    sleeps = []
    starts = []

    def clock():
        nonlocal now
        now += 1
        return now

    def sleep(seconds):
        nonlocal now
        duration = round(seconds * 1e9)
        sleeps.append(duration)
        now += duration

    samples = benchmark.deadline_schedule(
        lambda: starts.append(now), 3, 100, spin_ns=20, clock=clock, sleep=sleep
    )
    assert sleeps and all(duration < 100 for duration in sleeps)
    assert len(starts) == 3
    assert (samples[:, 1] >= samples[:, 0]).all()
    np.testing.assert_array_equal(samples[:, 0], [0, 100, 200])


def test_across_seed_sd_and_misses_are_not_thresholded_after_averaging(tmp_path):
    mean, sd = benchmark.mean_sd([1, 2, 3, 4, 5])
    assert mean == 3
    assert sd == pytest.approx(np.sqrt(2.5))
    runs = []
    for seed, duration in enumerate([2_000_000, 100_000, 100_000, 100_000, 100_000]):
        metric = {
            "name": "Full callback",
            "wall_us": seed + 1,
            "thread_cpu_us": seed + 1,
        }
        runs.append(
            {
                "config": {"hidden": 32, "device": "cpu", "seed": seed},
                "validation": {"passed": True},
                "breakdown": {"stages": [metric]},
                "advance_breakdown": {"stages": [metric]},
                "deadlines": {
                    "period_ms": 1,
                    "windows": 1,
                    "samples": [[0, 0, duration]],
                    "missed_deadlines": int(duration > 1_000_000),
                    "callback_budget_overruns": int(duration > 1_000_000),
                    "miss_rate": int(duration > 1_000_000),
                    "response_ms": {"maximum": duration / 1e6},
                    "caller_thread_cpu_percent": 100,
                },
            }
        )
    summary = benchmark.summarize_seeds(runs)
    assert summary["sd_defined"] and summary["seed_count"] == 5
    assert summary["breakdown"]["stages"][0]["wall_us_sd"] == pytest.approx(
        np.sqrt(2.5)
    )
    d = summary["deadlines"]
    assert d["total_missed_deadlines"] == 1
    assert d["total_windows"] == 5
    assert d["missed_deadlines_mean"] == pytest.approx(0.2)
    assert d["missed_deadlines_sd"] == pytest.approx(0.2 * np.sqrt(5))
    series = benchmark.interval_seed_series(runs)
    assert series["response_ms"][0] == pytest.approx(0.48)
    assert series["response_ms_sd"][0] == pytest.approx(0.38 * np.sqrt(5))
    assert series["miss_pct"][0] == pytest.approx(20)
    assert series["miss_pct_sd"][0] == pytest.approx(20 * np.sqrt(5))
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"cases": [{"config": runs[0]["config"], "runs": runs}]})
    )
    assert benchmark.plot_report(report_path).read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("seeds", [[1, 1], [-1, 2]])
def test_seed_ids_must_be_distinct_and_nonnegative(seeds, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--seeds", *map(str, seeds), "--output", str(tmp_path / "new")],
    )
    with pytest.raises(SystemExit):
        benchmark.main()
    assert not (tmp_path / "new").exists()
