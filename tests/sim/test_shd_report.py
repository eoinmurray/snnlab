"""Retained-report provenance and actual-kernel accounting, without training."""

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmarks/shd/results/shd-59e3371bb384"
spec = importlib.util.spec_from_file_location(
    "shd_report_plot", ROOT / "benchmarks/shd/plot.py"
)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def test_streaming_decoder_handles_large_events_and_escaped_strings(tmp_path):
    events = [
        {"name": 'escaped "value"', "args": {"padding": "x" * (1024 * 1024 + 7)}},
        {"name": "last"},
    ]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"schemaVersion": 1, "traceEvents": events}))
    assert list(plot.trace_events(path)) == events


def test_extractor_excludes_gpu_annotation_spans_and_ignores_event_order(tmp_path):
    events = [
        {"ph": "X", "cat": "kernel", "name": "actual", "ts": 1100, "dur": 2},
        {
            "ph": "X",
            "cat": "gpu_user_annotation",
            "name": "snnlab::forward",
            "ts": 1000,
            "dur": 999999,
        },
        {
            "ph": "X",
            "cat": "user_annotation",
            "name": "snnlab::neuron/E1",
            "ts": 1050,
            "dur": 20,
        },
        {
            "ph": "X",
            "cat": "user_annotation",
            "name": "snnlab::forward",
            "ts": 1000,
            "dur": 150000,
        },
        {"ph": "X", "cat": "kernel", "name": "actual", "ts": 200000, "dur": 3},
    ]
    raw = tmp_path / "trace.json"
    raw.write_text(json.dumps({"traceEvents": events}))
    result = plot.extract_trace(raw, tmp_path / "excerpt.json")
    assert result["kernels"] == [{"name": "actual", "calls": 2, "duration_us": 5}]
    assert sorted(row["lane"] for row in result["events"]) == [
        "CUDA kernels",
        "Host: neuron calls",
    ]
    assert result["raw_sha256"] == plot.digest(raw)
    assert result["events"][0]["start_ms"] == 0.1


def test_incomplete_trace_is_rejected(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"traceEvents": [{"name": "unfinished')
    with pytest.raises(ValueError, match="incomplete"):
        list(plot.trace_events(path))


@pytest.fixture
def retained(tmp_path):
    destination = tmp_path / plot.RUN_ID
    shutil.copytree(
        SOURCE, destination, ignore=shutil.ignore_patterns("*.png", "*.svg", "*.pdf")
    )
    return destination


def test_retained_report_uses_complete_baseline_and_early_diagnostic(retained):
    cfg, metrics, timing, tracing, excerpt, analysis = plot.analyze(retained)
    assert analysis["baseline"]["final_validation_accuracy"] == pytest.approx(375 / 816)
    assert analysis["baseline"]["examples_per_s"] == pytest.approx(
        73400 / timing["training_s"]
    )
    assert sum(row["calls"] for row in excerpt["kernels"]) == 210551
    assert [row["batch"] for row in tracing["captured_updates"]] == [6, 7, 8]
    for row in tracing["activity"]:
        assert row["mean_rate_hz"] == pytest.approx(row["density"] * 1000)
    assert all(row["zero_fraction"] == 0 for row in tracing["gradients"])
    assert all(
        row["nonzero_blocks"] == row["blocks"]
        for row in tracing["weights"]
        if row["parameter"] != "score_gain.weight"
    )


def test_changed_batch_size_cannot_silently_relabel_report(retained):
    path = retained / "configuration.json"
    config = json.loads(path.read_text())
    config["configuration"]["batch_size"] = 32
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configuration differs"):
        plot.analyze(retained)


def test_report_links_and_published_figures_match_retained_run():
    content = (ROOT / "docs/content/docs/shd-benchmark.mdx").read_text()
    links = re.findall(r"\]\((/shd-benchmark/[^)]+)\)", content)
    assert links
    for link in links:
        assert (ROOT / "docs/public" / link.lstrip("/")).is_file(), link
    public = ROOT / "docs/public/shd-benchmark" / plot.RUN_ID
    manifest = json.loads((SOURCE / "presentation.json").read_text())
    assert manifest["renderer_sha256"] == plot.digest(plot.__file__)
    for name, expected in manifest["figures"].items():
        assert plot.digest(SOURCE / name) == plot.digest(public / name) == expected
