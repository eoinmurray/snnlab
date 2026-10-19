"""Tracing is bounded, honest about scope and numerically non-interfering."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from snnlab.sim import GraphExecutor, TraceSpec
from snnlab.sim.diagnostics.tracing import TraceSession, current_trace, timeline_metrics
from snnlab.sim.requests.spec import ExecutionSpec
from snnlab.sim.training.trainer import train
from tests.sim.test_execution_epochs import request


@pytest.mark.parametrize("mode", ["monitor", "profile", "both"])
def test_modes_preserve_training_and_rng_and_capture_only_selected_updates(
    tmp_path, mode
):
    spec = request()
    baseline = train(spec)
    result = train(
        replace(
            spec,
            tracing=TraceSpec(
                tmp_path / mode, mode=mode, skip_updates=1, active_updates=2
            ),
        )
    )
    for name in baseline.parameters:
        torch.testing.assert_close(
            baseline.parameters[name], result.parameters[name], rtol=0, atol=0
        )
    for name in baseline.gradients:
        torch.testing.assert_close(
            baseline.gradients[name], result.gradients[name], rtol=0, atol=0
        )
    for name, state in baseline.optimizer_state.items():
        for key, value in state.items():
            torch.testing.assert_close(
                value, result.optimizer_state[name][key], rtol=0, atol=0
            )
    torch.testing.assert_close(
        baseline.training_checkpoint.rng_state,
        result.training_checkpoint.rng_state,
        rtol=0,
        atol=0,
    )
    assert baseline.metrics["epochs"] == result.metrics["epochs"]
    assert (
        baseline.training_checkpoint.data_state == result.training_checkpoint.data_state
    )
    summary = result.metrics["tracing"]
    assert summary["status"] == "complete"
    assert [row["update_index"] for row in summary["captured_updates"]] == [1, 2]
    assert current_trace() is None
    assert all(
        item["status"] == "unavailable" for item in summary["unavailable"].values()
    )
    assert summary["window_wall_s"] > 0
    assert summary["captured_examples"] == 3
    assert summary["instrumented_examples_per_s"] > 0
    assert (tmp_path / mode / "trace.json").exists() == (mode in {"profile", "both"})
    if mode in {"monitor", "both"}:
        inputs = next(
            row for row in summary["activity"] if row["signal"] == "input/events"
        )
        assert inputs["density"] == 0.5
        assert inputs["example_steps"] == 9
        assert inputs["silent_example_step_fraction"] == 0
        assert summary["memory"]
    else:
        assert not summary["activity"]
    if mode in {"profile", "both"}:
        exported = json.loads((tmp_path / mode / "trace.json").read_text())
        names = {event.get("name") for event in exported["traceEvents"]}
        assert "snnlab::forward" in names
        assert "snnlab::backward" in names
        assert (tmp_path / mode / "operators.csv").read_text()


def test_disabled_tracing_never_opens_profiler_or_creates_output(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("disabled tracing must not open a session or profiler")

    monkeypatch.setattr(TraceSession, "__enter__", forbidden)
    monkeypatch.setattr(torch.profiler, "profile", forbidden)
    result = train(replace(request(), tracing=TraceSpec(tmp_path / "off", mode="off")))
    assert "tracing" not in result.metrics
    assert not (tmp_path / "off").exists()


@pytest.mark.parametrize(
    "skip,active,status,count", [(0, 99, "truncated", 6), (99, 2, "empty", 0)]
)
def test_short_and_empty_windows_are_reported(tmp_path, skip, active, status, count):
    result = train(
        replace(
            request(),
            tracing=TraceSpec(
                tmp_path / status,
                mode="profile",
                skip_updates=skip,
                active_updates=active,
            ),
        )
    )
    summary = result.metrics["tracing"]
    assert summary["status"] == status
    assert len(summary["captured_updates"]) == count
    assert (tmp_path / status / "trace.json").exists() == bool(count)


def test_known_activity_and_weight_sparsity(tmp_path):
    model = SimpleNamespace(
        plan=SimpleNamespace(dt_ms=1.0, populations=[{"id": "E", "spiking": True}]),
        _neuron_steps={},
    )
    weights = {"w": torch.tensor([[1.0, 0], [0, 0]])}
    spikes = torch.tensor([[[1.0, 0], [0, 0]], [[0, 0], [0, 1.0]]])
    with TraceSession(
        TraceSpec(tmp_path / "known", skip_updates=0, active_updates=1, block_size=1),
        "cpu",
    ) as session:
        session.bind(model)
        session.begin_update(0, 1, 1, weights)
        session.observe_inputs({"x": spikes})
        for step in spikes:
            session.observe_spikes({"E": step})
        session.end_update(weights, last=True)
    summary = session.summary
    assert summary["weights"][0]["zero_fraction"] == 0.75
    assert summary["weights"][0]["zero_block_fraction"] == 0.75
    for item in summary["activity"]:
        assert item["density"] == 0.25
        assert item["mean_rate_hz"] == 250
        assert item["per_neuron_rate_hz"] == [250, 250]
        assert item["silent_example_step_fraction"] == 0.5
        assert item["globally_silent_step_fraction"] == 0


def test_timeline_uses_interval_union_and_copy_bytes_without_inventing_dram_bandwidth():
    events = [
        {"ph": "X", "cat": "kernel", "ts": 0, "dur": 10},
        {"ph": "X", "cat": "kernel", "ts": 5, "dur": 10},
        {"ph": "X", "cat": "kernel", "ts": 20, "dur": 5},
        {
            "ph": "X",
            "cat": "gpu_memcpy",
            "ts": 15,
            "dur": 5,
            "name": "Memcpy HtoD",
            "args": {"bytes": 5000},
        },
        {"ph": "X", "cat": "gpu_memcpy", "ts": 25, "dur": 1, "args": {}},
    ]
    summary = timeline_metrics(events, 0.0001)
    assert summary["kernel_interval_union_s"] == pytest.approx(0.00002)
    assert summary["kernel_coverage_fraction"] == pytest.approx(0.2)
    assert summary["max_internal_kernel_gap_s"] == pytest.approx(0.000005)
    assert summary["kernel_and_copy_interval_union_s"] == pytest.approx(0.000026)
    assert summary["copies"][0]["GB_per_s"] == 1
    assert summary["copies"][1]["GB_per_s"] is None


@pytest.mark.parametrize(
    "fields",
    [
        {"mode": "invented"},
        {"skip_updates": -1},
        {"active_updates": 0},
        {"active_updates": True},
        {"block_size": 0},
        {"profile_memory": 1},
    ],
)
def test_invalid_trace_specs(tmp_path, fields):
    with pytest.raises((TypeError, ValueError)):
        TraceSpec(tmp_path / "bad", **fields)


def test_trace_requires_typed_training_request(tmp_path):
    with pytest.raises(TypeError, match="TraceSpec"):
        replace(request(), tracing={})
    with pytest.raises(ValueError, match="requires kind='train'"):
        ExecutionSpec(kind="simulate", tracing=TraceSpec(tmp_path / "bad"))


def test_failure_closes_profiler_and_restores_context(tmp_path, monkeypatch):
    original = GraphExecutor._forward

    def fail(model, *args, **kwargs):
        session = current_trace()
        if session is not None and session.active:
            raise RuntimeError("injected forward failure")
        return original(model, *args, **kwargs)

    monkeypatch.setattr(GraphExecutor, "_forward", fail)
    with pytest.raises(RuntimeError, match="injected forward failure"):
        train(
            replace(
                request(),
                tracing=TraceSpec(
                    tmp_path / "failed",
                    mode="profile",
                    skip_updates=0,
                    active_updates=1,
                ),
            )
        )
    assert current_trace() is None
    summary = json.loads((tmp_path / "failed/summary.json").read_text())
    assert summary["status"] == "failed"
    assert "injected forward failure" in summary["error"]
    assert (tmp_path / "failed/trace.json").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_stream_timing_and_profiler_activity(tmp_path):
    result = train(
        replace(
            request(),
            device="cuda:0",
            compile_mode="default",
            tracing=TraceSpec(
                tmp_path / "cuda", mode="both", skip_updates=1, active_updates=1
            ),
        )
    )
    summary = result.metrics["tracing"]
    assert summary["timeline"]["kernel_count"] > 0
    assert any(row.get("cuda_stream_elapsed_s", 0) > 0 for row in summary["phases"])
    assert summary["memory"][-1]["cuda_peak_allocated_bytes"] > 0


def test_tracing_does_not_enter_resume_identity(tmp_path):
    from snnlab.sim import save_training_checkpoint

    spec = request()
    full = train(spec)
    partial = train(
        replace(
            spec,
            updates=2,
            tracing=TraceSpec(
                tmp_path / "partial-trace",
                mode="both",
                skip_updates=0,
                active_updates=1,
            ),
        )
    )
    checkpoint = save_training_checkpoint(
        tmp_path / "checkpoint", partial.training_checkpoint
    )
    resumed = train(replace(spec, checkpoint=checkpoint))
    for name in full.parameters:
        torch.testing.assert_close(
            full.parameters[name], resumed.parameters[name], rtol=0, atol=0
        )
    torch.testing.assert_close(
        full.training_checkpoint.rng_state,
        resumed.training_checkpoint.rng_state,
        rtol=0,
        atol=0,
    )
    assert full.training_checkpoint.data_state == resumed.training_checkpoint.data_state
