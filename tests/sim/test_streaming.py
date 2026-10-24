"""Prepared-stream numerical continuation, ownership and lifecycle contracts."""

import copy

import pytest
import torch

from snnlab import lang
from snnlab.sim import (
    GraphExecutor,
    Performance,
    StreamSession,
    plan_graph,
    prepare_stream,
)


def circuit():
    net = lang.Network("stream-contract", dt=0.1 * lang.ms)
    source = net.input(
        "events", shape=("time", "batch", 3), signal_type="spikes", unit="spike"
    )
    cells = net.population(
        "cells",
        size=4,
        neuron=lang.CUBA_LIF(tau_mem=20 * lang.ms, refractory=0.2 * lang.ms),
    )
    net.connect(
        source,
        cells.excitatory,
        name="drive",
        synapse=lang.ExponentialCurrent(tau=2 * lang.ms),
        weight=lang.Normal(10, 1),
        delay=0.3 * lang.ms,
    )
    net.connect(
        cells.spikes,
        cells.excitatory,
        name="recur",
        connection="recurrent",
        delay=0.1 * lang.ms,
        synapse=lang.ExponentialCurrent(tau=3 * lang.ms),
        weight=lang.Constant(0.1),
    )
    net.output("spikes", cells.spikes)
    net.output("voltage", cells.voltage)
    net.output("echo", source)
    return net


def eager():
    return Performance(compile_mode=None, cuda_graphs="off", precision="highest")


@pytest.fixture(params=[False, True])
def stream_policy(request, monkeypatch):
    if not request.param:
        return eager()
    original = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kw: original(fn, backend="eager", fullgraph=True, dynamic=False),
    )
    return Performance(compile_block_steps=3, precision="highest")


@pytest.mark.parametrize("compiled", [False, True])
def test_prepared_windows_match_uninterrupted_and_restored_state(compiled, monkeypatch):
    original = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kw: original(fn, backend="eager", fullgraph=True, dynamic=False),
    )
    graph = lang.compile(circuit()).graph
    reference = GraphExecutor(plan_graph(graph), seed=13, compile_mode=None)
    values = (
        torch.rand(40, 2, 3, generator=torch.Generator().manual_seed(3)) > 0.4
    ).float()
    with torch.inference_mode():
        expected = reference({"events": values}, diagnostics=False)
    policy = (
        Performance(compile_block_steps=5, precision="highest") if compiled else eager()
    )
    with prepare_stream(
        graph, window_steps=5, batch_size=2, seed=13, device="cpu", performance=policy
    ) as stream:
        assert isinstance(stream, StreamSession)
        assert stream.completed_steps == 0
        with pytest.raises(ValueError, match="no state"):
            stream.export_state()
        results = []
        snapshot = None
        for start in range(0, 40, 5):
            results.append(stream.advance({"events": values[start : start + 5]}))
            if start == 15:
                snapshot = stream.export_state()
        assert stream.completed_steps == 40
        for name in expected.outputs:
            actual = torch.cat([r.outputs[name] for r in results])
            torch.testing.assert_close(
                actual, expected.outputs[name], atol=1e-5, rtol=1e-5
            )
        assert expected.outputs["spikes"].count_nonzero() > 0
        final = stream.export_state()
        for group in (
            "voltages",
            "refractory",
            "currents",
            "population_histories",
            "input_histories",
        ):
            for name, value in getattr(expected.runtime_state, group).items():
                torch.testing.assert_close(getattr(final, group)[name], value)
        assert results[-1].runtime_state is None and results[-1].model is None
        assert results[-1].numpy().time_ms[0] == pytest.approx(3.5)
        stream.reset(snapshot)
        assert stream.completed_steps == 20
        restored = stream.advance({"events": values[20:25]})
        for name, value in results[4].outputs.items():
            torch.testing.assert_close(restored.outputs[name], value)
        stream.reset()
        assert stream.completed_steps == 0
        first = stream.advance({"events": values[:5]})
        torch.testing.assert_close(
            first.outputs["voltage"], results[0].outputs["voltage"]
        )
    with pytest.raises(RuntimeError, match="closed"):
        stream.advance({"events": values[:5]})
    stream.close()


def test_outputs_snapshots_and_source_model_are_independent(stream_policy):
    graph = lang.compile(circuit()).graph
    source = GraphExecutor(
        plan_graph(graph),
        seed=7,
        compile_mode=None,
        trainable_parameters=("drive.weight",),
    )
    source.train()
    source_weights = {
        name: value.detach().clone() for name, value in source.parameter_map().items()
    }
    with prepare_stream(source, window_steps=5, performance=stream_policy) as stream:
        assert source.training
        assert source.parameter_map()["drive.weight"].requires_grad
        first = stream.advance({"events": torch.ones(5, 1, 3)})
        saved = {name: value.clone() for name, value in first.outputs.items()}
        snapshot = stream.export_state()
        signature_metadata = copy.deepcopy(snapshot.compatibility)
        snapshot.voltages["cells"].fill_(999)
        snapshot.compatibility.clear()
        with torch.no_grad():
            source.parameter_map()["drive.weight"].fill_(0)
        stream.advance({"events": torch.zeros(5, 1, 3)})
        for name, value in saved.items():
            torch.testing.assert_close(first.outputs[name], value)
        fresh = stream.export_state()
        assert fresh.compatibility == signature_metadata
        assert fresh.voltages["cells"].max() < 999
        stream.reset()
        repeated = stream.advance({"events": torch.ones(5, 1, 3)})
        torch.testing.assert_close(
            repeated.outputs["voltage"], first.outputs["voltage"]
        )
    assert source.parameter_map()["drive.weight"].count_nonzero() == 0
    torch.testing.assert_close(
        source.parameter_map()["recur.weight"], source_weights["recur.weight"]
    )


def test_invalid_windows_and_restore_leave_progress_unchanged(stream_policy):
    with prepare_stream(circuit(), window_steps=5, performance=stream_policy) as stream:
        stream.advance({"events": torch.ones(5, 1, 3)})
        for inputs in (
            {},
            {"events": torch.ones(4, 1, 3)},
            {"events": torch.ones(5, 1, 3, dtype=torch.float64)},
            {"events": torch.ones(5, 1, 3, requires_grad=True)},
        ):
            with pytest.raises(ValueError):
                stream.advance(inputs)
            assert stream.completed_steps == 5
        state = stream.export_state()
        state.signature = "incompatible"
        with pytest.raises(ValueError, match="incompatible"):
            stream.reset(state)
        assert stream.completed_steps == 5
        state = stream.export_state()
        state.voltages["cells"] = torch.zeros(1, 9)
        with pytest.raises(ValueError):
            stream.reset(state)
        assert stream.completed_steps == 5


def test_retention_and_frozen_graph_metadata():
    graph = lang.compile(circuit()).graph
    with prepare_stream(
        graph, window_steps=5, performance=eager(), retain_outputs=False
    ) as stream:
        graph["populations"][0]["size"] = 999
        result = stream.advance({"events": torch.ones(5, 1, 3)})
        assert result.outputs == {}
        assert stream.export_state().voltages["cells"].shape == (1, 4)
        info = stream.info
        info["execution_path"].clear()
        assert stream.info["execution_path"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_steps": 0},
        {"window_steps": True},
        {"window_steps": 5, "batch_size": 0},
        {"window_steps": 5, "warmup_windows": 1},
        {"window_steps": 5, "dtype": torch.int64},
    ],
)
def test_invalid_preparation(kwargs):
    with pytest.raises((TypeError, ValueError)):
        prepare_stream(circuit(), performance=eager(), **kwargs)


def test_parameterless_stream_owns_passthrough_buffers(stream_policy):
    net = lang.Network("passthrough", dt=0.1 * lang.ms)
    events = net.input(
        "events", shape=("time", "batch", 3), signal_type="spikes", unit="spike"
    )
    net.output("echo", events)
    with prepare_stream(net, window_steps=5, performance=stream_policy) as stream:
        result = stream.advance({"events": torch.ones(5, 1, 3)})
        stream.advance({"events": torch.zeros(5, 1, 3)})
        assert result.outputs["echo"].sum() == 15
        assert not torch.is_inference(result.outputs["echo"])
        result.outputs["echo"].fill_(7)
        fresh = stream.advance({"events": torch.zeros(5, 1, 3)})
        assert fresh.outputs["echo"].count_nonzero() == 0
        assert stream.completed_steps == 15


def test_bundle_path_and_float64_state_roundtrip(tmp_path):
    bundle = lang.compile(circuit())
    bundle.write(tmp_path / "network.bundle")
    with prepare_stream(
        tmp_path / "network.bundle",
        window_steps=5,
        dtype=torch.float64,
        performance=eager(),
    ) as stream:
        stream.advance({"events": torch.ones(5, 1, 3, dtype=torch.float64)})
        state = stream.export_state()
        assert state.voltages["cells"].dtype == torch.float64
        state.completed_steps = -1
        with pytest.raises(ValueError, match="nonnegative"):
            stream.reset(state)
        assert stream.completed_steps == 5


@pytest.mark.parametrize(
    "constructor",
    [
        lang.CUBA_LIF,
        lang.COBA_LIF,
        lang.CUBA_ALIF,
        lang.COBA_ALIF,
        lang.CUBA_ADEX,
        lang.COBA_ADEX,
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("block_steps", [3, 7])
def test_persistent_blocks_match_executor_with_long_delays_gains_and_restore(
    constructor,
    dtype,
    block_steps,
    monkeypatch,
    tmp_path,
):
    from snnlab.sim import load_runtime_state, save_runtime_state

    original = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kw: original(fn, backend="eager", fullgraph=True, dynamic=False),
    )
    net = lang.Network("persistent", dt=0.1 * lang.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cells = net.population(
        "cells",
        size=3,
        neuron=constructor(tau_mem=20 * lang.ms, refractory=0.3 * lang.ms),
    )
    current = cells.neuron.kind.startswith("cuba_")
    synapse = lang.ExponentialCurrent if current else lang.AMPA
    drive = net.connect(
        events,
        cells.excitatory,
        name="drive",
        synapse=synapse(tau=2 * lang.ms),
        weight=lang.Constant(15 if current else 0.2),
        delay=1.1 * lang.ms,
        modulation=lang.StepGain(at=2.3 * lang.ms, before=1, after=0.4),
    )
    net.connect(
        cells.spikes,
        cells.excitatory,
        name="recur",
        connection="recurrent",
        delay=0.9 * lang.ms,
        synapse=synapse(tau=3 * lang.ms),
        weight=lang.Constant(0.1),
    )
    net.output("voltage", cells.voltage)
    net.output("spikes", cells.spikes)
    net.output("drive_trace", drive.current if current else drive.conductance)
    net.output("echo", events)
    if "alif" in cells.neuron.kind or "adex" in cells.neuron.kind:
        net.output("adaptation", cells.state("adaptation"))
    policy = Performance(compile_block_steps=block_steps, precision="highest")
    fast = prepare_stream(
        net, window_steps=7, batch_size=2, dtype=dtype, performance=policy
    )
    regular = prepare_stream(
        net, window_steps=7, batch_size=2, dtype=dtype, performance=policy
    )
    assert fast.info["persistent_state"]
    # A/B against the original, independently initialized compiled executor.
    regular._blocks = None
    values = (
        torch.rand(84, 2, 2, generator=torch.Generator().manual_seed(9)) > 0.3
    ).to(dtype)
    first = None
    snapshot = None
    for begin in range(0, len(values), 7):
        a = fast.advance({"events": values[begin : begin + 7]})
        b = regular.advance({"events": values[begin : begin + 7]})
        for name, expected in b.outputs.items():
            torch.testing.assert_close(a.outputs[name], expected, atol=0, rtol=0)
        if begin == 0:
            first = {name: value.clone() for name, value in a.outputs.items()}
            a.outputs["voltage"].fill_(999)
            a.outputs["echo"].zero_()
        if begin == 35:
            snapshot = fast.export_state()
    observed, expected = fast.export_state(), regular.export_state()
    for group in (
        "voltages",
        "refractory",
        "conductances",
        "currents",
        "population_histories",
        "input_histories",
        "custom_state",
    ):
        assert getattr(observed, group).keys() == getattr(expected, group).keys()
        for name, value in getattr(expected, group).items():
            torch.testing.assert_close(
                getattr(observed, group)[name], value, atol=0, rtol=0
            )
    # Portable export, independent restore, and replay across a scheduled gain.
    save_runtime_state(tmp_path / "state", snapshot)
    restored = load_runtime_state(tmp_path / "state")
    fast.reset(restored)
    restored.voltages["cells"].fill_(999)
    for begin in range(42, 84, 7):
        fast.advance({"events": values[begin : begin + 7]})
    for name, value in expected.voltages.items():
        torch.testing.assert_close(
            fast.export_state().voltages[name], value, atol=0, rtol=0
        )
    fast.reset()
    replay = fast.advance({"events": values[:7]})
    for name, value in first.items():
        torch.testing.assert_close(replay.outputs[name], value, atol=0, rtol=0)
    # No general forward or per-window state snapshot is needed after preparation.
    monkeypatch.setattr(
        fast, "_execute", lambda *a: pytest.fail("general executor entered")
    )
    monkeypatch.setattr(fast._blocks, "snapshot", lambda: pytest.fail("state copied"))
    fast.advance({"events": values[7:14]})
    fast.close()
    regular.close()


def test_failed_persistent_window_does_not_commit_partial_state(monkeypatch):
    # Exercise a failure after a complete numerical block, before window commit.
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    stream = prepare_stream(
        circuit(),
        window_steps=7,
        performance=Performance(compile_block_steps=3, precision="highest"),
    )
    values = {"events": torch.ones(7, 1, 3)}
    stream.advance(values)
    before = stream.export_state()
    expected = stream.advance(values)
    stream.reset(before)
    kernel = stream._blocks.kernel
    calls = 0

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted window")
        return kernel(*args)

    monkeypatch.setattr(stream._blocks, "kernel", interrupted)
    with pytest.raises(RuntimeError, match="interrupted window"):
        stream.advance(values)
    assert stream.completed_steps == before.completed_steps
    after = stream.export_state()
    for group in (
        "voltages",
        "refractory",
        "currents",
        "population_histories",
        "input_histories",
    ):
        for name, value in getattr(before, group).items():
            torch.testing.assert_close(
                getattr(after, group)[name], value, atol=0, rtol=0
            )
    monkeypatch.setattr(stream._blocks, "kernel", kernel)
    actual = stream.advance(values)
    for name, value in expected.outputs.items():
        torch.testing.assert_close(actual.outputs[name], value, atol=0, rtol=0)
    stream.close()


def test_persistent_trace_fallback_preserves_progress(monkeypatch):
    from importlib import import_module
    from types import SimpleNamespace

    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    stream = prepare_stream(
        circuit(),
        window_steps=5,
        performance=Performance(compile_block_steps=3, precision="highest"),
    )
    values = {"events": torch.ones(5, 1, 3)}
    stream.advance(values)
    state = stream.export_state()
    expected = stream.advance(values)
    stream.reset(state)
    monkeypatch.setattr(
        import_module("snnlab.sim.runtime.streaming"),
        "current_trace",
        lambda: SimpleNamespace(active=True),
    )
    actual = stream.advance(values)
    assert not stream.info["persistent_state"]
    assert stream.completed_steps == 10
    for name, value in expected.outputs.items():
        torch.testing.assert_close(actual.outputs[name], value, atol=0, rtol=0)
    stream.close()
