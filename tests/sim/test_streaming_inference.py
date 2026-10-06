"""Online reductions, bounded sinks, measurement continuation and sparse identity."""

import copy
import json
import random
import weakref
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from snnlab.sim.execution import (
    GraphExecutor,
    MeasurementWindow,
    NPZRecordingSink,
    RecordingSpec,
    SignalRecording,
    load_runtime_state,
    plan_graph,
    save_runtime_state,
    simulate,
    validate_inference_artifacts,
    write_inference_artifacts,
)
from tests.sim._execution_builders import standard_readout_graph
from tests.sim.test_batch_encoding import dataset
from tests.sim.test_epoch_observations import request
from tests.sim.test_voltage_sampling import fixture, inputs


def reductions_graph(legacy=False):
    graph = copy.deepcopy(fixture(spiking=True).graph)
    graph["outputs"] = [
        row for row in graph["outputs"] if not row["id"].endswith("_trace")
    ]
    if legacy:
        graph.pop("voltage_sampling", None)
    return graph


def model(graph):
    return GraphExecutor(plan_graph(graph), seed=13)


def data():
    return {
        "events": inputs(),
        "valid": torch.tensor(
            [[1, 0], [1, 1], [0, 1], [1, 1], [1, 1]], dtype=torch.bool
        ),
    }


@pytest.mark.parametrize("legacy", [False, True])
def test_automatic_reductions_match_dense_without_histories(legacy):
    graph = reductions_graph(legacy)
    executor = model(graph)
    dense, _ = executor._forward(data(), diagnostics=False)
    state = torch.get_rng_state().clone()
    streamed = executor(data(), diagnostics=False)
    assert torch.equal(state, torch.get_rng_state())
    assert len(streamed.metrics["online_reductions"]) == 10
    assert streamed.metrics["retained_signal_samples"] == {}
    for name, value in dense.outputs.items():
        torch.testing.assert_close(streamed.outputs[name], value, rtol=2e-6, atol=2e-6)
    for name, value in dense.runtime_state.voltages.items():
        assert torch.equal(streamed.runtime_state.voltages[name], value)


@pytest.mark.parametrize("kind", ["count", "rate", "final"])
def test_standard_readout_linear_paths(kind):
    graph = standard_readout_graph(kind, duration=1.2)
    executor = model(graph)
    events = {"events": torch.ones(12, 2, 2)}
    dense, _ = executor._forward(events, diagnostics=False)
    streamed = executor(events, diagnostics=False)
    assert streamed.metrics["online_reductions"]
    assert not streamed.metrics["retained_signal_samples"]
    for name, value in dense.outputs.items():
        torch.testing.assert_close(streamed.outputs[name], value, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("cuts", [(1, 3), (2, 4), (1, 2, 3, 4)])
def test_measurement_window_continues_exactly(cuts, tmp_path):
    graph = reductions_graph()
    window = MeasurementWindow(1, 5)
    full = model(graph)(data(), diagnostics=False, measurement=window)
    state = None
    start = 0
    for end in (*cuts, 5):
        result = model(graph)(
            {name: value[start:end] for name, value in data().items()},
            diagnostics=False,
            measurement=window,
            runtime_state=state,
        )
        assert result.metrics["measurement_complete"] == (end == 5)
        assert not result.metrics["retained_signal_samples"]
        save_runtime_state(tmp_path, result.runtime_state)
        state = load_runtime_state(tmp_path)
        start = end
    for name, value in full.outputs.items():
        assert torch.equal(result.outputs[name], value)
    assert state.reduction_state == full.runtime_state.reduction_state
    for name, value in full.runtime_state.reduction_tensors.items():
        assert torch.equal(state.reduction_tensors[name], value)
    assert json.loads((tmp_path / "manifest.json").read_text())["schema_version"] == 4


def test_measurement_rejects_changed_removed_and_corrupt_state(tmp_path):
    graph = reductions_graph()
    window = MeasurementWindow(0, 5)
    state = model(graph)(
        {name: value[:2] for name, value in data().items()},
        diagnostics=False,
        measurement=window,
    ).runtime_state
    tail = {name: value[2:] for name, value in data().items()}
    for changed in (None, MeasurementWindow(1, 5)):
        with pytest.raises(ValueError, match="policy"):
            model(graph)(
                tail, diagnostics=False, measurement=changed, runtime_state=state
            )
    broken = state.detached()
    broken.reduction_tensors.pop(next(iter(broken.reduction_tensors)))
    with pytest.raises(ValueError, match="keys"):
        model(graph)(tail, diagnostics=False, measurement=window, runtime_state=broken)
    save_runtime_state(tmp_path, state)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["reduction_state"]["completed_steps"] = 10
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest"):
        load_runtime_state(tmp_path)


def test_dense_consumer_and_autograd_keep_required_history():
    graph = fixture(spiking=True).graph
    dense = model(graph)(data(), diagnostics=True)
    assert dense.metrics["retained_signal_samples"]
    executor = GraphExecutor(plan_graph(graph), trainable_parameters=["drive.weight"])
    result = executor(data(), diagnostics=False)
    assert not result.metrics["online_reductions"]
    result.outputs["pre_sum_full_result"].sum().backward()
    assert executor.parameter_map()["drive.weight"].grad is not None
    with pytest.raises(ValueError, match="autograd"):
        executor(data(), measurement=MeasurementWindow(0, 5))


def test_sparse_and_selected_dense_recording_identity():
    graph = reductions_graph()
    recording = RecordingSpec(
        (
            SignalRecording("readout.spikes", cells=(0,), kind="spike_events"),
            SignalRecording("readout.voltage", cells=(0,)),
        ),
        window=MeasurementWindow(1, 4),
    )
    result = model(graph)(data(), diagnostics=False, recording=recording)
    dense = model(fixture(spiking=True).graph)(data(), diagnostics=True)
    torch.testing.assert_close(
        result.recorded_signals["readout.voltage"], dense.outputs["post_trace"][1:4]
    )
    spike_trace = RecordingSpec((SignalRecording("readout.spikes"),))
    spikes = model(graph)(
        data(), diagnostics=False, recording=spike_trace
    ).recorded_signals["readout.spikes"]
    indices = spikes[1:4].nonzero()
    indices[:, 0] += 1
    assert torch.equal(result.recorded_signals["readout.spikes"], indices)
    assert result.recorded_signals["readout.spikes"].dtype == torch.int64
    assert not result.metrics["retained_signal_samples"]
    for name, value in dense.outputs.items():
        if name in result.outputs:
            torch.testing.assert_close(result.outputs[name], value)


def test_sink_does_not_change_rng_and_retains_no_recordings():
    graph = reductions_graph()
    blocks = []

    def sink(block):
        random.random()
        np.random.rand()
        torch.rand(2)
        blocks.append(block)
        block.values.zero_()

    state = torch.get_rng_state().clone()
    result = model(graph)(
        data(),
        diagnostics=False,
        recording=RecordingSpec(
            (SignalRecording("readout.spikes", kind="spike_events"),),
            sink=sink,
            sink_id="test/v1",
        ),
        retain_outputs=False,
    )
    assert torch.equal(state, torch.get_rng_state())
    assert result.outputs == {} and result.recorded_signals == {}
    assert any(block.kind == "output" for block in blocks)
    assert not result.metrics["retained_signal_samples"]
    baseline = model(graph)(data(), diagnostics=False)
    for name, value in baseline.runtime_state.voltages.items():
        assert torch.equal(result.runtime_state.voltages[name], value)


@pytest.mark.parametrize("steps", [20, 200])
def test_accumulator_and_sink_memory_does_not_scale_with_time(steps):
    graph = request().graph
    sizes = []
    result = model(graph)(
        {"events": torch.ones(steps, 2, 2)},
        diagnostics=False,
        measurement=MeasurementWindow(0, steps),
        recording=RecordingSpec(
            (SignalRecording("out.spikes", kind="spike_events"),),
            sink=lambda block: sizes.append(block.values.numel()),
            sink_id="sizes/v1",
        ),
        retain_outputs=False,
    )
    assert not result.metrics["retained_signal_samples"]
    assert not result.recorded_signals and not result.outputs
    assert max(sizes) <= 24
    assert (
        sum(value.numel() for value in result.runtime_state.reduction_tensors.values())
        <= 8
    )


def test_dataset_sink_bounds_retained_samples_and_keeps_global_batches(tmp_path):
    spec = request()
    blocks = []
    result = simulate(
        replace(
            spec,
            kind="infer",
            training=None,
            input_bindings=(dataset(tmp_path, count=9),),
            targets={},
            validation=None,
            recording=RecordingSpec(
                (SignalRecording("out.spikes", kind="spike_events"),),
                sink=blocks.append,
                sink_id="dataset/v1",
            ),
            retain_outputs=False,
            batch_size=3,
        )
    )
    assert (
        result.outputs == {}
        and result.recorded_signals == {}
        and result.final_state == {}
    )
    assert result.runtime_state is None
    assert {block.batch_offset for block in blocks} == {0, 3, 6}
    for block in blocks:
        if block.kind == "spike_events" and block.values.numel():
            assert int(block.values[:, 1].min()) >= block.batch_offset
            assert int(block.values[:, 1].max()) < block.batch_offset + 3
    assert result.metrics["online_reductions"]


def test_npz_sink_and_cache_retention_identity(tmp_path):
    graph = reductions_graph()
    sink = NPZRecordingSink(tmp_path / "sink")
    recording = RecordingSpec(
        (SignalRecording("readout.spikes", kind="spike_events"),),
        sink=sink,
        sink_id=sink.identity,
    )
    result = model(graph)(
        data(), diagnostics=False, recording=recording, retain_outputs=False
    )
    lines = (tmp_path / "sink" / "manifest.jsonl").read_text().splitlines()
    assert len(lines) == 15
    assert (
        np.load(tmp_path / "sink" / json.loads(lines[0])["file"])["values"].dtype
        == np.int64
    )
    root = tmp_path / "artifacts"
    write_inference_artifacts(root, result, graph=graph, seed=13)
    validate_inference_artifacts(
        root, expected_retention=result.metrics["inference_retention"]
    )
    with pytest.raises(ValueError, match="retention"):
        validate_inference_artifacts(root, expected_retention=None)
    with pytest.raises(ValueError, match="empty"):
        NPZRecordingSink(tmp_path / "sink")


class LiveAllocations(TorchDispatchMode):
    def __init__(self, excluded):
        self.excluded = excluded
        self.live = {}
        self.peak = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for key, (size, refs) in list(self.live.items()):
            refs = {
                identifier: ref for identifier, ref in refs.items() if ref() is not None
            }
            if refs:
                self.live[key] = size, refs
            else:
                del self.live[key]
        for value in tree_flatten(result)[0]:
            if not isinstance(value, torch.Tensor):
                continue
            storage = value.untyped_storage()
            key = storage.data_ptr()
            if key in self.excluded:
                continue
            size, refs = self.live.setdefault(key, (storage.nbytes(), {}))
            refs[id(value)] = weakref.ref(value)
        self.peak = max(self.peak, sum(size for size, refs in self.live.values()))
        return result


def test_actual_live_tensor_allocation_is_bounded_by_batch():
    peaks = []
    for steps in (20, 200):
        graph = request().graph
        executor = model(graph)
        events = torch.ones(steps, 2, 2)
        excluded = {events.untyped_storage().data_ptr()}
        excluded.update(
            value.untyped_storage().data_ptr() for value in executor.parameters()
        )
        tracker = LiveAllocations(excluded)
        with tracker:
            executor({"events": events}, diagnostics=False, retain_outputs=False)
        peaks.append(tracker.peak)
    assert peaks[1] <= peaks[0] * 1.2


def test_segment_resets_and_window_accumulators_share_boundary_ownership(tmp_path):
    from snnlab.sim.execution import BoundarySchedule, DecisionSegments, ResetVoltage
    from tests.sim.test_spike_replay import stream

    graph = copy.deepcopy(request().graph)
    count = copy.deepcopy(graph["operations"][0])
    count.update(id="totals", kind="reduce_sum", sources=["out.spikes"], unit="spike")
    graph["operations"].append(count)
    graph["outputs"].append({"id": "counts", "signal": "totals.value"})
    boundaries = BoundarySchedule(steps=(0, 2, 4))
    kwargs = dict(
        diagnostics=False,
        measurement=MeasurementWindow(0, 6),
        resets=(ResetVoltage("out", boundaries),),
        decisions=DecisionSegments(boundaries, 6, ("out",), ("out",)),
    )
    full = model(graph)({"events": stream()}, **kwargs)
    state = None
    records = []
    start = 0
    for end in (1, 2, 4, 6):
        result = model(graph)(
            {"events": stream()[start:end]}, runtime_state=state, **kwargs
        )
        records.extend(result.decisions)
        save_runtime_state(tmp_path, result.runtime_state)
        state = load_runtime_state(tmp_path)
        start = end
    assert records == full.decisions
    assert torch.equal(result.outputs["counts"], full.outputs["counts"])
    for batch in range(2):
        assert (
            result.outputs["counts"][batch].tolist()
            == np.sum(
                [
                    row["output_spike_counts"]["out"]
                    for row in records
                    if row["batch"] == batch
                ],
                axis=0,
            ).tolist()
        )


def test_masked_rate_window_matches_dense_and_continuation():
    graph = standard_readout_graph("rate", mask=True)
    values = {
        "events": torch.ones(5, 2, 2),
        "valid": torch.tensor(
            [[1, 0], [1, 1], [0, 1], [1, 1], [0, 1]], dtype=torch.bool
        ),
    }
    dense, _ = model(graph)._forward(values, diagnostics=False)
    full = model(graph)(values, diagnostics=False, measurement=MeasurementWindow(0, 5))
    head = model(graph)(
        {name: value[:1] for name, value in values.items()},
        diagnostics=False,
        measurement=MeasurementWindow(0, 5),
    )
    tail = model(graph)(
        {name: value[1:] for name, value in values.items()},
        diagnostics=False,
        measurement=MeasurementWindow(0, 5),
        runtime_state=head.runtime_state,
    )
    for name, value in dense.outputs.items():
        torch.testing.assert_close(full.outputs[name], value)
        assert torch.equal(tail.outputs[name], full.outputs[name])


def test_cli_streams_to_npz_without_retained_outputs(tmp_path):
    from snnlab.sim.bundle import load_graph_bundle
    from snnlab.sim.tool import main
    from tests.sim.test_spike_replay import graph_fixture, stream

    bundle = tmp_path / "network.bundle"
    graph_fixture(bundle_path=bundle)
    _, graph = load_graph_bundle(bundle)
    inputs_path = tmp_path / "events.npy"
    np.save(inputs_path, stream().numpy())
    directory = tmp_path / "sink"
    out = tmp_path / "out"
    main(
        [
            "sim",
            "--executor",
            "graph",
            "--bundle",
            str(bundle),
            "--input-file",
            str(inputs_path),
            "--out-dir",
            str(out),
            "--device",
            "cpu",
            "--no-diagnostics",
            "--no-retain-outputs",
            "--recording",
            json.dumps(
                {
                    "signals": [{"signal": "source.spikes", "kind": "spike_events"}],
                    "directory": str(directory),
                }
            ),
        ]
    )
    rows = [
        json.loads(line)
        for line in (directory / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 24
    assert rows[0]["unit"] == "spike"
    assert all(row["end_step"] == row["start_step"] + 1 for row in rows)
    validate_inference_artifacts(out, graph=graph)
    assert np.load(out / "outputs.npz").files == []


def test_recording_artifacts_empty_window_and_invalid_cells(tmp_path):
    graph = reductions_graph()
    spec = RecordingSpec(
        (SignalRecording("readout.voltage", cells=(0,)),),
        window=MeasurementWindow(10, 20),
    )
    result = model(graph)(data(), diagnostics=False, recording=spec)
    assert result.recorded_signals["readout.voltage"].shape == (0, 2, 1)
    write_inference_artifacts(tmp_path, result, graph=graph, seed=13)
    assert validate_inference_artifacts(tmp_path)["schema_version"] == 2
    for cells in ((), (1,), (-1,), (0, 0)):
        with pytest.raises(ValueError):
            model(graph)(
                data(),
                recording=RecordingSpec(
                    (SignalRecording("readout.spikes", cells=cells),)
                ),
            )


def test_fractional_mask_keeps_weighted_mean_semantics():
    values = data()
    values["valid"] = torch.full((5, 2), 0.05)
    graph = reductions_graph()
    dense, _ = model(graph)._forward(values, diagnostics=False)
    streamed = model(graph)(values, diagnostics=False)
    for name, value in dense.outputs.items():
        torch.testing.assert_close(streamed.outputs[name], value, rtol=2e-6, atol=2e-6)


def test_actual_dataset_retention_allocation_is_bounded_by_batch(tmp_path):
    from snnlab.sim.dataset_provider import DatasetProvider, simulate_dataset

    peaks = []
    for count in (9, 90):
        graph = request().graph
        provider = DatasetProvider(
            graph, dataset(tmp_path, count=count, name=f"samples-{count}")
        )
        executor = model(graph)
        tracker = LiveAllocations(set())
        with tracker:
            result = simulate_dataset(
                executor,
                provider,
                batch_size=3,
                diagnostics=False,
                interventions=(),
                runtime_state=None,
                recording=RecordingSpec(sink=lambda block: None, sink_id="discard/v1"),
                retain_outputs=False,
            )
        assert not result.outputs and not result.final_state
        peaks.append(tracker.peak)
    assert peaks[1] <= peaks[0] * 1.2


def test_unsupported_reductions_keep_dense_path_and_reject_explicit_window():
    graph = standard_readout_graph("cumulative")
    cumulative = graph["outputs"][0]["signal"]
    reduction = copy.deepcopy(graph["operations"][0])
    reduction.update(
        id="total",
        kind="reduce_sum",
        sources=[cumulative],
        shape=["batch", 2],
        parameters=[],
        config={"over": "time"},
    )
    graph["operations"].append(reduction)
    graph["outputs"] = [{"id": "count", "signal": "total.value"}]
    values = {"events": torch.ones(5, 2, 2)}
    dense, _ = model(graph)._forward(values, diagnostics=False)
    result = model(graph)(values, diagnostics=False)
    assert not result.metrics["online_reductions"]
    torch.testing.assert_close(result.outputs["count"], dense.outputs["count"])
    with pytest.raises(ValueError, match="cannot stream"):
        model(graph)(values, measurement=MeasurementWindow(0, 5))


@pytest.mark.parametrize("dtype", [torch.float64, torch.bool])
def test_parameterless_reductions_preserve_promoted_output_dtype(dtype):
    from snnlab import lang as snn

    net = snn.Network("dtypes", dt=1 * snn.ms)
    source = net.input("values", shape=("time", "batch"), signal_type="mask")
    total = snn.ops.reduce(source, operation="sum", over="time", name="sum_values")
    net.output("total", total)
    graph = snn.compile(net).graph
    values = {"values": torch.ones(5, 2, dtype=dtype)}
    dense, _ = model(graph)._forward(values, diagnostics=False)
    result = model(graph)(values, diagnostics=False)
    assert result.outputs["total"].dtype == dense.outputs["total"].dtype
    assert torch.equal(result.outputs["total"], dense.outputs["total"])


def test_sink_aggregate_metadata_describes_full_measurement_coverage():
    graph = reductions_graph()
    blocks = []
    recording = RecordingSpec(sink=blocks.append, sink_id="coverage/v1")
    window = MeasurementWindow(1, 5)
    head = model(graph)(
        {name: value[:2] for name, value in data().items()},
        diagnostics=False,
        measurement=window,
        recording=recording,
        retain_outputs=False,
    )
    assert all((block.start_step, block.end_step) == (1, 2) for block in blocks)
    assert all(block.measurement_window == (1, 5) for block in blocks)
    blocks.clear()
    model(graph)(
        {name: value[2:] for name, value in data().items()},
        diagnostics=False,
        measurement=window,
        recording=recording,
        retain_outputs=False,
        runtime_state=head.runtime_state,
    )
    assert all((block.start_step, block.end_step) == (1, 5) for block in blocks)
