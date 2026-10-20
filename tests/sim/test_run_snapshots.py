"""Run snapshot lifecycle and faithful resolved parameter exports."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from snnlab import sim
from snnlab.sim.requests import handlers
from snnlab.sim.requests.spec import ExecutionSpec
from snnlab.sim.training.trainer import train
from tests.sim._execution_builders import coupled_graph, direct_train_bundle


def request(kind="simulate", **options):
    bundle = direct_train_bundle()
    values = dict(
        kind=kind,
        graph=bundle.graph,
        device="cpu",
        input_bindings=(
            sim.PoissonInputBinding(
                input_id="events",
                steps_count=3,
                batch_size=2,
                rates_hz=(100.0,),
                seed=7,
            ),
        ),
    )
    if kind == "train":
        inputs = torch.zeros(3, 2, 2)
        inputs[:, 0, 0] = 1
        inputs[:, 1, 1] = 1
        values.update(
            training=bundle.training,
            targets={"label": torch.tensor([0, 1])},
            updates=2,
            input_bindings=(sim.DenseArrayBinding("events", inputs),),
        )
    values.update(options)
    return ExecutionSpec(**values)


def read_snapshot(path):
    metadata = json.loads((path / "snapshot.json").read_text())
    arrays = {}
    if metadata["parameters"]:
        with np.load(path / "parameters.npz", allow_pickle=False) as data:
            arrays = {
                name: data[row["key"]].copy()
                for name, row in metadata["parameters"].items()
            }
    return metadata, arrays


@pytest.mark.parametrize(
    "handler,kind",
    [
        (handlers.simulate, "simulate"),
        (handlers.execute_request, "simulate"),
    ],
)
def test_path_shorthand_exports_resolved_overrides(tmp_path, handler, kind):
    spec = request(
        kind,
        snapshot=tmp_path / "run",
        retain_parameters=False,
        options={
            "inference_overrides": {
                "duration_ms": 0.8,
                "timestep_ms": 0.2,
                "input_rate_hz": 250.0,
            }
        },
    )
    baseline = handlers.simulate(replace(spec, snapshot=None, retain_parameters=True))
    result = handler(spec)
    metadata, arrays = read_snapshot(tmp_path / "run")
    assert metadata["kind"] == kind
    assert metadata["stage"] == "start"
    assert metadata["execution"]["timestep_ms"] == 0.2
    assert metadata["execution"]["protocol"]["timing"]["duration_ms"] == 0.8
    assert metadata["execution"]["effective_poisson_bindings"][0]["rates_hz"] == [250.0]
    assert metadata["execution"]["device_resolved"] == "cpu"
    assert metadata["network"]["timebase"]["dt"]["value"] == 0.2
    assert not result.parameters
    for name, tensor in baseline.parameters.items():
        np.testing.assert_array_equal(arrays[name], tensor.detach().numpy())
    torch.testing.assert_close(
        result.outputs["class_scores"], baseline.outputs["class_scores"], rtol=0, atol=0
    )


def test_training_both_stages_resume_and_independent_copies(tmp_path):
    spec = request("train", snapshot=sim.SnapshotSpec(tmp_path / "train", stage="both"))
    baseline = train(replace(spec, snapshot=None))
    result = train(spec)
    start, initial = read_snapshot(tmp_path / "train" / "start")
    end, final = read_snapshot(tmp_path / "train" / "end")
    assert start["execution"]["completed_updates"] == 0
    assert end["execution"]["completed_updates"] == 2
    assert end["training"] == spec.training
    assert start["request"]["input_bindings"][0]["value"]["sha256"]
    assert any(not np.array_equal(initial[name], final[name]) for name in initial)
    for name, tensor in result.parameters.items():
        np.testing.assert_array_equal(final[name], tensor.numpy())
        torch.testing.assert_close(tensor, baseline.parameters[name], rtol=0, atol=0)
    sim.save_training_checkpoint(tmp_path / "checkpoint", result.training_checkpoint)
    resumed = train(
        replace(
            spec,
            checkpoint=tmp_path / "checkpoint",
            snapshot=sim.SnapshotSpec(tmp_path / "resume", stage="both"),
        )
    )
    resume_start, restored = read_snapshot(tmp_path / "resume" / "start")
    assert resume_start["execution"]["completed_updates"] == 2
    assert resumed.snapshots["end"].metadata["execution"]["completed_updates"] == 4
    for name in initial:
        np.testing.assert_array_equal(restored[name], final[name])
    with torch.no_grad():
        for value in result.model.parameter_map().values():
            value.add_(100)
    sim.write_run_snapshot(tmp_path / "copy", result.snapshots["start"])
    _, copied = read_snapshot(tmp_path / "copy")
    for name in initial:
        np.testing.assert_array_equal(copied[name], initial[name])


def test_inference_snapshot_loads_checkpoint_weights(tmp_path):
    trained = train(request("train"))
    sim.save_training_checkpoint(tmp_path / "checkpoint", trained.training_checkpoint)
    result = handlers.simulate(
        request(snapshot=tmp_path / "simulate", checkpoint=tmp_path / "checkpoint")
    )
    metadata, arrays = read_snapshot(tmp_path / "simulate")
    assert metadata["execution"]["checkpoint"]["completed_updates"] == 2
    for name, value in trained.parameters.items():
        np.testing.assert_array_equal(arrays[name], value.numpy())
    assert result.snapshots["start"].metadata["stage"] == "start"


def test_metadata_only_end_and_disabled(tmp_path):
    result = handlers.simulate(
        request(
            snapshot=sim.SnapshotSpec(tmp_path / "meta", parameters=False, stage="end")
        )
    )
    metadata, arrays = read_snapshot(tmp_path / "meta")
    assert metadata["parameters_included"] is False
    assert metadata["stage"] == "end"
    assert not arrays
    assert not (tmp_path / "meta" / "parameters.npz").exists()
    assert not result.snapshots["end"].parameters
    assert handlers.simulate(request()).snapshots == {}


def test_writer_refuses_overwrite_and_invalid_metadata(tmp_path):
    snapshot = sim.RunSnapshot(
        {"schema": "snnlab.run-snapshot/v1"}, {"weight": torch.tensor([1.0])}
    )
    sim.write_run_snapshot(tmp_path / "saved", snapshot)
    before = (tmp_path / "saved" / "snapshot.json").read_bytes()
    with pytest.raises(FileExistsError):
        sim.write_run_snapshot(tmp_path / "saved", snapshot)
    assert (tmp_path / "saved" / "snapshot.json").read_bytes() == before
    with pytest.raises(ValueError):
        sim.write_run_snapshot(
            tmp_path / "bad", sim.RunSnapshot({"value": float("nan")}, {})
        )
    assert not (tmp_path / "bad").exists()


def test_writer_preserves_bfloat16_bits(tmp_path):
    tensor = torch.tensor([1.25, -0.5], dtype=torch.bfloat16)
    sim.write_run_snapshot(tmp_path, sim.RunSnapshot({}, {"weight": tensor}))
    metadata, arrays = read_snapshot(tmp_path)
    assert metadata["parameters"]["weight"]["dtype"] == "bfloat16"
    torch.testing.assert_close(
        torch.from_numpy(arrays["weight"]).view(torch.bfloat16), tensor
    )


@pytest.mark.parametrize("option", [42, "", sim.SnapshotSpec("unused")])
def test_snapshot_request_validation(option):
    with pytest.raises((TypeError, ValueError)):
        ExecutionSpec(kind="build", snapshot=option)


def test_snapshot_stage_validation():
    with pytest.raises(ValueError, match="stage"):
        sim.SnapshotSpec("unused", stage="selected")
    with pytest.raises(TypeError, match="boolean"):
        sim.SnapshotSpec("unused", parameters="yes")


def test_projection_scales_are_captured_after_application(tmp_path):
    graph = coupled_graph()
    spec = ExecutionSpec(
        kind="simulate",
        graph=graph,
        device="cpu",
        input_bindings=tuple(
            sim.PoissonInputBinding(
                input_id=name, steps_count=2, rates_hz=(0.0,), seed=7
            )
            for name in ("drive_a", "drive_b")
        ),
    )
    baseline = handlers.simulate(spec)
    projection = graph["projections"][0]
    result = handlers.simulate(
        replace(
            spec,
            snapshot=tmp_path / "scaled",
            options={
                "inference_overrides": {"projection_scales": {projection["id"]: 2.0}}
            },
        )
    )
    _, arrays = read_snapshot(tmp_path / "scaled")
    name = projection["parameters"][0]
    np.testing.assert_array_equal(arrays[name], baseline.parameters[name].numpy() * 2)
    torch.testing.assert_close(result.parameters[name], torch.from_numpy(arrays[name]))


def test_both_simulation_stages_and_runtime_continuation(tmp_path):
    first = handlers.simulate(request())
    result = handlers.simulate(
        request(snapshot=sim.SnapshotSpec(tmp_path, stage="both")),
        runtime_state=first.runtime_state,
    )
    start, _ = read_snapshot(tmp_path / "start")
    end, _ = read_snapshot(tmp_path / "end")
    assert (
        start["execution"]["runtime_state_signature"] == first.runtime_state.signature
    )
    assert (
        start["execution"]["runtime_state_completed_steps"]
        == first.runtime_state.completed_steps
    )
    assert set(result.snapshots) == {"start", "end"}
    assert start["stage"] == "start" and end["stage"] == "end"


def test_no_parameter_network_can_be_snapshotted(tmp_path):
    from snnlab import lang

    net = lang.Network("identity", dt=0.1 * lang.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    net.output("spikes", events)
    spec = replace(
        request(snapshot=tmp_path), graph=lang.compile(net, target=None).graph
    )
    handlers.simulate(spec)
    metadata, arrays = read_snapshot(tmp_path)
    assert metadata["parameters_included"] is True
    assert not arrays


def test_snapshot_supports_callable_recording_sinks(tmp_path):
    sink = sim.NPZRecordingSink(tmp_path / "blocks")
    graph = coupled_graph()
    signal = graph["populations"][0]["id"] + ".spikes"
    spec = request(
        graph=graph,
        input_bindings=tuple(
            sim.PoissonInputBinding(
                input_id=name, steps_count=2, rates_hz=(0.0,), seed=7
            )
            for name in ("drive_a", "drive_b")
        ),
        snapshot=tmp_path / "snapshot",
        recording=sim.RecordingSpec(
            signals=(sim.SignalRecording(signal),),
            sink=sink,
            sink_id=sink.identity,
        ),
    )
    handlers.simulate(spec)
    metadata, _ = read_snapshot(tmp_path / "snapshot")
    recorded_sink = metadata["request"]["recording"]["sink"]
    assert recorded_sink["directory"] == str(tmp_path / "blocks")
    assert recorded_sink["identity"] == sink.identity
    assert (tmp_path / "blocks" / "manifest.jsonl").is_file()
