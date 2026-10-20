"""Retained/sink recordings share axes, exact storage and sample coordinates."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from snnlab import sim
from tests.sim._execution_builders import coupled_graph


def request(*, kind="dense", sink=None):
    return dict(
        graph=coupled_graph(),
        input_bindings=tuple(
            sim.PoissonInputBinding(name, 6, (1000.0,), 3, batch_size=3)
            for name in ("drive_a", "drive_b")
        ),
        device="cpu",
        performance=sim.Performance(compile_mode=None),
        recording=sim.RecordingSpec(
            (sim.SignalRecording("a_E.spikes", cells=(3, 0), kind=kind),),
            window=sim.MeasurementWindow(1, 5),
            sink=sink,
            sink_id=sink.identity if sink else None,
            block_steps=2,
        ),
    )


@pytest.mark.parametrize("kind", ["dense", "spike_events"])
def test_retained_export_and_streamed_blocks_agree(tmp_path, kind):
    result = sim.simulate(**request(kind=kind))
    full = result.numpy().recording
    selected = result.numpy(batch=2)
    row = selected.recording.info["a_E.spikes"]
    assert row.cell_ids == (3, 0)
    assert row.sample_ids == (2,)
    assert row.start_step == 1 and row.end_step == 5
    if kind == "dense":
        np.testing.assert_allclose(row.time_ms, [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_array_equal(
            selected.recorded_signals["a_E.spikes"], full.signals["a_E.spikes"][:, 2]
        )
    else:
        assert np.all(selected.recorded_signals["a_E.spikes"][:, 1] == 2)
    result.write_recording(tmp_path / "selected", batch=2)
    saved = sim.RecordingData.read(tmp_path / "selected")
    np.testing.assert_array_equal(
        saved.signals["a_E.spikes"], selected.recorded_signals["a_E.spikes"]
    )
    assert saved.info["a_E.spikes"].selected_batch == 2
    sink = sim.NPZRecordingSink(tmp_path / "sink")
    sim.simulate(**request(kind=kind, sink=sink))
    streamed = sim.RecordingData.read(tmp_path / "sink")
    np.testing.assert_array_equal(
        streamed.signals["a_E.spikes"], full.signals["a_E.spikes"]
    )
    with pytest.raises(FileExistsError):
        result.write_recording(tmp_path / "selected")
    result.write_recording(tmp_path / "selected", overwrite=True)
    with pytest.raises(ValueError, match="unavailable"):
        result.write_recording(tmp_path / "absent", signals=("missing",))


def test_declared_axes_strides_and_bfloat16_do_not_mutate_autograd(tmp_path):
    source = torch.arange(12, dtype=torch.bfloat16).reshape(2, 3, 2).requires_grad_()
    tensor = source * 2
    info = sim.SignalInfo(
        ("batch", "time", "cell"),
        unit="mV",
        cell_ids=(4, 7),
        sample_ids=("sample-a", "sample-b"),
        step_indices=(5, 8, 11),
        dt_ms=0.2,
        start_step=5,
        end_step=12,
        batch_size=2,
    )
    result = sim.ExecutionResult(
        "graph",
        outputs={"value": tensor},
        recorded_signals={"E.voltage": tensor},
        _batch_size=2,
        _output_axes={"value": ("batch", "time", 2)},
        _recording_info={"E.voltage": info},
    )
    sample = result.numpy(batch=1)
    assert sample.recorded_signals["E.voltage"].dtype == np.float32
    assert sample.recording.info["E.voltage"].source_dtype == "bfloat16"
    np.testing.assert_allclose(
        sample.recording.info["E.voltage"].time_ms, [1.0, 1.6, 2.2]
    )
    sample.recorded_signals["E.voltage"][:] = 0
    tensor.sum().backward()
    torch.testing.assert_close(source.grad, torch.full_like(source, 2))
    result.write_recording(tmp_path / "saved", batch=1)
    saved = sim.RecordingData.read(tmp_path / "saved")
    np.testing.assert_array_equal(
        saved.signals["E.voltage"], tensor[1].detach().float().numpy()
    )
    with np.load(tmp_path / "saved" / "block-00000000.npz") as archive:
        np.testing.assert_array_equal(
            archive["values"], tensor[1].detach().view(torch.uint16).numpy()
        )


@pytest.mark.parametrize(
    "axes,shape",
    [
        (("cell",), (2,)),
        (("time", "cell"), (0, 2)),
        (("cell", "time", "batch"), (2, 3, 2)),
    ],
)
def test_absent_axes_empty_signals_and_permutations(axes, shape):
    steps = tuple(range(shape[axes.index("time")])) if "time" in axes else None
    info = sim.SignalInfo(
        axes, cell_ids=(5, 9), step_indices=steps, dt_ms=0.1, batch_size=2
    )
    result = sim.ExecutionResult(
        "graph",
        recorded_signals={"value": torch.ones(shape)},
        _batch_size=2,
        _recording_info={"value": info},
    )
    sample = result.numpy(batch=1)
    assert sample.recording.info["value"].axes == tuple(
        axis for axis in axes if axis != "batch"
    )
    expected = (
        torch.ones(shape).select(axes.index("batch"), 1).numpy()
        if "batch" in axes
        else np.ones(shape)
    )
    np.testing.assert_array_equal(sample.recorded_signals["value"], expected)
    with pytest.raises(IndexError):
        result.numpy(batch=2)


def test_empty_window_and_corrupt_artifacts(tmp_path):
    options = request()
    options["recording"] = replace(
        options["recording"], window=sim.MeasurementWindow(10, 12)
    )
    result = sim.simulate(**options)
    assert result.numpy(batch=0).recorded_signals["a_E.spikes"].shape == (0, 2)
    result.write_recording(tmp_path / "empty")
    assert sim.RecordingData.read(tmp_path / "empty").signals["a_E.spikes"].shape == (
        0,
        3,
        2,
    )
    result = sim.simulate(**request())
    result.write_recording(tmp_path / "corrupt")
    (tmp_path / "corrupt" / "block-00000000.npz").write_bytes(b"broken")
    with pytest.raises(ValueError, match="digest"):
        sim.RecordingData.read(tmp_path / "corrupt")


@pytest.mark.accelerator
@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_accelerator_conversion_returns_independent_cpu_arrays(device):
    available = (
        torch.cuda.is_available()
        if device == "cuda"
        else torch.backends.mps.is_available()
    )
    if not available:
        pytest.skip(f"{device} unavailable")
    source = torch.ones((2, 2), device=device, requires_grad=True)
    result = sim.ExecutionResult(
        "graph",
        recorded_signals={"x": source * 2},
        _batch_size=2,
        _recording_info={"x": sim.SignalInfo(("batch", "cell"), batch_size=2)},
    )
    result.numpy(batch=0).recorded_signals["x"][:] = 0
    result.recorded_signals["x"].sum().backward()
    torch.testing.assert_close(source.grad, torch.full_like(source, 2))
