"""Lossless dtype preservation in runtime/checkpoint/recording artifacts."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from snnlab import lang
from snnlab.sim import (
    DenseArrayBinding,
    GraphExecutor,
    NPZRecordingSink,
    RecordingSpec,
    SignalRecording,
    load_runtime_state,
    load_training_checkpoint,
    plan_graph,
    save_runtime_state,
    save_training_checkpoint,
)
from snnlab.sim.artifacts.tensors import numpy_payload, tensor_payload
from snnlab.sim.requests.spec import ExecutionSpec
from snnlab.sim.training.trainer import train
from tests.sim._execution_builders import direct_train_bundle
from tests.sim.test_gpu_efficiency import circuit


def test_bfloat16_storage_preserves_all_bits_including_nan_and_signed_zero():
    bits = torch.tensor([0, 32768, 16256, 32640, 32705], dtype=torch.uint16)
    value = bits.view(torch.bfloat16)
    array, storage = numpy_payload(value)
    row = {"dtype": "bfloat16", "shape": [5], **storage}
    restored = tensor_payload(array, row, label="probe")
    assert torch.equal(restored.view(torch.uint16), bits)
    with pytest.raises(ValueError, match="unsupported tensor storage"):
        tensor_payload(array, {**row, "tensor_storage": "unknown/v1"}, label="probe")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_portable_runtime_continuation_keeps_model_dtype(dtype, tmp_path):
    graph, _ = circuit(lang.CUBA_ALIF)
    model = GraphExecutor(plan_graph(graph), compile_mode=None).to(dtype=dtype)
    values = torch.ones(41, 2, 2, dtype=dtype)
    with torch.no_grad():
        full = model({"events": values}, diagnostics=False)
        first = model({"events": values[:17]}, diagnostics=False)
        save_runtime_state(tmp_path / "runtime", first.runtime_state)
        state = load_runtime_state(tmp_path / "runtime")
        tail = model({"events": values[17:]}, runtime_state=state, diagnostics=False)
    assert all(value.dtype == dtype for value in state.voltages.values())
    torch.testing.assert_close(
        torch.cat((first.outputs["voltage"], tail.outputs["voltage"])),
        full.outputs["voltage"],
        atol=0,
        rtol=0,
    )


def test_bfloat16_checkpoint_parameters_and_optimizer_moments_round_trip(tmp_path):
    bundle = direct_train_bundle()
    checkpoint = train(
        ExecutionSpec(
            kind="train",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(DenseArrayBinding("events", torch.ones(3, 2, 2)),),
            targets={"label": torch.tensor([0, 1])},
            updates=1,
            device="cpu",
            compile_mode=None,
        )
    ).training_checkpoint
    parameters = {
        name: value.bfloat16() for name, value in checkpoint.parameters.items()
    }
    optimizer = {
        name: {
            key: value.bfloat16()
            if isinstance(value, torch.Tensor) and key != "step"
            else value
            for key, value in state.items()
        }
        for name, state in checkpoint.optimizer_state.items()
    }
    checkpoint = replace(checkpoint, parameters=parameters, optimizer_state=optimizer)
    save_training_checkpoint(tmp_path / "checkpoint", checkpoint)
    restored = load_training_checkpoint(tmp_path / "checkpoint")
    for name, expected in parameters.items():
        assert restored.parameters[name].dtype == torch.bfloat16
        assert torch.equal(
            restored.parameters[name].view(torch.uint16), expected.view(torch.uint16)
        )
    for name, state in optimizer.items():
        for key, expected in state.items():
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(
                    restored.optimizer_state[name][key], expected, atol=0, rtol=0
                )


def test_bfloat16_npz_recording_has_explicit_lossless_storage_policy(tmp_path):
    graph, _ = circuit(lang.CUBA_LIF)
    model = GraphExecutor(plan_graph(graph), compile_mode=None).bfloat16()
    sink = NPZRecordingSink(tmp_path / "recordings")
    model(
        {"events": torch.ones(9, 2, 2, dtype=torch.bfloat16)},
        diagnostics=False,
        recording=RecordingSpec(
            (SignalRecording("cells.voltage"),),
            sink=sink,
            sink_id=sink.identity,
            block_steps=4,
        ),
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "recordings/manifest.jsonl").read_text().splitlines()
    ]
    dense = next(row for row in rows if row["signal"] == "cells.voltage")
    assert dense["dtype"] == "bfloat16"
    assert dense["tensor_storage"] == "snnlab.bfloat16-bits/v1"
    with np.load(
        tmp_path / "recordings" / dense["file"], allow_pickle=False
    ) as archive:
        restored = tensor_payload(archive["values"], dense, label="recording")
    assert restored.dtype == torch.bfloat16 and restored.shape == (4, 2, 3)


def test_bfloat16_inference_cache_records_lossless_float32_expansion(tmp_path):
    from snnlab.sim.artifacts.inference import (
        validate_inference_artifacts,
        write_inference_artifacts,
    )

    graph, _ = circuit(lang.CUBA_LIF)
    model = GraphExecutor(plan_graph(graph), compile_mode=None).bfloat16()
    result = model(
        {"events": torch.ones(9, 2, 2, dtype=torch.bfloat16)}, diagnostics=False
    )
    manifest = write_inference_artifacts(tmp_path, result, graph=graph, seed=0)
    assert manifest["tensor_storage"]["schema"] == "snnlab.inference-bfloat16/v1"
    validate_inference_artifacts(tmp_path, graph=graph, seed=0)
    with np.load(tmp_path / "outputs.npz", allow_pickle=False) as archive:
        for name, expected in result.outputs.items():
            np.testing.assert_array_equal(
                archive[name], expected.detach().float().numpy()
            )
