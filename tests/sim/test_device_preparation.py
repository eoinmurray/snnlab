"""Compact device encoding, sample-local event indexing and safe prefetch."""

import random
import threading
from dataclasses import replace

import numpy as np
import pytest
import torch

from snnlab import extensions as E
from snnlab.sim import DatasetEncoder, DatasetSnapshotBinding
from snnlab.sim.inputs.events import EventIndex
from snnlab.sim.inputs.provider import DatasetProvider, iter_batches
from tests.sim._execution_builders import direct_train_bundle
from tests.sim.test_gpu_efficiency import dataset

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA unavailable"
        ),
    ),
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="MPS unavailable"
        ),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
def test_poisson_draws_on_selected_device_and_prefetch_is_exact(
    tmp_path, device, monkeypatch
):
    graph = direct_train_bundle().graph
    provider = DatasetProvider(graph, dataset(tmp_path, graph), device=device)
    calls = []
    original = torch.rand

    def rand(*args, **kwargs):
        calls.append((torch.device(kwargs.get("device", "cpu")).type, args))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "rand", rand)
    state = torch.get_rng_state().clone()
    requests = [
        (torch.arange(begin, min(begin + 2, 5)), "train", 3, None)
        for begin in range(0, 5, 2)
    ]
    reference = list(iter_batches(provider, requests))
    actual = list(iter_batches(provider, requests, prefetch=2))
    assert calls and all(backend == device for backend, _ in calls)
    assert torch.equal(torch.get_rng_state(), state)
    for (_, expected, meta), (_, tensors, other_meta) in zip(
        reference, actual, strict=True
    ):
        torch.testing.assert_close(
            tensors["events"], expected["events"], atol=0, rtol=0
        )
        assert meta == other_meta
    policy = actual[0][2]["encoder"].get("random_policy")
    if device == "cpu":
        assert policy is None
        # The retained CPU stream is the historical float32 generator.
        seed = actual[0][2]["spike_encoding_seed"]
        expected = (
            original(3, 2, 2, generator=torch.Generator().manual_seed(seed))
            < torch.tensor([[0, 1], [1, 0]]).reshape(1, 2, 2) * 0.1
        )
        torch.testing.assert_close(
            actual[0][1]["events"], expected.float(), atol=0, rtol=0
        )
    else:
        assert policy["schema"] == "snnlab.poisson-device/v1"
        assert policy["backend"] == device
        assert policy["cross_device_bitwise"] is False
        assert (
            provider.protocol["dataset_binding"]["encoder"]["random_policy"] == policy
        )


@pytest.mark.parametrize("device", DEVICES)
def test_indexed_event_batches_preserve_unsorted_events_collisions_and_sample_order(
    tmp_path, device, monkeypatch
):
    graph = direct_train_bundle().graph
    dt_ms = graph["timebase"]["dt"]["value"]
    samples = np.array([4, 0, 3, 0, 4, 0])
    times = np.array([0, dt_ms, 2 * dt_ms, dt_ms, 2.9 * dt_ms, 0.999999 * dt_ms])
    channels = np.array([1, 0, 0, 0, 1, 1])
    path = tmp_path / "events.npz"
    np.savez(
        path,
        labels=np.arange(5) % 2,
        event_sample=samples,
        event_time_ms=times,
        event_channel=channels,
    )
    binding = DatasetSnapshotBinding(
        path,
        "events",
        "events",
        "train",
        DatasetEncoder("event_bin", duration_ms=3 * dt_ms),
    )
    provider = DatasetProvider(graph, binding, device=device)
    monkeypatch.setattr(
        EventIndex, "__init__", lambda *a, **kw: pytest.fail("index rebuilt per batch")
    )
    requests = [
        (torch.tensor(selected), "inference", 0, None)
        for selected in ([4, 0], [3, 2], [1])
    ]
    for request, tensors, meta in iter_batches(provider, requests, prefetch=2):
        selected = request[0].tolist()
        expected = torch.zeros(3, len(selected), 2)
        retained = 0
        for sample, time, channel in zip(samples, times, channels, strict=True):
            if sample in selected:
                expected[int(float(time) / dt_ms), selected.index(sample), channel] = 1
                retained += 1
        torch.testing.assert_close(tensors["events"].cpu(), expected, atol=0, rtol=0)
        assert meta["encoder"]["retained_events"] == retained
        assert meta["encoder"]["binary_collisions"] == retained - expected.sum()


@pytest.mark.parametrize("device_aware", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_custom_prefetch_runs_on_caller_and_preserves_global_rng(
    tmp_path, monkeypatch, device_aware, device
):
    monkeypatch.setattr(E, "_REGISTRY", {})
    calls = []
    caller = threading.get_ident()

    def legacy(arrays, selected, *, dt_ms, channels, config, seed):
        calls.append((threading.get_ident(), "cpu"))
        return torch.full(
            (3, len(selected), channels),
            float(random.random() + np.random.random() + torch.rand(()) > 1.5),
        )

    def modern(arrays, selected, *, dt_ms, channels, config, seed, device="cpu"):
        calls.append((threading.get_ident(), torch.device(device).type))
        generator = torch.Generator(device=device).manual_seed(seed)
        return (
            torch.rand(3, len(selected), channels, generator=generator, device=device)
            < 0.5
        ).float()

    E.register_encoder("test.prefetch/v1", modern if device_aware else legacy)
    graph = direct_train_bundle().graph
    binding = replace(
        dataset(tmp_path, graph),
        encoder=DatasetEncoder("custom", definition="test.prefetch/v1"),
    )
    provider = DatasetProvider(graph, binding, device=device)
    calls.clear()
    state = torch.get_rng_state().clone()
    py_state = random.getstate()
    np_state = np.random.get_state()
    requests = [
        (torch.arange(begin, min(begin + 2, 5)), "train", 3, None)
        for begin in range(0, 5, 2)
    ]
    reference = list(iter_batches(provider, requests))
    actual = list(iter_batches(provider, requests, prefetch=2))
    assert len(calls) == 6
    assert all(thread == caller for thread, _ in calls)
    assert all(backend == (device if device_aware else "cpu") for _, backend in calls)
    assert torch.equal(state, torch.get_rng_state())
    assert py_state == random.getstate()
    assert np.array_equal(np_state[1], np.random.get_state()[1])
    for (_, expected, meta), (_, tensors, other_meta) in zip(
        reference, actual, strict=True
    ):
        torch.testing.assert_close(
            tensors["events"], expected["events"], atol=0, rtol=0
        )
        assert meta == other_meta
        if device_aware and device != "cpu":
            assert meta["encoder"]["preparation_policy"] == "snnlab.encoder-device/v1"
        elif device == "cpu":
            assert "preparation_device" not in meta["encoder"]


def test_invalid_events_outside_current_selection_are_still_rejected(tmp_path):
    path = tmp_path / "invalid.npz"
    np.savez(
        path,
        labels=np.array([0, 1]),
        event_sample=np.array([1]),
        event_time_ms=np.array([float("nan")]),
        event_channel=np.array([0]),
    )
    binding = DatasetSnapshotBinding(
        path,
        "events",
        "events",
        "train",
        DatasetEncoder("event_bin", duration_ms=1),
        sample_cap=1,
    )
    with pytest.raises(ValueError, match="out of snapshot bounds"):
        DatasetProvider(direct_train_bundle().graph, binding)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sparse", [False, True])
def test_replay_stages_window_and_routing_before_step(device, sparse):
    from snnlab import lang
    from snnlab.sim import DenseSpikeReplay, ReplaySpikes, SparseSpikeReplay
    from snnlab.sim.runtime.interventions import prepare_interventions
    from tests.sim.test_gpu_efficiency import circuit

    graph, _ = circuit(lang.CUBA_LIF)
    values = torch.zeros(13, 2, 3)
    values[1, 0, 2] = 1
    values[4, 1, 1] = 1
    values[8, 0, 0] = 1
    values[11, 1, 2] = 1
    if sparse:
        coordinates = values.nonzero()
        replay = SparseSpikeReplay.from_events(
            steps=coordinates[:, 0],
            batches=coordinates[:, 1],
            cells=coordinates[:, 2],
            steps_count=13,
            batch_size=2,
            cells_count=3,
            dt_ms=0.5,
        )
    else:
        replay = DenseSpikeReplay.from_tensor(values, dt_ms=0.5)
    prepared, identity = prepare_interventions(
        (ReplaySpikes("cells", replay),),
        graph=graph,
        seed=7,
        start_step=3,
        steps_count=7,
        batch_size=2,
    )
    item = prepared[0]
    item.stage(device=device, dtype=torch.float16, start_step=3, steps_count=7)
    assert all(tensor.device.type == device for tensor in item.tensors)
    assert all(type(offset) is int for offset in item.offsets)
    assert item.metadata == identity["resolved"][0]
    for step in range(3, 10):
        actual = item.apply(
            torch.zeros(2, 3, device=device, dtype=torch.float16), step, 0
        )
        torch.testing.assert_close(actual.cpu(), values[step].half(), atol=0, rtol=0)
