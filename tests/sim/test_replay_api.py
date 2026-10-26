"""Supported replay interfaces, seeded draws and authenticated fresh bundles."""

import json

import numpy as np
import pytest
import torch

from snnlab import lang, sim
from snnlab.sim.inputs import datasets as dataset_impl


def replay_network(duration=1.2, model=lang.COBA_LIF):
    net = lang.Network("replay", dt=0.1 * lang.ms)
    events = net.input(
        "events", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    cells = net.population(
        "cells",
        size=1,
        neuron=model(tau_mem=20 * lang.ms, refractory=duration * lang.ms),
    )
    net.connect(
        events,
        cells.excitatory,
        name="drive",
        synapse=lang.AMPA(tau=5 * lang.ms),
        weight=lang.Constant(100),
        initialization_scaling="direct",
    )
    scores = lang.SpikeCount(source=cells.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    net.output("spikes", cells.spikes)
    net.output("voltage", cells.voltage)
    return net, scores


@pytest.mark.parametrize(
    "duration,count", [(0, 0), (0.1, 1), (0.2, 2), (0.6, 6), (1.2, 12)]
)
def test_fresh_legacy_bundle_authenticates_training_and_schedule(
    duration, count, tmp_path
):
    net, scores = replay_network(duration)
    training = lang.TrainSpec(
        objectives=[lang.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            lang.ParameterGroup([p["id"] for p in net.parameters], name="all", lr=1e-3)
        ],
        optimizer=lang.AdamW(),
    )
    current = lang.compile(net, training=training)
    legacy = lang.compile(net, training=training, refractory_timing="legacy_coba")
    assert current.graph["refractory_timing"] == "following_steps"
    assert legacy.graph["refractory_timing"] == "legacy_coba"
    assert current.manifest["graph_digest"] != legacy.manifest["graph_digest"]
    assert legacy.training["graph_digest"] == legacy.manifest["graph_digest"]
    path = legacy.write(tmp_path / "replay.bundle")
    loaded = lang.load_bundle(path)
    assert loaded.graph == legacy.graph and loaded.training == legacy.training
    legacy_plan = sim.plan_graph(loaded.graph)
    assert legacy_plan.refractory["cells"]["refractory_steps"] == count
    assert legacy_plan.refractory["cells"]["held_steps"] == max(count - 1, 0)
    model = sim.GraphExecutor(legacy_plan)
    inputs = torch.ones(40, 1, 1)
    full = model({"events": inputs})
    assert torch.where(full.outputs["spikes"].flatten() > 0)[0].tolist() == list(
        range(0, 40, max(count, 1))
    )
    first = model({"events": inputs[:5]})
    state_path = sim.save_runtime_state(tmp_path / "state", first.runtime_state)
    state = sim.load_runtime_state(state_path)
    resumed = model({"events": inputs[5:]}, runtime_state=state)
    for key in ("spikes", "voltage"):
        assert torch.equal(
            torch.cat((first.outputs[key], resumed.outputs[key])), full.outputs[key]
        )
    with pytest.raises(ValueError, match="incompatible with graph plan"):
        sim.GraphExecutor(sim.plan_graph(current.graph))(
            {"events": inputs[5:]}, runtime_state=state
        )
    graph_file = path / "graph.json"
    data = json.loads(graph_file.read_text())
    data["refractory_timing"] = "following_steps"
    graph_file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest"):
        lang.load_bundle(path)


@pytest.mark.parametrize("timing", [None, "nearest", "", 1])
def test_compile_rejects_unknown_schedule(timing):
    net, _ = replay_network()
    with pytest.raises(ValueError, match="refractory_timing"):
        lang.compile(net, refractory_timing=timing)


def test_legacy_authoring_rejects_inexact_durations_and_preserves_other_models():
    net, _ = replay_network(0.25)
    with pytest.raises(ValueError, match="compatible timestep"):
        lang.compile(net, refractory_timing="legacy_coba")
    net, _ = replay_network(model=lang.COBA_ALIF)
    plan = sim.plan_graph(lang.compile(net, refractory_timing="legacy_coba").graph)
    assert plan.refractory["cells"]["held_steps"] == 12
    assert plan.refractory["cells"]["timing"] == "following_steps"


DEVICES = (
    ["cpu"]
    + (["mps"] if torch.backends.mps.is_available() else [])
    + (["cuda"] if torch.cuda.is_available() else [])
)
GENERATORS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("generator_device", GENERATORS)
@pytest.mark.parametrize("rates", [250.0, [0.0, 250.0]])
def test_encoder_preserves_draw_order_and_generator_device(
    device, generator_device, rates
):
    pixels = torch.tensor([[-1, 0.25, 2], [1, 0.5, 0]], device=device)
    generator = torch.Generator(device=generator_device).manual_seed(17)
    reference_generator = torch.Generator(device=generator_device).manual_seed(17)
    result = sim.encode_images_poisson(pixels, 9, 0.5, rates, generator)
    draws = torch.rand(
        9, 2, 3, device=generator_device, generator=reference_generator
    ).to(device)
    p = torch.as_tensor(rates, device=device).expand(2).reshape(1, 2, 1) * 0.5 / 1000
    expected = (draws < pixels.clamp(0, 1).unsqueeze(0) * p).float()
    assert result.shape == (9, 2, 3) and result.dtype == torch.float32
    assert torch.equal(result, expected)
    assert torch.equal(generator.get_state(), reference_generator.get_state())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("generator_device", GENERATORS)
def test_poisson_occupancy_preserves_draw_order(device, generator_device):
    generator = torch.Generator(device=generator_device).manual_seed(31)
    reference = torch.Generator(device=generator_device).manual_seed(31)
    result = sim.poisson_spikes(500, (9, 2, 3), 1, generator, device=device)
    expected = (
        (torch.rand(9, 2, 3, device=generator_device, generator=reference) < 0.5)
        .float()
        .to(device)
    )
    assert torch.equal(result, expected)
    assert torch.equal(generator.get_state(), reference.get_state())
    assert result.dtype == torch.float32


@pytest.mark.parametrize(
    "rates,match",
    [([-1, 0], "non-negative"), ([1, 2, 3], "scalar or shape"), (1001, "above one")],
)
def test_encoder_validation(rates, match):
    with pytest.raises(ValueError, match=match):
        sim.encode_images_poisson(torch.ones(2, 3), 4, 1, rates)


def test_global_rng_encoding_and_randomized_reference_initialization():
    torch.manual_seed(23)
    state = torch.random.get_rng_state()
    result = sim.encode_images_poisson(torch.ones(2, 3), 4, 1, 200)
    torch.random.set_rng_state(state)
    assert torch.equal(result, (torch.rand(4, 2, 3) < 0.2).float())
    torch.manual_seed(29)
    v, ref = sim.init_lif_state(2, 3, "cpu", randomize=True, ref_mean=4, ref_std=2)
    torch.manual_seed(29)
    assert torch.equal(v, -65 + 15 * torch.rand(2, 3))
    assert torch.equal(ref, (torch.randn(2, 3) * 2 + 4).clamp(min=0).long())


def test_public_reference_step_and_surrogate_gradient():
    u = torch.tensor([-1.0, 0, 1], requires_grad=True)
    spikes = sim.fast_sigmoid_spike(u, 2)
    assert spikes.tolist() == [0, 1, 1]
    spikes.sum().backward()
    torch.testing.assert_close(u.grad, 2 / (1 + 2 * u.detach().abs()) ** 2)
    v = torch.full((1, 1), -65.0)
    ref = torch.ones(1, 1, dtype=torch.long)

    def spike(v):
        return sim.fast_sigmoid_spike(v + 50, 1)

    old = sim.lif_step_expeuler(
        v, ref, torch.full_like(v, 100), None, 1, 0.05, 12, spike, dt_override=0.1
    )
    new = sim.lif_step_expeuler(
        v,
        ref,
        torch.full_like(v, 100),
        None,
        1,
        0.05,
        12,
        spike,
        dt_override=0.1,
        refractory_after_step=True,
    )
    assert old[1].item() == 1 and old[2].item() == 12
    assert new[1].item() == 0 and new[2].item() == 0


def test_shd_official_events_and_deterministic_caps(monkeypatch, tmp_path):
    import h5py

    from snnlab.sim.inputs.datasets import _read_shd_split

    originals = {}
    for split, size in [("train", 12), ("test", 8)]:
        path = tmp_path / f"{split}.h5"
        with h5py.File(path, "w") as f:
            f["labels"] = np.arange(size, dtype=np.int64)
            group = f.create_group("spikes")
            group["units"] = np.arange(size * 2).reshape(size, 2)
            group["times"] = np.tile([0.1, 0.2], (size, 1))
        originals[split] = path
    monkeypatch.setattr(dataset_impl, "_download_shd", lambda split: originals[split])
    train_x, train_y = _read_shd_split("train")
    test_x, test_y = _read_shd_split("test")
    xtr, xte, ytr, yte = sim.load_dataset("shd", max_samples=5, split=True)
    for x, y, original_x, original_y in [
        (xtr, ytr, train_x, train_y),
        (xte, yte, test_x, test_y),
    ]:
        indices = np.random.RandomState(42).choice(len(original_y), 5, replace=False)
        np.testing.assert_array_equal(y, original_y[indices])
        assert x.dtype == object and y.dtype == np.int64
        for sample, index in zip(x, indices, strict=True):
            np.testing.assert_array_equal(sample[0], original_x[index][0])
            np.testing.assert_array_equal(sample[1], original_x[index][1])
            assert sample[0].dtype == np.int16 and sample[1].dtype == np.float32
    x, y = sim.load_dataset("shd")
    assert len(x) == 20
    np.testing.assert_array_equal(y, np.concatenate((train_y, test_y)))


def test_mnist_normalization_subsampling_and_validation_partition(monkeypatch):
    from types import SimpleNamespace

    from sklearn.model_selection import train_test_split
    from torchvision import datasets

    pixels = torch.arange(100 * 784).remainder(256).to(torch.uint8).reshape(100, 28, 28)
    labels = torch.arange(100).remainder(2)
    test_pixels = pixels[:12].flip(0)
    monkeypatch.setattr(
        datasets,
        "MNIST",
        lambda *, train, **kw: SimpleNamespace(
            data=pixels if train else test_pixels,
            targets=labels if train else labels[:12],
        ),
    )
    x = pixels.numpy().reshape(100, 784).astype(np.float32) / 255
    indices = np.random.RandomState(42).choice(100, 40, replace=False)
    capped_x, capped_y = sim.load_dataset("mnist", max_samples=40)
    np.testing.assert_array_equal(capped_x, x[indices])
    np.testing.assert_array_equal(capped_y, labels.numpy()[indices])
    expected = train_test_split(
        capped_x, capped_y, test_size=4, random_state=42, stratify=capped_y
    )
    actual = sim.load_dataset("mnist", max_samples=40, split=True)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    official = sim.load_dataset(
        "mnist", max_samples=40, split=True, evaluation_split="test"
    )
    np.testing.assert_array_equal(
        official[1], test_pixels.numpy().reshape(12, 784).astype(np.float32) / 255
    )
    assert len(official[1]) == 12
