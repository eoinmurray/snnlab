"""Independent spatial arithmetic, training and persistence acceptance checks."""

import copy
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from snnlab import lang, sim
from snnlab.sim.requests import handlers
from snnlab.sim.requests.spec import ExecutionSpec
from snnlab.sim.training.trainer import train


def circuit(*, current=True, inhibitory=False, delay=0, mode="avg", pooled=False):
    net = lang.Network("spatial", dt=1 * lang.ms)
    source = net.input(
        "events",
        shape=("time", "batch", 40),
        spatial_shape=(2, 4, 5),
        signal_type="spikes",
        unit="spike",
    )
    if pooled:
        source = lang.pool2d(
            source, kernel_size=(2, 2), stride=1, name="input_pool", mode=mode
        )
    shape = (3, 1, 2) if pooled else (3, 2, 3)
    cells = net.population(
        "cells",
        spatial_shape=shape,
        neuron=lang.CUBA_LIF(threshold_mv=1000)
        if current
        else lang.COBA_LIF(tau_mem=20 * lang.ms),
    )
    projection = net.connect(
        source,
        cells.inhibitory if inhibitory else cells.excitatory,
        name="conv",
        connectivity=lang.Conv2d(kernel_size=(2, 3), stride=(2, 1), padding=(0, 0)),
        synapse=lang.ExponentialCurrent(tau=2 * lang.ms)
        if current
        else (lang.GABA(tau=2 * lang.ms) if inhibitory else lang.AMPA(tau=2 * lang.ms)),
        weight=lang.Constant(0.1),
        initialization_scaling="direct",
        delay=delay * lang.ms,
    )
    net.output("filtered", projection.current if current else projection.conductance)
    net.output("voltage", cells.voltage)
    return net, projection


def model(net, trainable=()):
    return sim.GraphExecutor(
        sim.plan_graph(lang.compile(net, target="tools/snnsim").graph),
        trainable_parameters=trainable,
    )


@pytest.mark.parametrize("current", [True, False])
@pytest.mark.parametrize("inhibitory", [True, False])
@pytest.mark.parametrize("delay", [0, 2])
def test_convolution_synaptic_filter_and_gradients_match_pytorch(
    current, inhibitory, delay
):
    net, projection = circuit(current=current, inhibitory=inhibitory, delay=delay)
    runner = model(net, (projection.weight.id,))
    weight = runner.parameter_map()[projection.weight.id]
    assert weight.shape == (12, 3)  # Shared kernel, not a 40-by-18 dense matrix.
    with torch.no_grad():
        weight.copy_(torch.arange(36).reshape(12, 3) / 100)
    torch.manual_seed(8)
    events = torch.randint(2, (5, 2, 40)).float()
    reference_weight = weight.detach().clone().requires_grad_()
    kernels = reference_weight.T.reshape(3, 2, 2, 3)
    drive = F.conv2d(events.reshape(-1, 2, 4, 5), kernels, stride=(2, 1)).reshape(
        5, 2, 18
    )
    previous = torch.zeros_like(drive[0])
    reference = []
    for t in range(5):
        previous = previous * math.exp(-0.5) + (drive[t - delay] if t >= delay else 0)
        reference.append(previous)
    reference = torch.stack(reference)
    actual = runner({"events": events}).outputs["filtered"]
    torch.testing.assert_close(actual, reference)
    coefficients = torch.linspace(-1, 2, actual.numel()).reshape_as(actual)
    (actual * coefficients).sum().backward()
    (reference * coefficients).sum().backward()
    torch.testing.assert_close(weight.grad, reference_weight.grad)


@pytest.mark.parametrize("mode", ["avg", "max"])
def test_pooling_forward_gradient_and_chained_convolution(mode):
    net, projection = circuit(pooled=True, mode=mode)
    pool = net._signal("input_pool.value")
    net.output("pooled", pool)
    runner = model(net, (projection.weight.id,))
    values = torch.linspace(0, 1, 80).reshape(1, 2, 40).requires_grad_()
    reference_values = values.detach().clone().requires_grad_()
    function = F.avg_pool2d if mode == "avg" else F.max_pool2d
    reference = function(reference_values.reshape(2, 2, 4, 5), 2, 1).reshape(1, 2, 24)
    actual = runner({"events": values}).outputs
    torch.testing.assert_close(actual["pooled"], reference)
    kernels = runner.parameter_map()[projection.weight.id].T.reshape(3, 2, 2, 3)
    expected_drive = F.conv2d(
        reference.reshape(2, 2, 3, 4), kernels, stride=(2, 1)
    ).reshape(1, 2, 6)
    torch.testing.assert_close(actual["filtered"], expected_drive)
    actual["pooled"].square().sum().backward()
    reference.square().sum().backward()
    torch.testing.assert_close(values.grad, reference_values.grad)


def spiking_chain(mode="avg", delay=0):
    net = lang.Network("chain", dt=1 * lang.ms)
    source = net.input(
        "events",
        shape=("time", "batch", 16),
        spatial_shape=(1, 4, 4),
        signal_type="spikes",
        unit="spike",
    )
    # Deliberately reverse lexical order to check scheduling through operations.
    hidden = net.population(
        "z_hidden",
        spatial_shape=(2, 4, 4),
        neuron=lang.CUBA_LIF(
            tau_mem=5 * lang.ms, resting_mv=0, reset_mv=0, threshold_mv=0.4
        ),
    )
    first = net.connect(
        source,
        hidden.excitatory,
        name="first",
        connectivity=lang.Conv2d(kernel_size=3, padding=1),
        synapse=lang.ExponentialCurrent(tau=2 * lang.ms),
        weight=lang.Constant(0.2),
        initialization_scaling="direct",
    )
    pooled = lang.pool2d(hidden.spikes, kernel_size=2, name="pool", mode=mode)
    output = net.population(
        "a_output",
        spatial_shape=(2, 1, 1),
        spiking=False,
        neuron=lang.LeakyIntegrator(tau=2 * lang.ms),
    )
    second = net.connect(
        pooled,
        output.excitatory,
        name="second",
        connectivity=lang.Conv2d(kernel_size=2),
        synapse=lang.LeakyIntegrator(tau=2 * lang.ms),
        weight=lang.SignedNormal(mean=0, std=0.1),
        initialization_scaling="direct",
        delay=delay * lang.ms,
    )
    scores = lang.reduce(output.voltage, operation="mean", over="time", name="scores")
    net.output("class_scores", scores)
    net.output("hidden_spikes", hidden.spikes)
    training = lang.TrainSpec(
        objectives=(lang.CrossEntropy(prediction=scores, target="label"),),
        parameter_groups=(
            lang.ParameterGroup((first.weight, second.weight), name="all", lr=0.01),
        ),
        optimizer=lang.AdamW(weight_decay=0),
        presentation_duration=4 * lang.ms,
    )
    return net, training


@pytest.mark.parametrize("mode", ["avg", "max"])
def test_spiking_chain_trains_and_checkpoint_snapshot_roundtrip(tmp_path, mode):
    net, training = spiking_chain(mode)
    bundle = lang.compile(net, training=training, target="tools/snnsim")
    path = tmp_path / "network.bundle"
    bundle.write(path)
    loaded = lang.load_bundle(path)
    assert loaded.graph == bundle.graph
    assert any(
        "connectivity:conv2d" in row["features"]
        for row in bundle.manifest["required_capabilities"]["elements"]
    )
    torch.manual_seed(9)
    events = torch.randint(2, (4, 2, 16)).float()
    checkpoint = tmp_path / "trained.checkpoint"
    result = train(
        ExecutionSpec(
            kind="train",
            bundle=path,
            device="cpu",
            input_bindings=(sim.DenseArrayBinding("events", events),),
            targets={"label": torch.tensor([0, 1])},
            updates=2,
            save_final_checkpoint=checkpoint,
            snapshot=sim.SnapshotSpec(tmp_path / "snapshots", stage="both"),
        )
    )
    for name in ("first.weight", "second.weight"):
        assert result.gradients[name].abs().sum() > 0
        assert not torch.equal(
            result.snapshots["start"].parameters[name], result.parameters[name]
        )
    inferred = handlers.simulate(
        ExecutionSpec(
            kind="simulate",
            bundle=path,
            checkpoint=checkpoint,
            device="cpu",
            input_bindings=(sim.DenseArrayBinding("events", events),),
        )
    )
    torch.testing.assert_close(
        inferred.parameters["first.weight"], result.parameters["first.weight"]
    )
    for stage in ("start", "end"):
        snapshot = json.loads(
            (tmp_path / "snapshots" / stage / "snapshot.json").read_text()
        )
        assert snapshot["network"]["projections"][0]["connectivity"]["kind"] == "conv2d"
        assert snapshot["parameters"]["first.weight"]["shape"] == [9, 2]
        with np.load(tmp_path / "snapshots" / stage / "parameters.npz") as arrays:
            assert arrays[snapshot["parameters"]["first.weight"]["key"]].shape == (9, 2)


@pytest.mark.parametrize("mode", ["avg", "max"])
def test_pooling_delays_continue_exactly_and_geometry_changes_reject_state(
    tmp_path, mode
):
    net, _ = spiking_chain(mode, delay=3)
    runner = model(net)
    events = torch.ones(7, 1, 16)
    whole = runner({"events": events})
    first = runner({"events": events[:2]})
    sim.save_runtime_state(tmp_path / "state", first.runtime_state)
    state = sim.load_runtime_state(tmp_path / "state")
    rest = runner({"events": events[2:]}, runtime_state=state)
    torch.testing.assert_close(
        rest.runtime_state.voltages["a_output"],
        whole.runtime_state.voltages["a_output"],
    )
    changed = copy.deepcopy(lang.compile(net).graph)
    changed["operations"][0]["config"]["mode"] = "max" if mode == "avg" else "avg"
    changed["operations"][0]["signal_type"] = (
        "spikes" if mode == "avg" else "continuous"
    )
    with pytest.raises(ValueError, match="incompatible"):
        sim.GraphExecutor(sim.plan_graph(changed))(
            {"events": events[2:]}, runtime_state=state
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda g: g["inputs"][0].update(spatial_shape=[1, 4, 5]),
        lambda g: g["populations"][0].update(spatial_shape=[3, 3, 2]),
        lambda g: g["projections"][0]["connectivity"].update(stride=[0, 1]),
        lambda g: g["projections"][0]["connectivity"].update(kernel_size=[20, 20]),
        lambda g: g["parameters"][0].update(shape=[3, 40]),
        lambda g: g["projections"][0]["connectivity"].update(padding=[True, 0]),
    ],
)
def test_invalid_serialized_geometry_fails_before_execution(mutation):
    net, _ = circuit()
    graph = copy.deepcopy(lang.compile(net).graph)
    mutation(graph)
    assert lang.validate_graph(graph).errors
    with pytest.raises((ValueError, TypeError)):
        sim.plan_graph(graph)


def test_invalid_authoring_shapes_and_pooling_contracts():
    for kwargs in (
        {"kernel_size": 0},
        {"kernel_size": 2, "stride": True},
        {"kernel_size": 2, "padding": -1},
    ):
        with pytest.raises(ValueError):
            lang.Conv2d(**kwargs)
    net = lang.Network("invalid")
    with pytest.raises(ValueError, match="does not match"):
        net.population("bad", size=3, spatial_shape=(1, 2, 2), neuron=lang.CUBA_LIF())
    source = net.input(
        "events",
        shape=("time", "batch", 16),
        spatial_shape=(1, 4, 4),
        signal_type="spikes",
        unit="spike",
    )
    assert lang.pool2d(source, kernel_size=2, name="avg").signal_type == "continuous"
    assert (
        lang.pool2d(source, kernel_size=2, name="max", mode="max").signal_type
        == "spikes"
    )
    with pytest.raises(ValueError):
        lang.pool2d(source, kernel_size=2, padding=2, name="bad_padding")


def test_shared_kernel_across_spatial_extents_accumulates_gradients():
    net, projection = circuit()
    target = net.population("other", spatial_shape=(3, 3, 3), neuron=lang.CUBA_LIF())
    other = net.connect(
        net._signal("events.value"),
        target.excitatory,
        name="shared",
        connectivity=lang.Conv2d(kernel_size=(2, 3)),
        weight=projection.weight,
        synapse=lang.ExponentialCurrent(tau=2 * lang.ms),
    )
    net.output("other_current", other.current)
    runner = model(net, (projection.weight.id,))
    assert len(runner.parameter_map()) == 1
    weight = runner.parameter_map()[projection.weight.id]
    reference = weight.detach().clone().requires_grad_()
    events = torch.arange(80).reshape(1, 2, 40).remainder(3).eq(0).float()
    actual = runner({"events": events}).outputs
    (actual["filtered"].sum() + actual["other_current"].sum()).backward()
    images = events.reshape(2, 2, 4, 5)
    kernels = reference.T.reshape(3, 2, 2, 3)
    (
        F.conv2d(images, kernels, stride=(2, 1)).sum() + F.conv2d(images, kernels).sum()
    ).backward()
    torch.testing.assert_close(weight.grad, reference.grad)


@pytest.mark.parametrize("mode", ["avg", "max"])
def test_pooling_padding_recording_and_online_reductions(mode):
    net = lang.Network("pool_recording", dt=1 * lang.ms)
    source = net.input(
        "events",
        shape=("time", "batch", 6),
        spatial_shape=(1, 2, 3),
        signal_type="spikes",
        unit="spike",
    )
    pooled = lang.pool2d(
        source, kernel_size=3, padding=1, stride=1, mode=mode, name="pool"
    )
    second = lang.pool2d(pooled, kernel_size=2, stride=1, mode=mode, name="pool_again")
    mean = lang.reduce(second, operation="mean", over="time", name="mean")
    net.output("mean_output", mean)
    runner = model(net)
    values = torch.tensor(
        [[[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]]
    )
    function = F.avg_pool2d if mode == "avg" else F.max_pool2d
    kwargs = {"count_include_pad": False} if mode == "avg" else {}
    expected_pool = function(values.reshape(2, 1, 2, 3), 3, 1, 1, **kwargs)
    expected_second = function(expected_pool, 2, 1).reshape(2, 1, 2)
    with torch.no_grad():
        result = runner(
            {"events": values},
            recording=sim.RecordingSpec(signals=(sim.SignalRecording(second.id),)),
        )
    torch.testing.assert_close(result.outputs["mean_output"], expected_second.mean(0))
    torch.testing.assert_close(result.recorded_signals[second.id], expected_second)
    assert "mean.value" in result.metrics["online_reductions"]


def test_spatial_input_works_with_existing_sparse_event_binding():
    net, _ = circuit()
    graph = lang.compile(net).graph
    values = torch.arange(120).reshape(3, 1, 40).remainder(5).eq(0).float()
    steps, batches, channels = values.nonzero(as_tuple=True)
    sparse = handlers.simulate(
        ExecutionSpec(
            kind="simulate",
            graph=graph,
            device="cpu",
            input_bindings=(
                sim.EventStreamBinding("events", steps, batches, channels, 3, 1),
            ),
        )
    )
    dense = handlers.simulate(
        ExecutionSpec(
            kind="simulate",
            graph=graph,
            device="cpu",
            input_bindings=(sim.DenseArrayBinding("events", values),),
        )
    )
    torch.testing.assert_close(
        sparse.outputs["filtered"], dense.outputs["filtered"], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda g: g["operations"][0]["config"].update(mode="median"),
        lambda g: g["operations"][0].update(spatial_shape=[2, 4, 1]),
        lambda g: g["operations"][0].update(signal_type="spikes"),
        lambda g: g["operations"][0]["config"].update(padding=[5, 5]),
        lambda g: g["operations"][0].update(sources=["pool.value"]),
    ],
)
def test_invalid_serialized_pooling_fails_validation(mutation):
    net, _ = spiking_chain()
    graph = copy.deepcopy(lang.compile(net).graph)
    # Serialized operation ordering puts pool before scores.
    mutation(graph)
    assert lang.validate_graph(graph).errors
    with pytest.raises((ValueError, TypeError)):
        sim.plan_graph(graph)


def test_same_population_convolution_requires_explicit_autapses_and_is_causal():
    net, _ = spiking_chain()
    hidden = net._signal("z_hidden.spikes")
    kwargs = dict(
        connectivity=lang.Conv2d(kernel_size=3, padding=1),
        synapse=lang.ExponentialCurrent(tau=2 * lang.ms),
        weight=lang.Constant(0.1),
        initialization_scaling="direct",
        connection="recurrent",
    )
    with pytest.raises(ValueError, match="allow_autapses=True"):
        net.connect(hidden, "z_hidden.excitatory", name="rejected_self", **kwargs)
    feedback = net.connect(
        hidden,
        "z_hidden.excitatory",
        name="self_feedback",
        allow_autapses=True,
        **kwargs,
    )
    assert feedback.connectivity.kind == "conv2d"
    net.output("feedback_current", feedback.current)
    runner = model(net)
    result = runner({"events": torch.ones(4, 1, 16)})
    spikes = result.outputs["hidden_spikes"].reshape(4, 2, 4, 4)
    assert spikes.sum() > 0
    expected = []
    previous = torch.zeros(1, 32)
    for t in range(4):
        drive = (
            F.conv2d(
                spikes[t - 1 : t], torch.full((2, 2, 3, 3), 0.1), padding=1
            ).flatten(1)
            if t
            else 0
        )
        previous = previous * math.exp(-0.5) + drive
        expected.append(previous)
    torch.testing.assert_close(
        result.outputs["feedback_current"], torch.stack(expected)
    )
