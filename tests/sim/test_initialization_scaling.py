"""Declarative scaling preserves legacy initializers and retained weights."""

import copy

import pytest
import torch

from snnlab import lang as snn
from snnlab.sim.execution import (
    DenseArrayBinding,
    ExecutionSpec,
    GraphExecutor,
    plan_graph,
    simulate,
    train,
)
from tests.sim._execution_builders import direct_train_bundle


def readout_net(scaling=None, initializer=None):
    net = snn.Network("scaling", dt=0.1 * snn.ms)
    source = net.input(
        "events", shape=("time", "batch", 8), signal_type="spikes", unit="spike"
    )
    output = snn.readouts.MeanVoltage(
        source=source,
        classes=3,
        name="scores",
        weight=initializer or snn.LowerClampedNormal(0.05, 0.04),
        initialization_scaling=scaling,
    )
    net.output("class_scores", output)
    return net, source


def executor(graph, seed=17):
    return GraphExecutor(plan_graph(graph), seed=seed)


def test_direct_readout_matches_legacy_draw_and_bundle_roundtrip(tmp_path):
    net, _ = readout_net("direct")
    bundle = snn.compile(net)
    bundle.write(tmp_path / "network.bundle")
    loaded = snn.load_bundle(tmp_path / "network.bundle")
    assert loaded.graph == bundle.graph
    model = executor(loaded.graph)
    torch.manual_seed(17)
    expected = torch.randn(8, 3).mul_(0.04).add_(0.05).clamp_(min=0)
    torch.testing.assert_close(
        model.parameter_map()["scores_projection.weight"], expected, rtol=0, atol=0
    )
    assert (
        model.initialization_metadata["scores_projection.weight"]["scaling"] == "direct"
    )


@pytest.mark.parametrize("zeroing", ["bernoulli", "exact_k"])
@pytest.mark.parametrize("scaling", ["direct", "fan_in_normalized"])
def test_zeroing_and_rng_order_match_independent_draws(zeroing, scaling):
    initializer = snn.LowerClampedNormal(
        0.5, 0.4, initial_zero_fraction=0.5, zeroing=zeroing
    )
    net, _ = readout_net(scaling, initializer)
    model = executor(snn.compile(net).graph)
    torch.manual_seed(17)
    expected = torch.randn(8, 3).mul_(0.4).add_(0.5).clamp_(min=0)
    if zeroing == "exact_k":
        mask = torch.zeros(8, 3)
        for column in range(3):
            mask[torch.randperm(8)[:4], column] = 1
        expected = expected * mask * 2
    else:
        expected = expected * (torch.rand(8, 3) > 0.5).float() / 0.5
    expected = expected / 8
    if scaling == "direct":
        expected = expected * 8
    torch.testing.assert_close(
        model.parameter_map()["scores_projection.weight"], expected, rtol=0, atol=0
    )


def test_old_graph_without_field_keeps_default_and_rng():
    net, _ = readout_net()
    graph = snn.compile(net).graph
    old = copy.deepcopy(graph)
    for row in old["parameters"]:
        row.pop("initialization_scaling", None)
    new_model, old_model = executor(graph), executor(old)
    torch.testing.assert_close(
        new_model.parameter_map()["scores_projection.weight"],
        old_model.parameter_map()["scores_projection.weight"],
        rtol=0,
        atol=0,
    )
    assert new_model.initialization_metadata == old_model.initialization_metadata


def test_shared_parameter_inherits_policy_and_initializes_once():
    net, source = readout_net("direct")
    weights = net.parameter(
        "shared",
        shape=(3, 8),
        unit="uS",
        initializer=snn.Constant(2.0),
        initialization_scaling="direct",
    )
    for name in ("a", "b"):
        target = net.population(
            name, size=3, neuron=snn.LeakyIntegrator(tau=2 * snn.ms), spiking=False
        )
        net.connect(
            source,
            target.excitatory,
            name=f"to_{name}",
            synapse=snn.LeakyIntegrator(tau=2 * snn.ms),
            weight=weights,
        )
    calls = []
    original = torch.full

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    from unittest.mock import patch

    with patch("torch.full", counted):
        model = executor(snn.compile(net).graph)
    assert len(calls) == 1
    torch.testing.assert_close(model.parameter_map()["shared"], torch.full((8, 3), 2.0))
    with pytest.raises(ValueError, match="conflicting"):
        net.connect(
            source,
            target.excitatory,
            name="conflict",
            synapse=snn.LeakyIntegrator(tau=2 * snn.ms),
            weight=weights,
            initialization_scaling="fan_in_normalized",
        )


@pytest.mark.parametrize("bad", ["unknown", 3, [], None])
def test_serialized_invalid_scaling_rejected(bad):
    net, _ = readout_net()
    graph = snn.compile(net).graph
    graph["parameters"][0]["initialization_scaling"] = bad
    assert any(
        "initialization_scaling" in error.message
        for error in snn.validate_graph(graph).errors
    )
    with pytest.raises(ValueError, match="initialization_scaling"):
        executor(graph)


@pytest.mark.parametrize("scaling", ["direct", "fan_in_normalized"])
def test_checkpoint_load_and_training_resume_preserve_stored_weights(tmp_path, scaling):
    bundle = direct_train_bundle()
    bundle.graph["parameters"][0]["initialization_scaling"] = scaling
    bundle.graph["parameters"][0]["initializer"] = snn.Constant(0.4).json()
    # Recompute training identity after deliberately changing the fixture graph.
    from snnlab.lang.compiler import digest

    bundle.training["graph_digest"] = digest(bundle.graph)
    events = torch.zeros(3, 2, 2)
    events[:, 0, 0] = 1
    events[:, 1, 1] = 1
    common = dict(
        executor="graph",
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(DenseArrayBinding("events", events),),
        targets={"label": torch.tensor([0, 1])},
        seed=17,
    )
    full = train(ExecutionSpec(kind="train", **common, updates=2))
    first = train(
        ExecutionSpec(
            kind="train",
            **common,
            updates=1,
            save_final_checkpoint=tmp_path / "checkpoint",
        )
    )
    resumed = train(
        ExecutionSpec(
            kind="train", **common, updates=1, checkpoint=tmp_path / "checkpoint"
        )
    )
    restored = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            input_bindings=common["input_bindings"],
            checkpoint=tmp_path / "checkpoint",
            seed=999,
        )
    )
    for name in full.parameters:
        torch.testing.assert_close(
            full.parameters[name], resumed.parameters[name], rtol=0, atol=0
        )
        torch.testing.assert_close(
            first.parameters[name], restored.parameters[name], rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "scaling, expected", [(None, 2.0), ("direct", 2.0), ("fan_in_normalized", 0.25)]
)
def test_operation_parameter_scaling_and_old_default(scaling, expected):
    net = snn.Network("linear_scaling")
    source = net.input(
        "events", shape=("time", "batch", 8), signal_type="spikes", unit="spike"
    )
    value = snn.ops.linear(source, size=3, name="linear")
    net.output("output", value)
    graph = snn.compile(net).graph
    row = graph["parameters"][0]
    row["initializer"] = snn.Constant(2.0).json()
    if scaling is not None:
        row["initialization_scaling"] = scaling
    torch.testing.assert_close(
        executor(graph).parameter_map()["linear.weight"],
        torch.full((8, 3), expected),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("bad", ["unknown", [], 1])
def test_authoring_rejects_invalid_scaling(bad):
    with pytest.raises(ValueError, match="initialization_scaling"):
        readout_net(bad)
    with pytest.raises(ValueError, match="initialization_scaling"):
        snn.Network("bad").parameter(
            "weight",
            shape=(3, 8),
            initializer=snn.Constant(1),
            initialization_scaling=bad,
        )


def test_legacy_shared_projection_rng_is_preserved():
    net, source = readout_net()
    target = net.population(
        "other", size=3, neuron=snn.LeakyIntegrator(tau=2 * snn.ms), spiking=False
    )
    weight = net.projections[0]["parameters"][0]
    net.connect(
        source,
        target.excitatory,
        name="z_shared",
        synapse=snn.LeakyIntegrator(tau=2 * snn.ms),
        weight=snn.ParameterRef(net, weight),
    )
    graph = snn.compile(net).graph
    graph["parameters"][0].pop("initialization_scaling")
    model = executor(graph)
    torch.manual_seed(17)
    torch.randn(8, 3)
    expected = torch.randn(8, 3).mul_(0.04).add_(0.05).clamp_(min=0) / 8
    torch.testing.assert_close(model.parameter_map()[weight], expected, rtol=0, atol=0)
