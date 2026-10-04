"""Focused acceptance tests for the typed seam and graph executor."""

from __future__ import annotations

import pytest
import torch

from snnlab import lang as snn
from snnlab.lang import training
from snnlab.sim.execution import (
    DenseArrayBinding,
    ExecutionSpec,
    build,
    execute_request,
    train,
)


def _coupled_graph(*, direction="reciprocal", delay_ms=0.1):
    net = snn.Network("coupled_ping_gate", dt=0.1 * snn.ms)
    drive_a = net.input(
        "drive_a", shape=("time", "batch", 3), signal_type="spikes", unit="spike"
    )
    drive_b = net.input(
        "drive_b", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    a = snn.components.ping(net, name="a", n_e=4, n_i=1, source=drive_a)
    b = snn.components.ping(net, name="b", n_e=6, n_i=2, source=drive_b)
    if direction in {"unidirectional", "reciprocal"}:
        net.connect(
            a.I.spikes,
            b.E.inhibitory,
            name="a_I_to_b_E",
            synapse=snn.GABA(tau=9 * snn.ms),
            weight=snn.Constant(4.0),
            constraint=snn.NonNegative(),
            connection="feedback",
            delay=delay_ms * snn.ms,
        )
    if direction == "reciprocal":
        net.connect(
            b.I.spikes,
            a.E.inhibitory,
            name="b_I_to_a_E",
            synapse=snn.GABA(tau=9 * snn.ms),
            weight=snn.Constant(4.0),
            constraint=snn.NonNegative(),
            connection="feedback",
            delay=delay_ms * snn.ms,
        )
    net.expose(a.E.spikes, a.I.spikes, b.E.spikes, b.I.spikes, name="coupled")
    return snn.compile(net, target=None).graph


def test_typed_request_defaults_to_graph_and_graph_training_requires_recipe():
    request = ExecutionSpec(kind="build", graph=_coupled_graph(), device="cpu")
    assert request.executor == "graph"
    result = execute_request(request)
    assert result.executor == "graph"
    assert result.model is not None
    assert result.parameters
    request = ExecutionSpec(kind="build", executor="legacy")
    assert request.device == "auto"
    assert build(request).metrics["routing"] == "legacy"
    try:
        train(ExecutionSpec(kind="train", graph=_coupled_graph(), device="cpu"))
    except ValueError as exc:
        assert "training recipe" in str(exc)
    else:
        raise AssertionError("graph training must require an explicit recipe")


def _direct_train_bundle():
    net = snn.Network("direct_train", dt=0.1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    scores = snn.readouts.SpikeCount(source=events, classes=2, name="scores")
    snn.ops.linear(events, size=1, name="shadow")
    net.output("class_scores", scores)
    parameter_ids = [row["id"] for row in net.parameters]
    recipe = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            training.ParameterGroup(
                ["scores_projection.weight"], name="readout", lr=0.1
            ),
            training.ParameterGroup(
                [pid for pid in parameter_ids if pid != "scores_projection.weight"],
                name="frozen",
                lr=0,
                frozen=True,
            ),
        ],
        optimizer=training.AdamW(weight_decay=0.0),
        presentation_duration=0.3 * snn.ms,
    )
    return snn.compile(net, training=recipe, target=None)


def test_graph_training_updates_named_parameters_and_optimizer_state_deterministically():
    bundle = _direct_train_bundle()
    inputs = torch.zeros(3, 2, 2)
    inputs[:, 0, 0] = 1
    inputs[:, 1, 1] = 1
    spec = ExecutionSpec(
        kind="train",
        executor="graph",
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(DenseArrayBinding("events", inputs),),
        targets={"label": torch.tensor([0, 1])},
        seed=17,
        updates=5,
    )
    first = train(spec)
    second = train(spec)
    losses = [row["loss"] for row in first.metrics["updates"]]
    assert losses[-1] < losses[0]
    assert first.metrics["trainable_parameters"] == ["scores_projection.weight"]
    assert set(first.optimizer_state) == {"scores_projection.weight"}
    assert set(first.gradients) == {"scores_projection.weight"}
    assert torch.count_nonzero(first.parameters["shadow.weight"]) == 0
    torch.testing.assert_close(
        first.parameters["scores_projection.weight"],
        second.parameters["scores_projection.weight"],
        rtol=0,
        atol=0,
    )
    assert first.metrics["updates"] == second.metrics["updates"]


def test_graph_one_step_matches_direct_pytorch_gradient_weight_and_adamw_state():
    bundle = _direct_train_bundle()
    inputs = torch.zeros(3, 2, 2)
    inputs[:, 0, 0] = 1
    inputs[:, 1, 1] = 1
    labels = torch.tensor([0, 1])
    result = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(DenseArrayBinding("events", inputs),),
            targets={"label": labels},
            seed=17,
        )
    )
    direct = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = torch.optim.AdamW([direct], lr=0.1, weight_decay=0.0)
    loss = torch.nn.functional.cross_entropy((inputs @ direct).sum(dim=0), labels)
    loss.backward()
    expected_gradient = direct.grad.detach().clone()
    optimizer.step()
    torch.testing.assert_close(
        result.gradients["scores_projection.weight"], expected_gradient, rtol=0, atol=0
    )
    torch.testing.assert_close(
        result.parameters["scores_projection.weight"], direct.detach(), rtol=0, atol=0
    )
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            result.optimizer_state["scores_projection.weight"][key],
            optimizer.state[direct][key],
            rtol=0,
            atol=0,
        )


def test_graph_training_authenticates_recipe_from_bundle(tmp_path):
    bundle = _direct_train_bundle()
    root = bundle.write(tmp_path / "train.bundle")
    inputs = torch.zeros(3, 2, 2)
    inputs[:, 0, 0] = 1
    inputs[:, 1, 1] = 1
    result = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            bundle=root,
            input_bindings=(DenseArrayBinding("events", inputs),),
            targets={"label": torch.tensor([0, 1])},
            seed=17,
        )
    )
    assert result.metrics["training_schema"] == "snnlang.training/v1"
    assert result.metrics["updates"][0]["loss"] == pytest.approx(0.6931471824645996)


def test_graph_training_backpropagates_through_recurrence_and_spike_budget():
    net = snn.Network("tiny_recurrent_train", dt=0.1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cell = snn.components.ping(net, name="cell", n_e=4, n_i=2, source=events)
    input_parameter = next(
        row for row in net.parameters if row["id"] == "cell_input.weight"
    )
    input_parameter["initializer"] = snn.Constant(100.0).json()
    scores = snn.readouts.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    net.expose(cell.E.spikes, name="cell_E.spikes")
    net.expose(cell.I.spikes, name="cell_I.spikes")
    parameter_ids = [row["id"] for row in net.parameters]
    recipe = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[training.ParameterGroup(parameter_ids, name="all", lr=1e-3)],
        optimizer=training.AdamW(weight_decay=0.0),
        regularizers=[
            training.SpikeBudgetPenalty(
                signals=(cell.E.spikes, cell.I.spikes),
                ceiling_hz=0.0,
                strength=0.01,
            )
        ],
        surrogate=training.FastSigmoid(slope=1.0),
        presentation_duration=3 * snn.ms,
    )
    bundle = snn.compile(net, training=recipe, target=None)
    result = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(DenseArrayBinding("events", torch.ones(30, 2, 2)),),
            targets={"label": torch.tensor([0, 1])},
            seed=3,
        )
    )
    assert "cell_E_to_I.weight" in result.gradients
    assert "cell_I_to_E.weight" in result.gradients
    e_rates = result.diagnostics["cell_E.spikes"].sum(dim=0).mean(dim=1) / 0.003
    i_rates = result.diagnostics["cell_I.spikes"].sum(dim=0).mean(dim=1) / 0.003
    expected = 0.01 * torch.stack((e_rates.square(), i_rates.square())).mean()
    assert result.metrics["updates"][0]["components"][
        "regularizer[0]"
    ] == pytest.approx(float(expected.detach()))

    disabled = train(
        ExecutionSpec(
            kind="train",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(DenseArrayBinding("events", torch.ones(30, 2, 2)),),
            targets={"label": torch.tensor([0, 1])},
            seed=3,
            diagnostics=False,
        )
    )
    assert not disabled.diagnostics
    assert disabled.metrics["diagnostics"] is False
    assert disabled.metrics["updates"] == result.metrics["updates"]
    for name, value in result.outputs.items():
        torch.testing.assert_close(disabled.outputs[name], value, rtol=0, atol=0)
    for name, value in result.gradients.items():
        torch.testing.assert_close(disabled.gradients[name], value, rtol=0, atol=0)
