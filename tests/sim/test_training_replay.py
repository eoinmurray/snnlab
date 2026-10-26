"""Automatic CUDA replay preserves native training, batching and resume contracts."""

import copy

import pytest
import torch

from snnlab import lang, sim
from snnlab.sim.cli import execution_spec_from_args, parse_args
from snnlab.sim.requests.spec import ExecutionSpec
from snnlab.sim.training.replay import TrainingReplay

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def request(*, device="cpu", recurrent=False, block_steps=1, samples=5):
    if recurrent:
        net = lang.Network("replay", dt=1 * lang.ms)
        source = net.input(
            "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
        )
        cells = net.population("cells", size=4, neuron=lang.CUBA_LIF())
        for name, signal, delay in (
            ("drive", source, 0),
            ("feedback", cells.spikes, 2),
        ):
            net.connect(
                signal,
                cells.excitatory,
                name=name,
                synapse=lang.ExponentialCurrent(tau=3 * lang.ms),
                weight=lang.Constant(4 if name == "drive" else 0.1),
                delay=delay * lang.ms,
                allow_autapses=False,
                constraint=lang.NonNegative(),
            )
        scores = lang.MeanVoltage(source=cells.spikes, classes=2, name="scores")
        net.output("prediction", scores)
        net.expose(cells.spikes, name="cell_spikes")
        net.expose(cells.voltage, name="cell_voltage")
        recipe = lang.TrainSpec(
            objectives=(lang.CrossEntropy(prediction=scores, target="label"),),
            parameter_groups=(
                lang.ParameterGroup(
                    [row["id"] for row in net.parameters], name="all", lr=0.01
                ),
            ),
            optimizer=lang.AdamW(weight_decay=0.01),
            regularizers=(
                lang.SpikeBudgetPenalty(
                    signals=(cells.spikes,), ceiling_hz=0, strength=0.001
                ),
            ),
            gradient_clip=0.5,
            presentation_duration=9 * lang.ms,
        )
        bundle = lang.compile(net, training=recipe)
        steps = 9
    else:
        net = lang.Network("replay-direct", dt=0.1 * lang.ms)
        source = net.input(
            "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
        )
        scores = lang.SpikeCount(source=source, classes=2, name="scores")
        lang.linear(source, size=1, name="shadow")
        net.output("class_scores", scores)
        # An unused trainable leaf must keep grad=None and no AdamW moments.
        recipe = lang.TrainSpec(
            objectives=(lang.CrossEntropy(prediction=scores, target="label"),),
            parameter_groups=(
                lang.ParameterGroup(
                    [row["id"] for row in net.parameters], name="all", lr=0.1
                ),
            ),
            optimizer=lang.AdamW(weight_decay=0),
            presentation_duration=0.3 * lang.ms,
        )
        bundle = lang.compile(net, training=recipe)
        steps = 3
    values = (
        torch.rand(steps, samples, 2, generator=torch.Generator().manual_seed(91)) < 0.6
    ).float()
    return dict(
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(sim.DenseArrayBinding("events", values),),
        targets={"label": torch.arange(samples) % 2},
        device=device,
        seed=7,
        epochs=2,
        batch_size=2,
        diagnostics=False,
        compile_block_steps=block_steps,
        precision="highest",
    )


def compare(actual, expected):
    for group in ("outputs", "diagnostics", "parameters", "gradients"):
        left, right = getattr(actual, group), getattr(expected, group)
        assert left.keys() == right.keys()
        for name, value in right.items():
            torch.testing.assert_close(left[name], value, atol=3e-4, rtol=3e-4)
    assert actual.optimizer_state.keys() == expected.optimizer_state.keys()
    for name, state in expected.optimizer_state.items():
        assert actual.optimizer_state[name].keys() == state.keys()
        for key, value in state.items():
            candidate = actual.optimizer_state[name][key]
            if key == "step":
                # Existing checkpoint loading moves the step counter to CUDA;
                # fresh non-capturable AdamW keeps it on CPU. Compare its value.
                candidate = candidate.to(value.device)
            torch.testing.assert_close(candidate, value, atol=3e-4, rtol=3e-4)
    assert (
        actual.training_checkpoint.completed_updates
        == expected.training_checkpoint.completed_updates
    )
    assert (
        actual.training_checkpoint.execution_protocol
        == expected.training_checkpoint.execution_protocol
    )
    assert (
        actual.training_checkpoint.data_state == expected.training_checkpoint.data_state
    )
    assert len(actual.metrics["epochs"]) == len(expected.metrics["epochs"])
    for a, b in zip(actual.metrics["epochs"], expected.metrics["epochs"], strict=True):
        for key in ("train_loss", "train_accuracy", "validation_loss"):
            if key in b:
                assert a[key] == pytest.approx(b[key], abs=3e-4, rel=3e-4)


def test_cpu_default_falls_back_without_touching_cuda(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("CPU execution must not capture CUDA work")

    monkeypatch.setattr(torch.cuda, "CUDAGraph", unexpected)
    kwargs = request()
    kwargs["compile_mode"] = None
    actual = sim.train(**kwargs)
    expected = sim.train(**kwargs, cuda_graphs="off")
    compare(actual, expected)
    assert actual.metrics["cuda_graphs"]["replayed_updates"] == 0
    assert "CUDA device required" in actual.metrics["cuda_graphs"]["fallback_reasons"]


def test_replay_policy_validation():
    assert ExecutionSpec(kind="train").cuda_graphs == "auto"
    with pytest.raises(ValueError, match="auto or off"):
        ExecutionSpec(kind="train", cuda_graphs="force")
    assert ExecutionSpec(kind="simulate", cuda_graphs="off").cuda_graphs == "off"


@pytest.mark.parametrize("policy", ["auto", "off"])
def test_cli_replay_policy_reaches_the_training_field(policy):
    args = parse_args(["train", "--device", "cpu", "--cuda-graphs", policy])
    spec = execution_spec_from_args(args)
    assert spec.cuda_graphs == policy
    assert "cuda_graphs" not in spec.options


@CUDA
@pytest.mark.parametrize("recurrent,block_steps", [(False, 1), (True, 1), (True, 4)])
def test_default_replay_matches_normal_updates_with_partial_batches(
    recurrent, block_steps
):
    kwargs = request(device="cuda", recurrent=recurrent, block_steps=block_steps)
    kwargs["validation"] = sim.ValidationSpec(
        input_bindings=kwargs["input_bindings"], targets=kwargs["targets"]
    )
    kwargs["test"] = sim.TestSpec(
        input_bindings=kwargs["input_bindings"], targets=kwargs["targets"]
    )
    expected = sim.train(**kwargs, cuda_graphs="off")
    actual = sim.train(**kwargs)
    compare(actual, expected)
    if not recurrent:
        assert actual.model.parameter_map()["shadow.weight"].requires_grad
        assert "shadow.weight" not in actual.gradients
        assert actual.optimizer_state["shadow.weight"] == {}
    replay = actual.metrics["cuda_graphs"]
    assert replay["captures"] == 2
    assert replay["replayed_updates"] == 6
    assert not replay["fallback_reasons"]
    assert actual.metrics["test"]["loss"] == pytest.approx(
        expected.metrics["test"]["loss"], abs=3e-4, rel=3e-4
    )
    for name in ("rng_state",):
        torch.testing.assert_close(
            getattr(actual.training_checkpoint, name),
            getattr(expected.training_checkpoint, name),
            atol=0,
            rtol=0,
        )
    for device, state in expected.training_checkpoint.accelerator_rng_states.items():
        torch.testing.assert_close(
            actual.training_checkpoint.accelerator_rng_states[device],
            state,
            atol=0,
            rtol=0,
        )


@CUDA
def test_replay_preserves_requested_dense_diagnostics():
    kwargs = request(device="cuda", recurrent=True, block_steps=4)
    kwargs["diagnostics"] = True
    expected = sim.train(**kwargs, cuda_graphs="off")
    actual = sim.train(**kwargs)
    compare(actual, expected)
    assert actual.metrics["cuda_graphs"]["replayed_updates"] == 6
    assert actual.diagnostics
    assert all(value.shape[:2] == (9, 1) for value in actual.diagnostics.values())


@CUDA
def test_replay_resume_can_switch_policy_and_keeps_initial_validation(tmp_path):
    kwargs = request(device="cuda", recurrent=True, block_steps=4)
    kwargs["shuffle"] = True
    kwargs["validation"] = sim.ValidationSpec(
        input_bindings=kwargs["input_bindings"], targets=kwargs["targets"]
    )
    expected = sim.train(**kwargs, cuda_graphs="off")
    partial = sim.train(**kwargs, updates=4, save_final_checkpoint=tmp_path / "partial")
    assert partial.metrics["cuda_graphs"]["replayed_updates"] == 3
    assert (
        partial.metrics["cuda_graphs"]["fallback_reasons"]["shape occurs only once"]
        == 1
    )
    assert "validation_loss" in partial.metrics["epochs"][0]
    resumed = sim.train(**kwargs, checkpoint=tmp_path / "partial")
    compare(resumed, expected)
    assert resumed.metrics["cuda_graphs"]["replayed_updates"] == 0
    resumed_off = sim.train(
        **kwargs, checkpoint=tmp_path / "partial", cuda_graphs="off"
    )
    compare(resumed_off, expected)


@CUDA
def test_capture_rejection_falls_back_once_and_restores_training(monkeypatch):
    calls = []

    def reject(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("operation not permitted when stream is capturing")

    kwargs = request(device="cuda", samples=4)
    expected = sim.train(**kwargs, cuda_graphs="off")
    monkeypatch.setattr(torch.cuda, "graph", reject)
    actual = sim.train(**kwargs)
    compare(actual, expected)
    assert len(calls) == 1
    assert actual.metrics["cuda_graphs"]["fallback_reasons"] == {
        "operation unsupported during CUDA capture": 4
    }


@CUDA
def test_unrelated_runtime_error_is_not_hidden(monkeypatch):
    def reject(*args, **kwargs):
        raise RuntimeError("bad numerical implementation")

    monkeypatch.setattr(torch.cuda, "graph", reject)
    with pytest.raises(RuntimeError, match="bad numerical implementation"):
        sim.train(**request(device="cuda"))


def test_custom_dynamics_are_rejected_before_warmup():
    kwargs = request()
    built = sim.build(
        graph=kwargs["graph"],
        training=kwargs["training"],
        device="cpu",
        compile_mode=None,
    )
    training = copy.deepcopy(kwargs["training"])
    training["objectives"][0]["kind"] = "custom_objective"
    replay = TrainingReplay(
        ExecutionSpec(kind="train"),
        built.model,
        training,
        [(0, 0, torch.arange(2))] * 2,
    )
    assert "custom objective or regularizer" in replay.reasons
