"""Named observations are physically meaningful and do not change training."""

import copy
import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from snnlab import lang as snn
from snnlab.sim.epoch_observations import evaluation_rng
from snnlab.sim.execution import (
    DenseArrayBinding,
    EpochObservations,
    ExecutionSpec,
    GraphExecutor,
    ObservationProbe,
    ValidationSpec,
    load_training_checkpoint,
    train,
)


def request():
    net = snn.Network("audit", dt=1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    out = net.population(
        "out",
        size=2,
        neuron=snn.LeakyIntegrator(tau=2 * snn.ms, soft_reset_threshold=1.0),
        spiking=True,
    )
    projection = net.connect(
        events,
        out.excitatory,
        name="drive",
        synapse=snn.LeakyIntegrator(tau=2 * snn.ms),
        weight=snn.LowerClampedNormal(4.0, 0.1),
    )
    mean = snn.ops.reduce(
        out.pre_reset_voltage, operation="mean", over="time", name="mean"
    )
    net.output("scores", mean)
    recipe = snn.TrainSpec(
        objectives=[snn.training.CrossEntropy(prediction=mean, target="label")],
        parameter_groups=[
            snn.training.ParameterGroup([projection.weight], name="weights", lr=0.01)
        ],
        optimizer=snn.training.AdamW(weight_decay=0),
        gradient_clip=0.001,
    )
    bundle = snn.compile(net, training=recipe)
    events_value = torch.zeros(4, 5, 2)
    labels = torch.tensor([0, 1, 0, 1, 0])
    events_value[:, torch.arange(5), labels] = 1
    binding = DenseArrayBinding("events", events_value)
    return ExecutionSpec(
        kind="train",
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(binding,),
        targets={"label": labels},
        validation=ValidationSpec(input_bindings=(binding,), targets={"label": labels}),
        epochs=2,
        batch_size=2,
        seed=23,
        device="cpu",
        diagnostics=False,
    )


def observations(spec):
    return EpochObservations(
        population_rates=("out",),
        parameter_norms=("drive.weight",),
        output_activity=("out",),
        gradient_norms=("drive.weight",),
        probes={
            "reference": ObservationProbe(input_bindings=spec.input_bindings, seed=31),
            "draw_1": ObservationProbe(
                input_bindings=spec.input_bindings, seed=47, split="validation"
            ),
        },
    )


def test_named_units_activity_parameter_norms_and_completed_epoch_metadata():
    spec = request()
    result = train(replace(spec, observations=observations(spec)))
    rows = result.metrics["epochs"]
    assert [r["epoch"] for r in rows] == [0, 1, 2]
    assert [r["update"] for r in rows] == [0, 3, 6]
    assert rows[0]["phase"] == "initial"
    assert (
        rows[0]["observations"]["gradients"]["pre_clip_l2_norm"]["drive.weight"] is None
    )
    final = rows[-1]["observations"]
    activity = final["train"]
    assert activity["duration_s"] == 0.004
    assert activity["sample_count"] == 5
    with torch.no_grad():
        _, signals = result.model._forward(
            {"events": spec.input_bindings[0].value},
            diagnostics=False,
            required_signals=("out.spikes",),
        )
    spikes = signals["out.spikes"]
    assert activity["population_rates_hz"]["out"] == pytest.approx(
        float(spikes.sum()) / (5 * 2 * 0.004)
    )
    output = activity["output_activity"]["out"]
    assert output["spike_totals"] == spikes.sum(dim=(0, 1)).long().tolist()
    assert sum(output["class_spike_shares"]) == pytest.approx(1)
    assert output["silent_sample_fraction"] == pytest.approx(
        float((spikes.sum(dim=(0, 2)) == 0).float().mean())
    )
    assert final["parameters"]["l2_norm"]["drive.weight"] == pytest.approx(
        float(torch.linalg.vector_norm(result.parameters["drive.weight"]))
    )
    gradient = final["gradients"]
    assert gradient["updates"] == 3
    assert gradient["missing_updates"]["drive.weight"] == 0
    assert gradient["post_clip_l2_norm"]["drive.weight"] <= 0.00101
    assert (
        gradient["pre_clip_l2_norm"]["drive.weight"]
        >= gradient["post_clip_l2_norm"]["drive.weight"]
    )
    assert final["probes"]["reference"]["split"] == "reference"
    assert final["probes"]["draw_1"]["split"] == "validation"
    assert final["probes"]["draw_1"]["evaluation_seed"] == 47
    assert result.diagnostics == {}


def test_observations_and_extra_stochastic_evaluations_do_not_change_training(
    monkeypatch,
):
    original = GraphExecutor._forward

    def noisy_evaluation(self, *args, **kwargs):
        if not torch.is_grad_enabled():
            torch.rand(7)
            np.random.random(3)
            random.random()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GraphExecutor, "_forward", noisy_evaluation)
    spec = request()
    random.seed(9)
    np.random.seed(9)
    plain = train(spec)
    expected_next = (random.random(), np.random.random(), torch.rand(3))
    random.seed(9)
    np.random.seed(9)
    measured = train(replace(spec, observations=observations(spec)))
    actual_next = (random.random(), np.random.random(), torch.rand(3))
    assert expected_next[:2] == actual_next[:2]
    torch.testing.assert_close(expected_next[2], actual_next[2], rtol=0, atol=0)
    assert measured.metrics["updates"] == plain.metrics["updates"]
    for name in plain.parameters:
        torch.testing.assert_close(
            measured.parameters[name], plain.parameters[name], rtol=0, atol=0
        )
        for key, value in plain.optimizer_state[name].items():
            torch.testing.assert_close(
                measured.optimizer_state[name][key], value, rtol=0, atol=0
            )
    torch.testing.assert_close(
        measured.training_checkpoint.rng_state,
        plain.training_checkpoint.rng_state,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("updates", [1, 3])
def test_resumed_history_and_partial_gradient_totals_match_uninterrupted(
    tmp_path, updates
):
    spec = request()
    spec = replace(spec, observations=observations(spec))
    full = train(spec)
    first = train(
        replace(spec, updates=updates, save_final_checkpoint=tmp_path / "checkpoint")
    )
    assert (
        json.loads((tmp_path / "checkpoint" / "manifest.json").read_text())[
            "schema_version"
        ]
        == 3
    )
    loaded = load_training_checkpoint(tmp_path / "checkpoint")
    assert loaded.observation_state == first.training_checkpoint.observation_state
    resumed = train(replace(spec, checkpoint=tmp_path / "checkpoint"))
    assert resumed.metrics["epochs"] == full.metrics["epochs"]
    for name in full.parameters:
        torch.testing.assert_close(
            full.parameters[name], resumed.parameters[name], rtol=0, atol=0
        )
    assert first.metrics["epochs"][-1]["epoch"] == (0 if updates == 1 else 1)
    with pytest.raises(ValueError, match="observation configuration"):
        train(
            replace(
                spec,
                checkpoint=tmp_path / "checkpoint",
                observations=EpochObservations(population_rates=("out",)),
            )
        )
    with pytest.raises(ValueError, match="observation configuration"):
        train(replace(spec, checkpoint=tmp_path / "checkpoint", observations=None))


def test_history_corruption_and_old_checkpoint_opt_in_fail_explicitly(tmp_path):
    spec = request()
    train(
        replace(
            spec,
            observations=observations(spec),
            updates=1,
            save_final_checkpoint=tmp_path / "audited",
        )
    )
    path = tmp_path / "audited" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["observation_state"]["history"][0]["epoch"] = 99
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="observation history digest"):
        load_training_checkpoint(tmp_path / "audited")
    train(replace(spec, updates=1, save_final_checkpoint=tmp_path / "plain"))
    with pytest.raises(ValueError, match="no epoch observation history"):
        train(
            replace(
                spec, observations=observations(spec), checkpoint=tmp_path / "plain"
            )
        )


@pytest.mark.parametrize(
    "field, name",
    [
        ("population_rates", "missing"),
        ("parameter_norms", "missing"),
        ("output_activity", "missing"),
        ("gradient_norms", "missing"),
    ],
)
def test_unknown_names_rejected(field, name):
    with pytest.raises(ValueError, match="missing"):
        train(replace(request(), observations=EpochObservations(**{field: (name,)})))


def test_silence_is_explicit_and_evaluation_rng_restores_on_error():
    spec = request()
    binding = DenseArrayBinding("events", torch.zeros(4, 2, 2))
    selected = replace(
        observations(spec),
        probes={"silent": ObservationProbe(input_bindings=(binding,))},
    )
    result = train(replace(spec, observations=selected))
    output = result.metrics["epochs"][-1]["observations"]["probes"]["silent"][
        "output_activity"
    ]["out"]
    assert output["silent_sample_fraction"] == 1
    assert output["class_spike_shares"] is None
    assert output["status"] == "no_spikes"
    saved = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with evaluation_rng(10):
            torch.rand(5)
            raise RuntimeError("fixture")
    torch.testing.assert_close(saved, torch.get_rng_state(), rtol=0, atol=0)


def test_activity_observation_does_not_request_dense_spike_history(monkeypatch):
    original = GraphExecutor._forward
    seen = []

    def inspect(self, *args, **kwargs):
        if kwargs.get("observation_populations"):
            seen.append(kwargs)
            assert "out.spikes" not in kwargs.get("required_signals", ())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GraphExecutor, "_forward", inspect)
    spec = request()
    train(replace(spec, observations=observations(spec)))
    assert seen


def test_missing_gradients_and_optimizer_state_survive_resume(tmp_path):
    from snnlab.lang.compiler import digest

    spec = request()
    graph, recipe = copy.deepcopy(spec.graph), copy.deepcopy(spec.training)
    graph["parameters"].append(
        {
            "id": "unused",
            "shape": [1, 2],
            "unit": "1",
            "initializer": snn.Constant(2.0).json(),
        }
    )
    graph["operations"].append(
        {
            "id": "unused_linear",
            "kind": "linear",
            "sources": ["events.value"],
            "parameters": ["unused"],
            "shape": ["time", "batch", 1],
            "unit": "spike",
            "config": {},
        }
    )
    recipe["parameter_groups"][0]["parameters"].append("unused")
    recipe["resolved_parameters"]["trainable"].append("unused")
    recipe["graph_digest"] = digest(graph)
    selected = replace(observations(spec), gradient_norms=("drive.weight", "unused"))
    spec = replace(spec, graph=graph, training=recipe, observations=selected)
    train(replace(spec, updates=1, save_final_checkpoint=tmp_path / "checkpoint"))
    result = train(replace(spec, checkpoint=tmp_path / "checkpoint"))
    gradient = result.metrics["epochs"][-1]["observations"]["gradients"]
    assert gradient["pre_clip_l2_norm"]["unused"] is None
    assert gradient["missing_updates"]["unused"] == 3
    assert result.optimizer_state["unused"] == {}


def test_fixed_validation_draws_are_repeatable_and_separately_identified():
    from snnlab.sim.execution import PoissonInputBinding

    spec = request()
    selected = replace(
        observations(spec),
        probes={
            f"draw_{seed}": ObservationProbe(
                input_bindings=(
                    PoissonInputBinding("events", 8, (500.0,), seed, batch_size=3),
                ),
                seed=seed,
                split="validation",
            )
            for seed in (17, 23)
        },
    )
    one = train(replace(spec, observations=selected))
    two = train(replace(spec, observations=selected))
    assert one.metrics["epochs"] == two.metrics["epochs"]
    probes = one.metrics["epochs"][0]["observations"]["probes"]
    assert set(probes) == {"draw_17", "draw_23"}
    assert probes["draw_17"]["input_protocol"] != probes["draw_23"]["input_protocol"]


def test_history_logical_coordinates_are_checked(tmp_path):
    from snnlab.sim.execution import save_training_checkpoint

    spec = request()
    spec = replace(spec, observations=observations(spec))
    first = train(replace(spec, updates=1))
    first.training_checkpoint.observation_state["history"][0]["epoch"] = 5
    save_training_checkpoint(tmp_path / "checkpoint", first.training_checkpoint)
    with pytest.raises(ValueError, match="not contiguous"):
        train(replace(spec, checkpoint=tmp_path / "checkpoint"))
