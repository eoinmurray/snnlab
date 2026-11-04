"""Native fine-tuning contracts with materialized numerical reference inputs."""

import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from snnlab import extensions, lang, sim
from snnlab.sim.inputs.provider import DatasetProvider, iter_batches
from tests.sim._execution_builders import direct_train_bundle

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/native-finetuning/example.py"
spec = importlib.util.spec_from_file_location("native_finetuning_example", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)
extensions.register_encoder("examples.private_inhibition/v1", example.encode)

EAGER = sim.Performance(compile_mode=None, precision="highest", dataset_prefetch=0)


def data(tmp_path, *, cells_count=2):
    path = tmp_path / "dataset.npz"
    np.savez(
        path,
        features=np.asarray([[0.8, 0.1], [0.1, 0.8]] * 4, dtype=np.float32),
        labels=np.asarray([0, 1] * 4),
        ids=np.arange(10, 18),
    )
    bundle = example.circuit(cells_count=cells_count)
    signature = sim.runtime_state_signature(sim.plan_graph(bundle.graph))
    encoder = sim.DatasetEncoder(
        "custom",
        definition="examples.private_inhibition/v1",
        config={"signature": signature, "private_channels": cells_count},
        rng_policy="sample_v1",
        seed=17,
    )
    return bundle, sim.DatasetSnapshotBinding(
        path,
        "drive",
        "fixture",
        "train",
        encoder,
        target_id="label",
        sample_ids_key="ids",
    )


def test_coordinated_rng_is_identity_based_and_evaluation_is_isolated(tmp_path):
    bundle, binding = data(tmp_path)
    provider = DatasetProvider(bundle.graph, binding, encoding_seeds=(101, 102, 103))
    before = torch.get_rng_state().clone()
    full = provider.batch([0, 1, 2], phase="train", epoch=2).tensors
    reordered = provider.batch([2, 0, 1], phase="train", epoch=2).tensors
    next_epoch = provider.batch([0, 1, 2], phase="train", epoch=3).tensors
    for name in full:
        torch.testing.assert_close(
            full[name], reordered[name][:, [1, 2, 0]], rtol=0, atol=0
        )
    torch.testing.assert_close(full["drive"], next_epoch["drive"], rtol=0, atol=0)
    assert not torch.equal(full["artificial"], next_epoch["artificial"])
    validation = provider.batch(
        [0, 1], phase="validation", epoch=0, draw_seed=101
    ).tensors
    repeated = provider.batch(
        [0, 1], phase="validation", epoch=99, draw_seed=101
    ).tensors
    different = provider.batch([0, 1], phase="validation", draw_seed=102).tensors
    torch.testing.assert_close(
        validation["artificial"], repeated["artificial"], rtol=0, atol=0
    )
    assert not torch.equal(validation["artificial"], different["artificial"])
    for depth in (0, 1, 2):
        pieces = list(
            iter_batches(
                provider, [([i], "train", 2, None) for i in range(3)], prefetch=depth
            )
        )
        for name in full:
            torch.testing.assert_close(
                full[name],
                torch.cat([piece[1][name] for piece in pieces], dim=1),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            full.initial_state.conductances["private"],
            torch.cat(
                [piece[1].initial_state.conductances["private"] for piece in pieces]
            ),
            rtol=0,
            atol=0,
        )
    assert torch.equal(before, torch.get_rng_state())


def test_builtin_sample_rng_survives_partition_and_order(tmp_path):
    bundle = direct_train_bundle()
    path = tmp_path / "features.npz"
    np.savez(
        path,
        features=np.ones((5, 2), dtype=np.float32),
        labels=np.arange(5) % 2,
        ids=np.arange(5) + 10,
    )
    binding = sim.DatasetSnapshotBinding(
        path,
        "events",
        "fixture",
        "train",
        sim.DatasetEncoder(
            "rate_poisson",
            duration_ms=2,
            max_rate_hz=2000,
            rng_policy="sample_v1",
            seed=7,
        ),
        sample_ids_key="ids",
    )
    provider = DatasetProvider(bundle.graph, binding)
    full = provider.batch([0, 1, 2], phase="train", epoch=2).tensors["events"]
    ordered = provider.batch([2, 0, 1], phase="train", epoch=2).tensors["events"]
    torch.testing.assert_close(full, ordered[:, [1, 2, 0]], rtol=0, atol=0)
    parts = torch.cat(
        [
            provider.batch([i], phase="train", epoch=2).tensors["events"]
            for i in range(3)
        ],
        dim=1,
    )
    torch.testing.assert_close(full, parts, rtol=0, atol=0)


def test_pending_event_once_fresh_neurons_and_portable_initial_state(tmp_path):
    bundle, binding = data(tmp_path)
    model = sim.build(bundle=bundle, device="cpu", performance=EAGER).model
    resolved, _ = sim.resolve_dataset_snapshot_binding(
        bundle.graph, replace(binding, sample_indices=(0, 1))
    )
    state = resolved.tensors.initial_state
    state.conductances["private"].requires_grad_()
    path = state.save(tmp_path / "initial")
    restored = sim.PresentationState.load(path)
    inputs = {name: torch.zeros_like(value) for name, value in resolved.tensors.items()}
    actual, _ = model._forward(inputs, initial_state=state)
    fresh, _ = model._forward(inputs)
    expected = (
        state.conductances["private"].detach() * math.exp(-0.1 / 6)
        + 0.02 * state.input_histories["artificial"][0]
    )
    final_expected = expected * math.exp(-0.1 / 6) ** 19
    torch.testing.assert_close(
        actual.runtime_state.conductances["private"],
        final_expected,
        rtol=2e-6,
        atol=1e-8,
    )
    assert actual.runtime_state.completed_steps == 20
    assert not actual.runtime_state.input_histories["artificial"].any()
    torch.testing.assert_close(
        actual.runtime_state.voltages["scores_integrator"],
        fresh.runtime_state.voltages["scores_integrator"],
    )
    replayed, _ = model._forward(inputs, initial_state=restored)
    torch.testing.assert_close(
        actual.outputs["class_scores"], replayed.outputs["class_scores"]
    )
    assert state.conductances["private"].grad is None
    assert actual.runtime_state.refractory["E"].eq(0).all()
    for bad, pattern in [
        (replace(state, signature="wrong"), "signature"),
        (replace(state, conductances={"unknown": torch.zeros(2, 2)}), "unknown"),
        (
            replace(state, conductances={"private": torch.zeros(1, 2)}),
            "shape/dtype/device",
        ),
        (
            replace(state, conductances={"private": torch.full((2, 2), float("nan"))}),
            "finite",
        ),
        (
            replace(state, input_histories={"artificial": torch.zeros(2, 2, 2)}),
            "shape/dtype/device",
        ),
    ]:
        with pytest.raises(ValueError, match=pattern):
            model._forward(inputs, initial_state=bad)


def train_materialized(*, microbatch=None, **overrides):
    bundle = direct_train_bundle()
    bundle.training["gradient_clip"] = 0.01
    values = (
        torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ]
            ]
        )
        .expand(3, -1, -1)
        .clone()
    )
    kwargs = dict(
        bundle=bundle,
        input_bindings=[sim.DenseArrayBinding("events", values)],
        targets={"label": torch.tensor([0, 1, 1, 0, 1, 0, 1])},
        epochs=2,
        batch_size=4,
        batching=sim.Batching(microbatch_size=microbatch),
        metrics=sim.TrainingMetrics(update_history=True),
        seed=5,
        device="cpu",
        performance=EAGER,
    )
    kwargs.update(overrides)
    return sim.train(**kwargs)


def assert_training_equal(actual, expected):
    # Declared float32 tolerance, including AdamW first and second moments.
    for name in expected.parameters:
        torch.testing.assert_close(
            actual.parameters[name], expected.parameters[name], rtol=2e-5, atol=2e-6
        )
    for name in expected.gradients:
        torch.testing.assert_close(
            actual.gradients[name], expected.gradients[name], rtol=2e-5, atol=2e-6
        )
    for name, states in expected.optimizer_state.items():
        for key, value in states.items():
            torch.testing.assert_close(
                actual.optimizer_state[name][key], value, rtol=2e-5, atol=2e-6
            )
    assert (
        actual.training_checkpoint.completed_updates
        == expected.training_checkpoint.completed_updates
    )


@pytest.mark.parametrize("microbatch", [1, 2, 3])
def test_accumulation_matches_loss_gradients_adamw_moments_and_remainder(microbatch):
    full = train_materialized()
    actual = train_materialized(microbatch=microbatch)
    assert_training_equal(actual, full)
    assert [row["loss"] for row in actual.metrics["updates"]] == pytest.approx(
        [row["loss"] for row in full.metrics["updates"]], abs=2e-6
    )
    assert actual.metrics["batch_counts"]["optimizer_updates"] == 4
    assert actual.metrics["batch_counts"]["examples"] == 14
    assert actual.metrics["batch_counts"]["microbatches"] == 2 * (
        math.ceil(4 / microbatch) + math.ceil(3 / microbatch)
    )
    assert (
        "microbatch gradient accumulation"
        in actual.metrics["cuda_graphs"]["fallback_reasons"]
    )
    assert actual.parameters["shadow.weight"].equal(full.parameters["shadow.weight"])


def test_accumulation_checkpoint_boundary_and_fresh_weights_optimizer(tmp_path):
    complete = train_materialized(microbatch=2)
    partial = train_materialized(
        microbatch=2, updates=1, save_final_checkpoint=tmp_path / "partial"
    )
    assert partial.training_checkpoint.data_state == {"epoch": 0, "batch": 1}
    resumed = train_materialized(microbatch=2, checkpoint=tmp_path / "partial")
    assert_training_equal(resumed, complete)
    fresh = train_materialized(
        microbatch=2,
        weights=sim.WeightInitialization(partial.training_checkpoint),
        updates=1,
    )
    assert fresh.training_checkpoint.completed_updates == 1
    assert fresh.metrics["resumed_from_update"] == 0
    assert all(float(states["step"]) == 1 for states in fresh.optimizer_state.values())
    fine_complete = train_materialized(
        microbatch=2, weights=sim.WeightInitialization(partial.training_checkpoint)
    )
    train_materialized(
        microbatch=2,
        weights=sim.WeightInitialization(partial.training_checkpoint),
        updates=1,
        save_final_checkpoint=tmp_path / "fine",
    )
    fine_resumed = train_materialized(
        microbatch=2,
        weights=sim.WeightInitialization(partial.training_checkpoint),
        checkpoint=tmp_path / "fine",
    )
    assert_training_equal(fine_resumed, fine_complete)


def test_public_two_input_training_initial_validation_draws_and_accumulation(tmp_path):
    bundle, binding = data(tmp_path, cells_count=3)
    reports = []
    validation = replace(binding, split="validation", sample_indices=(6, 7))
    training = replace(binding, sample_indices=tuple(range(6)))
    result = sim.train(
        bundle=bundle,
        input_bindings=[training],
        validation=sim.ValidationSpec(
            input_bindings=[validation], encoding_seeds=(101, 102, 103)
        ),
        epochs=2,
        batch_size=4,
        batching=sim.Batching(order="shuffle", microbatch_size=2),
        metrics=sim.TrainingMetrics(train="evaluate"),
        performance=EAGER,
        on_epoch_end=reports.append,
        device="cpu",
        save_final_checkpoint=tmp_path / "final",
        save_selected_checkpoint=tmp_path / "selected",
    )
    assert len(result.metrics["epochs"]) == 3
    assert all(
        row["validation_encoding_draws"] == [101, 102, 103]
        for row in result.metrics["epochs"]
    )
    assert result.training_checkpoint.completed_updates == 4
    assert [report.microbatches for report in reports] == [3, 6]
    assert (
        result.metrics["execution_protocol"]["presentation_initialization"]["signature"]
        == binding.encoder.config["signature"]
    )
    assert (
        "per-presentation initial state"
        in result.metrics["cuda_graphs"]["fallback_reasons"]
    )
    assert sim.load_training_checkpoint(tmp_path / "selected").completed_updates <= 4


def test_weights_reject_missing_extra_shape_dtype_and_constraints():
    baseline = train_materialized(updates=1)
    source = baseline.parameters
    for bad in [
        {k: v for k, v in source.items() if k != "shadow.weight"},
        {**source, "unknown": torch.ones(1)},
        {**source, "shadow.weight": torch.ones(19, 2)},
        {**source, "shadow.weight": source["shadow.weight"].double()},
    ]:
        with pytest.raises(ValueError):
            train_materialized(updates=1, weights=sim.WeightInitialization(bad))
    mapped = {"old": source["scores_projection.weight"]}
    result = train_materialized(
        updates=1,
        weights=sim.WeightInitialization(
            mapped,
            mapping={"scores_projection.weight": "old"},
            new_parameters=("shadow.weight",),
        ),
    )
    assert (
        result.metrics["initialization"]["weights_only"]["scaling"]
        == "effective_no_rescaling"
    )


def test_planned_update_schedule_keeps_all_examples():
    policy = sim.Batching(order="shuffle", microbatch_size=64)
    sizes = policy.sizes(6300, 256)
    assert sizes == [256] * 24 + [156]
    assert len(sizes) * 20 == 500
    assert sum(sizes) == 6300


@pytest.mark.parametrize("block_steps", [1, 4])
def test_one_to_one_delayed_filter_values_gradients_and_first_delivery(block_steps):

    def runner(diagonal):
        net = lang.Network("diagonal_fixture", dt=0.1 * lang.ms)
        source = net.input(
            "private", shape=("time", "batch", 3), signal_type="continuous", unit="1"
        )
        cells = net.population("E", size=3, neuron=lang.COBA_LIF(tau_mem=20 * lang.ms))
        projection = net.connect(
            source,
            cells.inhibitory,
            name="gaba",
            synapse=lang.GABA(tau=6 * lang.ms),
            weight=lang.Constant(0.02),
            initialization_scaling="direct",
            delay=0.1 * lang.ms,
            connectivity=lang.OneToOne() if diagonal else None,
        )
        net.output("g", projection.conductance)
        net.output("spikes", cells.spikes)
        net.output("v", cells.voltage)
        graph = lang.compile(net, target="tools/snnsim").graph
        model = sim.GraphExecutor(
            sim.plan_graph(graph),
            trainable_parameters=("gaba.weight",),
            compile_mode="default" if block_steps > 1 else None,
            compile_block_steps=block_steps,
            precision="highest",
        )
        if not diagonal:
            with torch.no_grad():
                model.parameter_map()["gaba.weight"].copy_(torch.eye(3) * 0.02)
        return model

    diagonal, dense = runner(True), runner(False)
    inputs = {
        "private": torch.rand(8, 2, 3, generator=torch.Generator().manual_seed(9))
    }
    g = torch.arange(6).reshape(2, 3).float() * 0.01
    pending = torch.ones(1, 2, 3)
    results = []
    for model in (diagonal, dense):
        initial = sim.PresentationState(
            sim.runtime_state_signature(model.plan),
            conductances={"gaba": g},
            input_histories={"private": pending},
        )
        result, _ = model._forward(inputs, initial_state=initial)
        result.outputs["g"].square().sum().backward()
        results.append(result)
    torch.testing.assert_close(
        results[0].outputs["g"][0],
        math.exp(-0.1 / 6) * g + 0.02 * pending[0],
        rtol=2e-6,
        atol=1e-8,
    )
    for name in ("g", "v", "spikes"):
        torch.testing.assert_close(
            results[0].outputs[name], results[1].outputs[name], rtol=2e-5, atol=2e-6
        )
    torch.testing.assert_close(
        diagonal.parameter_map()["gaba.weight"].grad.flatten(),
        dense.parameter_map()["gaba.weight"].grad.diagonal(),
        rtol=2e-5,
        atol=2e-6,
    )
    assert diagonal.parameter_map()["gaba.weight"].numel() == 3
    if block_steps > 1:
        assert results[0].metrics["execution_path"]["selected"] == "compiled_blocks"


def malformed_encoder(arrays, selected, *, dt_ms, channels, config, seed, context):
    size = len(selected)
    inputs = {"drive": torch.zeros(20, size, 2), "artificial": torch.zeros(20, size, 2)}
    mode = config["mode"]
    if mode == "missing":
        inputs.pop("drive")
    elif mode == "unknown":
        inputs["unknown"] = torch.zeros(20, size, 2)
    elif mode == "samples":
        inputs["drive"] = torch.zeros(20, size + 1, 2)
    elif mode == "time":
        inputs["drive"] = torch.zeros(21, size, 2)
    elif mode == "spikes":
        inputs["drive"] += 0.5
    elif mode == "dtype":
        inputs["drive"] = inputs["drive"].double()
    return inputs


@pytest.mark.parametrize(
    "mode", ["missing", "unknown", "samples", "time", "spikes", "dtype"]
)
def test_coordinated_encoder_rejects_invalid_inputs(tmp_path, mode):

    extensions.register_encoder(f"tests.malformed-input-{mode}/v1", malformed_encoder)
    bundle, binding = data(tmp_path)
    bad = replace(
        binding,
        encoder=replace(
            binding.encoder,
            definition="tests.malformed-input/v1",
            config={"mode": mode},
        ),
    )
    with pytest.raises(ValueError):
        DatasetProvider(bundle.graph, bad)


class NonfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        return values.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient * float("nan")


def nonfinite_loss(scores, targets, config):
    return NonfiniteGradient.apply(scores).sum()


def test_nonfinite_accumulated_gradient_fails_before_optimizer(monkeypatch):

    extensions.register_objective(
        "tests.nonfinite-accumulation/v1", nonfinite_loss, classification=True
    )
    bundle = direct_train_bundle()
    bundle.training["objectives"][0].update(
        kind="custom_objective", definition="tests.nonfinite-accumulation/v1", config={}
    )
    stepped = []
    monkeypatch.setattr(
        torch.optim.AdamW, "step", lambda *args, **kwargs: stepped.append(True)
    )
    with pytest.raises(RuntimeError, match="nonfinite|non-finite"):
        train_materialized(microbatch=2, bundle=bundle, updates=1)
    assert not stepped


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_accumulation_with_identical_materialized_inputs():
    expected = train_materialized(device="cuda")
    actual = train_materialized(
        microbatch=2,
        device="cuda",
        performance=sim.Performance(
            compile_mode="default", compile_block_steps=4, precision="highest"
        ),
    )
    assert_training_equal(actual, expected)


def test_post_clip_gradients_match_unsplit_at_each_optimizer_boundary(monkeypatch):
    original = torch.optim.AdamW.step
    captured = []

    def record(optimizer, *args, **kwargs):
        captured.append(
            [
                parameter.grad.detach().clone()
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
        )
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", record)
    train_materialized()
    full = captured[:]
    captured.clear()
    train_materialized(microbatch=2)
    assert len(captured) == len(full) == 4
    for expected, actual in zip(full, captured, strict=True):
        for left, right in zip(expected, actual, strict=True):
            torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)
        norm = torch.linalg.vector_norm(
            torch.cat([value.flatten() for value in actual])
        )
        assert float(norm) <= 0.010001


def test_weights_constraints_and_snapshot_provenance(tmp_path):
    bundle, binding = data(tmp_path)
    built = sim.build(bundle=bundle, device="cpu", performance=EAGER)
    negative = {
        **built.parameters,
        "private.weight": -built.parameters["private.weight"],
    }
    with pytest.raises(ValueError, match="constraints"):
        sim.train(
            bundle=bundle,
            weights=sim.WeightInitialization(negative),
            input_bindings=[binding],
            epochs=1,
            device="cpu",
            performance=EAGER,
        )
    result = sim.train(
        bundle=bundle,
        weights=sim.WeightInitialization(built.parameters),
        input_bindings=[binding],
        epochs=1,
        batch_size=4,
        batching=sim.Batching(microbatch_size=2),
        device="cpu",
        performance=EAGER,
        training_timing=True,
        snapshot=tmp_path / "snapshot",
    )
    snapshot = sim.NetworkSnapshot.read(tmp_path / "snapshot")
    assert snapshot.load_parameters()
    assert result.metrics["initialization"]["weights_only"]["source_tensor_sha256"]
    assert result.metrics["training_timing"]["environment"]["device"] == "cpu"
    assert (
        result.metrics["training_timing"]["device_memory"]["peak_allocated_bytes"]
        is None
    )


def test_coordinated_accumulation_matches_unsplit_synaptic_initialization(tmp_path):
    bundle, binding = data(tmp_path, cells_count=3)
    kwargs = dict(
        bundle=bundle,
        input_bindings=[binding],
        epochs=2,
        batch_size=5,
        metrics=sim.TrainingMetrics(update_history=True),
        seed=7,
        device="cpu",
        performance=EAGER,
    )
    full = sim.train(**kwargs)
    actual = sim.train(**kwargs, batching=sim.Batching(microbatch_size=2))
    assert_training_equal(actual, full)
    assert [row["loss"] for row in actual.metrics["updates"]] == pytest.approx(
        [row["loss"] for row in full.metrics["updates"]], abs=2e-6
    )
    assert (actual.parameters["sensory.weight"] >= 0).all()
    assert (actual.parameters["scores_projection.weight"] >= 0).all()
    torch.testing.assert_close(
        actual.parameters["private.weight"],
        full.parameters["private.weight"],
        rtol=0,
        atol=0,
    )


def test_full_scored_trial_mean_is_pre_reset_and_starts_fresh(tmp_path):
    bundle, binding = data(tmp_path)
    graph = bundle.graph.copy()
    graph["outputs"] = [
        *graph["outputs"],
        {"id": "pre_reset", "signal": "scores_integrator.pre_reset_voltage"},
    ]
    model = sim.build(graph=graph, device="cpu", performance=EAGER).model
    state = sim.PresentationState(
        sim.runtime_state_signature(model.plan),
        conductances={"private": torch.tensor([[0.01, 0.03]])},
        input_histories={"artificial": torch.ones(1, 1, 2)},
    )
    inputs = {"drive": torch.ones(2000, 1, 2), "artificial": torch.zeros(2000, 1, 2)}
    actual = model(inputs, initial_state=state, diagnostics=False)
    repeated = model(inputs, initial_state=state, diagnostics=False)
    assert actual.runtime_state.completed_steps == 2000
    assert actual.outputs["pre_reset"].shape[0] == 2000
    assert actual.outputs["pre_reset"].abs().sum() > 0
    torch.testing.assert_close(
        actual.outputs["class_scores"],
        actual.outputs["pre_reset"].mean(dim=0),
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        actual.outputs["pre_reset"], repeated.outputs["pre_reset"], rtol=0, atol=0
    )


def test_coordinated_resume_preserves_rng_and_initial_state_identity(tmp_path):
    bundle, binding = data(tmp_path)
    kwargs = dict(
        bundle=bundle,
        input_bindings=[binding],
        epochs=2,
        batch_size=5,
        batching=sim.Batching(order="shuffle", microbatch_size=2),
        metrics=sim.TrainingMetrics(update_history=True),
        seed=7,
        device="cpu",
        performance=EAGER,
    )
    complete = sim.train(**kwargs)
    partial = sim.train(**kwargs, updates=1, save_final_checkpoint=tmp_path / "partial")
    assert partial.training_checkpoint.reporting_state["microbatches"] == 3
    resumed = sim.train(**kwargs, checkpoint=tmp_path / "partial")
    assert_training_equal(resumed, complete)
    encoding = complete.metrics["updates"][0]["encoding"]["microbatches"][0]
    assert {row["domain"] for row in encoding["rng_streams"]} == {
        "sensory",
        "inhibition",
        "spike_encoding",
        "rate_selection",
    }
    assert encoding["initial_state"]["provenance"]["preparation"]["sample_ids"]
