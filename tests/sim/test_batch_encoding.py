"""Batch-local encoding, independent streams, fixed evaluation and exact resume."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from snnlab.sim.dataset_provider import DatasetProvider
from snnlab.sim.execution import (
    CheckpointSelection,
    DatasetEncoder,
    DatasetSnapshotBinding,
    EpochObservations,
    ExecutionSpec,
    GraphExecutor,
    ValidationSpec,
    load_training_checkpoint,
    save_training_checkpoint,
    simulate,
    train,
)
from tests.sim.test_epoch_observations import request


def dataset(tmp_path, count=9, encoder=None, name="data"):
    path = tmp_path / f"{name}.npz"
    labels = np.arange(count) % 2
    features = np.full((count, 2), 0.25, dtype=np.float32)
    features[np.arange(count), labels] = 1
    np.savez_compressed(path, features=features, labels=labels)
    return DatasetSnapshotBinding(
        path=path,
        input_id="events",
        target_id="label",
        dataset_id=name,
        split="train",
        encoder=encoder
        or DatasetEncoder("rate_poisson", duration_ms=20, max_rate_hz=500, seed=19),
    )


def training_request(tmp_path, *, validation=False, audits=False):
    spec = request()
    binding = dataset(tmp_path)
    val = (
        ValidationSpec(
            input_bindings=(replace(binding, split="validation"),),
            encoding_seeds=(101, 103),
        )
        if validation
        else None
    )
    return replace(
        spec,
        input_bindings=(binding,),
        targets={},
        validation=val,
        epochs=3,
        batch_size=3,
        shuffle=True,
        observations=EpochObservations(
            population_rates=("out",),
            parameter_norms=("drive.weight",),
            gradient_norms=("drive.weight",),
        )
        if audits
        else None,
        checkpoint_selection=CheckpointSelection() if validation else None,
    )


def test_only_current_batch_is_encoded_and_source_is_memory_mapped(
    tmp_path, monkeypatch
):
    binding = dataset(tmp_path, count=97)
    spec = request()
    sizes = []
    original = torch.rand

    def tracked(*args, **kwargs):
        if len(args) == 3:
            sizes.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "rand", tracked)
    result = train(
        replace(
            spec,
            input_bindings=(binding,),
            targets={},
            validation=None,
            epochs=1,
            batch_size=7,
        )
    )
    assert max(sizes) <= 7
    assert len(result.metrics["updates"]) == 14
    assert len(result.metrics["updates"][-1]["encoding"]["sample_indices"]) == 6
    provider = DatasetProvider(spec.graph, binding)
    assert isinstance(provider.snapshot[0]["features"], np.memmap)
    assert not hasattr(provider, "tensors")
    assert provider.protocol["batch_encoding"]["schema"] == "snnlab.batch-encoding/v1"


def test_training_draws_are_fresh_evaluation_fixed_and_rng_independent(tmp_path):
    provider = DatasetProvider(request().graph, dataset(tmp_path))
    indices = torch.arange(3)
    rng = torch.get_rng_state().clone()
    first = provider.batch(indices, phase="train", epoch=0).tensors["events"]
    second = provider.batch(indices, phase="train", epoch=1).tensors["events"]
    assert not torch.equal(first, second)
    fixed = provider.batch(indices, phase="validation", epoch=0, draw_seed=101).tensors[
        "events"
    ]
    later = provider.batch(indices, phase="validation", epoch=7, draw_seed=101).tensors[
        "events"
    ]
    torch.testing.assert_close(fixed, later, rtol=0, atol=0)
    distinct = provider.batch(indices, phase="validation", draw_seed=103).tensors[
        "events"
    ]
    assert not torch.equal(fixed, distinct)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("updates", [1, 3, 4])
@pytest.mark.parametrize("audits", [False, True])
@pytest.mark.parametrize("categorical", [False, True])
def test_resume_matches_uninterrupted_parameters_optimizer_rng_and_selection(
    tmp_path, updates, audits, categorical
):
    spec = training_request(tmp_path, validation=True, audits=audits)
    if categorical:
        spec = replace(
            spec,
            input_bindings=(
                replace(
                    spec.input_bindings[0],
                    encoder=DatasetEncoder(
                        "rate_poisson", duration_ms=20, rates_hz=(100, 500), seed=19
                    ),
                ),
            ),
        )
    full = train(spec)
    first = train(
        replace(spec, updates=updates, save_final_checkpoint=tmp_path / "checkpoint")
    )
    resumed = train(replace(spec, checkpoint=tmp_path / "checkpoint"))
    for name in full.parameters:
        torch.testing.assert_close(
            full.parameters[name], resumed.parameters[name], rtol=0, atol=0
        )
    assert (
        full.selected_checkpoint.selection_record
        == resumed.selected_checkpoint.selection_record
    )
    assert torch.equal(
        full.training_checkpoint.rng_state, resumed.training_checkpoint.rng_state
    )
    assert (
        first.metrics["updates"] + resumed.metrics["updates"] == full.metrics["updates"]
    )
    for name, state in full.training_checkpoint.optimizer_state.items():
        for key, value in state.items():
            other = resumed.training_checkpoint.optimizer_state[name][key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, other, rtol=0, atol=0)
            else:
                assert value == other
    if audits:
        assert full.metrics["epochs"] == resumed.metrics["epochs"]


def test_categorical_rates_are_independent_per_presentation_and_spike_stream_is_separate(
    tmp_path,
):
    binding = dataset(
        tmp_path,
        count=64,
        encoder=DatasetEncoder(
            "rate_poisson", duration_ms=20, rates_hz=(0, 1000), seed=19
        ),
    )
    provider = DatasetProvider(request().graph, binding)
    batch = provider.batch(torch.arange(64), phase="train", epoch=0)
    rates = batch.protocol["dataset_binding"]["encoder"]["selected_rates_hz"]
    assert set(rates) == {0, 1000}
    assert (
        provider.last_batch["rate_selection_seed"]
        != provider.last_batch["spike_encoding_seed"]
    )
    for index, rate in enumerate(rates):
        if rate == 0:
            assert batch.tensors["events"][:, index].sum() == 0
    fixed = replace(
        binding,
        encoder=DatasetEncoder(
            "rate_poisson", duration_ms=20, max_rate_hz=500, seed=19
        ),
    )
    categorical = replace(
        binding,
        encoder=DatasetEncoder(
            "rate_poisson", duration_ms=20, rates_hz=(500,), seed=19
        ),
    )
    a = DatasetProvider(request().graph, fixed).batch(
        torch.arange(6), phase="train", epoch=2
    )
    b = DatasetProvider(request().graph, categorical).batch(
        torch.arange(6), phase="train", epoch=2
    )
    torch.testing.assert_close(a.tensors["events"], b.tensors["events"], rtol=0, atol=0)


def test_validation_draw_aggregation_is_sample_weighted_and_repeatable(tmp_path):
    spec = training_request(tmp_path, validation=True)
    recipe = spec.training.copy()
    recipe["parameter_groups"] = [
        {**group, "lr": 0} for group in recipe["parameter_groups"]
    ]
    spec = replace(spec, training=recipe)
    multiple = train(spec)
    rows = multiple.metrics["epochs"]
    assert all(
        row["validation_cross_entropies"] == rows[0]["validation_cross_entropies"]
        for row in rows
    )
    singles = [
        train(
            replace(spec, validation=replace(spec.validation, encoding_seeds=(seed,)))
        )
        for seed in (101, 103)
    ]
    expected = (
        sum(result.metrics["epochs"][0]["validation_loss"] for result in singles) / 2
    )
    assert rows[0]["validation_loss"] == pytest.approx(expected)
    assert rows[0]["validation_encoding_draws"] == [101, 103]


def test_exact_prebinned_and_event_replay_are_preserved(tmp_path):
    graph = request().graph
    values = np.zeros((4, 5, 2), dtype=np.uint8)
    values[0, 0, 1] = 1
    values[2, 3, 0] = 1
    path = tmp_path / "binned.npz"
    np.savez(path, features=values, labels=np.arange(5) % 2)
    binding = DatasetSnapshotBinding(
        path, "events", "binned", "train", DatasetEncoder("prebinned_spikes")
    )
    provider = DatasetProvider(graph, binding)
    combined = torch.cat(
        [
            provider.batch(
                torch.arange(start, min(start + 2, 5)), phase="train", epoch=7
            ).tensors["events"]
            for start in range(0, 5, 2)
        ],
        dim=1,
    )
    torch.testing.assert_close(
        combined, torch.tensor(values, dtype=torch.float32), rtol=0, atol=0
    )
    event_path = tmp_path / "events.npz"
    np.savez(
        event_path,
        labels=np.arange(5) % 2,
        event_sample=np.array([0, 3]),
        event_time_ms=np.array([0.0, 2.0]),
        event_channel=np.array([1, 0]),
    )
    event = replace(
        binding, path=event_path, encoder=DatasetEncoder("event_bin", duration_ms=4)
    )
    event_provider = DatasetProvider(graph, event)
    combined = torch.cat(
        [
            event_provider.batch(torch.arange(start, min(start + 2, 5))).tensors[
                "events"
            ]
            for start in range(0, 5, 2)
        ],
        dim=1,
    )
    torch.testing.assert_close(
        combined, torch.tensor(values, dtype=torch.float32), rtol=0, atol=0
    )


def test_snapshot_selection_tail_and_invalid_fields(tmp_path):
    binding = replace(
        dataset(tmp_path, count=13), sample_cap=9, shuffle=True, order_seed=11
    )
    provider = DatasetProvider(request().graph, binding)
    assert (
        provider.selected.tolist()
        == torch.randperm(13, generator=torch.Generator().manual_seed(11))[:9].tolist()
    )
    assert provider.targets[0].value.tolist() == [
        index % 2 for index in provider.selected
    ]
    for encoder in [
        DatasetEncoder(
            "rate_poisson", duration_ms=20, max_rate_hz=500, rates_hz=(100,)
        ),
        DatasetEncoder("rate_poisson", duration_ms=20, rates_hz=()),
        DatasetEncoder("rate_poisson", duration_ms=20, rates_hz=(-1,)),
        DatasetEncoder("rate_poisson", duration_ms=20, rates_hz=(1001,)),
    ]:
        with pytest.raises(ValueError):
            DatasetProvider(request().graph, replace(binding, encoder=encoder))


def test_inference_encodes_batches_without_rebuilding_model(tmp_path, monkeypatch):
    binding = dataset(tmp_path, count=11)
    sizes = []
    original = GraphExecutor._forward

    def record(self, inputs, **kwargs):
        sizes.append((id(self), inputs["events"].shape[1]))
        return original(self, inputs, **kwargs)

    monkeypatch.setattr(GraphExecutor, "_forward", record)
    result = simulate(
        ExecutionSpec(
            kind="infer",
            graph=request().graph,
            input_bindings=(binding,),
            batch_size=4,
            device="cpu",
            seed=23,
        )
    )
    assert [size for _, size in sizes] == [4, 4, 3]
    assert len({model for model, _ in sizes}) == 1
    assert result.outputs["scores"].shape == (11, 2)
    assert result.runtime_state is None
    assert len(result.metrics["encoding_batches"]) == 3


def test_changed_encoder_draws_and_old_protocol_cannot_resume(tmp_path):
    spec = training_request(tmp_path, validation=True)
    train(replace(spec, updates=1, save_final_checkpoint=tmp_path / "checkpoint"))
    changed = replace(
        spec,
        input_bindings=(
            replace(
                spec.input_bindings[0],
                encoder=replace(spec.input_bindings[0].encoder, seed=20),
            ),
        ),
    )
    with pytest.raises(ValueError, match="execution protocol"):
        train(replace(changed, checkpoint=tmp_path / "checkpoint"))
    with pytest.raises(ValueError, match="selection policy or evaluation identity"):
        train(
            replace(
                spec,
                validation=replace(spec.validation, encoding_seeds=(107,)),
                checkpoint=tmp_path / "checkpoint",
            )
        )
    old = load_training_checkpoint(tmp_path / "checkpoint")
    protocol = dict(old.execution_protocol)
    protocol.pop("batch_encoding")
    save_training_checkpoint(
        tmp_path / "old", replace(old, execution_protocol=protocol)
    )
    with pytest.raises(ValueError, match="execution protocol"):
        train(replace(spec, checkpoint=tmp_path / "old"))


def test_provider_uses_snapshot_even_if_original_path_is_replaced(tmp_path):
    binding = dataset(tmp_path)
    provider = DatasetProvider(request().graph, binding)
    first = provider.batch(torch.arange(4)).tensors["events"]
    binding.path.write_bytes(b"replaced")
    second = provider.batch(torch.arange(4)).tensors["events"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_default_dataset_request_is_automatically_batched_for_one_epoch(
    tmp_path, monkeypatch
):
    spec = request()
    binding = dataset(tmp_path, count=65)
    sizes = []
    original = GraphExecutor._forward

    def track(self, inputs, **kwargs):
        sizes.append(inputs["events"].shape[1])
        return original(self, inputs, **kwargs)

    monkeypatch.setattr(GraphExecutor, "_forward", track)
    result = train(
        replace(
            spec,
            input_bindings=(binding,),
            targets={},
            validation=None,
            epochs=0,
            batch_size=None,
        )
    )
    assert max(sizes) == 32
    assert len(result.metrics["updates"]) == 3
    assert result.training_checkpoint.data_state == {"epoch": 1, "batch": 0}


def test_encoding_peak_does_not_grow_with_dataset_size(tmp_path, monkeypatch):
    largest = []
    original = torch.rand

    def track(*args, **kwargs):
        if len(args) == 3:
            largest.append(args[0] * args[1] * args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "rand", track)
    for count in (9, 97):
        largest.clear()
        spec = request()
        train(
            replace(
                spec,
                input_bindings=(dataset(tmp_path, count=count),),
                targets={},
                validation=None,
                epochs=1,
                batch_size=3,
            )
        )
        assert max(largest) == 20 * 3 * 2


def test_cli_dataset_encoding_is_automatic_and_categorical_rates_are_recorded(tmp_path):
    import json

    from snnlab.sim.tool import main
    from tests.sim._execution_builders import direct_train_bundle

    bundle = direct_train_bundle().write(tmp_path / "network.bundle")
    binding = dataset(tmp_path, count=7)
    out = tmp_path / "out"
    main(
        [
            "train",
            "--executor",
            "graph",
            "--bundle",
            str(bundle),
            "--dataset-file",
            str(binding.path),
            "--dataset-encoder",
            "rate-poisson",
            "--dataset-target-id",
            "label",
            "--input-dataset-id",
            "fixture",
            "--input-split",
            "train",
            "--epochs",
            "2",
            "--batch-size",
            "3",
            "--t-ms",
            "4",
            "--max-samples",
            "7",
            "--input-rates",
            "0",
            "1000",
            "--out-dir",
            str(out),
            "--device",
            "cpu",
        ]
    )
    metrics = json.loads((out / "metrics.json").read_text())
    assert (
        metrics["execution_protocol"]["batch_encoding"]["schema"]
        == "snnlab.batch-encoding/v1"
    )
    assert metrics["execution_protocol"]["dataset_binding"]["encoder"]["rates_hz"] == [
        0.0,
        1000.0,
    ]
    assert len(metrics["updates"]) == 6
    assert len(metrics["updates"][-1]["encoding"]["sample_indices"]) == 1
