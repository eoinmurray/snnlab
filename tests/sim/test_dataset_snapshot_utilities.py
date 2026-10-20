"""Dataset archives preserve caller rows and enter the existing native pipeline."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from snnlab import sim
from tests.sim._execution_builders import direct_train_bundle


def dataset(tmp_path):
    labels = np.asarray([0, 1, 0, 1, 0], dtype=np.int32)
    features = np.eye(2, dtype=np.float32)[labels]
    partitions = {
        "train": sim.DatasetPartition(features, labels, np.asarray([91, 7, 42, 8, 3])),
        "validation": sim.DatasetPartition(
            features[:3], labels[:3], np.asarray([15, 26, 37])
        ),
        "test": sim.DatasetPartition(features[:2], labels[:2], np.asarray([50, 60])),
    }
    snapshot = sim.DatasetSnapshot.from_partitions(
        dataset_id="fixture",
        partitions=partitions,
        disjoint=tuple(partitions),
        preprocessing={"operation": "caller-supplied identity"},
    )
    path = snapshot.write(tmp_path / "data.npz")
    return snapshot, sim.DatasetSnapshot.read(path)


def encoder():
    return sim.DatasetEncoder("rate_poisson", duration_ms=0.3, max_rate_hz=1000, seed=5)


def test_named_archive_preserves_arrays_and_identity_selected_bindings(tmp_path):
    source, saved = dataset(tmp_path)
    assert saved.counts == {"train": 5, "validation": 3, "test": 2}
    assert saved.preprocessing == source.preprocessing
    for name in source.partitions:
        for field in ("features", "labels", "sample_ids"):
            expected, actual = (
                getattr(source.partitions[name], field),
                getattr(saved.partitions[name], field),
            )
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="saved"):
        source.binding("train", input_id="events", encoder=encoder())
    binding = saved.binding(
        "train",
        input_id="events",
        target_id="label",
        encoder=encoder(),
        sample_ids=(42, 91),
    )
    result = sim.simulate(
        bundle=direct_train_bundle(),
        input_bindings=(binding,),
        batch_size=1,
        device="cpu",
        performance=sim.Performance(compile_mode=None),
    )
    assert result._batch_size == 2
    protocol = result.metrics["execution_protocol"]
    assert protocol["dataset_binding"]["selected_indices"] == [2, 0]
    assert protocol["dataset_binding"]["sample_ids"] == [42, 91]
    assert protocol["dataset"]["split"] == "train"
    assert [row["sample_ids"] for row in result.metrics["encoding_batches"]] == [
        [42],
        [91],
    ]
    with pytest.raises(ValueError, match="absent"):
        saved.binding("test", input_id="events", encoder=encoder(), sample_ids=(91,))


def test_named_bindings_match_legacy_keys_numerically_and_preserve_rng(tmp_path):
    _, saved = dataset(tmp_path)
    bundle = direct_train_bundle()
    binding = saved.binding(
        "train", input_id="events", target_id="label", encoder=encoder()
    )
    legacy = sim.DatasetSnapshotBinding(
        binding.path,
        binding.input_id,
        binding.dataset_id,
        binding.split,
        binding.encoder,
        target_id=binding.target_id,
        feature_key=binding.feature_key,
        label_key=binding.label_key,
    )
    options = dict(
        bundle=bundle,
        epochs=2,
        batch_size=2,
        seed=29,
        device="cpu",
        performance=sim.Performance(compile_mode=None),
    )
    result = sim.train(**options, input_bindings=(binding,))
    baseline = sim.train(**options, input_bindings=(legacy,))
    assert result.metrics["epochs"] == baseline.metrics["epochs"]
    for name, value in result.parameters.items():
        torch.testing.assert_close(value, baseline.parameters[name], rtol=0, atol=0)
    torch.testing.assert_close(
        result.training_checkpoint.rng_state, baseline.training_checkpoint.rng_state
    )
    validation = saved.binding(
        "validation", input_id="events", target_id="label", encoder=encoder()
    )
    test = saved.binding(
        "test", input_id="events", target_id="label", encoder=encoder()
    )
    measured = sim.train(
        **options,
        input_bindings=(binding,),
        validation=sim.ValidationSpec(
            input_bindings=(validation,), encoding_seeds=(11, 12)
        ),
        test=sim.TestSpec(input_bindings=(test,), encoding_seeds=(21, 22)),
    )
    sim.TrainingHistory.from_metrics(measured.metrics).validate(
        require_complete=True,
        expected_examples={"validation": 6, "test": 4},
        expected_encoding_draws={"validation": (11, 12), "test": (21, 22)},
    )


def test_duplicate_overlap_corruption_and_bound_digest_are_checked(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        sim.DatasetPartition(np.ones((2, 2)), np.zeros(2), np.asarray([1, 1]))
    partition = sim.DatasetPartition(np.ones((2, 2)), np.zeros(2), np.asarray([1, 2]))
    with pytest.raises(ValueError, match="overlap"):
        sim.DatasetSnapshot.from_partitions(
            dataset_id="bad",
            partitions={"a": partition, "b": partition},
            disjoint=("a", "b"),
        )
    _, saved = dataset(tmp_path)
    binding = saved.binding("train", input_id="events", encoder=encoder())
    with pytest.raises(FileExistsError):
        saved.write(saved.path)
    arrays = {}
    with np.load(saved.path) as archive:
        arrays.update({key: archive[key] for key in archive.files})
    arrays[binding.feature_key][0, 0] = 0.123
    np.savez(saved.path, **arrays)
    with pytest.raises(ValueError, match="integrity"):
        sim.DatasetSnapshot.read(saved.path)
    with pytest.raises(ValueError, match="digest"):
        sim.simulate(
            bundle=direct_train_bundle(),
            input_bindings=(binding,),
            device="cpu",
            performance=sim.Performance(compile_mode=None),
        )


def test_explicit_stratified_policy_does_not_touch_global_rng():
    x, y, ids = np.arange(60).reshape(20, 3), np.arange(20) % 3, np.arange(20) + 100
    state = np.random.get_state()
    options = dict(
        features=x,
        labels=y,
        sample_ids=ids,
        fractions={"train": 0.6, "validation": 0.2, "test": 0.2},
        seed=17,
        policy="stratified-v1",
    )
    first = sim.stratified_split(**options)
    second = sim.stratified_split(**options)
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(state, after))
    for name, partition in first.items():
        np.testing.assert_array_equal(partition.sample_ids, second[name].sample_ids)
        assert np.all(np.diff(partition.sample_ids) > 0)
        np.testing.assert_array_equal(partition.features, x[partition.sample_ids - 100])
    assert (
        sorted(np.concatenate([p.sample_ids for p in first.values()])) == ids.tolist()
    )
    assert first["train"].provenance["policy"] == "stratified-v1"
    with pytest.raises(ValueError, match="sum"):
        sim.stratified_split(**{**options, "fractions": {"train": 0.7, "test": 0.4}})


def test_batched_recording_exports_preserve_original_dataset_identities(tmp_path):
    _, saved = dataset(tmp_path)
    binding = saved.binding("train", input_id="events", encoder=encoder())
    recording = sim.RecordingSpec(
        (sim.SignalRecording("events.value", cells=(1, 0)),), block_steps=2
    )
    options = dict(
        bundle=direct_train_bundle(),
        input_bindings=(binding,),
        batch_size=2,
        device="cpu",
        performance=sim.Performance(compile_mode=None),
    )
    retained = sim.simulate(**options, recording=recording)
    assert retained.numpy(batch=4).recording.info["events.value"].sample_ids == (3,)
    sink = sim.NPZRecordingSink(tmp_path / "sink")
    sim.simulate(
        **options, recording=replace(recording, sink=sink, sink_id=sink.identity)
    )
    streamed = sim.RecordingData.read(tmp_path / "sink")
    assert streamed.info["events.value"].sample_ids == (91, 7, 42, 8, 3)
    np.testing.assert_array_equal(
        streamed.signals["events.value"],
        retained.numpy().recorded_signals["events.value"],
    )


def test_sparse_batch_tiles_preserve_original_identities(tmp_path):
    sink = sim.NPZRecordingSink(tmp_path / "sink")
    for offset, identities in ((0, (91, 7)), (2, (42, 8)), (4, (3,))):
        sink(
            sim.RecordingBlock(
                "E.spikes",
                "spike_events",
                torch.tensor([[1, offset, 1]]),
                0,
                3,
                0.1,
                batch_offset=offset,
                cells=(1, 0),
                unit="spike",
                axes=("event", "coordinate"),
                sample_ids=identities,
                batch_size=offset + len(identities),
            )
        )
    data = sim.RecordingData.read(tmp_path / "sink")
    assert data.info["E.spikes"].sample_ids == (91, 7, 42, 8, 3)
    assert data.select_batch(4).info["E.spikes"].sample_ids == (3,)


def test_prebinned_partitions_preserve_native_time_major_layout(tmp_path):
    labels = np.asarray([0, 1, 0])
    spikes = np.zeros((3, 3, 2), dtype=np.uint8)
    spikes[:, np.arange(3), labels] = 1
    snapshot = sim.DatasetSnapshot.from_partitions(
        dataset_id="prebinned",
        partitions={
            "train": sim.DatasetPartition(
                spikes, labels, np.asarray(["a", "b", "c"]), feature_sample_axis=1
            )
        },
    )
    saved = sim.DatasetSnapshot.read(snapshot.write(tmp_path / "spikes.npz"))
    assert saved.partitions["train"].feature_sample_axis == 1
    np.testing.assert_array_equal(saved.partitions["train"].features, spikes)
    binding = saved.binding(
        "train",
        input_id="events",
        target_id="label",
        sample_ids=("c", "a"),
        encoder=sim.DatasetEncoder("prebinned_spikes"),
    )
    result = sim.train(
        bundle=direct_train_bundle(),
        input_bindings=(binding,),
        epochs=1,
        batch_size=1,
        device="cpu",
        performance=sim.Performance(compile_mode=None),
    )
    assert result.metrics["execution_protocol"]["dataset_binding"]["sample_ids"] == [
        "c",
        "a",
    ]
    assert result.metrics["epochs"][-1]["train_examples"] == 2
    with pytest.raises(ValueError, match="sample-leading"):
        saved.binding("train", input_id="events", encoder=encoder())


def test_custom_encoder_receives_canonical_named_partition_arrays(
    tmp_path, monkeypatch
):
    from snnlab import extensions

    monkeypatch.setattr(extensions, "_REGISTRY", {})
    calls = []

    def encode(arrays, selected, *, dt_ms, channels, config, seed):
        calls.append(arrays["sample_ids"][selected].tolist())
        values = np.asarray(arrays["features"][selected])
        assert arrays["labels"][selected].shape == (len(selected),)
        return torch.from_numpy(values).unsqueeze(0).repeat(3, 1, 1)

    extensions.register_encoder("test.utility-partition/v1", encode)
    _, saved = dataset(tmp_path)
    binding = saved.binding(
        "train",
        input_id="events",
        sample_ids=(42, 91),
        encoder=sim.DatasetEncoder("custom", definition="test.utility-partition/v1"),
    )
    sim.simulate(
        bundle=direct_train_bundle(),
        input_bindings=(binding,),
        batch_size=2,
        device="cpu",
        performance=sim.Performance(compile_mode=None),
    )
    assert calls[-1] == [42, 91]
