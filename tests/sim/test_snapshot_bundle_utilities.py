"""Structured snapshots and authored bundles preserve native execution contracts."""

import copy
import json
from dataclasses import replace

import pytest
import torch

from snnlab import lang, sim
from snnlab.sim.training.trainer import train
from tests.sim._execution_builders import coupled_graph, direct_train_bundle
from tests.sim.test_final_test_evaluation import test_request


def test_bundle_training_matches_graph_recipe_outputs_checkpoints_and_rng(tmp_path):
    bundle = direct_train_bundle()
    spec = test_request()
    baseline = train(spec)
    before = copy.deepcopy((bundle.graph, bundle.training, bundle.manifest))
    result = train(
        replace(
            spec,
            bundle=bundle,
            graph=None,
            training=None,
            snapshot=sim.SnapshotSpec(tmp_path / "snapshots", stage="both"),
            on_epoch_end=sim.EpochLogger(stream=(tmp_path / "progress.log").open("w")),
        )
    )
    assert (bundle.graph, bundle.training, bundle.manifest) == before
    for field in ("outputs", "parameters", "gradients"):
        for name, value in getattr(result, field).items():
            torch.testing.assert_close(
                value, getattr(baseline, field)[name], rtol=0, atol=0
            )
    for key in (
        "epochs",
        "test",
        "selection_record",
        "checkpoint_selection",
        "execution_protocol",
    ):
        assert result.metrics[key] == baseline.metrics[key]
    for name, state in result.optimizer_state.items():
        for key, value in state.items():
            torch.testing.assert_close(
                value, baseline.optimizer_state[name][key], rtol=0, atol=0
            )
    for field in ("graph_digest", "training_digest", "data_state", "completed_updates"):
        assert getattr(result.training_checkpoint, field) == getattr(
            baseline.training_checkpoint, field
        )
    torch.testing.assert_close(
        result.training_checkpoint.rng_state,
        baseline.training_checkpoint.rng_state,
        rtol=0,
        atol=0,
    )
    path = result.write_metrics(tmp_path / "metrics.json")
    sim.TrainingHistory.read(path).validate(require_complete=True)
    assert torch.equal(torch.get_rng_state(), baseline.training_checkpoint.rng_state)


def test_metadata_only_snapshot_and_explicit_parameter_integrity(tmp_path, monkeypatch):
    options = dict(
        graph=coupled_graph(),
        input_bindings=tuple(
            sim.PoissonInputBinding(n, 2, (0.0,), 1) for n in ("drive_a", "drive_b")
        ),
        device="cpu",
        performance=sim.Performance(compile_mode=None),
    )
    result = sim.simulate(
        **options, snapshot=sim.SnapshotSpec(tmp_path / "snapshot", stage="end")
    )
    snapshot = sim.NetworkSnapshot.read(tmp_path / "snapshot")
    assert snapshot.schema == "snnlab.run-snapshot/v1"
    assert snapshot.stage == "end"
    assert snapshot.execution.timestep_ms == 0.1
    assert snapshot.populations["a_E"].parameters["threshold_mv"].unit == "mV"
    projection = snapshot.projections["a_E_to_I"]
    assert projection.source_id == "a_E" and projection.target_id == "a_I"
    assert projection.scaling == "fan_in_normalized"
    assert projection.matrix_shape == (1, 4)
    assert projection.runtime_matrix_shape == (4, 1)
    assert (
        snapshot.parameter_info["a_E_to_I.weight"].initialization.statistics.count == 4
    )
    assert snapshot.parameter_info["a_E_to_I.weight"].frozen is True
    parameters = snapshot.load_parameters(names=("a_E_to_I.weight",))
    torch.testing.assert_close(
        parameters["a_E_to_I.weight"],
        result.parameters["a_E_to_I.weight"],
        rtol=0,
        atol=0,
    )
    captured = sim.NetworkSnapshot.from_snapshot(result.snapshots["end"])
    torch.testing.assert_close(
        captured.load_parameters()["a_E_to_I.weight"], parameters["a_E_to_I.weight"]
    )
    # Metadata reading must still work after an array is lost/corrupt.
    (tmp_path / "snapshot" / "parameters.npz").write_bytes(b"broken")
    sim.NetworkSnapshot.read(tmp_path / "snapshot")
    with pytest.raises(ValueError, match="digest"):
        snapshot.validate(verify_parameters=True)
    with pytest.raises(ValueError, match="digest"):
        sim.NetworkSnapshot.read(tmp_path / "snapshot", parameters=True)
    metadata_path = tmp_path / "snapshot" / "snapshot.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["network"]["name"] = "tampered"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="graph digest"):
        sim.NetworkSnapshot.read(metadata_path)


def test_bundle_build_path_override_validation_and_assets(tmp_path):
    bundle = direct_train_bundle()
    path = bundle.write(tmp_path / "network.bundle")
    options = dict(device="cpu", seed=9, performance=sim.Performance(compile_mode=None))
    for source in (bundle, path):
        model = sim.build(bundle=source, **options)
        baseline = sim.build(graph=bundle.graph, **options)
        for name, value in model.parameters.items():
            torch.testing.assert_close(value, baseline.parameters[name], rtol=0, atol=0)
        # Explicit graph ignores the other graph source, as before.
        sim.build(bundle=source, graph=bundle.graph, **options)
        sim.build(bundle=source, training=bundle.training, **options)
    mutated = copy.deepcopy(bundle)
    mutated.graph["parameters"][0]["shape"] = [-1, 2]
    with pytest.raises(ValueError):
        sim.build(bundle=mutated, **options)
    net = lang.Network("asset")
    net.asset("weights", media_type="application/octet-stream")
    asset = tmp_path / "source.bin"
    asset.write_bytes(b"weights")
    with_asset = lang.compile(net, assets={"weights": asset})
    sim.build(bundle=with_asset, **options)
    asset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset digest"):
        sim.build(bundle=with_asset, **options)


def test_snapshot_exact_bfloat16_and_no_parameter_capture(tmp_path):
    bundle = direct_train_bundle()
    result = sim.simulate(
        bundle=bundle,
        input_bindings=(sim.DenseArrayBinding("events", torch.zeros(3, 1, 2)),),
        snapshot=sim.SnapshotSpec(tmp_path / "metadata", parameters=False),
        device="cpu",
        performance=sim.Performance(compile_mode=None),
    )
    snapshot = sim.NetworkSnapshot.read(tmp_path / "metadata")
    assert snapshot.load_parameters() == {}
    with pytest.raises(ValueError, match="unavailable"):
        snapshot.load_parameters(names=("scores_projection.weight",))
    captured = result.snapshots["start"]
    tensors = {
        "scores_projection.weight": torch.tensor(
            [[1.0, -1.0], [2.0, -2.0]], dtype=torch.bfloat16
        )
    }
    sim.write_run_snapshot(
        tmp_path / "bf16", sim.RunSnapshot(captured.metadata, tensors)
    )
    actual = sim.NetworkSnapshot.read(tmp_path / "bf16").load_parameters()
    assert actual["scores_projection.weight"].dtype == torch.bfloat16
    torch.testing.assert_close(
        actual["scores_projection.weight"], tensors["scores_projection.weight"]
    )


def test_in_memory_snapshot_metadata_does_not_copy_parameters(tmp_path, monkeypatch):
    bundle = direct_train_bundle()
    result = sim.simulate(
        bundle=bundle,
        input_bindings=(sim.DenseArrayBinding("events", torch.zeros(3, 1, 2)),),
        device="cpu",
        performance=sim.Performance(compile_mode=None),
        snapshot=tmp_path / "snapshot",
    )

    def no_clone(*args, **kwargs):
        raise AssertionError("metadata reader copied a parameter tensor")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "clone", no_clone)
        snapshot = sim.NetworkSnapshot.from_snapshot(result.snapshots["start"])
        assert (
            snapshot.parameter_info["scores_projection.weight"].stored.dtype
            == "float32"
        )
    parameters = snapshot.load_parameters()
    parameters["scores_projection.weight"].fill_(123)
    assert (
        torch.count_nonzero(
            result.snapshots["start"].parameters["scores_projection.weight"]
        )
        == 0
    )
