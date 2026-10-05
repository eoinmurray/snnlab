"""Checkpoint selection follows evaluated weights and survives interrupted training."""

import copy
import json
from dataclasses import replace

import pytest
import torch

from snnlab.sim.execution import (
    CheckpointSelection,
    SelectionMetric,
    build,
    execution_spec_from_args,
    load_training_checkpoint,
    train,
)
from snnlab.sim.tool import parse_args
from tests.sim.test_epoch_observations import observations, request


def policy(**kwargs):
    return CheckpointSelection(
        metric=SelectionMetric("cross_entropy"),
        tie_break=(SelectionMetric("accuracy", direction="max"),),
        **kwargs,
    )


def assert_same_candidate(left, right):
    assert left.selection_record == right.selection_record
    assert left.completed_updates == right.completed_updates
    for name in left.parameters:
        torch.testing.assert_close(
            left.parameters[name], right.parameters[name], rtol=0, atol=0
        )
    assert torch.equal(left.rng_state, right.rng_state)
    for name, states in left.optimizer_state.items():
        for key, value in states.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(
                    value, right.optimizer_state[name][key], rtol=0, atol=0
                )
            else:
                assert value == right.optimizer_state[name][key]


def test_validation_ranks_disagree_and_saved_weights_match_score(tmp_path):
    spec = request()
    validation = replace(spec.validation, targets={"label": 1 - spec.targets["label"]})
    recipe = copy.deepcopy(spec.training)
    recipe["objectives"][0]["weight"] = 0.25
    spec = replace(
        spec,
        training=recipe,
        validation=validation,
        checkpoint_selection=policy(),
        save_selected_checkpoint=tmp_path / "winner",
    )
    result = train(spec)
    rows = result.metrics["epochs"][1:]
    assert rows[-1]["train_loss"] < rows[0]["train_loss"]
    assert rows[-1]["validation_loss"] > rows[0]["validation_loss"]
    selected = result.selected_checkpoint
    assert selected.selection_record["epoch"] == 1
    assert selected.completed_updates == 3
    assert result.training_checkpoint.completed_updates == 6
    loaded = load_training_checkpoint(tmp_path / "winner")
    assert_same_candidate(selected, loaded)
    model = build(spec).model
    with torch.no_grad():
        for name, parameter in model.parameter_map().items():
            parameter.copy_(loaded.parameters[name])
        outputs = model.forward(
            {"events": spec.input_bindings[0].value}, diagnostics=False
        ).outputs
        ce = torch.nn.functional.cross_entropy(
            outputs["scores"], validation.targets["label"]
        )
    assert selected.selection_record["values"][0] == pytest.approx(float(ce))
    assert selected.selection_record["values"][0] == pytest.approx(
        rows[0]["validation_loss"] / 0.25
    )
    assert (
        selected.selection_record["evaluation"]["aggregation"] == "sample_weighted_mean"
    )


@pytest.mark.parametrize("updates", [1, 3, 4])
@pytest.mark.parametrize("audited", [False, True])
def test_resume_keeps_best_so_far_and_matches_uninterrupted(tmp_path, updates, audited):
    spec = request()
    spec = replace(
        spec,
        validation=replace(
            spec.validation, targets={"label": 1 - spec.targets["label"]}
        ),
        checkpoint_selection=policy(),
        observations=observations(spec) if audited else None,
    )
    full = train(spec)
    first = train(
        replace(spec, updates=updates, save_final_checkpoint=tmp_path / "final")
    )
    loaded = load_training_checkpoint(tmp_path / "final")
    assert (loaded.best_checkpoint is None) == (first.selected_checkpoint is None)
    resumed = train(replace(spec, checkpoint=tmp_path / "final"))
    assert_same_candidate(full.selected_checkpoint, resumed.selected_checkpoint)
    for name in full.parameters:
        torch.testing.assert_close(
            full.parameters[name], resumed.parameters[name], rtol=0, atol=0
        )
    if updates >= 3:
        assert resumed.selected_checkpoint.completed_updates == 3
    assert (
        json.loads((tmp_path / "final" / "manifest.json").read_text())["schema_version"]
        == 4
    )


def test_accuracy_tie_break_and_exact_tie_keeps_earliest():
    selection = policy()
    assert selection.better([0.5, 0.9], [0.5, 0.8])
    assert not selection.better([0.5, 0.8], [0.5, 0.9])
    assert not selection.better([0.5, 0.9], [0.5, 0.9])
    assert selection.better([0.4, 0.1], [0.5, 0.9])
    spec = request()
    recipe = copy.deepcopy(spec.training)
    for group in recipe["parameter_groups"]:
        group["lr"] = 0.0
    result = train(replace(spec, training=recipe, checkpoint_selection=policy()))
    assert result.selected_checkpoint.selection_record["epoch"] == 1
    initial = train(
        replace(
            spec,
            training=recipe,
            checkpoint_selection=policy(include_initial=True),
            updates=1,
        )
    )
    assert initial.selected_checkpoint.completed_updates == 0
    assert initial.selected_checkpoint.selection_record["phase"] == "initial"


def test_no_completed_candidate_is_explicit(tmp_path):
    spec = replace(request(), checkpoint_selection=policy(), updates=1)
    assert train(spec).selected_checkpoint is None
    with pytest.raises(ValueError, match="no eligible"):
        train(replace(spec, save_selected_checkpoint=tmp_path / "winner"))


@pytest.mark.parametrize(
    "change",
    [
        {"split": "unknown"},
        {"metric": SelectionMetric("cross_entropy", objective=10)},
        {"metric": SelectionMetric("cross_entropy", direction="sideways")},
        {"include_initial": 1},
        {"tie_break": (SelectionMetric("cross_entropy"),)},
    ],
)
def test_invalid_policies_fail(change):
    with pytest.raises((ValueError, TypeError)):
        train(replace(request(), checkpoint_selection=replace(policy(), **change)))


def test_missing_validation_and_nonfinite_metric_fail():
    with pytest.raises(ValueError, match="requires ValidationSpec"):
        train(replace(request(), validation=None, checkpoint_selection=policy()))
    with pytest.raises(ValueError, match="positive epochs"):
        train(
            replace(
                request(),
                epochs=0,
                validation=None,
                checkpoint_selection=policy(split="train"),
            )
        )
    for value in [float("nan"), float("inf"), None]:
        with pytest.raises(ValueError, match="missing or non-finite"):
            policy().scores({"cross_entropies": {"objective[0]": value}})


def test_resume_rejects_changed_policy_or_validation_and_authenticates_winner(tmp_path):
    spec = replace(request(), checkpoint_selection=policy())
    train(replace(spec, updates=3, save_final_checkpoint=tmp_path / "final"))
    for changed in [
        replace(spec, checkpoint_selection=policy(include_initial=True)),
        replace(spec, checkpoint_selection=None),
        replace(
            spec,
            validation=replace(
                spec.validation, targets={"label": 1 - spec.targets["label"]}
            ),
        ),
    ]:
        with pytest.raises(ValueError, match="selection policy or evaluation identity"):
            train(replace(changed, checkpoint=tmp_path / "final"))
    child = tmp_path / "final" / "selected" / "manifest.json"
    manifest = json.loads(child.read_text())
    manifest["completed_updates"] = 99
    child.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="candidate manifest digest"):
        load_training_checkpoint(tmp_path / "final")


@pytest.mark.parametrize("audited", [False, True])
def test_selected_checkpoint_itself_can_resume(tmp_path, audited):
    spec = request()
    spec = replace(
        spec,
        checkpoint_selection=policy(),
        observations=observations(spec) if audited else None,
    )
    first = train(
        replace(spec, updates=3, save_selected_checkpoint=tmp_path / "selected")
    )
    resumed = train(replace(spec, checkpoint=tmp_path / "selected"))
    full = train(spec)
    assert_same_candidate(resumed.selected_checkpoint, full.selected_checkpoint)
    assert first.selected_checkpoint.best_checkpoint is None


def test_serialized_cli_policy_and_named_legacy_default():
    selection = policy()
    value = json.dumps(selection.to_dict())
    spec = execution_spec_from_args(
        parse_args(["train", "--executor", "graph", "--checkpoint-selection", value])
    )
    assert spec.checkpoint_selection.to_dict() == selection.to_dict()
    result = train(request())
    assert (
        result.selected_checkpoint.selection_record["policy"]["mode"]
        == "legacy_training_batch_loss"
    )


def test_selection_preserves_training_trajectory_and_rng():
    spec = request()
    legacy = train(spec)
    epoch = train(replace(spec, checkpoint_selection=policy()))
    assert legacy.metrics["updates"] == epoch.metrics["updates"]
    for name in legacy.parameters:
        torch.testing.assert_close(
            legacy.parameters[name], epoch.parameters[name], rtol=0, atol=0
        )
    assert torch.equal(
        legacy.training_checkpoint.rng_state, epoch.training_checkpoint.rng_state
    )


def test_old_checkpoint_cannot_reconstruct_selection_history(tmp_path):
    from snnlab.sim.execution import save_training_checkpoint

    spec = request()
    first = train(replace(spec, updates=1))
    old = replace(
        first.training_checkpoint,
        selection_contract=None,
        selection_record=None,
        best_checkpoint=None,
    )
    save_training_checkpoint(tmp_path / "old", old)
    with pytest.raises(ValueError, match="cannot reconstruct missing candidates"):
        train(replace(spec, checkpoint_selection=policy(), checkpoint=tmp_path / "old"))


def test_selection_metadata_and_candidate_tensor_corruption_fail(tmp_path):
    spec = replace(request(), checkpoint_selection=policy())
    train(replace(spec, updates=3, save_final_checkpoint=tmp_path / "final"))
    manifest_path = tmp_path / "final" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    original = manifest_path.read_text()
    manifest["selection_metadata"]["record"]["values"][0] = 123
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="selection metadata digest"):
        load_training_checkpoint(tmp_path / "final")
    manifest_path.write_text(original)
    archive = tmp_path / "final" / "selected" / "tensors.npz"
    archive.write_bytes(archive.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="tensors digest"):
        load_training_checkpoint(tmp_path / "final")
