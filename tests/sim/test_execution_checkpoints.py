"""Focused acceptance tests for the typed seam and graph executor."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from snnlab.sim.conformance import canonical_json_tensor, compare_conformance_layers
from snnlab.sim.execution import (
    DenseArrayBinding,
    ExecutionSpec,
    capture_training_rng_state,
    load_training_checkpoint,
    restore_training_rng_state,
    save_training_checkpoint,
    simulate,
    train,
)
from tests.sim._execution_builders import (
    direct_train_bundle as _direct_train_bundle,
)


def test_training_checkpoint_round_trip_and_resume_are_exact(tmp_path):
    bundle = _direct_train_bundle()
    inputs = torch.zeros(3, 2, 2)
    inputs[:, 0, 0] = 1
    inputs[:, 1, 1] = 1
    common = dict(
        kind="train",
        executor="graph",
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(DenseArrayBinding("events", inputs),),
        targets={"label": torch.tensor([0, 1])},
        seed=17,
    )
    uninterrupted = train(ExecutionSpec(**common, updates=4))
    checkpoint_dir = tmp_path / "checkpoint"
    first_half = train(
        ExecutionSpec(
            **common,
            updates=2,
            save_final_checkpoint=checkpoint_dir,
            save_selected_checkpoint=tmp_path / "selected",
        )
    )
    loaded = load_training_checkpoint(checkpoint_dir)
    assert loaded.completed_updates == 2
    assert loaded.graph_digest == bundle.training["graph_digest"]
    assert (tmp_path / "selected" / "manifest.json").is_file()
    resumed = train(
        ExecutionSpec(
            **common,
            checkpoint=checkpoint_dir,
            updates=2,
        )
    )
    assert resumed.metrics["resumed_from_update"] == 2
    assert [row["update"] for row in resumed.metrics["updates"]] == [3, 4]
    assert resumed.metrics["updates"] == uninterrupted.metrics["updates"][2:]
    assert uninterrupted.training_checkpoint is not None
    assert resumed.training_checkpoint is not None
    for name in uninterrupted.parameters:
        torch.testing.assert_close(
            resumed.parameters[name], uninterrupted.parameters[name], rtol=0, atol=0
        )
    conformance = compare_conformance_layers(
        "checkpoint-resume",
        {
            "topology": {"graph": canonical_json_tensor(bundle.graph)},
            "initialization": {
                "metadata": canonical_json_tensor(
                    uninterrupted.metrics["initialization"]
                )
            },
            "parameters": uninterrupted.parameters,
            "gradients": uninterrupted.gradients,
            "outputs": uninterrupted.outputs,
            "optimizer": {
                f"{name}.{state}": value
                for name, values in uninterrupted.optimizer_state.items()
                for state, value in values.items()
                if isinstance(value, torch.Tensor)
            },
            "checkpoint": {
                "data_state": canonical_json_tensor(
                    uninterrupted.training_checkpoint.data_state
                )
            },
        },
        {
            "topology": {"graph": canonical_json_tensor(bundle.graph)},
            "initialization": {
                "metadata": canonical_json_tensor(resumed.metrics["initialization"])
            },
            "parameters": resumed.parameters,
            "gradients": resumed.gradients,
            "outputs": resumed.outputs,
            "optimizer": {
                f"{name}.{state}": value
                for name, values in resumed.optimizer_state.items()
                for state, value in values.items()
                if isinstance(value, torch.Tensor)
            },
            "checkpoint": {
                "data_state": canonical_json_tensor(
                    resumed.training_checkpoint.data_state
                )
            },
        },
    )
    conformance.require_passed()
    for name in uninterrupted.optimizer_state:
        for state in uninterrupted.optimizer_state[name]:
            torch.testing.assert_close(
                resumed.optimizer_state[name][state],
                uninterrupted.optimizer_state[name][state],
                rtol=0,
                atol=0,
            )
    assert first_half.training_checkpoint is not None
    assert resumed.training_checkpoint.completed_updates == 4


def test_accelerator_rng_checkpoint_round_trip_and_topology_restore(
    tmp_path, monkeypatch
):
    bundle = _direct_train_bundle()
    trained = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(DenseArrayBinding("events", torch.zeros(3, 1, 2)),),
            targets={"label": torch.tensor([0])},
        )
    )
    checkpoint = trained.training_checkpoint
    assert checkpoint is not None
    checkpoint.rng_backend = "cuda"
    checkpoint.accelerator_rng_states = {
        "cuda:0": torch.tensor([1, 2, 3], dtype=torch.uint8),
        "cuda:1": torch.tensor([4, 5, 6], dtype=torch.uint8),
    }
    root = save_training_checkpoint(tmp_path / "cuda-checkpoint", checkpoint)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["rng_backend"] == "cuda"
    assert manifest["accelerator_rng_devices"] == ["cuda:0", "cuda:1"]
    loaded = load_training_checkpoint(root)
    assert loaded.rng_backend == "cuda"
    assert set(loaded.accelerator_rng_states) == {"cuda:0", "cuda:1"}

    restored = []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all", lambda states: restored.extend(states)
    )
    restore_training_rng_state(loaded, "cuda")
    assert [state.tolist() for state in restored] == [[1, 2, 3], [4, 5, 6]]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="topology"):
        restore_training_rng_state(loaded, "cuda")


def test_accelerator_rng_capture_names_every_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: [
            torch.tensor([7], dtype=torch.uint8),
            torch.tensor([8], dtype=torch.uint8),
        ],
    )
    backend, states = capture_training_rng_state("cuda:1")
    assert backend == "cuda"
    assert {name: value.tolist() for name, value in states.items()} == {
        "cuda:0": [7],
        "cuda:1": [8],
    }


def test_v1_cpu_checkpoint_remains_loadable(tmp_path):
    bundle = _direct_train_bundle()
    trained = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(DenseArrayBinding("events", torch.zeros(3, 1, 2)),),
            targets={"label": torch.tensor([0])},
        )
    )
    assert trained.training_checkpoint is not None
    root = save_training_checkpoint(tmp_path / "v1", trained.training_checkpoint)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["schema_version"] = 1
    manifest.pop("rng_backend")
    manifest.pop("accelerator_rng_devices")
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")
    loaded = load_training_checkpoint(root)
    assert loaded.rng_backend == "cpu"
    assert loaded.accelerator_rng_states == {}


def test_training_checkpoint_rejects_partial_parameter_mapping(tmp_path):
    bundle = _direct_train_bundle()
    inputs = torch.zeros(3, 2, 2)
    result = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(DenseArrayBinding("events", inputs),),
            targets={"label": torch.tensor([0, 1])},
        )
    )
    checkpoint = result.training_checkpoint
    assert checkpoint is not None
    checkpoint.parameters.pop("shadow.weight")
    root = save_training_checkpoint(tmp_path / "partial", checkpoint)
    with pytest.raises(ValueError, match="parameter names mismatch"):
        train(
            ExecutionSpec(
                kind="train",
                executor="graph",
                graph=bundle.graph,
                training=bundle.training,
                input_bindings=(DenseArrayBinding("events", inputs),),
                targets={"label": torch.tensor([0, 1])},
                checkpoint=root,
            )
        )


def test_graph_inference_loads_portable_selected_checkpoint_with_provenance(tmp_path):
    bundle = _direct_train_bundle()
    inputs = torch.zeros(3, 2, 2)
    inputs[:, 0, 0] = 1
    inputs[:, 1, 1] = 1
    selected_path = tmp_path / "selected"
    trained = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(DenseArrayBinding("events", inputs),),
            targets={"label": torch.tensor([0, 1])},
            seed=17,
            updates=3,
            save_selected_checkpoint=selected_path,
        )
    )
    selected = load_training_checkpoint(selected_path)
    inferred = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            input_bindings=(DenseArrayBinding("events", inputs),),
            seed=17,
            checkpoint=selected_path,
        )
    )
    for name in selected.parameters:
        torch.testing.assert_close(
            inferred.parameters[name], selected.parameters[name], rtol=0, atol=0
        )
    assert inferred.metrics["checkpoint"] == {
        "format": "tools/snnsim.training-checkpoint/v1",
        "path": str(selected_path),
        "graph_digest": selected.graph_digest,
        "training_digest": selected.training_digest,
        "completed_updates": selected.completed_updates,
        "selected_loss": selected.selected_loss,
    }
    assert trained.selected_checkpoint is not None

    incompatible = copy.deepcopy(bundle.graph)
    incompatible["name"] = "different"
    with pytest.raises(ValueError, match="inference checkpoint graph digest"):
        simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=incompatible,
                input_bindings=(DenseArrayBinding("events", inputs),),
                checkpoint=selected_path,
            )
        )


def test_dataset_training_resume_preserves_shuffle_and_batch_position(tmp_path):
    bundle = _direct_train_bundle()
    inputs = torch.zeros(3, 5, 2)
    for sample in range(5):
        inputs[:, sample, sample % 2] = 1
    common = dict(
        kind="train",
        executor="graph",
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(DenseArrayBinding("events", inputs),),
        targets={"label": torch.tensor([0, 1, 0, 1, 0])},
        seed=23,
    )
    trajectory = {"epochs": 2, "batch_size": 2, "shuffle": True}
    uninterrupted = train(ExecutionSpec(**common, **trajectory))
    checkpoint_dir = tmp_path / "dataset-checkpoint"
    first = train(
        ExecutionSpec(
            **common,
            **trajectory,
            updates=1,
            save_final_checkpoint=checkpoint_dir,
        )
    )
    assert first.training_checkpoint is not None
    assert first.training_checkpoint.data_state == {"epoch": 0, "batch": 1}
    resumed = train(
        ExecutionSpec(
            **common,
            checkpoint=checkpoint_dir,
            **trajectory,
        )
    )
    assert resumed.metrics["updates"] == uninterrupted.metrics["updates"][1:]
    assert resumed.training_checkpoint is not None
    assert resumed.training_checkpoint.data_state == {"epoch": 2, "batch": 0}
    for name in uninterrupted.parameters:
        torch.testing.assert_close(
            resumed.parameters[name], uninterrupted.parameters[name], rtol=0, atol=0
        )
