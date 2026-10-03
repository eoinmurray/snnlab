"""Focused acceptance tests for the typed seam and graph executor."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from snnlab.lang.examples.build_examples import deep_network, ping_classifier
from snnlab.sim.execution import (
    ExecutionSpec,
    GraphRuntimeState,
    PoissonInputBinding,
    build,
    execute_request,
    execution_spec_from_args,
    graph_capability_issues,
    resolve_device,
    simulate,
    train,
)
from snnlab.sim.tool import parse_args
from tests.sim._execution_builders import coupled_graph as _coupled_graph


def test_legacy_and_bundle_cli_arguments_both_lower_to_typed_specs(tmp_path):
    legacy = execution_spec_from_args(parse_args(["sim"]))
    assert legacy.executor == "legacy" and legacy.bundle is None
    root = ping_classifier().write(tmp_path / "ping.bundle")
    graph = execution_spec_from_args(
        parse_args(["sim", "--bundle", str(root), "--executor", "graph"])
    )
    assert graph.executor == "graph" and graph.bundle == root
    called = []
    result = execute_request(
        legacy, legacy=lambda: called.append(True) or build(legacy)
    )
    assert called and result.executor == "legacy"


def test_graph_cli_accepts_ordered_intervention_syntax():
    args = parse_args(
        [
            "sim",
            "--intervention",
            "drop:cell_E=0.25",
            "--intervention",
            "add:cell_E=5",
            "--inference-timestep-ms",
            "0.05",
        ]
    )
    assert args.intervention == ["drop:cell_E=0.25", "add:cell_E=5"]
    assert args.inference_timestep_ms == 0.05


def test_graph_cli_resolves_explicit_device_and_recording_profile(
    tmp_path, monkeypatch
):
    root = ping_classifier().write(tmp_path / "ping.bundle")
    graph = execution_spec_from_args(
        parse_args(
            [
                "sim",
                "--bundle",
                str(root),
                "--executor",
                "graph",
                "--device",
                "cpu",
                "--recording",
                "observables",
            ]
        )
    )
    assert graph.device == "cpu"
    assert graph.recording == "observables"
    assert resolve_device("cpu") == "cpu"
    monkeypatch.setenv("PINGLAB_DEVICE", "cpu")
    assert resolve_device("auto") == "cpu"


def test_recording_profiles_select_full_observable_or_no_traces():
    graph = _coupled_graph()
    inputs = {"drive_a": torch.zeros(4, 1, 3), "drive_b": torch.zeros(4, 1, 2)}
    full = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs=inputs,
            recording="full",
        )
    )
    observables = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs=inputs,
            recording="observables",
        )
    )
    none = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs=inputs,
            recording="none",
        )
    )
    observable_names = {row["id"] for row in graph["observables"]}
    assert observable_names < set(full.recordings)
    assert set(observables.recordings) == observable_names
    assert not none.recordings
    assert full.metrics["recording"] == "full"
    assert observables.metrics["recording"] == "observables"


def _state_tensors(state: GraphRuntimeState):
    for group in (
        state.voltages,
        state.refractory,
        state.conductances,
        state.population_histories,
        state.input_histories,
    ):
        yield from group.values()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_graph_cpu_mps_parity_and_all_result_state_follows_device():
    graph = _coupled_graph()
    inputs = {"drive_a": torch.zeros(12, 1, 3), "drive_b": torch.zeros(12, 1, 2)}
    inputs["drive_a"][0, 0, 0] = 1.0
    cpu = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs=inputs,
            seed=23,
            device="cpu",
            recording="observables",
        )
    )
    mps = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs=inputs,
            seed=23,
            device="mps",
            recording="observables",
        )
    )
    assert mps.runtime_state is not None
    assert all(value.device.type == "mps" for value in mps.parameters.values())
    assert all(value.device.type == "mps" for value in mps.recordings.values())
    assert all(value.device.type == "mps" for value in mps.outputs.values())
    assert all(
        value.device.type == "mps" for value in _state_tensors(mps.runtime_state)
    )
    for name in cpu.recordings:
        torch.testing.assert_close(
            mps.recordings[name].cpu(),
            cpu.recordings[name],
            rtol=1e-5,
            atol=1e-6,
        )
    assert mps.metrics["device"] == "mps"


def test_representative_shd_checkpoint_and_recording_requests_remain_legacy():
    shd = execution_spec_from_args(
        parse_args(["train", "--dataset", "shd", "--max-samples", "8", "--epochs", "1"])
    )
    assert shd.executor == "legacy"
    assert shd.options["dataset"] == "shd"
    assert shd.options["max_samples"] == 8


def test_production_shaped_mnist_and_shd_graphs_execute_named_outputs():
    mnist = ping_classifier()
    shd = deep_network()
    assert mnist.graph["inputs"][0]["shape"] == ["time", "batch", 784]
    assert shd.graph["inputs"][0]["shape"] == ["time", "batch", 700]
    assert len(shd.graph["populations"]) == 6
    assert [row["target"] for row in shd.training["objectives"]] == ["gesture"]
    mnist_result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=mnist.graph,
            inputs={"image": torch.zeros(2, 1, 784)},
            recording="observables",
            seed=5,
        )
    )
    shd_result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=shd.graph,
            inputs={"events": torch.zeros(2, 1, 700)},
            recording="observables",
            seed=5,
        )
    )
    assert mnist_result.outputs["class_logits"].shape == (1, 10)
    assert shd_result.outputs["gesture_logits"].shape == (1, 20)
    assert set(shd_result.recordings) == {
        "association_E_spikes",
        "decision_E_spikes",
        "encoder_E_spikes",
    }


def test_production_shaped_deep_shd_recipe_trains_all_recurrent_layers():
    bundle = deep_network()
    result = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            inputs={"events": torch.zeros(2, 1, 700)},
            targets={"gesture": torch.tensor([0])},
            seed=9,
        )
    )
    assert set(result.gradients) == set(
        bundle.training["resolved_parameters"]["trainable"]
    )
    assert {
        "encoder_E_to_I.weight",
        "encoder_I_to_E.weight",
        "association_E_to_I.weight",
        "association_I_to_E.weight",
        "decision_E_to_I.weight",
        "decision_I_to_E.weight",
    } <= set(result.gradients)
    assert result.metrics["updates"][0]["components"]["regularizer[0]"] == 0.0


def test_production_ping_fine_timestep_and_variable_rate_protocol():
    bundle = ping_classifier()
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            poisson_bindings=(
                PoissonInputBinding(
                    "image", 2, 3, (0.0, 5.0, 25.0), 41, categorical=True
                ),
            ),
            recording="observables",
            seed=41,
            options={"inference_overrides": {"timestep_ms": 0.05}},
        )
    )
    protocol = result.metrics["execution_protocol"]
    assert protocol["timing"] == {
        "dt_ms": 0.05,
        "steps": 4,
        "duration_ms": pytest.approx(0.2),
    }
    assert set(protocol["inputs"][0]["realized_rates_hz"]) <= {0.0, 5.0, 25.0}
    assert result.outputs["class_logits"].shape == (3, 10)
    checkpoint = execution_spec_from_args(
        parse_args(["sim", "--load-weights", "checkpoint.pth", "--outputs", "rasters"])
    )
    assert checkpoint.executor == "legacy"
    assert checkpoint.checkpoint == Path("checkpoint.pth")
    assert checkpoint.options["outputs"] == ["rasters"]


def test_arbitrary_sizes_independent_inputs_and_all_population_recordings():
    graph = _coupled_graph()
    assert not graph_capability_issues(graph)
    inputs = {"drive_a": torch.zeros(8, 1, 3), "drive_b": torch.zeros(8, 1, 2)}
    result = simulate(
        ExecutionSpec(
            kind="simulate", executor="graph", graph=graph, inputs=inputs, seed=3
        )
    )
    assert result.executor == "graph"
    assert {
        "coupled_0",
        "coupled_1",
        "coupled_2",
        "coupled_3",
    } <= result.recordings.keys()
    assert result.recordings["coupled_0"].shape == (8, 1, 4)
    assert result.recordings["coupled_3"].shape == (8, 1, 2)
