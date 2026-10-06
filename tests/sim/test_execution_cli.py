"""Focused acceptance tests for the typed seam and graph executor."""

from __future__ import annotations

import pytest
import torch

from snnlab.sim.execution import (
    DenseArrayBinding,
    ExecutionSpec,
    GraphRuntimeState,
    PoissonInputBinding,
    execution_spec_from_args,
    graph_capability_issues,
    resolve_device,
    simulate,
    train,
)
from snnlab.sim.tool import parse_args
from tests.sim._bundle_builders import deep_network, ping_classifier
from tests.sim._execution_builders import coupled_graph as _coupled_graph


def test_training_cli_maps_controls_to_direct_fields(tmp_path):
    spec = execution_spec_from_args(
        parse_args(
            [
                "train",
                "--epochs",
                "3",
                "--batch-size",
                "7",
                "--input-shuffle",
                "--save-final-checkpoint",
                str(tmp_path / "final"),
                "--save-selected-checkpoint",
                str(tmp_path / "selected"),
            ]
        )
    )
    assert (spec.epochs, spec.batch_size, spec.shuffle) == (3, 7, True)
    assert spec.save_final_checkpoint == str(tmp_path / "final")
    assert spec.save_selected_checkpoint == str(tmp_path / "selected")
    assert not set(spec.options) & {
        "epochs",
        "batch_size",
        "shuffle",
        "updates",
        "save_final_checkpoint",
        "save_selected_checkpoint",
    }


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


def test_graph_cli_resolves_explicit_device_and_diagnostics(tmp_path, monkeypatch):
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
                "--no-diagnostics",
            ]
        )
    )
    assert graph.device == "cpu"
    assert graph.diagnostics is False
    assert resolve_device("cpu") == "cpu"
    monkeypatch.setenv("PINGLAB_DEVICE", "cpu")
    assert resolve_device("auto") == "cpu"


def test_diagnostics_default_to_exposed_signals_and_can_be_disabled():
    graph = _coupled_graph()
    inputs = {"drive_a": torch.zeros(4, 1, 3), "drive_b": torch.zeros(4, 1, 2)}
    spec = dict(
        kind="simulate",
        graph=graph,
        input_bindings=tuple(
            DenseArrayBinding(name, value) for name, value in inputs.items()
        ),
    )
    default = simulate(ExecutionSpec(**spec))
    disabled = simulate(ExecutionSpec(**spec, diagnostics=False))
    assert set(default.diagnostics) == {row["id"] for row in graph["observables"]}
    assert not disabled.diagnostics
    assert default.outputs
    for name, value in default.outputs.items():
        torch.testing.assert_close(disabled.outputs[name], value, rtol=0, atol=0)
    assert default.metrics["diagnostics"] is True
    assert disabled.metrics["diagnostics"] is False


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
            input_bindings=tuple(
                DenseArrayBinding(name, value) for name, value in (inputs).items()
            ),
            seed=23,
            device="cpu",
        )
    )
    mps = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            input_bindings=tuple(
                DenseArrayBinding(name, value) for name, value in (inputs).items()
            ),
            seed=23,
            device="mps",
        )
    )
    assert mps.runtime_state is not None
    assert all(value.device.type == "mps" for value in mps.parameters.values())
    assert all(value.device.type == "mps" for value in mps.diagnostics.values())
    assert all(value.device.type == "mps" for value in mps.outputs.values())
    assert all(
        value.device.type == "mps" for value in _state_tensors(mps.runtime_state)
    )
    for name in cpu.diagnostics:
        torch.testing.assert_close(
            mps.diagnostics[name].cpu(),
            cpu.diagnostics[name],
            rtol=1e-5,
            atol=1e-6,
        )
    assert mps.metrics["device"] == "mps"


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
            input_bindings=(DenseArrayBinding("image", torch.zeros(2, 1, 784)),),
            seed=5,
        )
    )
    shd_result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=shd.graph,
            input_bindings=(DenseArrayBinding("events", torch.zeros(2, 1, 700)),),
            seed=5,
        )
    )
    assert mnist_result.outputs["class_logits"].shape == (1, 10)
    assert shd_result.outputs["gesture_logits"].shape == (1, 20)
    assert set(shd_result.diagnostics) == {
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
            input_bindings=(DenseArrayBinding("events", torch.zeros(2, 1, 700)),),
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
            input_bindings=(
                PoissonInputBinding(
                    input_id="image",
                    steps_count=2,
                    batch_size=3,
                    rates_hz=(0.0, 5.0, 25.0),
                    seed=41,
                    categorical=True,
                ),
            ),
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


def test_arbitrary_sizes_independent_inputs_and_all_population_recordings():
    graph = _coupled_graph()
    assert not graph_capability_issues(graph)
    inputs = {"drive_a": torch.zeros(8, 1, 3), "drive_b": torch.zeros(8, 1, 2)}
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            input_bindings=tuple(
                DenseArrayBinding(name, value) for name, value in (inputs).items()
            ),
            seed=3,
        )
    )
    assert result.executor == "graph"
    assert {
        "coupled_0",
        "coupled_1",
        "coupled_2",
        "coupled_3",
    } <= result.diagnostics.keys()
    assert result.diagnostics["coupled_0"].shape == (8, 1, 4)
    assert result.diagnostics["coupled_3"].shape == (8, 1, 2)


def test_graph_is_the_only_cli_and_request_executor():
    assert execution_spec_from_args(parse_args(["sim"])).executor == "graph"
    with pytest.raises(SystemExit):
        parse_args(["sim", "--executor", "legacy"])
    with pytest.raises(ValueError, match="only the graph executor"):
        ExecutionSpec(kind="build", executor="legacy")
