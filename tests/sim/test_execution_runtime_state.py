"""Focused acceptance tests for the typed seam and graph executor."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from snnlab import lang as snn
from snnlab.sim.execution import (
    DenseArrayBinding,
    ExecutionSpec,
    GraphRuntimeState,
    load_runtime_state,
    plan_graph,
    runtime_state_signature,
    save_runtime_state,
    simulate,
)
from snnlab.sim.tool import main
from tests.sim._execution_builders import (
    coupled_graph as _coupled_graph,
)
from tests.sim._execution_builders import (
    state_tensors as _state_tensors,
)


def _continuation_case():
    graph = _coupled_graph(delay_ms=0.3)
    for projection in graph["projections"]:
        if projection["id"] in {"a_input", "b_input"}:
            projection["delay"] = {"value": 0.3, "unit": "ms"}
    inputs = {
        "drive_a": torch.zeros(1000, 2, 3),
        "drive_b": torch.zeros(1000, 2, 2),
    }
    inputs["drive_a"][::3, :, :] = 1.0
    inputs["drive_b"][1::4, :, :] = 1.0
    return graph, inputs


def _assert_state_equal(left: GraphRuntimeState, right: GraphRuntimeState):
    assert left.signature == right.signature
    assert left.completed_steps == right.completed_steps
    for group in (
        "voltages",
        "refractory",
        "conductances",
        "population_histories",
        "input_histories",
    ):
        left_values = getattr(left, group)
        right_values = getattr(right, group)
        assert left_values.keys() == right_values.keys()
        for name in left_values:
            torch.testing.assert_close(
                left_values[name], right_values[name], rtol=0, atol=0
            )


def test_split_run_exactly_preserves_spikes_voltages_conductances_and_dynamic_state():
    graph, inputs = _continuation_case()
    whole = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            input_bindings=tuple(
                DenseArrayBinding(name, value) for name, value in (inputs).items()
            ),
            seed=11,
        )
    )
    first = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            input_bindings=tuple(
                DenseArrayBinding(name, value)
                for name, value in (
                    {name: value[:500] for name, value in inputs.items()}
                ).items()
            ),
            seed=11,
        )
    )
    assert first.runtime_state is not None
    # The split occurs with live synapses, delayed events and refractory cells.
    assert any(
        torch.count_nonzero(x) for x in first.runtime_state.conductances.values()
    )
    assert any(
        torch.count_nonzero(x)
        for x in first.runtime_state.population_histories.values()
    )
    assert any(
        torch.count_nonzero(x) for x in first.runtime_state.input_histories.values()
    )
    assert any(torch.count_nonzero(x) for x in first.runtime_state.refractory.values())
    second = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            input_bindings=tuple(
                DenseArrayBinding(name, value)
                for name, value in (
                    {name: value[500:] for name, value in inputs.items()}
                ).items()
            ),
            seed=11,
        ),
        runtime_state=first.runtime_state,
    )
    for name, expected in whole.diagnostics.items():
        actual = torch.cat((first.diagnostics[name], second.diagnostics[name]))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert whole.runtime_state is not None and second.runtime_state is not None
    _assert_state_equal(second.runtime_state, whole.runtime_state)
    for name in whole.final_state:
        torch.testing.assert_close(
            second.final_state[name], whole.final_state[name], rtol=0, atol=0
        )


def test_runtime_state_portable_round_trip_preserves_dtype_and_values(tmp_path):
    graph, inputs = _continuation_case()
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            input_bindings=tuple(
                DenseArrayBinding(name, value)
                for name, value in (
                    {name: value[:13] for name, value in inputs.items()}
                ).items()
            ),
            seed=2,
        )
    )
    assert result.runtime_state is not None
    root = save_runtime_state(tmp_path / "state", result.runtime_state)
    loaded = load_runtime_state(root, device="cpu")
    _assert_state_equal(loaded, result.runtime_state)


def test_runtime_state_allows_weight_branch_but_rejects_structural_changes():
    graph, inputs = _continuation_case()
    zero_parameter = next(
        row for row in graph["parameters"] if row["id"] == "a_I_to_b_E.weight"
    )
    zero_parameter["initializer"] = {"kind": "constant", "value": 0.0}
    first = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            input_bindings=tuple(
                DenseArrayBinding(name, value)
                for name, value in (
                    {name: value[:12] for name, value in inputs.items()}
                ).items()
            ),
            seed=5,
        )
    )
    assert first.runtime_state is not None
    branch = copy.deepcopy(graph)
    cross_parameter = next(
        row for row in branch["parameters"] if row["id"] == "a_I_to_b_E.weight"
    )
    cross_parameter["initializer"] = {"kind": "constant", "value": 9.0}
    assert runtime_state_signature(plan_graph(branch)) == first.runtime_state.signature
    simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=branch,
            input_bindings=tuple(
                DenseArrayBinding(name, value)
                for name, value in (
                    {name: value[12:] for name, value in inputs.items()}
                ).items()
            ),
            seed=5,
        ),
        runtime_state=first.runtime_state,
    )

    incompatible_graphs = []
    resized = copy.deepcopy(graph)
    next(row for row in resized["populations"] if row["id"] == "a_E")["size"] += 1
    incompatible_graphs.append(resized)
    new_dt = copy.deepcopy(graph)
    new_dt["timebase"]["dt"]["value"] = 0.05
    incompatible_graphs.append(new_dt)
    new_delay = copy.deepcopy(graph)
    next(row for row in new_delay["projections"] if row["id"] == "a_I_to_b_E")["delay"][
        "value"
    ] = 0.4
    incompatible_graphs.append(new_delay)
    new_synapse = copy.deepcopy(graph)
    next(row for row in new_synapse["projections"] if row["id"] == "a_I_to_b_E")[
        "synapse"
    ]["tau"]["value"] = 8.0
    incompatible_graphs.append(new_synapse)
    missing_projection = copy.deepcopy(graph)
    missing_projection["projections"] = [
        row for row in missing_projection["projections"] if row["id"] != "a_I_to_b_E"
    ]
    incompatible_graphs.append(missing_projection)
    renamed_projection = copy.deepcopy(graph)
    next(row for row in renamed_projection["projections"] if row["id"] == "a_I_to_b_E")[
        "id"
    ] = "renamed"
    incompatible_graphs.append(renamed_projection)
    reshaped_parameter = copy.deepcopy(graph)
    next(
        row
        for row in reshaped_parameter["parameters"]
        if row["id"] == "a_I_to_b_E.weight"
    )["shape"][0] += 1
    incompatible_graphs.append(reshaped_parameter)
    for incompatible in incompatible_graphs:
        try:
            simulate(
                ExecutionSpec(
                    kind="simulate",
                    executor="graph",
                    graph=incompatible,
                    input_bindings=tuple(
                        DenseArrayBinding(name, value)
                        for name, value in (
                            {name: value[12:] for name, value in inputs.items()}
                        ).items()
                    ),
                    seed=5,
                ),
                runtime_state=first.runtime_state,
            )
        except ValueError as exc:
            assert "runtime state is incompatible" in str(exc)
            assert "graph." in str(exc)
        else:
            raise AssertionError("structurally incompatible runtime state must fail")


def test_runtime_state_validates_batch_shape_and_dtype():
    graph, inputs = _continuation_case()
    first = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            input_bindings=tuple(
                DenseArrayBinding(name, value)
                for name, value in (
                    {name: value[:10] for name, value in inputs.items()}
                ).items()
            ),
            seed=8,
        )
    )
    assert first.runtime_state is not None
    wrong_batch = {name: value[10:, :1] for name, value in inputs.items()}
    try:
        simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=graph,
                input_bindings=tuple(
                    DenseArrayBinding(name, value)
                    for name, value in (wrong_batch).items()
                ),
                seed=8,
            ),
            runtime_state=first.runtime_state,
        )
    except ValueError as exc:
        assert "shape expected" in str(exc)
    else:
        raise AssertionError("runtime-state batch mismatch must fail")
    bad_state = first.runtime_state.detached()
    bad_state.voltages["a_E"] = bad_state.voltages["a_E"].double()
    try:
        simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=graph,
                input_bindings=tuple(
                    DenseArrayBinding(name, value)
                    for name, value in (
                        {name: value[10:] for name, value in inputs.items()}
                    ).items()
                ),
                seed=8,
            ),
            runtime_state=bad_state,
        )
    except ValueError as exc:
        assert "dtype expected" in str(exc)
    else:
        raise AssertionError("runtime-state dtype mismatch must fail")


def test_graph_cli_runtime_state_round_trip_requires_bundle(tmp_path):
    graph, inputs = _continuation_case()
    bundle = snn.compiler.Bundle(
        graph=graph,
        training=None,
        manifest={
            "schema": "snnlang.bundle/v1",
            "graph_digest": snn.compiler.digest(graph),
            "files": [{"path": "graph.json", "digest": snn.compiler.digest(graph)}],
            "assets": [],
            "compiler": {"name": "test", "version": "1"},
            "target": None,
        },
        diagnostics=[],
        asset_sources={},
    ).write(tmp_path / "graph.bundle")
    first_inputs = tmp_path / "first.npz"
    second_inputs = tmp_path / "second.npz"
    np.savez(
        first_inputs, **{name: value[:19].numpy() for name, value in inputs.items()}
    )
    np.savez(
        second_inputs, **{name: value[19:].numpy() for name, value in inputs.items()}
    )
    state_one = tmp_path / "state-one"
    state_two = tmp_path / "state-two"
    assert (
        main(
            [
                "sim",
                "--executor",
                "graph",
                "--bundle",
                str(bundle),
                "--input-file",
                str(first_inputs),
                "--out-dir",
                str(tmp_path / "first-out"),
                "--save-runtime-state",
                str(state_one),
            ]
        )
        == 0
    )
    with np.load(tmp_path / "first-out" / "parameters.npz") as parameters:
        expected = {row["id"] for row in graph["parameters"]}
        assert set(parameters.files) == expected
        for row in graph["parameters"]:
            assert parameters[row["id"]].size == int(np.prod(row["shape"]))
    assert (
        main(
            [
                "sim",
                "--executor",
                "graph",
                "--bundle",
                str(bundle),
                "--input-file",
                str(second_inputs),
                "--out-dir",
                str(tmp_path / "second-out"),
                "--load-runtime-state",
                str(state_one),
                "--save-runtime-state",
                str(state_two),
            ]
        )
        == 0
    )
    assert load_runtime_state(state_two).completed_steps == 1000
    try:
        main(
            [
                "sim",
                "--load-runtime-state",
                str(state_one),
                "--out-dir",
                str(tmp_path / "legacy-out"),
            ]
        )
    except SystemExit as exc:
        assert "requires an explicit --bundle" in str(exc)
    else:
        raise AssertionError("legacy executor must reject graph-runtime-state flags")


def test_disabling_diagnostics_preserves_outputs_and_branch_state():
    graph = _coupled_graph()
    inputs = {"drive_a": torch.ones(12, 1, 3), "drive_b": torch.ones(12, 1, 2)}
    spec = dict(
        kind="simulate",
        graph=graph,
        input_bindings=tuple(
            DenseArrayBinding(name, value) for name, value in inputs.items()
        ),
        device="cpu",
        seed=42,
    )
    enabled = simulate(ExecutionSpec(**spec))
    disabled = simulate(ExecutionSpec(**spec, diagnostics=False))
    assert enabled.diagnostics and not disabled.diagnostics
    assert disabled.outputs.keys() == enabled.outputs.keys()
    for key in enabled.outputs:
        assert torch.equal(disabled.outputs[key], enabled.outputs[key])
    for expected, actual in zip(
        _state_tensors(enabled.runtime_state),
        _state_tensors(disabled.runtime_state),
        strict=True,
    ):
        assert torch.equal(expected, actual)


@pytest.mark.parametrize(
    "field,value", [("recording", "full"), ("recording_fields", ())]
)
def test_removed_recording_arguments_are_rejected(field, value):
    with pytest.raises(TypeError, match=field):
        ExecutionSpec(kind="simulate", **{field: value})
