"""Selective resets, explicit segment ownership and portable continuation."""

import json
from dataclasses import replace

import pytest
import torch

from snnlab.sim.execution import (
    Boundary,
    BoundarySchedule,
    DecisionSegments,
    DenseArrayBinding,
    ExecutionSpec,
    GraphExecutor,
    ResetVoltage,
    execution_spec_from_args,
    load_runtime_state,
    plan_graph,
    save_runtime_state,
    segment_identity,
    simulate,
    validate_inference_artifacts,
    write_inference_artifacts,
)
from snnlab.sim.tool import parse_args
from tests.sim.test_spike_replay import graph_fixture, stream


def policy():
    boundaries = BoundarySchedule(steps=(0, 2, 4))
    return (ResetVoltage("immediate", boundaries),), DecisionSegments(
        boundaries, 6, ("source",), ("source",)
    )


def run(data, *, state=None, resets=None, decisions=None):
    default_resets, default_decisions = policy()
    return GraphExecutor(plan_graph(graph_fixture()), seed=17)(
        {"events": data},
        runtime_state=state,
        diagnostics=False,
        resets=default_resets if resets is None else resets,
        decisions=default_decisions if decisions is None else decisions,
    )


@pytest.mark.parametrize("cuts", [(1, 3, 5), (2, 4), (3,), (1, 2, 3, 4, 5)])
def test_chunked_portable_continuation_matches(cuts, tmp_path):
    full = run(stream())
    state = None
    records = []
    outputs = []
    begin = 0
    for end in (*cuts, 6):
        result = run(stream()[begin:end], state=state)
        records.extend(result.decisions)
        outputs.append(result.outputs["immediate_out"])
        save_runtime_state(tmp_path, result.runtime_state)
        state = load_runtime_state(tmp_path)
        begin = end
    assert records == full.decisions
    assert torch.equal(torch.cat(outputs), full.outputs["immediate_out"])
    for group in (
        "voltages",
        "refractory",
        "conductances",
        "currents",
        "population_histories",
        "input_histories",
        "segment_counts",
    ):
        for name, expected in getattr(full.runtime_state, group).items():
            assert torch.equal(getattr(state, group)[name], expected)
    assert state.segment_state == full.runtime_state.segment_state


def test_selective_reset_preserves_other_dynamic_state():
    plain = run(stream(), resets=(), decisions=None)
    selective = run(stream())
    assert not torch.equal(
        plain.outputs["immediate_out"], selective.outputs["immediate_out"]
    )
    for name in ("source", "delayed"):
        assert torch.equal(
            plain.runtime_state.voltages[name], selective.runtime_state.voltages[name]
        )
    for group in (
        "refractory",
        "conductances",
        "currents",
        "population_histories",
        "input_histories",
    ):
        for name, expected in getattr(plain.runtime_state, group).items():
            assert torch.equal(getattr(selective.runtime_state, group)[name], expected)


def test_counts_and_per_batch_nonuniform_ownership():
    boundaries = BoundarySchedule(
        events=(Boundary(0), Boundary(2, (0,)), Boundary(3, (1,)), Boundary(4, (0,)))
    )
    decisions = DecisionSegments(boundaries, 6, ("source",), ("source",))
    result = run(stream(), resets=(), decisions=decisions)
    plain = run(stream(), resets=(), decisions=decisions)
    for row in result.decisions:
        expected = (
            plain.outputs["emitted"][row["start_step"] : row["end_step"], row["batch"]]
            .to(torch.int64)
            .sum(0)
        )
        assert row["output_spike_counts"]["source"] == expected.tolist()
        assert row["population_totals"]["source"] == int(expected.sum())
    assert [(r["batch"], r["start_step"], r["end_step"]) for r in result.decisions] == [
        (0, 0, 2),
        (1, 0, 3),
        (0, 2, 4),
        (0, 4, 6),
        (1, 3, 6),
    ]


def test_partial_counts_do_not_finalize_at_call_end():
    result = run(stream()[:1])
    assert result.decisions == []
    assert result.runtime_state.segment_counts["source"].shape == (2, 2)
    assert result.runtime_state.segment_counts["source"].dtype == torch.int64
    assert int(result.runtime_state.segment_counts["source"].sum()) > 0
    assert result.diagnostics == {}


def test_resume_rejects_changed_or_removed_policy():
    state = run(stream()[:1]).runtime_state
    resets, decisions = policy()
    with pytest.raises(ValueError, match="policy"):
        run(
            stream()[1:],
            state=state,
            resets=(),
            decisions=replace(decisions, end_step=7),
        )
    with pytest.raises(ValueError, match="policy"):
        GraphExecutor(plan_graph(graph_fixture()))(
            {"events": stream()[1:]}, runtime_state=state
        )
    plain = GraphExecutor(plan_graph(graph_fixture()))(
        {"events": stream()[:1]}
    ).runtime_state
    with pytest.raises(ValueError, match="unaudited"):
        run(stream()[1:], state=plain)


@pytest.mark.parametrize(
    "schedule",
    [
        BoundarySchedule(steps=(0, 0)),
        BoundarySchedule(steps=(2, 1)),
        BoundarySchedule(steps=(-1,)),
        BoundarySchedule(steps=(True,)),
        BoundarySchedule(steps=(0,), batches=()),
        BoundarySchedule(steps=(0,), batches=(2,)),
        BoundarySchedule(steps=(0,), events=(Boundary(1),)),
    ],
)
def test_malformed_boundaries_rejected(schedule):
    with pytest.raises(ValueError):
        run(stream(), resets=(ResetVoltage("immediate", schedule),))


def test_ambiguous_resets_and_nonspiking_counts_rejected():
    resets, decisions = policy()
    with pytest.raises(ValueError, match="overlapping"):
        run(stream(), resets=resets + resets)
    with pytest.raises(ValueError, match="spiking"):
        run(stream(), decisions=replace(decisions, population_totals=("immediate",)))
    with pytest.raises(ValueError, match="end_step"):
        run(stream(), decisions=replace(decisions, end_step=4))
    with pytest.raises(ValueError, match="finite"):
        run(stream(), resets=(replace(resets[0], value_mv=float("nan")),))


def test_runtime_metadata_corruption_rejected(tmp_path):
    save_runtime_state(tmp_path, run(stream()[:1]).runtime_state)
    path = tmp_path / "manifest.json"
    value = json.loads(path.read_text())
    value["segment_state"]["starts"][0] = 100
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="digest"):
        load_runtime_state(tmp_path)


def test_typed_simulation_cli_and_cache(tmp_path):
    graph = graph_fixture()
    resets, decisions = policy()
    result = simulate(
        ExecutionSpec(
            kind="infer",
            graph=graph,
            device="cpu",
            input_bindings=(DenseArrayBinding("events", stream()),),
            resets=resets,
            decisions=decisions,
            diagnostics=False,
        )
    )
    assert result.decisions == run(stream()).decisions
    write_inference_artifacts(tmp_path, result, graph=graph, seed=0)
    identity = segment_identity(resets, decisions, graph=graph, batch_size=2)
    validate_inference_artifacts(tmp_path, expected_segments=identity)
    with pytest.raises(ValueError, match="policy"):
        validate_inference_artifacts(tmp_path, expected_segments=None)
    assert (
        json.loads((tmp_path / "metrics.json").read_text())["decisions"]
        == result.decisions
    )
    args = parse_args(
        [
            "sim",
            "--executor",
            "graph",
            "--reset-voltage",
            json.dumps(
                {"population_id": "immediate", "boundaries": {"steps": [0, 2, 4]}}
            ),
            "--decisions",
            json.dumps(
                {
                    "boundaries": {"steps": [0, 2, 4]},
                    "end_step": 6,
                    "population_totals": ["source"],
                }
            ),
        ]
    )
    request = execution_spec_from_args(args)
    assert request.resets[0].population_id == "immediate"
    assert request.decisions.end_step == 6
    assert "decisions" not in request.options


def test_cli_writes_decisions_and_runtime_state(tmp_path):
    import numpy as np

    from snnlab.sim.tool import main

    bundle = tmp_path / "network.bundle"
    graph_fixture(bundle_path=bundle)
    inputs = tmp_path / "inputs.npy"
    np.save(inputs, stream().numpy())
    out = tmp_path / "out"
    state = tmp_path / "state"
    main(
        [
            "sim",
            "--executor",
            "graph",
            "--bundle",
            str(bundle),
            "--input-file",
            str(inputs),
            "--device",
            "cpu",
            "--out-dir",
            str(out),
            "--save-runtime-state",
            str(state),
            "--reset-voltage",
            json.dumps(
                {"population_id": "immediate", "boundaries": {"steps": [0, 2, 4]}}
            ),
            "--decisions",
            json.dumps(
                {
                    "boundaries": {"steps": [0, 2, 4]},
                    "end_step": 6,
                    "population_totals": ["source"],
                    "output_spike_counts": ["source"],
                }
            ),
        ]
    )
    assert (
        json.loads((out / "metrics.json").read_text())["decisions"]
        == run(stream()).decisions
    )
    assert load_runtime_state(state).segment_state["starts"] == [None, None]
    validate_inference_artifacts(out)


def test_reset_subset_only_changes_selected_batch():
    schedule = BoundarySchedule(steps=(2,), batches=(0,))
    model = GraphExecutor(plan_graph(graph_fixture()))
    plain = model({"events": stream()})
    reset = model(
        {"events": stream()}, resets=(ResetVoltage("immediate", schedule, 100.0),)
    )
    assert torch.equal(
        reset.outputs["immediate_out"][:, 1], plain.outputs["immediate_out"][:, 1]
    )
    assert torch.equal(
        reset.outputs["immediate_out"][:2, 0], plain.outputs["immediate_out"][:2, 0]
    )
    assert not torch.equal(
        reset.outputs["immediate_out"][2:, 0], plain.outputs["immediate_out"][2:, 0]
    )


def test_unsupported_neuron_reset_and_training_rejected():
    from snnlab import lang as snn
    from snnlab.sim.execution import execute_request, train

    graph = graph_fixture(neuron=snn.CUBA_LIF())
    with pytest.raises(ValueError, match="leaky-integrator"):
        GraphExecutor(plan_graph(graph))(
            {"events": stream()},
            resets=(ResetVoltage("source", BoundarySchedule(steps=(0,))),),
        )
    resets, decisions = policy()
    with pytest.raises(ValueError, match="simulation/inference"):
        train(ExecutionSpec(kind="train", graph=graph_fixture(), resets=resets))
    with pytest.raises(ValueError, match="graph executor"):
        execute_request(
            ExecutionSpec(kind="infer", executor="legacy", decisions=decisions),
            legacy=lambda: None,
        )
