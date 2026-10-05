"""Focused acceptance tests for the typed seam and graph executor."""

from __future__ import annotations

import pytest
import torch

from snnlab.sim.execution import (
    AddPoissonSpikes,
    DenseArrayBinding,
    DropSpikes,
    ExecutionSpec,
    GraphExecutor,
    PoissonInputBinding,
    build,
    plan_graph,
    resolve_poisson_input_bindings,
    simulate,
    train,
)
from tests.sim._execution_builders import (
    coupled_graph as _coupled_graph,
)
from tests.sim._execution_builders import (
    direct_train_bundle as _direct_train_bundle,
)
from tests.sim._execution_builders import (
    standard_readout_graph as _standard_readout_graph,
)


def test_poisson_binding_defaults_to_one_batch_item():
    graph = _standard_readout_graph("count")
    binding = PoissonInputBinding(
        input_id="events", steps_count=3, rates_hz=(0.0,), seed=7
    )
    resolved = resolve_poisson_input_bindings(graph, bindings=(binding,))
    assert binding.batch_size == 1
    assert resolved.tensors["events"].shape[:2] == (3, 1)


def test_fixed_rate_poisson_binding_has_exact_boundary_fixtures():
    graph = _standard_readout_graph("count")
    zero = resolve_poisson_input_bindings(
        graph,
        bindings=(
            PoissonInputBinding(
                input_id="events", steps_count=3, batch_size=2, rates_hz=(0.0,), seed=7
            ),
        ),
    )
    assert torch.count_nonzero(zero.tensors["events"]) == 0
    graph["timebase"]["dt"] = {"value": 1.0, "unit": "ms"}
    full = resolve_poisson_input_bindings(
        graph,
        bindings=(
            PoissonInputBinding(
                input_id="events",
                steps_count=3,
                batch_size=2,
                rates_hz=(1000.0,),
                seed=7,
            ),
        ),
    )
    assert torch.all(full.tensors["events"] == 1)
    assert full.protocol["binding_schema"] == "tools/snnsim.poisson-input-binding/v1"
    assert full.protocol["inputs"][0]["selection"] == "constant"


def test_graph_inference_overrides_poisson_duration_and_rate():
    graph = _standard_readout_graph("count")
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            input_bindings=(
                PoissonInputBinding(
                    input_id="events",
                    steps_count=2,
                    batch_size=1,
                    rates_hz=(0.0,),
                    seed=7,
                ),
            ),
            options={
                "inference_overrides": {
                    "duration_ms": 300.0,
                    "input_rate_hz": 10.0,
                }
            },
        )
    )
    protocol = result.metrics["execution_protocol"]
    assert protocol["timing"] == {
        "dt_ms": 100.0,
        "duration_ms": 300.0,
        "steps": 3,
    }
    assert protocol["inputs"][0]["rates_hz"] == [10.0]
    assert result.metrics["inference_overrides"] == {
        "schema": "tools/snnsim.inference-overrides/v1",
        "requested": {"duration_ms": 300.0, "input_rate_hz": 10.0},
        "resolved": {
            "duration_ms": 300.0,
            "timestep_ms": 100.0,
            "projection_scales": {},
            "input_rate_hz": 10.0,
        },
    }


def test_graph_inference_timestep_recompiles_and_preserves_duration(tmp_path):
    bundle = _direct_train_bundle()
    checkpoint = tmp_path / "checkpoint"
    train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(DenseArrayBinding("events", torch.ones(3, 1, 2)),),
            targets={"label": torch.tensor([0])},
            seed=7,
            save_final_checkpoint=checkpoint,
        )
    )
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            checkpoint=checkpoint,
            input_bindings=(
                PoissonInputBinding(
                    input_id="events",
                    steps_count=3,
                    batch_size=1,
                    rates_hz=(0.0,),
                    seed=13,
                ),
            ),
            options={"inference_overrides": {"timestep_ms": 0.05}},
        )
    )
    assert bundle.graph["timebase"]["dt"] == {"value": 0.1, "unit": "ms"}
    assert result.metrics["execution_protocol"]["timing"] == {
        "dt_ms": 0.05,
        "steps": 6,
        "duration_ms": pytest.approx(0.3),
    }
    provenance = result.metrics["inference_overrides"]
    assert provenance["resolved"]["timestep_ms"] == 0.05
    assert provenance["resolved"]["duration_ms"] == pytest.approx(0.3)
    assert result.metrics["source_graph_digest"] == bundle.training["graph_digest"]
    assert (
        result.metrics["effective_graph_digest"]
        != result.metrics["source_graph_digest"]
    )


def test_graph_inference_timestep_rejects_non_resampleable_inputs():
    graph = _standard_readout_graph("count")
    with pytest.raises(ValueError, match="resampleable Poisson"):
        simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=graph,
                input_bindings=(DenseArrayBinding("events", torch.zeros(2, 1, 2)),),
                options={"inference_overrides": {"timestep_ms": 50.0}},
            )
        )


def test_graph_inference_projection_scale_is_request_local():
    graph = _coupled_graph(direction="uncoupled")
    inputs = {
        "drive_a": torch.zeros(2, 1, 3),
        "drive_b": torch.zeros(2, 1, 2),
    }
    baseline = build(ExecutionSpec(kind="build", executor="graph", graph=graph, seed=5))
    projection = graph["projections"][0]
    parameter_id = projection["parameters"][0]
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            input_bindings=tuple(
                DenseArrayBinding(name, value) for name, value in (inputs).items()
            ),
            seed=5,
            options={
                "inference_overrides": {"projection_scales": {projection["id"]: 0.25}}
            },
        )
    )
    torch.testing.assert_close(
        result.parameters[parameter_id], baseline.parameters[parameter_id] * 0.25
    )
    assert graph["parameters"][0]["initializer"] != {"kind": "constant", "value": 0.25}


def test_graph_inference_overrides_reject_ambiguous_or_unknown_requests():
    graph = _standard_readout_graph("count")
    with pytest.raises(ValueError, match="require Poisson"):
        simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=graph,
                input_bindings=(DenseArrayBinding("events", torch.zeros(2, 1, 2)),),
                options={"inference_overrides": {"duration_ms": 100.0}},
            )
        )
    with pytest.raises(ValueError, match="unknown projections"):
        simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=graph,
                input_bindings=(DenseArrayBinding("events", torch.zeros(2, 1, 2)),),
                options={
                    "inference_overrides": {"projection_scales": {"missing": 1.0}}
                },
            )
        )


def test_graph_inference_interventions_are_ordered_and_recorded():
    graph = _coupled_graph(direction="uncoupled")
    inputs = {
        "drive_a": torch.zeros(3, 1, 3),
        "drive_b": torch.zeros(3, 1, 2),
    }
    add = AddPoissonSpikes(population_id="a_E", rate_hz=10000.0, seed=11)
    drop = DropSpikes(population_id="a_E", probability=1.0, seed=12)
    dropped = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            input_bindings=tuple(
                DenseArrayBinding(name, value) for name, value in (inputs).items()
            ),
            interventions=(
                add,
                drop,
            ),
        )
    )
    added = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            input_bindings=tuple(
                DenseArrayBinding(name, value) for name, value in (inputs).items()
            ),
            interventions=(
                drop,
                add,
            ),
        )
    )
    assert torch.count_nonzero(dropped.diagnostics["a_E.spikes"]) == 0
    assert torch.all(added.diagnostics["a_E.spikes"] == 1)
    provenance = added.metrics["inference_interventions"]
    assert provenance["schema"] == "tools/snnsim.inference-interventions/v1"
    assert [row["kind"] for row in provenance["requested"]] == [
        "drop_spikes",
        "add_poisson_spikes",
    ]
    assert provenance["requested"][0]["seed"] == drop.seed
    assert provenance["resolved"][1]["probability_per_step"] == 1.0


def test_graph_inference_intervention_stream_resumes_exactly():
    graph = _coupled_graph(direction="uncoupled")
    intervention = AddPoissonSpikes(population_id="a_E", rate_hz=5000.0, seed=31)
    full_model = GraphExecutor(plan_graph(graph), seed=4)
    full = full_model(
        {
            "drive_a": torch.zeros(4, 2, 3),
            "drive_b": torch.zeros(4, 2, 2),
        },
        interventions=(intervention,),
    )
    resumed_model = GraphExecutor(plan_graph(graph), seed=4)
    first = resumed_model(
        {
            "drive_a": torch.zeros(2, 2, 3),
            "drive_b": torch.zeros(2, 2, 2),
        },
        interventions=(intervention,),
    )
    second = resumed_model(
        {
            "drive_a": torch.zeros(2, 2, 3),
            "drive_b": torch.zeros(2, 2, 2),
        },
        runtime_state=first.runtime_state,
        interventions=(intervention,),
    )
    torch.testing.assert_close(
        full.diagnostics["a_E.spikes"],
        torch.cat((first.diagnostics["a_E.spikes"], second.diagnostics["a_E.spikes"])),
        rtol=0,
        atol=0,
    )


def test_graph_inference_interventions_reject_invalid_targets_and_values():
    graph = _coupled_graph(direction="uncoupled")
    inputs = {
        "drive_a": torch.zeros(1, 1, 3),
        "drive_b": torch.zeros(1, 1, 2),
    }
    with pytest.raises(ValueError, match="unknown population"):
        simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=graph,
                input_bindings=tuple(
                    DenseArrayBinding(name, value) for name, value in (inputs).items()
                ),
                interventions=(DropSpikes(population_id="missing", probability=0.5),),
            )
        )
    with pytest.raises(ValueError, match="rate times dt"):
        simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=graph,
                input_bindings=tuple(
                    DenseArrayBinding(name, value) for name, value in (inputs).items()
                ),
                interventions=(AddPoissonSpikes(population_id="a_E", rate_hz=10001.0),),
            )
        )
