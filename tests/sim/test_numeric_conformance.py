from __future__ import annotations

import json
from fractions import Fraction

import pytest
import torch

from snnlab import lang as snn
from snnlab.lang import training
from snnlab.sim import models as M
from snnlab.sim.conformance import (
    CONFORMANCE_REPORT_SCHEMA,
    ComparisonPolicy,
    canonical_json_tensor,
    compare_conformance_layers,
    remap_named_tensors,
    write_conformance_report,
)
from snnlab.sim.execution import (
    DenseArrayBinding,
    ExecutionSpec,
    GraphExecutor,
    build,
    export_legacy_parameters_v1,
    import_legacy_parameters_v1,
    legacy_parameter_map_v1,
    train,
)
from tests._circuits import author_ping
from tests.sim._execution_builders import expose_graph_diagnostics


def test_layered_conformance_requires_complete_exact_named_coverage(tmp_path):
    reference = {
        "parameters": {"input.weight": torch.tensor([[1.0, 2.0]])},
        "forward": {"logits": torch.tensor([[0.25, 0.75]])},
    }
    report = compare_conformance_layers("hand-checkable", reference, reference)
    assert report.passed
    report.require_passed()
    path = write_conformance_report(tmp_path / "conformance.json", report)
    payload = json.loads(path.read_text())
    assert payload["schema"] == CONFORMANCE_REPORT_SCHEMA
    assert payload["summary"] == {"comparisons": 2, "failed": 0, "passed": 2}

    incomplete = compare_conformance_layers(
        "missing", reference, {"parameters": reference["parameters"]}
    )
    assert not incomplete.passed
    assert incomplete.comparisons[0].reason == "missing from candidate"
    with pytest.raises(AssertionError, match="forward.logits"):
        incomplete.require_passed()


def test_numeric_policy_reports_error_and_never_hides_shape_or_dtype_mismatch():
    reference = {"gradients": {"readout.weight": torch.tensor([1.0, 2.0])}}
    close = {"gradients": {"readout.weight": torch.tensor([1.0, 2.00001])}}
    policy = {
        "gradients": {"readout.weight": ComparisonPolicy(mode="numeric", atol=2e-5)}
    }
    report = compare_conformance_layers("tolerant", reference, close, policies=policy)
    assert report.passed
    assert report.comparisons[0].max_abs_error == pytest.approx(1e-5, rel=0.01)

    wrong_dtype = {
        "gradients": {"readout.weight": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    }
    mismatch = compare_conformance_layers(
        "dtype", reference, wrong_dtype, policies=policy
    )
    assert mismatch.comparisons[0].reason == "dtype mismatch"


def test_conformance_rejects_implicit_or_unused_tolerance_rules():
    with pytest.raises(ValueError, match="exact.*tolerances"):
        ComparisonPolicy(atol=1e-6)
    with pytest.raises(ValueError, match="absent fields"):
        compare_conformance_layers(
            "unused",
            {"forward": {"logits": torch.zeros(1)}},
            {"forward": {"logits": torch.zeros(1)}},
            policies={"forward": {"rates": ComparisonPolicy(mode="numeric")}},
        )


def test_canonical_json_tensor_compares_structural_layers_independent_of_key_order():
    first = canonical_json_tensor({"b": [2, 3], "a": 1})
    second = canonical_json_tensor({"a": 1, "b": [2, 3]})
    assert torch.equal(first, second)


def test_explicit_name_remapping_rejects_partial_and_duplicate_maps():
    values = {"graph.a": torch.ones(1), "graph.b": torch.zeros(1)}
    assert set(
        remap_named_tensors(values, {"graph.a": "legacy.a", "graph.b": "legacy.b"})
    ) == {"legacy.a", "legacy.b"}
    with pytest.raises(ValueError, match="must be complete"):
        remap_named_tensors(values, {"graph.a": "legacy.a"})
    with pytest.raises(ValueError, match="duplicate destination"):
        remap_named_tensors(values, {"graph.a": "legacy.a", "graph.b": "legacy.a"})


@pytest.mark.parametrize("active_recurrence", [False, True])
@pytest.mark.parametrize("dt_ms", [0.05, 0.1, 0.2, 0.3, 0.6])
def test_minimal_legacy_and_graph_ping_forward_share_parameters_and_logits(
    active_recurrence,
    dt_ms,
):
    M.N_IN = 2
    M.N_OUT = 2
    M.dt = dt_ms
    M.T_ms = 200.0
    M.T_steps = int(Fraction("200") / Fraction(str(dt_ms)))
    net = snn.Network("legacy_graph_ping", dt=0.1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cell = author_ping(
        net,
        name="cell",
        n_e=4,
        n_i=1,
        source=events,
        include_silent_recurrence=True,
    )
    scores = snn.readouts.MeanVoltage(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_logits", scores)
    bundle = snn.compile(net)
    # Authoring defaults intentionally retain historical 12/6 step counters.
    # This explicit comparison graph instead represents the collection's
    # physical durations and production's one-update recurrent delay.
    bundle.graph["timebase"]["dt"]["value"] = dt_ms
    for population in bundle.graph["populations"]:
        if population["id"] in {"cell_E", "cell_I"}:
            neuron = population["neuron"]
            assert neuron["refractory_steps"] == (
                12 if population["id"] == "cell_E" else 6
            )
            physical_ms = "1.2" if population["id"] == "cell_E" else "0.6"
            count = Fraction(physical_ms) / Fraction(str(dt_ms))
            assert count.denominator == 1
            neuron["refractory_steps"] = int(count)
    for projection in bundle.graph["projections"]:
        if projection.get("connection") == "recurrent":
            projection["delay"]["value"] = dt_ms
    built = build(
        ExecutionSpec(
            kind="build",
            executor="graph",
            graph=expose_graph_diagnostics(bundle.graph),
            seed=7,
        )
    )
    assert isinstance(built.model, GraphExecutor)
    graph_model = built.model
    graph_parameters = graph_model.parameter_map()
    with torch.no_grad():
        graph_parameters["cell_input.weight"].fill_(10.0)
        for name in (
            "cell_E_to_E.weight",
            "cell_E_to_I.weight",
            "cell_I_to_E.weight",
            "cell_I_to_I.weight",
        ):
            graph_parameters[name].zero_()
        if active_recurrence:
            graph_parameters["cell_E_to_I.weight"].fill_(2.0)
            graph_parameters["cell_I_to_E.weight"].fill_(5.0)
        graph_parameters["scores_projection.weight"].copy_(
            torch.tensor([[1.0, 0.5], [0.25, 1.5], [1.25, 0.75], [0.5, 1.0]])
        )

    legacy = M.COBANet(
        hidden_sizes=[4],
        n_inh_per_layer={1: 1},
        refractory_e_ms=1.2,
        refractory_i_ms=0.6,
        refractory_policy="exact",
        readout_mode="mem-mean",
        w_in=(0.0, 0.0),
        w_hid=(0.0, 0.0),
        w_ee=(0.0, 0.0),
        w_ei=(0.0, 0.0),
        w_ie=(0.0, 0.0),
        w_ii=(0.0, 0.0),
    )
    legacy.recording = True
    mapping = legacy_parameter_map_v1(bundle.graph)
    exported = export_legacy_parameters_v1(bundle.graph, graph_parameters)
    imported = import_legacy_parameters_v1(bundle.graph, exported.parameters)
    with torch.no_grad():
        for name, value in imported.parameters.items():
            graph_parameters[name].copy_(value)
    legacy_parameters = dict(legacy.named_parameters())
    with torch.no_grad():
        for legacy_name, value in exported.parameters.items():
            legacy_parameters[legacy_name].copy_(value)

    inputs = torch.zeros(M.T_steps, 2, 2)
    inputs[:, 0, 0] = 1
    inputs[::2, 1, 1] = 1
    graph = graph_model({"events": inputs})
    legacy_logits = legacy(input_spikes=inputs)
    assert legacy.timing_metadata["duration_steps"] == inputs.shape[0]
    assert legacy.timing_metadata["nominal_duration_ms"] == 200.0
    assert legacy.timing_metadata["realized_duration_ms"] == pytest.approx(
        199.8 if dt_ms in (0.3, 0.6) else 200.0
    )
    if active_recurrence:
        assert torch.count_nonzero(legacy.spike_record["inh"]) > 0
        assert torch.count_nonzero(legacy.spike_record["gi_e_1"]) > 0
    report = compare_conformance_layers(
        "minimal-legacy-graph-ping",
        {
            "parameters": remap_named_tensors(graph.parameters, mapping),
            "forward": {
                "e_spikes": legacy.spike_record["hid"],
                "i_spikes": legacy.spike_record["inh"],
                "e_voltage": legacy.spike_record["v_e_1"],
                "i_voltage": legacy.spike_record["v_i_1"],
                "input_conductance": legacy.spike_record["ge_e_1"],
                "e_to_i_conductance": legacy.spike_record["ge_i_1"],
                "i_to_e_conductance": legacy.spike_record["gi_e_1"],
                "logits": legacy_logits.detach(),
            },
        },
        {
            "parameters": {
                name: value.detach() for name, value in legacy_parameters.items()
            },
            "forward": {
                "e_spikes": graph.diagnostics["cell_E.spikes"],
                "i_spikes": graph.diagnostics["cell_I.spikes"],
                "e_voltage": graph.diagnostics["cell_E.voltage"],
                "i_voltage": graph.diagnostics["cell_I.voltage"],
                "input_conductance": graph.diagnostics["cell_input.conductance"],
                "e_to_i_conductance": graph.diagnostics["cell_E_to_I.conductance"],
                "i_to_e_conductance": graph.diagnostics["cell_I_to_E.conductance"],
                "logits": graph.outputs["class_logits"].detach(),
            },
        },
        policies={
            "forward": {
                "logits": ComparisonPolicy(mode="numeric", atol=1e-6, rtol=1e-6)
            }
        },
    )
    report.require_passed()


def test_legacy_and_graph_four_update_trajectory_and_resume_are_conformant(tmp_path):
    M.N_IN = 2
    M.N_OUT = 2
    M.dt = 0.1
    M.T_ms = 4.0
    M.T_steps = 40
    net = snn.Network("legacy_graph_backward", dt=0.1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cell = author_ping(
        net,
        name="cell",
        n_e=4,
        n_i=1,
        source=events,
        include_silent_recurrence=True,
    )
    scores = snn.readouts.MeanVoltage(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_logits", scores)
    recurrent = [
        "cell_E_to_E.weight",
        "cell_E_to_I.weight",
        "cell_I_to_E.weight",
        "cell_I_to_I.weight",
    ]
    trainable = ["cell_input.weight", "scores_projection.weight", *recurrent]
    recipe = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            training.ParameterGroup(trainable, name="all", lr=0.01),
        ],
        optimizer=training.AdamW(weight_decay=0.0),
        surrogate=training.FastSigmoid(slope=5.0),
        presentation_duration=4.0 * snn.ms,
    )
    bundle = snn.compile(net, training=recipe)
    initial = build(
        ExecutionSpec(
            kind="build",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            seed=7,
        )
    )
    assert isinstance(initial.model, GraphExecutor)
    mapping = legacy_parameter_map_v1(bundle.graph)
    graph_neurons = {row["id"]: row["neuron"] for row in bundle.graph["populations"]}
    legacy = M.COBANet(
        hidden_sizes=[4],
        n_inh_per_layer={1: 1},
        refractory_e_ms=graph_neurons["cell_E"]["refractory_steps"] * M.dt,
        refractory_i_ms=graph_neurons["cell_I"]["refractory_steps"] * M.dt,
        refractory_policy="exact",
        readout_mode="mem-mean",
        w_in=(0.0, 0.0),
        w_hid=(0.0, 0.0),
        w_ee=(0.0, 0.0),
        w_ei=(0.0, 0.0),
        w_ie=(0.0, 0.0),
        w_ii=(0.0, 0.0),
        trainable_w_ee=True,
        trainable_w_ei=True,
        trainable_w_ie=True,
        trainable_w_ii=True,
    )
    legacy_parameters = dict(legacy.named_parameters())
    with torch.no_grad():
        for graph_name, legacy_name in mapping.items():
            legacy_parameters[legacy_name].copy_(initial.parameters[graph_name])

    inputs = torch.zeros(40, 2, 2)
    inputs[:, 0, 0] = 1
    inputs[::2, 1, 1] = 1
    labels = torch.tensor([0, 1])
    update_count = 4
    graph = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(DenseArrayBinding("events", inputs),),
            targets={"label": labels},
            seed=7,
            updates=update_count,
        )
    )
    checkpoint_path = tmp_path / "trajectory-checkpoint"
    first_half = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(DenseArrayBinding("events", inputs),),
            targets={"label": labels},
            seed=7,
            updates=2,
            save_final_checkpoint=checkpoint_path,
        )
    )
    resumed = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(DenseArrayBinding("events", inputs),),
            targets={"label": labels},
            seed=7,
            checkpoint=checkpoint_path,
            updates=2,
        )
    )
    assert [
        *first_half.metrics["updates"],
        *resumed.metrics["updates"],
    ] == graph.metrics["updates"]
    for name in graph.parameters:
        torch.testing.assert_close(
            resumed.parameters[name], graph.parameters[name], rtol=0, atol=0
        )
    optimizer = torch.optim.AdamW(
        [legacy_parameters[mapping[name]] for name in sorted(trainable)],
        lr=0.01,
        weight_decay=0.0,
    )
    legacy_losses = []
    legacy_gradients = {}
    for _ in range(update_count):
        optimizer.zero_grad(set_to_none=True)
        legacy_logits = legacy(input_spikes=inputs)
        legacy_loss = torch.nn.functional.cross_entropy(legacy_logits, labels)
        legacy_losses.append(legacy_loss.detach().clone())
        legacy_loss.backward()
        legacy_gradients = {
            mapping[name]: legacy_parameters[mapping[name]].grad.detach().clone()
            for name in trainable
        }
        optimizer.step()
        with torch.no_grad():
            for parameter in legacy_parameters.values():
                parameter.clamp_(min=0)
    assert torch.count_nonzero(legacy_gradients["W_ff.0"]) > 0
    assert torch.count_nonzero(legacy_gradients["W_ff.1"]) > 0
    assert any(
        torch.count_nonzero(legacy_gradients[mapping[name]]) > 0 for name in recurrent
    )
    legacy_optimizer = {}
    for name in sorted(legacy_gradients):
        for state, value in optimizer.state[legacy_parameters[name]].items():
            if isinstance(value, torch.Tensor):
                legacy_optimizer[f"{name}.{state}"] = value.detach()
    graph_optimizer = {
        f"{mapping[name]}.{state}": value
        for name, values in graph.optimizer_state.items()
        for state, value in values.items()
        if isinstance(value, torch.Tensor)
    }
    numeric = ComparisonPolicy(mode="numeric", atol=1e-6, rtol=1e-6)
    report = compare_conformance_layers(
        "legacy-graph-backward",
        {
            "loss": {"cross_entropy": torch.stack(legacy_losses)},
            "gradients": legacy_gradients,
            "parameters": {
                name: value.detach() for name, value in legacy_parameters.items()
            },
            "optimizer": legacy_optimizer,
        },
        {
            "loss": {
                "cross_entropy": torch.tensor(
                    [row["loss"] for row in graph.metrics["updates"]]
                )
            },
            "gradients": remap_named_tensors(
                graph.gradients, {name: mapping[name] for name in graph.gradients}
            ),
            "parameters": remap_named_tensors(graph.parameters, mapping),
            "optimizer": graph_optimizer,
        },
        policies={
            "loss": {"cross_entropy": numeric},
            "gradients": {name: numeric for name in legacy_gradients},
            "parameters": {name: numeric for name in legacy_parameters},
            "optimizer": {name: numeric for name in legacy_optimizer},
        },
    )
    report.require_passed()


def test_shuffled_dataset_trajectory_matches_independent_pytorch_loop(tmp_path):
    net = snn.Network("dataset_oracle", dt=0.1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    scores = snn.readouts.SpikeCount(source=events, classes=2, name="scores")
    snn.ops.linear(events, size=1, name="shadow")
    net.output("class_scores", scores)
    parameter_ids = [row["id"] for row in net.parameters]
    recipe = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            training.ParameterGroup(
                ["scores_projection.weight"], name="readout", lr=0.1
            ),
            training.ParameterGroup(
                [name for name in parameter_ids if name != "scores_projection.weight"],
                name="frozen",
                lr=0.0,
                frozen=True,
            ),
        ],
        optimizer=training.AdamW(weight_decay=0.0),
        epochs=2,
        presentation_duration=0.3 * snn.ms,
    )
    bundle = snn.compile(net, training=recipe)
    inputs = torch.zeros(3, 5, 2)
    for sample in range(5):
        inputs[:, sample, sample % 2] = 1
    labels = torch.tensor([0, 1, 0, 1, 0])
    seed = 23
    batch_size = 2
    graph = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(DenseArrayBinding("events", inputs),),
            targets={"label": labels},
            seed=seed,
            epochs=2,
            batch_size=batch_size,
            shuffle=True,
        )
    )
    initial = build(
        ExecutionSpec(
            kind="build",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            seed=seed,
        )
    )
    direct = torch.nn.Parameter(
        initial.parameters["scores_projection.weight"].detach().clone()
    )
    optimizer = torch.optim.AdamW([direct], lr=0.1, weight_decay=0.0)
    losses = []
    coordinates = []
    direct_gradient = None
    for epoch in range(2):
        order = torch.randperm(5, generator=torch.Generator().manual_seed(seed + epoch))
        for batch, indices in enumerate(order.split(batch_size)):
            optimizer.zero_grad(set_to_none=True)
            logits = (inputs[:, indices] @ direct).sum(dim=0)
            loss = torch.nn.functional.cross_entropy(logits, labels[indices])
            losses.append(loss.detach().clone())
            coordinates.append((epoch + 1, batch + 1))
            loss.backward()
            direct_gradient = direct.grad.detach().clone()
            optimizer.step()
    assert [
        (row["epoch"], row["batch"]) for row in graph.metrics["updates"]
    ] == coordinates
    assert direct_gradient is not None
    direct_optimizer = {
        state: value.detach()
        for state, value in optimizer.state[direct].items()
        if isinstance(value, torch.Tensor)
    }
    graph_optimizer = graph.optimizer_state["scores_projection.weight"]
    graph_losses = torch.tensor([row["loss"] for row in graph.metrics["updates"]])
    assert torch.allclose(torch.stack(losses), graph_losses, atol=1e-6, rtol=1e-6), (
        torch.stack(losses),
        graph_losses,
    )
    numeric = ComparisonPolicy(mode="numeric", atol=1e-6, rtol=1e-6)
    report = compare_conformance_layers(
        "dataset-order-oracle",
        {
            "loss": {"cross_entropy": torch.stack(losses)},
            "gradients": {"scores_projection.weight": direct_gradient},
            "parameters": {"scores_projection.weight": direct.detach()},
            "optimizer": direct_optimizer,
        },
        {
            "loss": {"cross_entropy": graph_losses},
            "gradients": {
                "scores_projection.weight": graph.gradients["scores_projection.weight"]
            },
            "parameters": {
                "scores_projection.weight": graph.parameters["scores_projection.weight"]
            },
            "optimizer": graph_optimizer,
        },
        policies={
            "loss": {"cross_entropy": numeric},
            "gradients": {"scores_projection.weight": numeric},
            "parameters": {"scores_projection.weight": numeric},
            "optimizer": {name: numeric for name in direct_optimizer},
        },
    )
    report.require_passed()

    checkpoint = tmp_path / "mid-epoch"
    train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(DenseArrayBinding("events", inputs),),
            targets={"label": labels},
            seed=seed,
            epochs=2,
            batch_size=batch_size,
            shuffle=True,
            updates=2,
            save_final_checkpoint=checkpoint,
        )
    )
    with pytest.raises(ValueError, match="execution protocol does not match"):
        train(
            ExecutionSpec(
                kind="train",
                executor="graph",
                graph=bundle.graph,
                training=bundle.training,
                input_bindings=(DenseArrayBinding("events", inputs),),
                targets={"label": labels},
                seed=seed,
                checkpoint=checkpoint,
                epochs=2,
                batch_size=batch_size,
                shuffle=False,
            )
        )
