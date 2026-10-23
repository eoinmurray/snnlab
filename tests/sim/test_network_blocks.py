"""Network block trajectories, cross-block gradients and explicit fallbacks."""

import pytest
import torch

from snnlab import lang
from snnlab.sim import GraphExecutor, RecordingSpec, SignalRecording, plan_graph
from snnlab.sim.cli import execution_spec_from_args, parse_args
from snnlab.sim.requests.spec import ExecutionSpec
from tests.sim.test_gpu_efficiency import circuit


@pytest.fixture
def capture_backend(monkeypatch):
    original = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kw: original(fn, backend="eager", fullgraph=True, dynamic=False),
    )


@pytest.mark.parametrize(
    "constructor",
    [
        lang.CUBA_LIF,
        lang.COBA_LIF,
        lang.CUBA_ALIF,
        lang.COBA_ALIF,
        lang.CUBA_ADEX,
        lang.COBA_ADEX,
    ],
)
@pytest.mark.parametrize("block_steps", [4, 16])
def test_blocks_match_reference_trajectories_gradients_adam_and_continuation(
    constructor, block_steps, capture_backend
):
    graph, parameter = circuit(constructor)
    results = []
    for size in (1, block_steps):
        model = GraphExecutor(
            plan_graph(graph),
            trainable_parameters=(parameter,),
            compile_mode="default" if size > 1 else None,
            compile_block_steps=size,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        inputs = torch.ones(41, 2, 2, requires_grad=True)
        result = model({"events": inputs}, diagnostics=False)
        assert result.outputs["spikes"].count_nonzero() > 0
        result.outputs["voltage"].sum().backward()
        gradient = model.parameter_map()[parameter].grad.clone()
        assert gradient.count_nonzero() > 0
        results.append((result, gradient, inputs.grad.clone(), optimizer))
        with torch.no_grad():
            first = model({"events": inputs[:19].detach()}, diagnostics=False)
            tail = model(
                {"events": inputs[19:].detach()},
                diagnostics=False,
                runtime_state=first.runtime_state,
            )
        torch.testing.assert_close(
            torch.cat((first.outputs["voltage"], tail.outputs["voltage"])),
            result.outputs["voltage"],
            atol=2e-5,
            rtol=2e-5,
        )
        optimizer.step()
        model.enforce_constraints()
    eager, actual = results
    assert actual[0].metrics["execution_path"]["selected"] == "compiled_blocks"
    for name in eager[0].outputs:
        torch.testing.assert_close(
            actual[0].outputs[name], eager[0].outputs[name], atol=2e-5, rtol=2e-5
        )
    for expected, value in zip(eager[1:3], actual[1:3], strict=True):
        torch.testing.assert_close(value, expected, atol=2e-5, rtol=2e-5)
    for expected, value in zip(
        eager[3].state.values(), actual[3].state.values(), strict=True
    ):
        for name in expected:
            torch.testing.assert_close(
                value[name], expected[name], atol=2e-5, rtol=2e-5
            )
    for name, expected in eager[0].model.parameter_map().items():
        torch.testing.assert_close(
            actual[0].model.parameter_map()[name], expected, atol=2e-5, rtol=2e-5
        )
    for group in (
        "voltages",
        "refractory",
        "conductances",
        "currents",
        "population_histories",
        "custom_state",
    ):
        for name, expected in getattr(eager[0].runtime_state, group).items():
            torch.testing.assert_close(
                getattr(actual[0].runtime_state, group)[name],
                expected,
                atol=2e-5,
                rtol=2e-5,
            )


def test_recording_falls_back_with_reason(capture_backend):
    graph, _ = circuit(lang.CUBA_LIF)
    result = GraphExecutor(plan_graph(graph), compile_block_steps=4)(
        {"events": torch.ones(9, 2, 2)},
        diagnostics=False,
        recording=RecordingSpec((SignalRecording("cells.voltage"),), block_steps=4),
    )
    assert result.metrics["execution_path"]["selected"] == "kernel_loop"
    assert (
        "online reductions or recording"
        in result.metrics["execution_path"]["fallback_reasons"]
    )
    assert result.recorded_signals["cells.voltage"].shape == (9, 2, 3)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_large_mps_blocks_use_documented_kernel_loop(capture_backend):
    graph, parameter = circuit(lang.CUBA_LIF)
    model = GraphExecutor(
        plan_graph(graph), compile_block_steps=16, trainable_parameters=(parameter,)
    ).to("mps")
    result = model({"events": torch.ones(13, 2, 2, device="mps")}, diagnostics=False)
    assert result.metrics["execution_path"]["selected"] == "kernel_loop"
    assert (
        "MPS block size above validated limit (4)"
        in result.metrics["execution_path"]["fallback_reasons"]
    )
    result.outputs["voltage"].sum().backward()
    assert torch.isfinite(model.parameter_map()[parameter].grad).all()


@pytest.mark.parametrize("value", [0, -1, 65, True, 1.5])
def test_invalid_block_sizes(value):
    with pytest.raises(ValueError, match="compile_block_steps"):
        ExecutionSpec(kind="build", compile_block_steps=value)
    graph, _ = circuit(lang.CUBA_LIF)
    with pytest.raises(ValueError, match="compile_block_steps"):
        GraphExecutor(plan_graph(graph), compile_block_steps=value)


def test_cli_block_size():
    spec = execution_spec_from_args(parse_args(["sim", "--compile-block-steps", "16"]))
    assert spec.compile_block_steps == 16


def test_checkpoint_protocol_binds_block_policy_and_preserves_default_identity(
    capture_backend, tmp_path
):
    from dataclasses import replace

    from snnlab.sim import DenseArrayBinding
    from snnlab.sim.training.trainer import train
    from tests.sim._execution_builders import direct_train_bundle

    bundle = direct_train_bundle()
    spec = ExecutionSpec(
        kind="train",
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(DenseArrayBinding("events", torch.ones(3, 2, 2)),),
        targets={"label": torch.tensor([0, 1])},
        device="cpu",
        diagnostics=False,
        epochs=2,
        updates=1,
        compile_block_steps=4,
        save_final_checkpoint=tmp_path / "checkpoint",
    )
    first = train(spec)
    assert (
        first.training_checkpoint.execution_protocol["network_blocks"]["block_steps"]
        == 4
    )
    with pytest.raises(ValueError, match="execution protocol"):
        train(
            replace(
                spec,
                checkpoint=tmp_path / "checkpoint",
                compile_block_steps=8,
                save_final_checkpoint=None,
            )
        )
    legacy = train(replace(spec, compile_block_steps=1, save_final_checkpoint=None))
    assert "network_blocks" not in legacy.training_checkpoint.execution_protocol


def test_partial_output_dependencies_keep_unused_parameter_gradients_none(
    capture_backend,
):
    net = lang.Network("partial-loss", dt=1 * lang.ms)
    source = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cells = net.population("cells", size=3, neuron=lang.CUBA_LIF())
    net.connect(
        source,
        cells.excitatory,
        name="drive",
        synapse=lang.ExponentialCurrent(tau=3 * lang.ms),
        weight=lang.Constant(2),
    )
    scores = lang.MeanVoltage(source=cells.spikes, classes=2, name="readout")
    net.output("voltage", cells.voltage)
    net.output("class_scores", scores)
    graph = lang.compile(net).graph
    parameters = tuple(row["id"] for row in graph["parameters"])
    model = GraphExecutor(
        plan_graph(graph), compile_block_steps=4, trainable_parameters=parameters
    )
    result = model({"events": torch.ones(13, 2, 2)}, diagnostics=False)
    result.outputs["voltage"].sum().backward()
    assert (
        "partial parameter dependencies in retained signals"
        in result.metrics["execution_path"]["fallback_reasons"]
    )
    readout = next(name for name in parameters if "readout" in name)
    assert model.parameter_map()[readout].grad is None


def test_multiple_input_gradients_preserve_unused_leaf_none(capture_backend):
    net = lang.Network("partial-inputs", dt=1 * lang.ms)
    for name in ("a", "b"):
        source = net.input(
            name, shape=("time", "batch", 1), signal_type="spikes", unit="spike"
        )
        cells = net.population(f"cells_{name}", size=1, neuron=lang.CUBA_LIF())
        net.connect(
            source,
            cells.excitatory,
            name=f"drive_{name}",
            synapse=lang.ExponentialCurrent(tau=3 * lang.ms),
            weight=lang.Constant(1),
        )
    net.output("voltage", net._signal("cells_a.voltage"))
    model = GraphExecutor(plan_graph(lang.compile(net).graph), compile_block_steps=4)
    inputs = {name: torch.ones(9, 1, 1, requires_grad=True) for name in ("a", "b")}
    result = model(inputs, diagnostics=False)
    assert (
        "multiple inputs with input gradients"
        in result.metrics["execution_path"]["fallback_reasons"]
    )
    result.outputs["voltage"].sum().backward()
    assert inputs["a"].grad is not None
    assert inputs["b"].grad is None


@pytest.mark.parametrize("pooled", [False, True])
def test_spatial_delayed_inputs_gradients_and_continuation(pooled, capture_backend):
    from tests.sim.test_spatial import circuit as spatial_circuit

    net, projection = spatial_circuit(delay=3, pooled=pooled)
    graph = lang.compile(net).graph
    results = []
    for size in (1, 4):
        model = GraphExecutor(
            plan_graph(graph),
            compile_mode=None if size == 1 else "default",
            compile_block_steps=size,
            trainable_parameters=(projection.weight.id,),
        )
        values = torch.linspace(0, 1, 11 * 2 * 40).reshape(11, 2, 40).requires_grad_()
        result = model({"events": values}, diagnostics=False)
        result.outputs["filtered"].square().sum().backward()
        results.append(
            (result, values.grad, model.parameter_map()[projection.weight.id].grad)
        )
        with torch.no_grad():
            first = model({"events": values[:2].detach()}, diagnostics=False)
            tail = model(
                {"events": values[2:].detach()},
                diagnostics=False,
                runtime_state=first.runtime_state,
            )
        torch.testing.assert_close(
            torch.cat((first.outputs["filtered"], tail.outputs["filtered"])),
            result.outputs["filtered"],
            atol=2e-5,
            rtol=2e-5,
        )
    for name in results[0][0].outputs:
        torch.testing.assert_close(
            results[0][0].outputs[name],
            results[1][0].outputs[name],
            atol=2e-5,
            rtol=2e-5,
        )
    for expected, actual in zip(results[0][1:], results[1][1:], strict=True):
        torch.testing.assert_close(expected, actual, atol=2e-5, rtol=2e-5)


def test_modulation_uses_absolute_arrival_time_across_blocks(capture_backend):
    from tests.sim.test_weight_modulation import network

    net, projection = network(lang.RampGain(end=8 * lang.ms), delay=3 * lang.ms)
    results = []
    for size in (1, 4):
        model = GraphExecutor(
            plan_graph(lang.compile(net).graph),
            compile_block_steps=size,
            compile_mode=None if size == 1 else "default",
        )
        with torch.no_grad():
            # Direct call with gradients enabled is also eligible. Avoid online
            # inference here so this checks modulation in the block function.
            first, _ = model._forward(
                {"events": torch.ones(5, 1, 1)}, diagnostics=False
            )
            tail, _ = model._forward(
                {"events": torch.ones(6, 1, 1)},
                diagnostics=False,
                runtime_state=first.runtime_state,
            )
        results.append(torch.cat((first.outputs["drive"], tail.outputs["drive"])))
    torch.testing.assert_close(*results, atol=2e-5, rtol=2e-5)


def test_maximum_ties_keep_first_timestep_gradient(capture_backend):
    graph, parameter = circuit(lang.CUBA_LIF)
    gradients = []
    for size in (1, 4):
        model = GraphExecutor(
            plan_graph(graph),
            compile_block_steps=size,
            compile_mode=None if size == 1 else "default",
            trainable_parameters=(parameter,),
        )
        inputs = torch.zeros(13, 2, 2, requires_grad=True)
        result = model({"events": inputs}, diagnostics=False)
        assert torch.equal(result.outputs["voltage"], torch.full((13, 2, 3), -65.0))
        result.outputs["voltage"].max(dim=0).values.sum().backward()
        gradients.append(inputs.grad)
        assert inputs.grad[0].count_nonzero() > 0
        assert inputs.grad[1:].count_nonzero() == 0
    torch.testing.assert_close(*gradients, atol=0, rtol=0)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_actual_inductor_block_forward_backward(device):
    graph, parameter = circuit(lang.CUBA_LIF)
    results = []
    for size in (1, 4):
        model = GraphExecutor(
            plan_graph(graph),
            compile_block_steps=size,
            compile_mode=None if size == 1 else "default",
            trainable_parameters=(parameter,),
        ).to(device)
        result = model(
            {"events": torch.ones(13, 2, 2, device=device)}, diagnostics=False
        )
        wrapper = result.metrics["compiler_wrapper"]
        assert wrapper["selected"] == (
            "eager" if size == 1 else "python" if device == "mps" else "cpp"
        )
        if size > 1 and device == "mps":
            assert "get_mps_lib" in wrapper["reason"]
        result.outputs["voltage"].sum().backward()
        results.append(
            (result.outputs["voltage"], model.parameter_map()[parameter].grad)
        )
    for expected, actual in zip(*results, strict=True):
        torch.testing.assert_close(expected, actual, atol=3e-4, rtol=3e-4)


def projection_batching_circuit(constructor, delay, feedback):
    net = lang.Network("projection-batching", dt=0.5 * lang.ms)
    source = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cells = net.population("cells", size=3, neuron=constructor())
    readouts = [
        net.population(name, size=2, neuron=constructor())
        for name in ("readout", "other")
    ]
    synapse = (
        lang.ExponentialCurrent if cells.neuron.kind.startswith("cuba_") else lang.AMPA
    )(tau=3 * lang.ms)
    net.connect(
        source,
        cells.excitatory,
        name="drive",
        synapse=synapse,
        delay=delay * lang.ms,
        weight=lang.Constant(0.8),
    )
    net.connect(
        cells.spikes,
        cells.excitatory,
        name="recurrence",
        synapse=synapse,
        connection="recurrent",
        delay=0.5 * lang.ms,
        weight=lang.Constant(0.1),
    )
    for readout in readouts:
        edge = net.connect(
            cells.spikes,
            readout.excitatory,
            name=f"to_{readout.id}",
            synapse=synapse,
            delay=delay * lang.ms,
            weight=lang.Constant(0.3),
        )
        net.output(
            f"{readout.id}_current",
            edge.current if cells.neuron.kind.startswith("cuba_") else edge.conductance,
        )
    if feedback:
        net.connect(
            readouts[0].spikes,
            cells.excitatory,
            name="feedback",
            synapse=synapse,
            delay=0.5 * lang.ms,
            weight=lang.Constant(0.1),
        )
    for pop in [cells, *readouts]:
        net.output(f"{pop.id}_voltage", pop.voltage)
        net.output(f"{pop.id}_spikes", pop.spikes)
    return lang.compile(net).graph


@pytest.mark.parametrize("constructor", [lang.CUBA_LIF, lang.COBA_ALIF])
@pytest.mark.parametrize("delay", [0, 1, 3])
@pytest.mark.parametrize("feedback", [False, True])
@pytest.mark.parametrize("all_gradients", [False, True])
def test_window_projection_batching_gradients_delays_and_continuation(
    constructor, delay, feedback, all_gradients, capture_backend
):
    graph = projection_batching_circuit(constructor, delay, feedback)
    if all_gradients:
        graph["outputs"] = [
            row for row in graph["outputs"] if row["id"].startswith("readout_")
        ]
    trainable = tuple(
        p["id"]
        for p in graph["parameters"]
        if (all_gradients and p["id"] != "to_other.weight") or p["id"] == "drive.weight"
    )
    results = []
    for block_steps in (1, 8):
        model = GraphExecutor(
            plan_graph(graph),
            compile_block_steps=block_steps,
            compile_mode="default" if block_steps > 1 else None,
            trainable_parameters=trainable,
        )
        inputs = torch.linspace(0.1, 1.9, 116).reshape(29, 2, 2).requires_grad_()
        result = model({"events": inputs}, diagnostics=False)
        if block_steps > 1:
            assert result.metrics["execution_path"]["selected"] == "compiled_blocks"
        loss = sum(value.sum() for value in result.outputs.values())
        loss.backward()
        results.append((result, inputs.grad, [p.grad for p in model.parameters()]))
        with torch.no_grad():
            first = model({"events": inputs[:3]}, diagnostics=False)
            tail = model(
                {"events": inputs[3:]},
                diagnostics=False,
                runtime_state=first.runtime_state,
            )
        for name, expected in result.outputs.items():
            torch.testing.assert_close(
                torch.cat((first.outputs[name], tail.outputs[name])),
                expected,
                atol=3e-5,
                rtol=3e-5,
            )
    expected, actual = results
    for name in expected[0].outputs:
        torch.testing.assert_close(
            actual[0].outputs[name], expected[0].outputs[name], atol=3e-5, rtol=3e-5
        )
    torch.testing.assert_close(actual[1], expected[1], atol=3e-5, rtol=3e-5)
    for a, b in zip(actual[2], expected[2], strict=True):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)


def test_window_projection_batching_matrix_call_count(capture_backend):
    graph = projection_batching_circuit(lang.CUBA_LIF, 0, False)
    model = GraphExecutor(
        plan_graph(graph), compile_mode="default", compile_block_steps=10
    )
    inputs = {"events": torch.ones(10, 1, 2)}
    with torch.no_grad():
        model(inputs, diagnostics=False)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            result = model(inputs, diagnostics=False)
    assert result.metrics["execution_path"]["selected"] == "compiled_blocks"
    counts = {row.key: row.count for row in profile.key_averages()}
    # Ten recurrent products plus one input and two terminal readout products.
    assert counts.get("aten::mm", 0) == 13


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_actual_inductor_batched_readout_forward_backward(device):
    graph = projection_batching_circuit(lang.CUBA_LIF, 1, False)
    graph["outputs"] = [
        row for row in graph["outputs"] if row["id"].startswith("readout_")
    ]
    results = []
    for steps in (1, 4):
        model = GraphExecutor(
            plan_graph(graph),
            compile_mode="default" if steps > 1 else None,
            compile_block_steps=steps,
            trainable_parameters=tuple(
                p["id"] for p in graph["parameters"] if p["id"] != "to_other.weight"
            ),
        ).to(device)
        values = torch.ones(17, 2, 2, device=device, requires_grad=True)
        result = model({"events": values}, diagnostics=False)
        if steps > 1:
            assert result.metrics["execution_path"]["selected"] == "compiled_blocks"
        sum(value.sum() for value in result.outputs.values()).backward()
        results.append(
            (result.outputs, values.grad, [p.grad for p in model.parameters()])
        )
    expected, actual = results
    for name in expected[0]:
        torch.testing.assert_close(
            actual[0][name], expected[0][name], atol=3e-5, rtol=3e-5
        )
    torch.testing.assert_close(actual[1], expected[1], atol=3e-5, rtol=3e-5)
    for a, b in zip(actual[2], expected[2], strict=True):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("feedback_mode", ["enabled", "disabled", "self"])
def test_batched_projection_gains_and_readout_dependencies(
    feedback_mode, capture_backend
):
    graph = projection_batching_circuit(lang.CUBA_LIF, 1, True)
    for row in graph["projections"]:
        row["modulation"] = {
            "kind": "ramp_gain",
            "start_ms": 0.5,
            "end_ms": 5.0,
            "before": 0.2,
            "after": 1.7,
        }
        if row["id"] == "feedback":
            if feedback_mode == "disabled":
                row["enabled"] = False
            elif feedback_mode == "self":
                row["target"] = "readout.excitatory"
                row["connection"] = "recurrent"
                parameter = next(
                    p for p in graph["parameters"] if p["id"] == row["parameters"][0]
                )
                parameter["shape"] = [2, 2]
    values = torch.ones(19, 2, 2)
    results = []
    with torch.no_grad():
        for size in (1, 4):
            model = GraphExecutor(
                plan_graph(graph),
                compile_mode="default" if size > 1 else None,
                compile_block_steps=size,
            )
            first = model({"events": values[:7]}, diagnostics=False)
            tail = model(
                {"events": values[7:]},
                diagnostics=False,
                runtime_state=first.runtime_state,
            )
            if size > 1:
                assert tail.metrics["execution_path"]["selected"] == "compiled_blocks"
            results.append((first, tail))
    for expected, actual in zip(*results, strict=True):
        for name in expected.outputs:
            torch.testing.assert_close(
                actual.outputs[name], expected.outputs[name], atol=3e-5, rtol=3e-5
            )
        for group in (
            "voltages",
            "currents",
            "population_histories",
            "input_histories",
        ):
            for name, state in getattr(expected.runtime_state, group).items():
                torch.testing.assert_close(
                    getattr(actual.runtime_state, group)[name],
                    state,
                    atol=3e-5,
                    rtol=3e-5,
                )
