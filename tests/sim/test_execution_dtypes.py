"""Floating runtime state, recurrent spikes and continuation follow model dtype."""

import pytest
import torch

from snnlab import lang
from snnlab.sim import GraphExecutor, MeasurementWindow, plan_graph
from snnlab.sim.graph.dynamics import fast_sigmoid_spike
from tests.sim.test_execution_extensions import custom_network
from tests.sim.test_gpu_efficiency import circuit

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA unavailable"
        ),
    ),
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="MPS unavailable"
        ),
    ),
]


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_surrogate_preserves_dtype_and_hard_spikes(dtype, scale):
    values = torch.tensor([-2, -0.1, 0, 0.1, 2], dtype=dtype, requires_grad=True)
    spikes = fast_sigmoid_spike(values, 5, scale)
    assert spikes.dtype == dtype
    torch.testing.assert_close(spikes, (values >= 0).to(dtype), atol=0, rtol=0)
    spikes.sum().backward()
    assert values.grad.dtype == dtype
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
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
def test_recurrent_state_gradients_and_split_continuation(constructor, dtype, device):
    if device == "mps" and dtype == torch.float64:
        pytest.skip("MPS does not support float64 tensors")
    graph, parameter = circuit(constructor)
    model = GraphExecutor(
        plan_graph(graph), compile_mode=None, trainable_parameters=(parameter,)
    ).to(device=device, dtype=dtype)
    values = torch.ones(41, 2, 2, device=device, dtype=dtype)
    result = model({"events": values}, diagnostics=False)
    assert result.outputs["spikes"].count_nonzero() > 0
    result.outputs["voltage"].sum().backward()
    gradient = model.parameter_map()[parameter].grad
    assert gradient.dtype == dtype and torch.isfinite(gradient).all()
    assert gradient.count_nonzero() > 0
    with torch.no_grad():
        first = model({"events": values[:17]}, diagnostics=False)
        tail = model(
            {"events": values[17:]},
            diagnostics=False,
            runtime_state=first.runtime_state,
        )
    for name, output in result.outputs.items():
        assert output.dtype == dtype
        if output.ndim == 3:
            torch.testing.assert_close(
                torch.cat((first.outputs[name], tail.outputs[name])),
                output,
                atol=0,
                rtol=0,
            )
    for group in (
        result.runtime_state.voltages,
        result.runtime_state.conductances,
        result.runtime_state.currents,
        result.runtime_state.population_histories,
        result.runtime_state.custom_state,
    ):
        for name, value in group.items():
            assert value.dtype == (
                torch.int64 if name.endswith("/refractory") else dtype
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_custom_callbacks_receive_model_dtype(dtype, monkeypatch):
    from snnlab import extensions as E

    monkeypatch.setattr(E, "_REGISTRY", {})
    net, _, _, projection = custom_network()
    model = GraphExecutor(
        plan_graph(lang.compile(net).graph),
        compile_mode=None,
        trainable_parameters=(projection.weight.id,),
    ).to(dtype=dtype)
    result = model({"inputs": torch.ones(13, 2, 1, dtype=dtype)}, diagnostics=False)
    result.outputs["spikes"].sum().backward()
    assert all(value.dtype == dtype for value in result.outputs.values())


def test_parameters_on_wrong_device_give_clear_error():
    graph, _ = circuit(lang.CUBA_LIF)
    model = GraphExecutor(plan_graph(graph), compile_mode=None).to("meta")
    with pytest.raises(ValueError, match="parameter .* execution device"):
        model({"events": torch.ones(3, 1, 2)})


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_online_reductions_keep_output_dtype(dtype):
    graph, _ = circuit(lang.CUBA_LIF)
    model = GraphExecutor(plan_graph(graph), compile_mode=None).to(dtype=dtype)
    with torch.no_grad():
        result = model(
            {"events": torch.ones(41, 2, 2, dtype=dtype)},
            diagnostics=False,
            measurement=MeasurementWindow(0, 41),
        )
    assert result.outputs["class_scores"].dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_online_low_precision_sum_does_not_stall_at_mantissa_limit(dtype):
    net = lang.Network("sum-dtype", dt=1 * lang.ms)
    source = net.input(
        "values", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    total = lang.reduce(source, operation="sum", over="time", name="total_op")
    net.output("total", total)
    model = GraphExecutor(plan_graph(lang.compile(net).graph), compile_mode=None)
    with torch.no_grad():
        result = model(
            {"values": torch.ones(4096, 1, 1, dtype=dtype)}, diagnostics=False
        )
    assert result.outputs["total"].dtype == dtype
    assert result.outputs["total"].item() == 4096


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_measurement_continues_from_portable_cpu_runtime_state():
    graph, _ = circuit(lang.CUBA_LIF)
    model = GraphExecutor(plan_graph(graph), compile_mode=None).to("mps")
    values = torch.ones(41, 2, 2, device="mps")
    window = MeasurementWindow(0, 41)
    with torch.no_grad():
        full = model({"events": values}, measurement=window, diagnostics=False)
        first = model({"events": values[:17]}, measurement=window, diagnostics=False)
        restored = first.runtime_state.detached(device="cpu")
        tail = model(
            {"events": values[17:]},
            measurement=window,
            runtime_state=restored,
            diagnostics=False,
        )
    torch.testing.assert_close(
        tail.outputs["class_scores"], full.outputs["class_scores"], atol=0, rtol=0
    )
