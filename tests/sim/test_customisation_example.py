"""Validate the actual documented custom callbacks against native execution."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from snnlab import extensions, lang, sim

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/customisation"


@pytest.fixture
def callbacks(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    monkeypatch.setattr(extensions, "_REGISTRY", {})
    spec = importlib.util.spec_from_file_location(
        "customisation_example", EXAMPLE / "example.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.register_definitions()
    return module


def bundle(*, custom_synapse=True, adaptation=0.35, tau_ms=5.0, tau_adapt_ms=100):
    net = lang.Network("customisation_test", dt=0.5 * lang.ms)
    source = net.input(
        "inputs", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    standard = net.population("standard", size=3, neuron=lang.CUBA_LIF())
    adaptive = net.population(
        "adaptive",
        size=3,
        neuron=lang.CustomNeuron(
            "example.adaptive_lif/v1",
            tau_mem_ms=20,
            tau_adapt_ms=tau_adapt_ms,
            adaptation_na=adaptation,
        ),
    )
    shared = net.parameter(
        "weights", shape=(3, 2), unit="nA", initializer=lang.Constant(5.0)
    )
    for cells in (standard, adaptive):
        synapse = (
            lang.CustomSynapse("example.exponential_current/v1", tau_ms=tau_ms)
            if custom_synapse
            else lang.ExponentialCurrent(tau=5 * lang.ms)
        )
        projection = net.connect(
            source,
            cells.excitatory,
            name=f"input_to_{cells.id}",
            synapse=synapse,
            weight=shared,
        )
        net.output(f"{cells.id}_spikes", cells.spikes)
        net.output(f"{cells.id}_voltage", cells.voltage)
        net.expose(projection.current, name=f"{cells.id}_current")
    net.expose(adaptive.state("adaptation"), name="adaptation")
    return lang.compile(net)


def spikes():
    return (
        torch.rand(64, 2, 2, generator=torch.Generator().manual_seed(19)) < 0.25
    ).float()


def simulate(compiled, values, *, compile_mode=None, runtime_state=None):
    return sim.simulate(
        graph=compiled.graph,
        input_bindings=(sim.DenseArrayBinding("inputs", values),),
        device="cpu",
        runtime_state=runtime_state,
        extension_validation="strict",
        performance=sim.Performance(compile_mode=compile_mode, precision="highest"),
    )


def compare(actual, expected):
    for name, value in expected.outputs.items():
        torch.testing.assert_close(actual.outputs[name], value, atol=1e-4, rtol=1e-5)
    for name, value in expected.diagnostics.items():
        torch.testing.assert_close(
            actual.diagnostics[name], value, atol=1e-4, rtol=1e-5
        )


def test_custom_synapse_matches_builtin_and_zero_adaptation_matches_lif(callbacks):
    values = spikes()
    custom = simulate(bundle(adaptation=0), values)
    reference = simulate(bundle(custom_synapse=False, adaptation=0), values)
    compare(custom, reference)
    torch.testing.assert_close(
        custom.outputs["standard_spikes"],
        custom.outputs["adaptive_spikes"],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        custom.outputs["standard_voltage"],
        custom.outputs["adaptive_voltage"],
        atol=1e-4,
        rtol=1e-5,
    )
    assert custom.outputs["standard_spikes"].sum() > 0


def test_compiled_callbacks_preserve_outputs_gradients_and_continuation(callbacks):
    compiled = bundle()
    eager_inputs = spikes().requires_grad_()
    compiled_inputs = eager_inputs.detach().clone().requires_grad_()
    expected = simulate(compiled, eager_inputs)
    actual = simulate(compiled, compiled_inputs, compile_mode="default")
    compare(actual, expected)
    gradients = [
        torch.autograd.grad(
            sum(value.sum() for value in result.outputs.values()), values
        )[0]
        for result, values in ((actual, compiled_inputs), (expected, eager_inputs))
    ]
    torch.testing.assert_close(gradients[0], gradients[1], atol=1e-3, rtol=1e-4)
    assert gradients[0].abs().sum() > 0
    values = spikes()
    first = simulate(compiled, values[:32], compile_mode="default")
    second = simulate(
        compiled, values[32:], compile_mode="default", runtime_state=first.runtime_state
    )
    for name, value in expected.outputs.items():
        torch.testing.assert_close(
            torch.cat((first.outputs[name], second.outputs[name])),
            value,
            atol=1e-4,
            rtol=1e-5,
        )
    assert second.runtime_state.completed_steps == 64


@pytest.mark.parametrize("precision", ["highest", "auto"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_documented_callbacks_replay_on_cuda_with_stable_state_dtype(
    callbacks, precision
):
    compiled = bundle()
    model = sim.build(
        graph=compiled.graph,
        device="cuda:0",
        performance=sim.Performance(precision=precision),
    ).model
    values = spikes().cuda()
    for _ in range(3):
        actual = model({"inputs": values}, diagnostics=False)
    assert actual.metrics["cuda_graphs"]["captures"] == 1
    assert actual.metrics["cuda_graphs"]["replayed_calls"] == 1
    saved = {name: value.clone() for name, value in actual.outputs.items()}
    changed = model({"inputs": values * 0}, diagnostics=False)
    model.cuda_graphs = "off"
    reference = model({"inputs": values * 0}, diagnostics=False)
    for name, value in reference.outputs.items():
        torch.testing.assert_close(changed.outputs[name], value)
        torch.testing.assert_close(actual.outputs[name], saved[name])
        assert value.dtype == torch.float32
    assert actual.metrics["execution_path"]["selected"] == "kernel_loop"
    assert (
        "registered neuron, synapse or surrogate"
        in actual.metrics["execution_path"]["fallback_reasons"]
    )


def test_callback_configuration_is_validated_before_stepping(callbacks):
    with pytest.raises(ValueError, match="tau_ms"):
        sim.build(
            graph=bundle(tau_ms=0).graph, performance=sim.Performance(compile_mode=None)
        )
    with pytest.raises(ValueError, match="tau_adapt_ms"):
        sim.build(
            graph=bundle(tau_adapt_ms=-1).graph,
            performance=sim.Performance(compile_mode=None),
        )
