"""Declared outputs and exposed diagnostics have independent retention."""

import pytest
import torch

from snnlab import lang
from snnlab.sim.execution import DenseArrayBinding, ExecutionSpec, simulate


def _network():
    net = lang.Network("diagnostics")
    inputs = net.input(
        "inputs", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cells = net.population("E", size=2, neuron=lang.COBA_LIF(tau_mem=20 * lang.ms))
    projection = net.connect(
        inputs,
        cells.excitatory,
        name="input_to_E",
        synapse=lang.AMPA(tau=2 * lang.ms),
        weight=lang.Constant(10),
    )
    net.output("spikes", cells.spikes)
    return net, inputs, cells, projection


def _run(net, **options):
    return simulate(
        ExecutionSpec(
            kind="simulate",
            graph=lang.compile(net, target="tools/snnsim").graph,
            input_bindings=(DenseArrayBinding("inputs", torch.ones(5, 1, 2)),),
            device="cpu",
            seed=3,
            **options,
        )
    )


def test_no_implicit_diagnostics_and_no_old_result_attribute():
    net, _, _, _ = _network()
    result = _run(net)
    assert result.outputs["spikes"].shape == (5, 1, 2)
    assert result.diagnostics == {}
    assert not hasattr(result, "recordings")


def test_exposed_voltage_conductance_and_input_return_by_name():
    net, inputs, cells, projection = _network()
    net.expose(inputs, name="stimulus")
    net.expose(cells.voltage, name="voltage")
    net.expose(projection.conductance, name="conductance")
    net.expose(cells.spikes, name="spike_diagnostic")
    enabled = _run(net)
    disabled = _run(net, diagnostics=False)
    assert set(enabled.diagnostics) == {
        "stimulus",
        "voltage",
        "conductance",
        "spike_diagnostic",
    }
    assert all(value.shape == (5, 1, 2) for value in enabled.diagnostics.values())
    assert all(not value.requires_grad for value in enabled.diagnostics.values())
    assert torch.equal(enabled.diagnostics["stimulus"], torch.ones(5, 1, 2))
    assert torch.count_nonzero(enabled.diagnostics["conductance"]) > 0
    assert torch.equal(
        enabled.outputs["spikes"], enabled.diagnostics["spike_diagnostic"]
    )
    assert disabled.diagnostics == {}
    assert torch.equal(enabled.outputs["spikes"], disabled.outputs["spikes"])


def test_exposed_operation_is_resolved_by_signal():
    net, inputs, _, _ = _network()
    scores = lang.readouts.SpikeCount(source=inputs, classes=2, name="scores")
    net.output("scores_output", scores)
    net.expose(scores, name="scores_diagnostic")
    result = _run(net)
    assert torch.equal(
        result.outputs["scores_output"], result.diagnostics["scores_diagnostic"]
    )


@pytest.mark.parametrize("value", ["full", 1, None])
def test_diagnostics_requires_boolean(value):
    net, _, _, _ = _network()
    with pytest.raises(TypeError, match="diagnostics must be boolean"):
        _run(net, diagnostics=value)
