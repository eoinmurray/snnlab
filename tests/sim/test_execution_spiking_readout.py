"""Regression coverage for checkpoint-backed graph spike-count inference."""

import math

import pytest
import torch

from snnlab import lang as snn
from snnlab.sim.execution import (
    DenseArrayBinding,
    ExecutionSpec,
    build,
    simulate,
)
from tests.sim._execution_builders import direct_train_bundle


def test_native_pytorch_checkpoint_restores_parameters_and_rejects_old_keys(tmp_path):
    bundle = direct_train_bundle()
    built = build(ExecutionSpec(kind="build", graph=bundle.graph, seed=7))
    checkpoint = tmp_path / "parameters.pt"
    torch.save(built.model.state_dict(), checkpoint)
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            graph=bundle.graph,
            seed=99,
            checkpoint=checkpoint,
            input_bindings=(DenseArrayBinding("events", torch.zeros(3, 1, 2)),),
        )
    )
    for name, expected in built.model.parameter_map().items():
        torch.testing.assert_close(result.parameters[name], expected, rtol=0, atol=0)
    assert result.metrics["checkpoint"]["format"] == "graph_torch_state_dict"
    torch.save({"W_ff.0": torch.zeros(2, 2)}, checkpoint)
    with pytest.raises(RuntimeError, match="state_dict"):
        simulate(
            ExecutionSpec(
                kind="simulate",
                graph=bundle.graph,
                checkpoint=checkpoint,
                input_bindings=(DenseArrayBinding("events", torch.zeros(3, 1, 2)),),
            )
        )


@pytest.mark.parametrize("spiking", [False, True])
def test_soft_reset_integrator_records_spikes_only_when_declared(spiking):
    net = snn.Network("readout", dt=0.1 * snn.ms)
    source = net.input(
        "drive", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    cell = net.population(
        "out",
        size=1,
        spiking=spiking,
        neuron=snn.LeakyIntegrator(tau=2 * snn.ms, soft_reset_threshold=1.0),
    )
    net.connect(
        source,
        cell.excitatory,
        name="input",
        weight=snn.Constant(1.0),
        synapse=snn.LeakyIntegrator(tau=2 * snn.ms),
    )
    net.expose(cell.voltage, name="out.voltage")
    if spiking:
        net.expose(cell.spikes, name="out.spikes")
    graph = snn.compile(net).graph
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            input_bindings=(DenseArrayBinding("drive", torch.ones(8, 1, 1)),),
        )
    )
    voltage = 0.0
    expected_spikes, expected_voltage = [], []
    beta = math.exp(-0.1 / 2)
    for _ in range(8):
        voltage = beta * voltage + (1 - beta) / 0.1
        spike = float(voltage > 1.0)
        voltage -= spike
        expected_spikes.append(spike if spiking else 0.0)
        expected_voltage.append(voltage)
    if spiking:
        torch.testing.assert_close(
            result.diagnostics["out.spikes"].flatten(), torch.tensor(expected_spikes)
        )
    else:
        assert "out.spikes" not in result.diagnostics
    torch.testing.assert_close(
        result.diagnostics["out.voltage"].flatten(), torch.tensor(expected_voltage)
    )
