"""Regression coverage for checkpoint-backed graph spike-count inference."""

import math

import pytest
import torch

from snnlab import lang as snn
from snnlab.lang.examples.build_examples import ping_classifier
from snnlab.sim.execution import (
    ExecutionSpec,
    build,
    export_legacy_parameters_v1,
    simulate,
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
    graph = snn.compile(net).graph
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs={"drive": torch.ones(8, 1, 1)},
            recording="full",
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
    torch.testing.assert_close(
        result.recordings["out.spikes"].flatten(), torch.tensor(expected_spikes)
    )
    torch.testing.assert_close(
        result.recordings["out.voltage"].flatten(), torch.tensor(expected_voltage)
    )


def test_graph_inference_restores_complete_legacy_checkpoint(tmp_path):
    graph = ping_classifier().graph
    built = build(ExecutionSpec(kind="build", executor="graph", graph=graph, seed=7))
    legacy = export_legacy_parameters_v1(graph, built.model.parameter_map())
    path = tmp_path / "weights.pth"
    torch.save(legacy.parameters, path)
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            seed=99,
            checkpoint=path,
            inputs={"image": torch.zeros(2, 1, 784)},
        )
    )
    for name, expected in built.model.parameter_map().items():
        torch.testing.assert_close(result.parameters[name], expected, rtol=0, atol=0)
    assert result.metrics["checkpoint"]["interchange"]["direction"] == "legacy_to_graph"
    incomplete = dict(legacy.parameters)
    incomplete.pop("W_ff.1")
    torch.save(incomplete, path)
    with pytest.raises(ValueError, match="requires exact keys"):
        simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=graph,
                checkpoint=path,
                inputs={"image": torch.zeros(2, 1, 784)},
            )
        )
