"""Population-specific COBA thresholds in the native graph executor."""

import torch

from snnlab import lang as snn
from snnlab.sim.execution import DenseArrayBinding, ExecutionSpec, simulate


def test_coba_threshold_is_population_specific():
    net = snn.Network("thresholds", dt=0.1 * snn.ms)
    drive = net.input(
        "drive", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    for name, threshold in (("low", -50.0), ("high", -48.0)):
        cell = net.population(
            name,
            size=1,
            neuron=snn.COBA_LIF(
                tau_mem=20 * snn.ms,
                capacitance_nf=1.0,
                leak_us=0.05,
                threshold_mv=threshold,
            ),
        )
        net.connect(
            drive,
            cell.excitatory,
            name=f"input_{name}",
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=snn.Constant(2.85),
        )
        net.expose(cell.spikes, name=f"{name}.spikes")
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=snn.compile(net).graph,
            input_bindings=(DenseArrayBinding("drive", torch.ones(1, 1, 1)),),
        )
    )
    assert result.diagnostics["low.spikes"].item() == 1.0
    assert result.diagnostics["high.spikes"].item() == 0.0
