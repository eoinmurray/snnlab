"""Population-specific COBA thresholds in the native graph executor."""

import torch

from snnlab import lang as snn
from snnlab.sim.execution import ExecutionSpec, simulate


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
                initial_voltage_mv=-49.0,
            ),
        )
        net.connect(
            drive,
            cell.excitatory,
            name=f"input_{name}",
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=snn.Constant(0.0),
        )
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=snn.compile(net).graph,
            inputs={"drive": torch.zeros(1, 1, 1)},
            recording="full",
        )
    )
    assert result.recordings["low.spikes"].item() == 1.0
    assert result.recordings["high.spikes"].item() == 0.0
