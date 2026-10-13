"""Independent snnTorch comparisons for current-based LIF dynamics.

Reference: https://snntorch.readthedocs.io/en/latest/snn.neurons_synaptic.html
Shift membrane voltage by the resting potential (mV), and scale current (nA)
by tau_mem / capacitance * (1 - beta). This gives snnTorch's discrete input.
Zero reset is equivalent when reset equals rest, with immediate reset and no
refractory period. Conductance dynamics and refractory timing are not covered.
Hard thresholds differ at exact equality (SNNLab >=, snnTorch >), so the
trajectory protocols avoid equality. Reset gradients differ too; gradient
comparisons deliberately use a subthreshold trajectory with reference reset
disabled, and a separate matched fast-sigmoid spike derivative.
"""

import math

import pytest
import torch

from snnlab import lang
from snnlab.sim import extensions as X
from snnlab.sim import models as M
from snnlab.sim.execution import (
    DenseArrayBinding,
    ExecutionSpec,
    simulate,
)

snn = pytest.importorskip("snntorch")
surrogate = snn.surrogate

pytestmark = [pytest.mark.integration, pytest.mark.snntorch]


@pytest.mark.parametrize("dt_ms", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("tau_ms, capacitance", [(20.0, 1.0), (5.0, 0.5)])
@pytest.mark.parametrize("spiking", [False, True], ids=["subthreshold", "spiking"])
def test_current_lif_voltage_and_spikes_match_snntorch(
    dt_ms, tau_ms, capacitance, spiking
):
    beta = math.exp(-dt_ms / tau_ms)
    gain = tau_ms / capacitance * (1 - beta)
    reference = snn.Leaky(
        beta=torch.tensor(beta, dtype=torch.float64),
        threshold=15.0,
        reset_mechanism="zero",
        reset_delay=False,
    ).double()
    config = lang.CUBA_LIF(tau_mem=tau_ms * lang.ms, capacitance_nf=capacitance).json()
    voltage = torch.tensor([[-61.0, -65.0], [-68.0, -63.0]], dtype=torch.float64)
    membrane = voltage + 65.0
    refractory = torch.zeros_like(voltage, dtype=torch.long)
    currents = torch.rand(
        (240, 2, 2), generator=torch.Generator().manual_seed(17), dtype=torch.float64
    )
    currents *= 5.0 if spiking else 0.1
    spike_count = 0
    for current in currents:
        voltage, spikes, refractory = X.current_lif(
            {"voltage": voltage, "refractory": refractory},
            current,
            torch.full_like(current, 0.02),
            dt_ms=dt_ms,
            config=config,
            spike_function=lambda value: (value >= 0).to(value.dtype),
        )
        ref_spikes, membrane = reference(gain * (current - 0.02), membrane)
        torch.testing.assert_close(voltage, membrane - 65.0, atol=1e-11, rtol=0)
        torch.testing.assert_close(
            spikes, ref_spikes, atol=0, rtol=0, check_dtype=False
        )
        spike_count += int(spikes.sum())
    assert (spike_count > 10) if spiking else (spike_count == 0)


@pytest.mark.parametrize("dt_ms", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("inhibitory", [False, True], ids=["excitation", "inhibition"])
def test_public_synaptic_network_matches_snntorch(dt_ms, inhibitory):
    """Check compiled execution, batch state, current decay, spikes and continuation."""
    net = lang.Network("snntorch-comparison", dt=dt_ms * lang.ms)
    inputs = net.input(
        "inputs", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cells = net.population("cells", size=3, neuron=lang.CUBA_LIF())
    projection = net.connect(
        inputs,
        cells.inhibitory if inhibitory else cells.excitatory,
        name="drive",
        synapse=lang.ExponentialCurrent(tau=3 * lang.ms),
        weight=lang.Constant(0.8),
        initialization_scaling="direct",
    )
    net.output("voltage", cells.voltage)
    net.output("spikes", cells.spikes)
    net.expose(projection.current, name="current")
    bundle = lang.compile(net)
    events = (
        torch.rand((120, 2, 2), generator=torch.Generator().manual_seed(41)) < 0.35
    ).float()

    def run(values, state=None):
        return simulate(
            ExecutionSpec(
                kind="simulate",
                graph=bundle.graph,
                device="cpu",
                input_bindings=(DenseArrayBinding("inputs", values),),
                runtime_state=state,
            )
        )

    result = run(events)
    first = run(events[:47])
    second = run(events[47:], first.runtime_state)
    for name in ("voltage", "spikes"):
        torch.testing.assert_close(
            torch.cat([first.outputs[name], second.outputs[name]]),
            result.outputs[name],
            atol=0,
            rtol=0,
        )

    beta = math.exp(-dt_ms / 20.0)
    gain = 20.0 * (1 - beta)
    reference = snn.Synaptic(
        alpha=math.exp(-dt_ms / 3.0),
        beta=beta,
        threshold=15.0,
        reset_mechanism="zero",
        reset_delay=False,
    )
    synapse = torch.zeros(2, 3)
    membrane = torch.zeros(2, 3)
    weights = torch.full((2, 3), 0.8)
    sign = -1 if inhibitory else 1
    voltages, spikes, currents = [], [], []
    for event in events:
        ref_spikes, synapse, membrane = reference(
            sign * gain * (event @ weights), synapse, membrane
        )
        voltages.append(membrane - 65.0)
        spikes.append(ref_spikes)
        currents.append(sign * synapse / gain)
    torch.testing.assert_close(
        result.outputs["voltage"], torch.stack(voltages), atol=5e-5, rtol=1e-6
    )
    torch.testing.assert_close(
        result.outputs["spikes"], torch.stack(spikes), atol=0, rtol=0
    )
    torch.testing.assert_close(
        result.diagnostics["current"], torch.stack(currents), atol=2e-6, rtol=1e-6
    )
    assert (
        result.outputs["spikes"].sum() > 10
        if not inhibitory
        else result.outputs["spikes"].sum() == 0
    )


def test_subthreshold_input_gradients_match_snntorch():
    """Independent unrolled voltage gradients; reset differentiation is excluded."""
    drive = (
        torch.linspace(0.01, 0.1, 30, dtype=torch.float64)
        .reshape(-1, 1, 1)
        .requires_grad_()
    )
    ref_drive = drive.detach().clone().requires_grad_()
    beta = math.exp(-1.0 / 20.0)
    gain = 20.0 * (1 - beta)
    reference = snn.Leaky(
        beta=torch.tensor(beta, dtype=torch.float64),
        threshold=15.0,
        reset_mechanism="none",
    ).double()
    voltage = torch.full((1, 1), -65.0, dtype=torch.float64)
    membrane = torch.zeros_like(voltage)
    refractory = torch.zeros_like(voltage, dtype=torch.long)
    config = lang.CUBA_LIF().json()
    for current, ref_current in zip(drive, ref_drive):
        voltage, spikes, refractory = X.current_lif(
            {"voltage": voltage, "refractory": refractory},
            current,
            torch.zeros_like(current),
            dt_ms=1.0,
            config=config,
            spike_function=lambda value: M.fast_sigmoid_spike(value, 1.0),
        )
        ref_spikes, membrane = reference(gain * ref_current, membrane)
        assert spikes.sum() == ref_spikes.sum() == 0
    (voltage + 65.0).sum().backward()
    membrane.sum().backward()
    torch.testing.assert_close(drive.grad, ref_drive.grad, atol=1e-12, rtol=0)
    assert torch.all(drive.grad > 0)


def test_fast_sigmoid_spike_gradients_match_snntorch():
    """Slope one matches both libraries' kernels, away from threshold equality."""
    values = torch.tensor(
        [-3.0, -0.1, 0.1, 3.0], dtype=torch.float64, requires_grad=True
    )
    ref_values = values.detach().clone().requires_grad_()
    spikes = M.fast_sigmoid_spike(values, 1.0)
    ref_spikes = surrogate.fast_sigmoid(slope=1.0)(ref_values)
    spikes.sum().backward()
    ref_spikes.sum().backward()
    torch.testing.assert_close(spikes, ref_spikes, atol=0, rtol=0, check_dtype=False)
    torch.testing.assert_close(values.grad, ref_values.grad, atol=1e-12, rtol=0)
