"""Independent snnTorch comparisons for current-based LIF dynamics.

References:
https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html
https://snntorch.readthedocs.io/en/latest/snn.neurons_synaptic.html
Shift membrane voltage by the resting potential (mV), and scale current (nA)
by tau_mem / capacitance * (1 - beta). This gives snnTorch's discrete input.
Zero reset is equivalent when reset equals rest, with immediate reset and no
refractory period. Conductance dynamics and refractory timing are not covered.
Hard thresholds differ at exact equality (SNNLab >=, snnTorch >), so the
trajectory protocols avoid equality. Reset gradients differ too; gradient
comparisons deliberately use subthreshold trajectories with reference reset
disabled, and a matched slope-one fast-sigmoid spike derivative. Public-network
checks include mixed excitation/inhibition, delayed recurrence, heterogeneous
weights, continuation, and input/weight gradients through synaptic filters.
"""

import math

import pytest
import torch

from snnlab import lang
from snnlab.sim import DenseArrayBinding, GraphExecutor, plan_graph
from snnlab.sim.graph import dynamics as M
from snnlab.sim.graph import extensions as X
from snnlab.sim.requests.handlers import simulate
from snnlab.sim.requests.spec import ExecutionSpec

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
    for current, ref_current in zip(drive, ref_drive, strict=True):
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


@pytest.mark.parametrize("dt_ms", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("delay_steps", [1, 3])
@pytest.mark.parametrize("tau_ms, capacitance", [(20.0, 1.0), (5.0, 0.5)])
def test_mixed_recurrent_network_matches_snntorch(
    dt_ms, delay_steps, tau_ms, capacitance
):
    """Independent synaptic filters and Leaky cells check delayed E/I wiring."""
    net = lang.Network("snntorch-recurrence", dt=dt_ms * lang.ms)
    inputs = net.input(
        "inputs", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cells = net.population(
        "cells",
        size=3,
        neuron=lang.CUBA_LIF(tau_mem=tau_ms * lang.ms, capacitance_nf=capacitance),
    )
    excitation = net.connect(
        inputs,
        cells.excitatory,
        name="excitation",
        synapse=lang.ExponentialCurrent(tau=3 * lang.ms),
        weight=lang.Constant(1),
        initialization_scaling="direct",
    )
    inhibition = net.connect(
        cells.spikes,
        cells.inhibitory,
        name="inhibition",
        synapse=lang.ExponentialCurrent(tau=7 * lang.ms),
        weight=lang.Constant(1),
        initialization_scaling="direct",
        connection="recurrent",
        delay=delay_steps * dt_ms * lang.ms,
        allow_autapses=False,
    )
    net.output("voltage", cells.voltage)
    net.output("spikes", cells.spikes)
    net.expose(excitation.current, name="excitatory_current")
    net.expose(inhibition.current, name="inhibitory_current")
    model = GraphExecutor(plan_graph(lang.compile(net).graph))
    # Independently specified, heterogeneous matrices catch transposition and
    # population broadcasting errors. Recurrent self-connections are excluded.
    excitatory_weights = torch.tensor(
        [[0.8, 1.1, 0.6], [1.3, 0.7, 1.0]], dtype=torch.float64
    )
    inhibitory_weights = torch.tensor(
        [[0.0, 0.3, 0.1], [0.2, 0.0, 0.4], [0.5, 0.2, 0.0]], dtype=torch.float64
    )
    with torch.no_grad():
        model.parameter_map()[excitation.weight.id].copy_(excitatory_weights)
        model.parameter_map()[inhibition.weight.id].copy_(inhibitory_weights)
    events = (
        torch.rand((160, 2, 2), generator=torch.Generator().manual_seed(53)) < 0.4
    ).float()
    with torch.no_grad():
        result = model({"inputs": events})
        first = model({"inputs": events[:47]})
        second = model({"inputs": events[47:]}, runtime_state=first.runtime_state)

    beta = math.exp(-dt_ms / tau_ms)
    gain = tau_ms / capacitance * (1 - beta)
    reference = snn.Leaky(
        beta=torch.tensor(beta, dtype=torch.float64),
        threshold=15.0,
        reset_mechanism="zero",
        reset_delay=False,
    ).double()
    # Synaptic supplies independent current decay. Its unused membrane cannot
    # spike or reset; the Leaky cell integrates the net physical E-I current.
    filters = [
        snn.Synaptic(
            alpha=torch.tensor(math.exp(-dt_ms / tau), dtype=torch.float64),
            beta=0.0,
            threshold=1e6,
            reset_mechanism="none",
        ).double()
        for tau in (3.0, 7.0)
    ]
    currents = [torch.zeros(2, 3, dtype=torch.float64) for _ in filters]
    membrane = torch.zeros(2, 3, dtype=torch.float64)
    spike_history = [torch.zeros_like(membrane) for _ in range(delay_steps)]
    expected = {
        key: []
        for key in ("voltage", "spikes", "excitatory_current", "inhibitory_current")
    }
    for step, event in enumerate(events):
        delayed_spikes = spike_history[step]
        for index, drive in enumerate(
            (event.double() @ excitatory_weights, delayed_spikes @ inhibitory_weights)
        ):
            _, currents[index], _ = filters[index](
                drive, currents[index], torch.zeros_like(membrane)
            )
        spikes, membrane = reference(gain * (currents[0] - currents[1]), membrane)
        spike_history.append(spikes.double())
        for key, value in zip(
            expected, (membrane - 65.0, spikes, currents[0], currents[1]), strict=True
        ):
            expected[key].append(value)

    for key, values in expected.items():
        actual = result.outputs if key in result.outputs else result.diagnostics
        left = first.outputs if key in first.outputs else first.diagnostics
        right = second.outputs if key in second.outputs else second.diagnostics
        # Production execution is float32; the independent oracle is float64.
        torch.testing.assert_close(
            actual[key],
            torch.stack(values),
            atol=0 if key == "spikes" else 5e-5,
            rtol=0 if key == "spikes" else 1e-6,
            check_dtype=False,
        )
        torch.testing.assert_close(
            torch.cat([left[key], right[key]]), actual[key], atol=0, rtol=0
        )
    assert result.outputs["spikes"].sum() > 10
    assert result.diagnostics["inhibitory_current"].max() > 0.1
    assert not torch.equal(
        result.outputs["spikes"][:, 0], result.outputs["spikes"][:, 1]
    )


@pytest.mark.parametrize("dt_ms", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("tau_ms, capacitance", [(20.0, 1.0), (5.0, 0.5)])
def test_compiled_synaptic_input_and_weight_gradients_match_snntorch(
    dt_ms, tau_ms, capacitance
):
    """Compare BPTT through public E/I projections and voltage/spike outputs."""
    net = lang.Network("snntorch-gradients", dt=dt_ms * lang.ms)
    inputs = net.input(
        "inputs", shape=("time", "batch", 2), signal_type="current", unit="nA"
    )
    cells = net.population(
        "cells",
        size=3,
        neuron=lang.CUBA_LIF(tau_mem=tau_ms * lang.ms, capacitance_nf=capacitance),
    )
    projections = [
        net.connect(
            inputs,
            target,
            name=name,
            synapse=lang.ExponentialCurrent(tau=3 * lang.ms),
            weight=lang.Constant(weight),
            initialization_scaling="direct",
        )
        for name, target, weight in (
            ("excitation", cells.excitatory, 0.2),
            ("inhibition", cells.inhibitory, 0.05),
        )
    ]
    net.output("voltage", cells.voltage)
    net.output("spikes", cells.spikes)
    model = GraphExecutor(
        plan_graph(lang.compile(net).graph),
        trainable_parameters=tuple(p.weight.id for p in projections),
        surrogate_slope=1.0,
    )
    weights = [model.parameter_map()[p.weight.id] for p in projections]
    reference_weights = [w.detach().double().clone().requires_grad_() for w in weights]
    drive = (
        0.1
        * torch.rand(
            (30, 2, 2),
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(71),
        )
    ).requires_grad_()
    reference_drive = drive.detach().double().clone().requires_grad_()
    result = model({"inputs": drive})
    beta = math.exp(-dt_ms / tau_ms)
    gain = tau_ms / capacitance * (1 - beta)
    reference = snn.Synaptic(
        alpha=torch.tensor(math.exp(-dt_ms / 3.0), dtype=torch.float64),
        beta=torch.tensor(beta, dtype=torch.float64),
        threshold=15.0,
        spike_grad=surrogate.fast_sigmoid(slope=1.0),
        reset_mechanism="none",
    ).double()
    synapse = torch.zeros(2, 3, dtype=torch.float64)
    membrane = torch.zeros_like(synapse)
    reference_voltages, reference_spikes = [], []
    for current in reference_drive:
        spikes, synapse, membrane = reference(
            gain * (current @ reference_weights[0] - current @ reference_weights[1]),
            synapse,
            membrane,
        )
        reference_voltages.append(membrane)
        reference_spikes.append(spikes)
    reference_voltage = torch.stack(reference_voltages)
    reference_spikes = torch.stack(reference_spikes)
    voltage = result.outputs["voltage"] + 65.0
    torch.testing.assert_close(
        voltage, reference_voltage, atol=5e-5, rtol=1e-6, check_dtype=False
    )
    assert result.outputs["spikes"].sum() == reference_spikes.sum() == 0
    # Silent spikes still carry a surrogate derivative, so this loss exercises
    # both output gradient paths without introducing incompatible reset gradients.
    (voltage.square().mean() + result.outputs["spikes"].sum()).backward()
    (reference_voltage.square().mean() + reference_spikes.sum()).backward()
    torch.testing.assert_close(
        drive.grad, reference_drive.grad, atol=1e-6, rtol=1e-5, check_dtype=False
    )
    assert drive.grad.abs().sum() > 0
    for weight, reference_weight in zip(weights, reference_weights, strict=True):
        torch.testing.assert_close(
            weight.grad, reference_weight.grad, atol=1e-6, rtol=1e-5, check_dtype=False
        )
        assert weight.grad.abs().sum() > 0


def test_threshold_equality_documents_snntorch_difference():
    """SNNLab uses >= while snnTorch uses >; equality is not conformance."""
    voltage = torch.tensor([[0.5]], dtype=torch.float64)
    config = lang.CUBA_LIF(
        tau_mem=1 * lang.ms, resting_mv=0.0, threshold_mv=1.0, reset_mv=0.0
    ).json()
    actual_voltage, actual_spikes, _ = X.current_lif(
        {"voltage": voltage, "refractory": torch.zeros_like(voltage, dtype=torch.long)},
        torch.full_like(voltage, 1.5),
        torch.zeros_like(voltage),
        dt_ms=math.log(2),
        config=config,
        spike_function=lambda value: (value >= 0).to(value.dtype),
    )
    reference = snn.Leaky(
        beta=0.5, threshold=1.0, reset_mechanism="zero", reset_delay=False
    )
    reference_spikes, reference_voltage = reference(
        torch.full_like(voltage, 0.75), voltage
    )
    assert actual_spikes.item() == 1 and actual_voltage.item() == 0
    assert reference_spikes.item() == 0 and reference_voltage.item() == 1
