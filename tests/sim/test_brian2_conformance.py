"""Independent Brian2 comparisons for snnsim's conductance-based LIF core.

These tests deliberately spell out the reference equations and parameters on
the Brian2 side instead of importing them from ``models``.  That separation is
important: sharing the implementation under test would allow the same mistake
to make both sides agree.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
import torch
from scipy.signal import find_peaks

from snnlab.sim import models as M
from snnlab.sim.config import build_net, set_sim_dt
from snnlab.sim.timing import duration_steps, refractory_steps

b2 = pytest.importorskip("brian2")

pytestmark = [pytest.mark.integration, pytest.mark.brian2]

# Publication-model specification, independently repeated for the reference
# implementation. Units are ms, mV, nF, and uS unless Brian2 units are attached.
EL_MV = -65.0
EE_MV = 0.0
EI_MV = -80.0
V_THRESHOLD_MV = -50.0
V_RESET_MV = -65.0


def _brian_lif(
    *,
    dt_ms,
    duration_ms,
    capacitance_nf,
    leak_us,
    refractory_ms,
    excitatory_us,
    inhibitory_us,
    initial_mv=-65.0,
):
    """Run the independently encoded LIF equation with fixed conductances."""
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    b2.defaultclock.dt = dt_ms * b2.ms
    namespace = {
        "E_L": EL_MV * b2.mV,
        "E_e": EE_MV * b2.mV,
        "E_i": EI_MV * b2.mV,
        "V_threshold": V_THRESHOLD_MV * b2.mV,
        "V_reset": V_RESET_MV * b2.mV,
        "C_m": capacitance_nf * b2.nfarad,
        "g_L": leak_us * b2.usiemens,
    }
    neurons = b2.NeuronGroup(
        1,
        """
        dv/dt = (-g_L*(v-E_L) - g_e*(v-E_e) - g_i*(v-E_i))/C_m
                : volt (unless refractory)
        g_e : siemens
        g_i : siemens
        """,
        threshold="v >= V_threshold",
        reset="v = V_reset",
        refractory=refractory_ms * b2.ms,
        method="exact",
        namespace=namespace,
    )
    neurons.v = initial_mv * b2.mV
    neurons.g_e = excitatory_us * b2.usiemens
    neurons.g_i = inhibitory_us * b2.usiemens
    voltages = b2.StateMonitor(neurons, "v", record=True, when="end")
    spikes = b2.SpikeMonitor(neurons)
    network = b2.Network(neurons, voltages, spikes)
    # Brian2 otherwise runs a partial final step. Match the whole-step trial
    # contract using independent rational arithmetic, not snnsim conversion.
    steps = int(Fraction(str(duration_ms)) / Fraction(str(dt_ms)))
    network.run(steps * dt_ms * b2.ms, namespace={})
    return np.asarray(voltages.v[0] / b2.mV), np.asarray(spikes.t / b2.ms)


def _snnsim_lif(
    *,
    dt_ms,
    duration_ms,
    capacitance_nf,
    leak_us,
    refractory_ms,
    excitatory_us,
    inhibitory_us,
    initial_mv=-65.0,
):
    """Run the snnsim LIF primitive with the same fixed-conductance protocol."""
    steps = duration_steps(duration_ms, dt_ms)
    voltage = torch.tensor([[initial_mv]], dtype=torch.float64)
    refractory = torch.zeros((1, 1), dtype=torch.long)
    g_e = torch.tensor([[excitatory_us]], dtype=torch.float64)
    g_i = torch.tensor([[inhibitory_us]], dtype=torch.float64)
    ref_count = refractory_steps(refractory_ms, dt_ms, policy="nearest")
    voltages = []
    spike_times = []
    for step in range(steps):
        voltage, spike, refractory = M.lif_step_expeuler(
            voltage,
            refractory,
            g_e,
            g_i,
            capacitance_nf,
            leak_us,
            ref_count,
            M.spike_biophysical,
            v_grad_dampen=1.0,
            dt_override=dt_ms,
        )
        voltages.append(voltage.item())
        if spike.item():
            spike_times.append(step * dt_ms)
    return np.asarray(voltages), np.asarray(spike_times)


@pytest.mark.parametrize(
    ("capacitance_nf", "leak_us", "refractory_ms"),
    [(1.0, 0.05, 3.0), (0.5, 0.1, 1.5)],
    ids=["excitatory-cell", "inhibitory-cell"],
)
def test_fixed_conductance_lif_matches_brian2_exactly(
    capacitance_nf, leak_us, refractory_ms
):
    """Subthreshold E/I trajectories agree with Brian2's exact solver."""
    protocol = dict(
        dt_ms=0.1,
        duration_ms=20.0,
        capacitance_nf=capacitance_nf,
        leak_us=leak_us,
        refractory_ms=refractory_ms,
        excitatory_us=0.01,
        inhibitory_us=0.02,
        initial_mv=-60.0,
    )
    snnsim_voltage, snnsim_spikes = _snnsim_lif(**protocol)
    brian_voltage, brian_spikes = _brian_lif(**protocol)

    np.testing.assert_allclose(snnsim_voltage, brian_voltage, atol=1e-11, rtol=0)
    np.testing.assert_array_equal(snnsim_spikes, brian_spikes)


def test_threshold_reset_and_refractory_match_brian2():
    """A repeatedly spiking cell agrees in voltage and spike timing."""
    protocol = dict(
        dt_ms=0.1,
        duration_ms=20.0,
        capacitance_nf=1.0,
        leak_us=0.05,
        refractory_ms=3.0,
        excitatory_us=0.5,
        inhibitory_us=0.0,
    )
    snnsim_voltage, snnsim_spikes = _snnsim_lif(**protocol)
    brian_voltage, brian_spikes = _brian_lif(**protocol)

    assert len(snnsim_spikes) >= 5, "protocol must exercise repeated refractoriness"
    np.testing.assert_allclose(snnsim_voltage, brian_voltage, atol=1e-11, rtol=0)
    np.testing.assert_allclose(snnsim_spikes, brian_spikes, atol=1e-12, rtol=0)


@pytest.mark.parametrize("dt_ms", [0.05, 0.1, 0.2, 0.3, 0.6])
@pytest.mark.parametrize("drive_us", [0.2, 100.0], ids=["voltage-release", "strong-drive"])
def test_collection_production_refractory_and_duration_match_brian2(
    monkeypatch, dt_ms, drive_us
):
    """Public production wiring agrees with independent E/I refractory clocks."""
    nominal_ms = 200.0
    expected_steps = int(Fraction("200") / Fraction(str(dt_ms)))
    monkeypatch.setattr(M, "N_IN", 1)
    monkeypatch.setattr(M, "N_OUT", 1)
    set_sim_dt(dt_ms, nominal_ms)
    net = build_net(
        "ping", hidden_sizes=[4], w_ee=(0.0, 0.0), w_ei=(0.0, 0.0),
        w_ie=(0.0, 0.0), w_ii=(0.0, 0.0), refractory_e_ms=1.2,
        refractory_i_ms=0.6, refractory_policy="exact",
    )
    net.recording = True
    # Production external drive adds to an exponentially decaying conductance.
    # Balance each decay after the first kick to hold that conductance fixed.
    e_drive = torch.full((expected_steps, 4), drive_us * (1 - math.exp(-dt_ms / 2)))
    i_drive = torch.full((expected_steps, 1), drive_us * (1 - math.exp(-dt_ms / 2)))
    e_drive[0] = drive_us
    i_drive[0] = drive_us
    with torch.no_grad():
        net(ext_g=e_drive, ext_g_i=i_drive)

    assert net.timing_metadata["nominal_duration_ms"] == nominal_ms
    assert net.timing_metadata["duration_steps"] == expected_steps
    assert net.timing_metadata["realized_duration_ms"] == pytest.approx(
        199.8 if dt_ms in (0.3, 0.6) else 200.0
    )
    for key, voltage_key, capacitance, leak, refractory_ms in (
        ("hid", "v_e_1", 1.0, 0.05, 1.2),
        ("inh", "v_i_1", 0.5, 0.1, 0.6),
    ):
        brian_voltage, brian_times = _brian_lif(
            dt_ms=dt_ms, duration_ms=nominal_ms, capacitance_nf=capacitance,
            leak_us=leak, refractory_ms=refractory_ms,
            excitatory_us=drive_us, inhibitory_us=0.0,
        )
        actual_voltage = net.spike_record[voltage_key][:, 0].numpy()
        actual_indices = torch.where(net.spike_record[key][:, 0] != 0)[0].numpy()
        expected_counter = Fraction(str(refractory_ms)) / Fraction(str(dt_ms))
        assert expected_counter.denominator == 1
        expected_counter = int(expected_counter)
        assert len(brian_voltage) == len(actual_voltage) == expected_steps
        assert len(actual_indices) > 5
        np.testing.assert_allclose(
            actual_indices * dt_ms, brian_times, atol=1e-10, rtol=0
        )
        # Production states use float32; Brian2's independent solver uses float64.
        np.testing.assert_allclose(actual_voltage, brian_voltage, atol=2e-4, rtol=0)
        if drive_us == 100.0:
            np.testing.assert_array_equal(np.diff(actual_indices), expected_counter)
        else:
            first_spike = actual_indices[0]
            release = first_spike + expected_counter
            np.testing.assert_array_equal(actual_voltage[first_spike:release], -65.0)
            assert actual_voltage[release] > -65.0
            assert brian_voltage[release] > -65.0


def test_exponential_synapse_matches_brian2_event_dynamics():
    """Decay-then-add conductance updates match native Brian2 synapses."""
    dt_ms = 0.1
    tau_ms = 2.0
    weight_us = 0.3
    steps = 10
    event_steps = {0, 3, 7}

    conductance = torch.zeros((1, 1), dtype=torch.float64)
    weight = torch.tensor([[weight_us]], dtype=torch.float64)
    snnsim_trace = []
    for step in range(steps):
        presynaptic_spike = torch.tensor(
            [[1.0 if step in event_steps else 0.0]], dtype=torch.float64
        )
        conductance = M.exp_synapse(
            conductance,
            presynaptic_spike,
            weight,
            math.exp(-dt_ms / tau_ms),
        )
        snnsim_trace.append(conductance.item())

    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    b2.defaultclock.dt = dt_ms * b2.ms
    target = b2.NeuronGroup(
        1,
        "dg/dt = -g/tau : siemens",
        method="exact",
        namespace={"tau": tau_ms * b2.ms},
    )
    source = b2.SpikeGeneratorGroup(
        1,
        np.zeros(len(event_steps), dtype=int),
        np.asarray(sorted(event_steps)) * dt_ms * b2.ms,
    )
    synapse = b2.Synapses(source, target, on_pre="g_post += weight")
    synapse.connect()
    monitor = b2.StateMonitor(target, "g", record=True, when="end")
    network = b2.Network(target, source, synapse, monitor)
    network.run(
        steps * dt_ms * b2.ms,
        namespace={"weight": weight_us * b2.usiemens},
    )
    brian_trace = np.asarray(monitor.g[0] / b2.usiemens)

    np.testing.assert_allclose(snnsim_trace, brian_trace, atol=1e-12, rtol=0)


def _snnsim_ei_pair(dt_ms, duration_ms=100.0):
    """Minimal reciprocal E-I circuit using snnsim's production primitives."""
    steps = int(round(duration_ms / dt_ms))
    voltage_e = torch.tensor([[-65.0]], dtype=torch.float64)
    voltage_i = voltage_e.clone()
    refractory_e = torch.zeros((1, 1), dtype=torch.long)
    refractory_i = refractory_e.clone()
    g_ee = torch.zeros_like(voltage_e)
    g_ei = torch.zeros_like(voltage_e)
    g_ie = torch.zeros_like(voltage_e)
    spike_e = torch.zeros_like(voltage_e)
    spike_i = torch.zeros_like(voltage_e)
    input_steps = {
        int(round(time_ms / dt_ms)) for time_ms in (5.0, 25.0, 45.0, 65.0, 85.0)
    }
    e_times, i_times = [], []

    for step in range(steps):
        external_kick = 0.15 if step in input_steps else 0.0
        g_ee = g_ee * math.exp(-dt_ms / 2.0) + external_kick
        g_ei = g_ei * math.exp(-dt_ms / 2.0) + spike_e * 1.0
        g_ie = g_ie * math.exp(-dt_ms / 6.0) + spike_i * 3.0
        voltage_e, spike_e, refractory_e = M.lif_step_expeuler(
            voltage_e,
            refractory_e,
            g_ee,
            g_ie,
            1.0,
            0.05,
            max(1, int(round(3.0 / dt_ms))),
            M.spike_biophysical,
            v_grad_dampen=1.0,
            dt_override=dt_ms,
        )
        voltage_i, spike_i, refractory_i = M.lif_step_expeuler(
            voltage_i,
            refractory_i,
            g_ei,
            None,
            0.5,
            0.1,
            max(1, int(round(1.5 / dt_ms))),
            M.spike_biophysical,
            v_grad_dampen=1.0,
            dt_override=dt_ms,
        )
        if spike_e.item():
            e_times.append(step * dt_ms)
        if spike_i.item():
            i_times.append(step * dt_ms)
    return np.asarray(e_times), np.asarray(i_times)


def _brian_ei_pair(dt_ms, duration_ms=100.0):
    """The same circuit encoded with Brian2's native ODEs and Synapses."""
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    b2.defaultclock.dt = dt_ms * b2.ms
    namespace = {
        "E_L": EL_MV * b2.mV,
        "E_e": EE_MV * b2.mV,
        "E_i": EI_MV * b2.mV,
        "V_threshold": V_THRESHOLD_MV * b2.mV,
        "V_reset": V_RESET_MV * b2.mV,
        "tau_ampa": 2.0 * b2.ms,
        "tau_gaba": 6.0 * b2.ms,
    }
    equations = """
        dv/dt = (-g_L*(v-E_L) - g_e*(v-E_e) - g_i*(v-E_i))/C_m
                : volt (unless refractory)
        dg_e/dt = -g_e/tau_ampa : siemens
        dg_i/dt = -g_i/tau_gaba : siemens
        C_m : farad (constant)
        g_L : siemens (constant)
    """
    excitatory = b2.NeuronGroup(
        1,
        equations,
        threshold="v >= V_threshold",
        reset="v = V_reset",
        refractory=3.0 * b2.ms,
        method="exponential_euler",
        namespace=namespace,
    )
    inhibitory = b2.NeuronGroup(
        1,
        equations,
        threshold="v >= V_threshold",
        reset="v = V_reset",
        refractory=1.5 * b2.ms,
        method="exponential_euler",
        namespace=namespace,
    )
    for neurons, capacitance, leak in ((excitatory, 1.0, 0.05), (inhibitory, 0.5, 0.1)):
        neurons.v = -65.0 * b2.mV
        neurons.g_e = 0.0 * b2.usiemens
        neurons.g_i = 0.0 * b2.usiemens
        neurons.C_m = capacitance * b2.nfarad
        neurons.g_L = leak * b2.usiemens

    input_times = np.asarray([5.0, 25.0, 45.0, 65.0, 85.0])
    source = b2.SpikeGeneratorGroup(
        1, np.zeros(len(input_times), dtype=int), input_times * b2.ms
    )
    external = b2.Synapses(source, excitatory, on_pre="g_e_post += 0.15*usiemens")
    e_to_i = b2.Synapses(excitatory, inhibitory, on_pre="g_e_post += 1.0*usiemens")
    i_to_e = b2.Synapses(inhibitory, excitatory, on_pre="g_i_post += 3.0*usiemens")
    external.connect()
    e_to_i.connect()
    i_to_e.connect()
    e_spikes = b2.SpikeMonitor(excitatory)
    i_spikes = b2.SpikeMonitor(inhibitory)
    network = b2.Network(
        excitatory, inhibitory, source, external, e_to_i, i_to_e, e_spikes, i_spikes
    )
    network.run(duration_ms * b2.ms, namespace={})
    return np.asarray(e_spikes.t / b2.ms), np.asarray(i_spikes.t / b2.ms)


@pytest.mark.parametrize("dt_ms", [0.25, 0.1, 0.05])
def test_reciprocal_ei_spikes_converge_with_brian2(dt_ms):
    """E-I spike counts agree and scheduling error is bounded by one timestep."""
    snnsim_populations = _snnsim_ei_pair(dt_ms)
    brian_populations = _brian_ei_pair(dt_ms)

    for snnsim_times, brian_times in zip(snnsim_populations, brian_populations):
        assert len(snnsim_times) == len(brian_times)
        assert len(snnsim_times) > 0, "protocol must recruit both populations"
        np.testing.assert_allclose(
            snnsim_times, brian_times, atol=dt_ms + 1e-12, rtol=0
        )


PING_N_E = 20
PING_N_I = 5
PING_DURATION_MS = 500.0
PING_DISCARD_MS = 50.0
PING_BIN_MS = 1.0


def _snnsim_ping_network(
    dt_ms, tau_gaba_ms=6.0, *, refractory_e_ms=3.0, refractory_i_ms=1.5
):
    """Run a heterogeneous 20E/5I PING circuit with tonic E-cell drive."""
    steps = int(round(PING_DURATION_MS / dt_ms))
    voltage_e = torch.linspace(-67.0, -63.0, PING_N_E, dtype=torch.float64)[None]
    voltage_i = torch.linspace(-66.0, -64.0, PING_N_I, dtype=torch.float64)[None]
    refractory_e = torch.zeros((1, PING_N_E), dtype=torch.long)
    refractory_i = torch.zeros((1, PING_N_I), dtype=torch.long)
    g_ei = torch.zeros((1, PING_N_I), dtype=torch.float64)
    g_ie = torch.zeros((1, PING_N_E), dtype=torch.float64)
    spike_e = torch.zeros((1, PING_N_E), dtype=torch.float64)
    spike_i = torch.zeros((1, PING_N_I), dtype=torch.float64)
    e_to_i = torch.full((PING_N_E, PING_N_I), 1.0 / PING_N_E, dtype=torch.float64)
    i_to_e = torch.full((PING_N_I, PING_N_E), 3.0 / PING_N_I, dtype=torch.float64)
    tonic_drive = torch.linspace(0.135, 0.165, PING_N_E, dtype=torch.float64)[None]
    e_events, i_events = [], []

    for step in range(steps):
        g_ei = g_ei * math.exp(-dt_ms / 2.0) + spike_e @ e_to_i
        g_ie = g_ie * math.exp(-dt_ms / tau_gaba_ms) + spike_i @ i_to_e
        voltage_e, spike_e, refractory_e = M.lif_step_expeuler(
            voltage_e,
            refractory_e,
            tonic_drive,
            g_ie,
            1.0,
            0.05,
            refractory_steps(refractory_e_ms, dt_ms, policy="nearest"),
            M.spike_biophysical,
            v_grad_dampen=1.0,
            dt_override=dt_ms,
        )
        voltage_i, spike_i, refractory_i = M.lif_step_expeuler(
            voltage_i,
            refractory_i,
            g_ei,
            None,
            0.5,
            0.1,
            refractory_steps(refractory_i_ms, dt_ms, policy="nearest"),
            M.spike_biophysical,
            v_grad_dampen=1.0,
            dt_override=dt_ms,
        )
        time_ms = step * dt_ms
        e_events.extend((time_ms, int(index)) for index in torch.where(spike_e[0])[0])
        i_events.extend((time_ms, int(index)) for index in torch.where(spike_i[0])[0])
    return np.asarray(e_events), np.asarray(i_events)


def _brian_ping_network(
    dt_ms, tau_gaba_ms=6.0, *, refractory_e_ms=3.0, refractory_i_ms=1.5
):
    """Run an independently encoded native-Brian2 version of the PING circuit."""
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    b2.defaultclock.dt = dt_ms * b2.ms
    namespace = {
        "E_L": EL_MV * b2.mV,
        "E_e": EE_MV * b2.mV,
        "E_i": EI_MV * b2.mV,
        "V_threshold": V_THRESHOLD_MV * b2.mV,
        "V_reset": V_RESET_MV * b2.mV,
        "tau_ampa": 2.0 * b2.ms,
        "tau_gaba": tau_gaba_ms * b2.ms,
    }
    equations = """
        dv/dt = (-g_L*(v-E_L) - g_drive*(v-E_e)
                 - g_exc*(v-E_e) - g_inh*(v-E_i))/C_m
                : volt (unless refractory)
        dg_exc/dt = -g_exc/tau_ampa : siemens
        dg_inh/dt = -g_inh/tau_gaba : siemens
        g_drive : siemens (constant)
        C_m : farad (constant)
        g_L : siemens (constant)
    """
    excitatory = b2.NeuronGroup(
        PING_N_E,
        equations,
        threshold="v >= V_threshold",
        reset="v = V_reset",
        refractory=refractory_e_ms * b2.ms,
        method="exponential_euler",
        namespace=namespace,
    )
    inhibitory = b2.NeuronGroup(
        PING_N_I,
        equations,
        threshold="v >= V_threshold",
        reset="v = V_reset",
        refractory=refractory_i_ms * b2.ms,
        method="exponential_euler",
        namespace=namespace,
    )
    excitatory.v = np.linspace(-67.0, -63.0, PING_N_E) * b2.mV
    inhibitory.v = np.linspace(-66.0, -64.0, PING_N_I) * b2.mV
    excitatory.C_m = 1.0 * b2.nfarad
    excitatory.g_L = 0.05 * b2.usiemens
    excitatory.g_drive = np.linspace(0.135, 0.165, PING_N_E) * b2.usiemens
    inhibitory.C_m = 0.5 * b2.nfarad
    inhibitory.g_L = 0.1 * b2.usiemens
    inhibitory.g_drive = 0.0 * b2.usiemens
    for neurons in (excitatory, inhibitory):
        neurons.g_exc = 0.0 * b2.usiemens
        neurons.g_inh = 0.0 * b2.usiemens

    e_to_i = b2.Synapses(
        excitatory,
        inhibitory,
        on_pre=f"g_exc_post += {1.0 / PING_N_E}*usiemens",
    )
    i_to_e = b2.Synapses(
        inhibitory,
        excitatory,
        on_pre=f"g_inh_post += {3.0 / PING_N_I}*usiemens",
    )
    e_to_i.connect()
    i_to_e.connect()
    e_spikes = b2.SpikeMonitor(excitatory)
    i_spikes = b2.SpikeMonitor(inhibitory)
    network = b2.Network(excitatory, inhibitory, e_to_i, i_to_e, e_spikes, i_spikes)
    network.run(PING_DURATION_MS * b2.ms, namespace={})

    e_events = np.column_stack(
        (np.asarray(e_spikes.t / b2.ms), np.asarray(e_spikes.i, dtype=int))
    )
    i_events = np.column_stack(
        (np.asarray(i_spikes.t / b2.ms), np.asarray(i_spikes.i, dtype=int))
    )
    return e_events, i_events


def _population_observables(events, population_size):
    """Return rate, gamma peak, synchrony, and 1 ms population spike trace."""
    bin_edges = np.arange(0.0, PING_DURATION_MS + PING_BIN_MS, PING_BIN_MS)
    spike_matrix = np.zeros((population_size, len(bin_edges) - 1))
    for neuron_index in range(population_size):
        neuron_times = events[events[:, 1] == neuron_index, 0]
        spike_matrix[neuron_index] = np.histogram(neuron_times, bin_edges)[0]
    keep = bin_edges[:-1] >= PING_DISCARD_MS
    spike_matrix = spike_matrix[:, keep]
    population_trace = spike_matrix.mean(axis=0)
    analysed_seconds = (PING_DURATION_MS - PING_DISCARD_MS) / 1000.0
    rate_hz = spike_matrix.sum() / (population_size * analysed_seconds)

    centred_trace = population_trace - population_trace.mean()
    frequencies = np.fft.rfftfreq(len(centred_trace), PING_BIN_MS / 1000.0)
    power = np.abs(np.fft.rfft(centred_trace)) ** 2
    gamma_band = (frequencies >= 30.0) & (frequencies <= 80.0)
    gamma_hz = frequencies[gamma_band][np.argmax(power[gamma_band])]

    individual_variance = np.var(spike_matrix, axis=1).mean()
    synchrony = np.var(population_trace) / individual_variance
    return {
        "rate_hz": rate_hz,
        "gamma_hz": gamma_hz,
        "synchrony": synchrony,
        "trace": population_trace,
    }


def _primary_e_to_i_lag_ms(e_events, i_events):
    """Median delay between corresponding E- and I-volley centres."""

    def volley_centres(events):
        times = np.sort(events[:, 0])
        volleys = np.split(times, np.where(np.diff(times) > 10.0)[0] + 1)
        return np.asarray([volley.mean() for volley in volleys if len(volley)])

    e_centres = volley_centres(e_events)
    i_centres = volley_centres(i_events)
    lags = []
    for e_centre in e_centres:
        following_i = i_centres[
            (i_centres >= e_centre)
            & (i_centres - e_centre < 10.0)
            & (i_centres >= PING_DISCARD_MS)
        ]
        if len(following_i):
            lags.append(following_i[0] - e_centre)
    assert len(lags) >= 10, "protocol must sustain repeated E-I gamma cycles"
    return float(np.median(lags))


@pytest.mark.parametrize("dt_ms", [0.25, 0.1, 0.05])
def test_ping_population_observables_match_brian2(dt_ms):
    """Rates, gamma rhythm, E-I lag, synchrony, and traces agree with Brian2."""
    snnsim_events = _snnsim_ping_network(dt_ms)
    brian_events = _brian_ping_network(dt_ms)

    for population_size, snnsim_population, brian_population in zip(
        (PING_N_E, PING_N_I), snnsim_events, brian_events
    ):
        snnsim = _population_observables(snnsim_population, population_size)
        brian = _population_observables(brian_population, population_size)
        assert 30.0 <= snnsim["gamma_hz"] <= 80.0
        assert snnsim["rate_hz"] == pytest.approx(brian["rate_hz"], abs=0.25)
        assert snnsim["gamma_hz"] == pytest.approx(brian["gamma_hz"], abs=2.3)
        assert snnsim["synchrony"] == pytest.approx(brian["synchrony"], abs=0.02)
        trace_correlation = np.corrcoef(snnsim["trace"], brian["trace"])[0, 1]
        assert trace_correlation >= 0.95

    snnsim_lag = _primary_e_to_i_lag_ms(*snnsim_events)
    brian_lag = _primary_e_to_i_lag_ms(*brian_events)
    assert snnsim_lag == pytest.approx(brian_lag, abs=dt_ms + 1e-12)


def test_brian2_gap_is_smaller_than_snnsim_timestep_sensitivity():
    """Cross-simulator error is below snnsim's own coarse/fine discretisation."""
    coarse_snnsim = _snnsim_ping_network(0.25)
    fine_snnsim = _snnsim_ping_network(0.05)
    fine_brian = _brian_ping_network(0.05)

    def summary_distance(left_events, right_events):
        scales = {"rate_hz": 20.0, "gamma_hz": 50.0, "synchrony": 1.0}
        squared_error = 0.0
        for population_size, left_population, right_population in zip(
            (PING_N_E, PING_N_I), left_events, right_events
        ):
            left = _population_observables(left_population, population_size)
            right = _population_observables(right_population, population_size)
            squared_error += sum(
                ((left[name] - right[name]) / scale) ** 2
                for name, scale in scales.items()
            )
        return math.sqrt(squared_error)

    cross_simulator_gap = summary_distance(fine_snnsim, fine_brian)
    timestep_sensitivity = summary_distance(coarse_snnsim, fine_snnsim)
    assert timestep_sensitivity > 0.05, "protocol must expose timestep sensitivity"
    assert cross_simulator_gap < timestep_sensitivity


def _cycle_sparsity_observables(events, dt_ms=0.1):
    """Mirror exp046's I-volley-anchored E-spikes-per-cycle measurement."""
    e_events, i_events = events
    steps = int(round(PING_DURATION_MS / dt_ms))
    e_spikes = np.zeros((steps, PING_N_E), dtype=np.int8)
    i_spikes = np.zeros((steps, PING_N_I), dtype=np.int8)
    e_spikes[
        np.rint(e_events[:, 0] / dt_ms).astype(int), e_events[:, 1].astype(int)
    ] = 1
    i_spikes[
        np.rint(i_events[:, 0] / dt_ms).astype(int), i_events[:, 1].astype(int)
    ] = 1

    # As in exp046, smooth the population-I count with a 1 ms Gaussian and
    # reject small peaks. A fixed 10 ms separation is below every measured
    # cycle period in this controlled sweep and suppresses within-volley doublets.
    population_i = i_spikes.sum(axis=1).astype(np.float64)
    sigma_steps = max(1.0, 1.0 / dt_ms)
    half_width = int(math.ceil(4.0 * sigma_steps))
    kernel_steps = np.arange(-half_width, half_width + 1)
    kernel = np.exp(-0.5 * (kernel_steps / sigma_steps) ** 2)
    kernel /= kernel.sum()
    smoothed_i = np.convolve(population_i, kernel, mode="same")
    peaks, _ = find_peaks(
        smoothed_i,
        distance=max(1, int(round(10.0 / dt_ms))),
        height=0.05 * smoothed_i.max(),
    )
    assert len(peaks) >= 5, "protocol must sustain repeated inhibitory cycles"

    boundaries = np.concatenate(
        ([0], ((peaks[:-1] + peaks[1:]) // 2).astype(int), [steps])
    )
    per_cycle = np.stack(
        [
            e_spikes[start:stop].sum(axis=0)
            for start, stop in zip(boundaries[:-1], boundaries[1:])
        ]
    )
    flat_counts = per_cycle.ravel()
    buckets = np.asarray(
        [
            np.count_nonzero(flat_counts == 0),
            np.count_nonzero(flat_counts == 1),
            np.count_nonzero(flat_counts == 2),
            np.count_nonzero(flat_counts >= 3),
        ],
        dtype=np.float64,
    )
    bucket_fractions = buckets / buckets.sum()
    duration_seconds = PING_DURATION_MS / 1000.0
    per_cell_rates = e_spikes.sum(axis=0) / duration_seconds
    return {
        "cycle_frequency_hz": len(peaks) / duration_seconds,
        "bucket_fractions": bucket_fractions,
        "maximum_cell_rate_hz": float(per_cell_rates.max()),
        "median_cell_rate_hz": float(np.median(per_cell_rates)),
        "cell_cycle_pairs": int(flat_counts.size),
    }


def test_gamma_gated_sparsity_tau_sweep_matches_brian2():
    """The exp041/046 cycle-sparsity relationship survives simulator exchange."""
    tau_gaba_sweep_ms = (4.5, 6.0, 9.0, 12.0, 18.0, 27.0)
    snnsim_rows = []
    brian_rows = []

    for tau_gaba_ms in tau_gaba_sweep_ms:
        refractory = {"refractory_e_ms": 1.2, "refractory_i_ms": 0.6}
        snnsim = _cycle_sparsity_observables(
            _snnsim_ping_network(0.1, tau_gaba_ms, **refractory)
        )
        brian = _cycle_sparsity_observables(
            _brian_ping_network(0.1, tau_gaba_ms, **refractory)
        )
        snnsim_rows.append(snnsim)
        brian_rows.append(brian)

        assert snnsim["cell_cycle_pairs"] == brian["cell_cycle_pairs"]
        np.testing.assert_allclose(
            snnsim["bucket_fractions"],
            brian["bucket_fractions"],
            atol=1e-12,
            rtol=0,
        )
        assert snnsim["cycle_frequency_hz"] == pytest.approx(
            brian["cycle_frequency_hz"], abs=1e-12
        )
        assert snnsim["maximum_cell_rate_hz"] == pytest.approx(
            brian["maximum_cell_rate_hz"], abs=1e-12
        )
        assert snnsim["median_cell_rate_hz"] == pytest.approx(
            brian["median_cell_rate_hz"], abs=1e-12
        )

        # Collection-specific acceptance contract: E cells are silent or emit
        # once in at least 95% of I-anchored cycles, and the busiest cell stays
        # close to the one-spike-per-cycle ceiling.
        assert snnsim["bucket_fractions"][:2].sum() >= 0.95
        assert snnsim["maximum_cell_rate_hz"] == pytest.approx(
            snnsim["cycle_frequency_hz"], abs=2.1
        )

    snnsim_frequencies = [row["cycle_frequency_hz"] for row in snnsim_rows]
    brian_frequencies = [row["cycle_frequency_hz"] for row in brian_rows]
    assert all(
        faster > slower
        for faster, slower in zip(snnsim_frequencies, snnsim_frequencies[1:])
    )
    np.testing.assert_allclose(snnsim_frequencies, brian_frequencies, atol=1e-12)
