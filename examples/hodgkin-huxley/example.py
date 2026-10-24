import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.special import exprel

from snnlab import extensions, lang, viz
from snnlab.sim import DenseArrayBinding, Performance, simulate

CAPACITANCE_NF = 0.01
G_NA_US, G_K_US, G_LEAK_US = 1.2, 0.36, 0.003
E_NA_MV, E_K_MV, E_LEAK_MV = 50.0, -77.0, -54.387
MAX_STEP_MS = 0.025


def _vtrap(x):
    small = x.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1 + x / 2 + x.square() / 12, safe / -torch.expm1(-safe))


def rates(voltage):
    return (
        (_vtrap((voltage + 40) / 10), 4 * torch.exp(-(voltage + 65) / 18)),
        (0.07 * torch.exp(-(voltage + 65) / 20), torch.sigmoid((voltage + 35) / 10)),
        (0.1 * _vtrap((voltage + 55) / 10), 0.125 * torch.exp(-(voltage + 65) / 80)),
    )


def initial_state(context):
    voltage = torch.full(
        context.shape, -65.0, device=context.device, dtype=context.dtype
    )
    state = {"voltage": voltage}
    for gate, (alpha, beta) in zip(("m", "h", "n"), rates(voltage)):
        state[gate] = alpha / (alpha + beta)
    return state


def _derivative(values, current):
    voltage, m, h, n = values
    ionic = (
        G_NA_US * m**3 * h * (voltage - E_NA_MV)
        + G_K_US * n**4 * (voltage - E_K_MV)
        + G_LEAK_US * (voltage - E_LEAK_MV)
    )
    gates = tuple(
        alpha * (1 - gate) - beta * gate
        for gate, (alpha, beta) in zip((m, h, n), rates(voltage))
    )
    return ((current - ionic) / CAPACITANCE_NF, *gates)


def step(context):
    keys = ("voltage", "m", "h", "n")
    values = tuple(context.state[key] for key in keys)
    current = (context.excitatory - context.inhibitory).to(values[0].dtype)
    substeps = max(1, math.ceil(context.dt_ms / MAX_STEP_MS))
    dt = context.dt_ms / substeps
    crossed = torch.zeros_like(values[0], dtype=torch.bool)
    for _ in range(substeps):
        k1 = _derivative(values, current)
        k2 = _derivative(tuple(y + dt / 2 * k for y, k in zip(values, k1)), current)
        k3 = _derivative(tuple(y + dt / 2 * k for y, k in zip(values, k2)), current)
        k4 = _derivative(tuple(y + dt * k for y, k in zip(values, k3)), current)
        updated = tuple(
            y + dt / 6 * (a + 2 * b + 2 * c + d)
            for y, a, b, c, d in zip(values, k1, k2, k3, k4)
        )
        crossed = crossed | ((values[0] < 0) & (updated[0] >= 0))
        values = updated
    return {**context.state, **dict(zip(keys, values))}, crossed.to(values[0].dtype)


def current_step(context):
    return {"value": context.drive.to(context.state["value"].dtype)}


DT_MS = 0.05
DURATION_MS = 100.0
PULSE_START_MS, PULSE_STOP_MS = 10.0, 70.0
CURRENT_NA = 0.1
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_PATH = OUTPUT_DIR / "hodgkin-huxley.png"


def prepare_results():
    if OUTPUT_DIR.is_symlink():
        OUTPUT_DIR.unlink()
    elif OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)


def build_example(dt_ms=DT_MS):

    extensions.register_neuron(
        "example.hodgkin_huxley/v1",
        step,
        initialize=initial_state,
        input_unit="nA",
        state_units={"m": "1", "h": "1", "n": "1"},
        capture_safe=True,
    )
    extensions.register_synapse(
        "example.injected_current/v1",
        current_step,
        output_unit="nA",
        capture_safe=True,
    )
    net = lang.Network("hodgkin_huxley", dt=dt_ms * lang.ms)
    pulse = net.input(
        "pulse", shape=("time", "batch", 1), signal_type="continuous", unit="1"
    )
    cells = net.population(
        "axon", size=1, neuron=lang.CustomNeuron("example.hodgkin_huxley/v1")
    )
    injection = net.connect(
        pulse,
        cells.excitatory,
        name="injection",
        synapse=lang.CustomSynapse("example.injected_current/v1"),
        weight=lang.Constant(CURRENT_NA),
        initialization_scaling="direct",
    )
    net.output("voltage", cells.voltage)
    net.output("spikes", cells.spikes)
    net.output("current", injection.current)
    for gate in ("m", "h", "n"):
        net.output(gate, cells.state(gate))
    bundle = lang.compile(net, target="tools/snnsim")
    stimulus = torch.zeros(round(DURATION_MS / dt_ms), 1, 1)
    stimulus[round(PULSE_START_MS / dt_ms) : round(PULSE_STOP_MS / dt_ms)] = 1
    return bundle, stimulus


def run_example(*, render_diagram=False):
    bundle, stimulus = build_example()
    if render_diagram:
        viz.render_diagram(
            lang.diagram(bundle, view="expanded"),
            OUTPUT_PATH.with_name("network.png"),
            scale=2,
            height_to_width_ratio=None,
        )
    result = simulate(
        graph=bundle.graph,
        device="cpu",
        input_bindings=(DenseArrayBinding("pulse", stimulus),),
        performance=Performance(compile_mode=None),
        extension_validation="strict",
    ).numpy(batch=0)
    return result


def main():
    prepare_results()
    result = run_example(render_diagram=True)
    outputs = {name: value.ravel() for name, value in result.outputs.items()}
    time = (np.arange(len(outputs["voltage"])) + 1) * DT_MS
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].plot(time, outputs["voltage"], color="#4c78a8", linewidth=1.2)
    spike_times = time[outputs["spikes"] > 0]
    axes[0].plot(
        spike_times,
        np.full_like(spike_times, 45),
        "v",
        color="#b33b33",
        label="Upward 0 mV crossing",
    )
    axes[0].set(
        ylabel="Voltage (mV)",
        title="Hodgkin–Huxley: continuous action potentials, no reset",
    )
    axes[0].legend(loc="upper right")
    for gate, color in zip(("m", "h", "n"), ("#e08b26", "#4c78a8", "#54a24b")):
        axes[1].plot(time, outputs[gate], label=gate, color=color)
    axes[1].set(ylabel="Gate probability", ylim=(0, 1))
    axes[1].legend(ncol=3)

    edges = np.arange(len(time) + 1) * DT_MS
    axes[2].stairs(outputs["current"], edges, color="#555555")
    axes[2].set(
        ylabel="Injected current (nA)", xlabel="Time (ms)", xlim=(0, DURATION_MS)
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT_PATH, dpi=160)
    plt.close(fig)
    plot_comparison(result)


def _rates(v):
    alpha = np.array(
        [
            1 / exprel(-(v + 40) / 10),
            0.07 * np.exp(-(v + 65) / 20),
            0.1 / exprel(-(v + 55) / 10),
        ]
    )
    beta = np.array(
        [
            4 * np.exp(-(v + 65) / 18),
            1 / (1 + np.exp(-(v + 35) / 10)),
            0.125 * np.exp(-(v + 65) / 80),
        ]
    )
    return alpha, beta


def solve_reference(times, *, pulse_start_ms=10.0, pulse_stop_ms=70.0, current_na=0.1):
    alpha, beta = _rates(-65)
    initial = np.r_[-65, alpha / (alpha + beta)]
    reference = np.empty((4, len(times)))
    event_times = []

    def crossing(t, y):
        return y[0]

    crossing.direction = 1
    for start, stop, current in (
        (0, pulse_start_ms, 0),
        (pulse_start_ms, pulse_stop_ms, current_na * 100),
        (pulse_stop_ms, float(times[-1]), 0),
    ):

        def derivative(t, y):
            v, m, h, n = y
            ionic = (
                120 * m**3 * h * (v - 50) + 36 * n**4 * (v + 77) + 0.3 * (v + 54.387)
            )
            a, b = _rates(v)
            return np.r_[current - ionic, a * (1 - y[1:]) - b * y[1:]]

        solution = solve_ivp(
            derivative,
            (start, stop),
            initial,
            method="DOP853",
            rtol=1e-10,
            atol=1e-12,
            dense_output=True,
            events=crossing,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        mask = (times > start) & (times <= stop)
        reference[:, mask] = solution.sol(times[mask])
        initial = solution.y[:, -1]
        event_times.extend(solution.t_events[0])
    return reference, np.asarray(event_times)


def compare(result, dt_ms=DT_MS):
    outputs = {name: value.ravel() for name, value in result.outputs.items()}
    times = (np.arange(len(outputs["voltage"])) + 1) * dt_ms
    reference, reference_spikes = solve_reference(
        times,
        pulse_start_ms=PULSE_START_MS,
        pulse_stop_ms=PULSE_STOP_MS,
        current_na=CURRENT_NA,
    )
    actual = np.stack([outputs[key] for key in ("voltage", "m", "h", "n")])
    error = actual - reference
    spike_times = times[outputs["spikes"] > 0]
    if len(spike_times) != len(reference_spikes):
        raise ValueError(
            f"Spike count mismatch: snnlab={len(spike_times)}, SciPy={len(reference_spikes)}"
        )
    metrics = {
        "network_dt_ms": dt_ms,
        "rk4_max_substep_ms": MAX_STEP_MS,
        "state_dtype": "float32",
        "reference_solver": "DOP853",
        "reference_rtol": 1e-10,
        "reference_atol": 1e-12,
        "voltage_max_abs_error_mv": float(np.abs(error[0]).max()),
        "voltage_rmse_mv": float(np.sqrt(np.mean(error[0] ** 2))),
        "gate_max_abs_error": {
            gate: float(np.abs(error[i]).max())
            for i, gate in enumerate(("m", "h", "n"), 1)
        },
        "snnlab_spike_times_ms": spike_times.tolist(),
        "scipy_spike_times_ms": reference_spikes.tolist(),
        "max_spike_time_error_ms": float(np.abs(spike_times - reference_spikes).max())
        if len(spike_times)
        else 0.0,
    }
    return times, actual, reference, metrics


def plot_comparison(result):
    times, actual, reference, metrics = compare(result)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].plot(times, actual[0], label="snnlab RK4, float32", color="#4c78a8")
    axes[0].plot(
        times,
        reference[0],
        "--",
        label="SciPy DOP853, float64",
        color="#e08b26",
        linewidth=1,
    )
    axes[0].set(
        ylabel="Voltage (mV)",
        title="HH extension compared with an independent SciPy solution",
    )
    axes[0].legend(loc="upper right")
    axes[1].plot(times, actual[0] - reference[0], color="#b33b33", linewidth=1)
    axes[1].set(ylabel="Voltage error (mV)")
    for index, gate in enumerate(("m", "h", "n"), 1):
        axes[2].plot(times, actual[index] - reference[index], label=gate)
    axes[2].set(ylabel="Gate error", xlabel="Time (ms)", xlim=(0, times[-1]))
    axes[2].legend(ncol=3)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT_DIR / "scipy-comparison.png", dpi=160)
    plt.close(fig)
    report = json.dumps(metrics, indent=2) + "\n"
    (OUTPUT_DIR / "comparison.json").write_text(report)


if __name__ == "__main__":
    import example

    example.main()
