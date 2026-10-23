"""Independent NumPy/SciPy reference and plots for the HH extension example.

Run: uv run python examples/hodgkin-huxley/scipy_comparison.py
Writes scipy-comparison.png, comparison.json and network.png beside this script.
The reference implements density equations independently of hh_neuron.py and
uses adaptive DOP853 integration, split exactly at the input discontinuities.
"""

import json
from pathlib import Path

import hodgkin_huxley as example
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import exprel

OUTPUT_DIR = Path(__file__).resolve().parent


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
    """Return (V, m, h, n) samples and upward 0 mV crossing times in ms.

    Density units: uF/cm^2, mS/cm^2, uA/cm^2, mV and ms. A 1,000 um^2 patch
    has area 1e-5 cm^2, so nA * 100 gives uA/cm^2. This reference never calls
    the extension's rates, derivative or integration routine.
    """
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


def compare(result, dt_ms=example.DT_MS):
    outputs = {name: value.ravel() for name, value in result.outputs.items()}
    times = (np.arange(len(outputs["voltage"])) + 1) * dt_ms
    reference, reference_spikes = solve_reference(
        times,
        pulse_start_ms=example.PULSE_START_MS,
        pulse_stop_ms=example.PULSE_STOP_MS,
        current_na=example.CURRENT_NA,
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
        "rk4_max_substep_ms": example.hh_neuron.MAX_STEP_MS,
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


def main():
    result = example.run_example(render_diagram=True)
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
    print(report, end="")


if __name__ == "__main__":
    main()
