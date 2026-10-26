"""Analyse a recurrent excitatory/inhibitory mean-field model.

Run: uv run python examples/meanfield/example.py
No spiking simulation or external dataset is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snnlab import lang
from snnlab import meanfield as mf
from snnlab.meanfield import closures

HERE = Path(__file__).resolve().parent


def configuration():
    return {
        "population_sizes": {
            "E": 4000,
            "I": 1000,
            "X": 4000,
        },
        "dt_ms": 0.05,
        "neuron": {
            "capacitance_nf": 0.25,
            "leak_us": 0.0167,
            "resting_mv": -70.0,
            "reset_mv": -60.0,
            "threshold_mv": -50.0,
            "excitatory_reversal_mv": 0.0,
            "inhibitory_reversal_mv": -80.0,
        },
        "refractory_ms": {
            "E": 2.0,
            "I": 1.0,
        },
        "tau_synapse_ms": 5.0,
        "mean_incoming_weights_us": {
            "E_to_E": 0.38571382833127793,
            "E_to_I": 0.7709475985337921,
            "I_to_E": 1.6070054247970984,
            "I_to_I": 1.6072907404674917,
            "X_to_E": 0.8676710825662012,
            "X_to_I": 0.86695020872046,
        },
        "external_rate_hz": 2.5,
        "voltage_noise_mv": 4.0,
        "rate_relaxation_ms": {
            "E": 20.0,
            "I": 5.0,
        },
        "alpha_points": 201,
        "gain_times_ms": [800.0, 1550.0],
        "gain_values": [1.0, 0.0],
        "duration_ms": 2300.0,
        "initial_e_perturbation_hz": 0.001,
        "eigenvalue_tolerance_per_ms": 1e-8,
        "sample_interval_ms": 0.5,
        "solver": {
            "method": "LSODA",
            "rtol": 1e-8,
            "rate_atol": 1e-10,
            "max_step_ms": 0.5,
        },
    }


def build_model(
    cfg=None,
    *,
    diagram_path=None,
):
    cfg = configuration() if cfg is None else cfg
    sizes = cfg["population_sizes"]
    neuron = cfg["neuron"]
    net = lang.Network(
        "recurrent_ei",
        dt=cfg["dt_ms"] * lang.ms,
    )
    external = net.input(
        "X",
        shape=("time", "batch", sizes["X"]),
        signal_type="spikes",
        unit="spike",
    )
    populations = {
        name: net.population(
            name,
            size=sizes[name],
            neuron=lang.COBA_LIF(
                **neuron,
                tau_mem=neuron["capacitance_nf"] / neuron["leak_us"] * lang.ms,
                refractory=cfg["refractory_ms"][name] * lang.ms,
            ),
        )
        for name in ("E", "I")
    }
    for name, coupling in cfg["mean_incoming_weights_us"].items():
        source, target = name.split("_to_")
        inhibitory = source == "I"
        fan_in = sizes[source] - int(source == target)
        net.connect(
            external if source == "X" else populations[source].spikes,
            populations[target].inhibitory
            if inhibitory
            else populations[target].excitatory,
            name=name,
            synapse=(lang.GABA if inhibitory else lang.AMPA)(
                tau=cfg["tau_synapse_ms"] * lang.ms
            ),
            weight=lang.Constant(coupling / fan_in),
            initialization_scaling="direct",
            connection="feedforward" if source == "X" else "recurrent",
        )
    bundle = lang.compile(net)
    if diagram_path is not None:
        bundle.visualise(diagram_path)
    model = mf.reduce(
        bundle,
        closure=closures.RateConductance(
            gain=closures.NoisyLIF(sigma=cfg["voltage_noise_mv"] * lang.mV),
            tau_rate=cfg["rate_relaxation_ms"],
            driving_force="resting",
        ),
        weights=mf.MeanIncomingWeights(
            cfg["mean_incoming_weights_us"],
            provenance="Explicit mean incoming conductance couplings",
        ),
        inputs={"X": mf.StationaryRate(cfg["external_rate_hz"] * lang.Hz)},
        external_filters="quasistatic",
    )
    control = mf.PiecewiseLinear(
        times=cfg["gain_times_ms"],
        values=cfg["gain_values"],
    )
    return model, control


def run(out_dir=HERE / "results"):
    cfg = configuration()
    out_dir = Path(out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    model, control = build_model(
        cfg,
        diagram_path=out_dir / "network.png",
    )
    conductance_per_hz = {
        f"{pathway}.conductance": cfg["tau_synapse_ms"] * coupling / 1000
        for pathway, coupling in cfg["mean_incoming_weights_us"].items()
        if not pathway.startswith("X_")
    }
    guess = {
        "E.rate": 2,
        "I.rate": 6,
    }
    guess.update(
        {
            name: scale * guess[name.split("_to_")[0] + ".rate"]
            for name, scale in conductance_per_hz.items()
        }
    )
    branch = mf.continuation(
        model,
        "X_to_I.gain",
        np.linspace(
            1,
            0,
            cfg["alpha_points"],
        ),
        guess=guess,
    )
    crossings = mf.find_stability_crossings(
        model,
        "X_to_I.gain",
        branch,
        tolerance=1e-10,
        eigenvalue_tolerance=cfg["eigenvalue_tolerance_per_ms"],
    )
    initial = dict(branch[0][1].state)
    initial["E.rate"] += cfg["initial_e_perturbation_hz"]
    trajectory = mf.integrate(
        model,
        duration=cfg["duration_ms"] * lang.ms,
        initial=initial,
        controls={"X_to_I.gain": control},
        sample_interval=cfg["sample_interval_ms"],
        method=cfg["solver"]["method"],
        rtol=cfg["solver"]["rtol"],
        atol=[
            cfg["solver"]["rate_atol"]
            * conductance_per_hz.get(
                name,
                1,
            )
            for name in model.states
        ],
        max_step=cfg["solver"]["max_step_ms"],
    )
    rows = [
        {
            "alpha": value,
            "state": list(eq.state.values()),
            "eigenvalues_real": stability.eigenvalues.real.tolist(),
            "eigenvalues_imag": stability.eigenvalues.imag.tolist(),
            "max_real_per_ms": stability.max_real,
        }
        for value, eq, stability in branch
    ]
    result = {
        "configuration": cfg,
        "states": list(model.states),
        "continuation": rows,
        "crossings": [
            {
                "alpha": crossing.value,
                "frequency_hz": crossing.frequency_hz,
                "oscillatory": crossing.oscillatory,
                "destabilizes_as_alpha_decreases": crossing.stable_before is True
                and crossing.stable_after is False,
            }
            for crossing in crossings
        ],
    }
    (out_dir / "analysis.json").write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n"
    )
    model.write(out_dir / "model.json")
    np.savez_compressed(
        out_dir / "trajectory.npz",
        time_ms=trajectory.time_ms,
        state=np.array(list(trajectory.state.values())).T,
    )
    plot(
        branch,
        crossings,
        trajectory,
        control,
        out_dir,
    )
    print(mf.summary(model))
    print(f"Oscillatory crossing candidates: {result['crossings']}")
    print(f"Results: {out_dir.resolve()}")
    return result, trajectory


def plot(
    branch,
    crossings,
    trajectory,
    control,
    out_dir,
):
    alpha = np.array([row[0] for row in branch])
    rates = np.array(
        [[row[1].state["E.rate"], row[1].state["I.rate"]] for row in branch]
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7),
        layout="constrained",
    )
    for index, name in enumerate(("E", "I")):
        axes[0, 0].plot(
            alpha,
            rates[:, index],
            label=name,
        )
        axes[1, 1].plot(
            trajectory.time_ms,
            trajectory.state[f"{name}.rate"],
            label=name,
            linewidth=0.8,
        )
    axes[0, 0].set(
        xlabel="Input→I gain α",
        ylabel="Equilibrium rate (Hz)",
    )
    axes[0, 0].legend()
    axes[0, 1].plot(
        alpha,
        [row[2].max_real for row in branch],
    )
    axes[0, 1].axhline(
        0,
        color="black",
        linewidth=0.8,
    )
    axes[0, 1].set(
        xlabel="Input→I gain α",
        ylabel="Largest Re λ (ms⁻¹)",
    )
    for crossing in crossings:
        axes[0, 1].axvline(
            crossing.value,
            color="grey",
            linestyle="--",
        )
    axes[1, 0].plot(
        trajectory.time_ms,
        [control(t) for t in trajectory.time_ms],
    )
    axes[1, 0].set(
        xlabel="Time (ms)",
        ylabel="Input→I gain α",
    )
    axes[1, 1].set(
        xlabel="Time (ms)",
        ylabel="Population rate (Hz)",
    )
    axes[1, 1].legend()
    fig.suptitle("Recurrent E/I mean field — decreasing external drive to inhibition")
    fig.savefig(
        out_dir / "analysis.png",
        dpi=160,
    )
    fig.savefig(out_dir / "analysis.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=HERE / "results",
    )
    run(parser.parse_args().out_dir)
