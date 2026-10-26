"""Two coupled E/I circuits with explicit delays and a delivery-time onset.

Run: uv run python examples/meanfield-delays/example.py
"""

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


def build_model(*, delays="explicit"):
    net = lang.Network("delayed_coupled_ei", dt=0.05 * lang.ms)
    populations = {
        name: net.population(
            name,
            size=100 if name.startswith("E") else 25,
            neuron=lang.COBA_LIF(
                tau_mem=15 * lang.ms,
                refractory=2 * lang.ms,
            ),
        )
        for name in ("E_A", "I_A", "E_B", "I_B")
    }
    weights = {}
    for source in populations:
        for target in populations:
            cross = source[-1] != target[-1]
            if cross and (source.startswith("I") or target.startswith("I")):
                continue
            inhibitory = source.startswith("I")
            name = f"{source}_to_{target}"
            weights[name] = 0.15 if inhibitory else (0.01 if cross else 0.03)
            net.connect(
                populations[source].spikes,
                populations[target].inhibitory
                if inhibitory
                else populations[target].excitatory,
                name=name,
                synapse=(lang.GABA if inhibitory else lang.AMPA)(
                    tau=(5 if inhibitory else 3) * lang.ms,
                ),
                delay=(2 if cross else 0.1) * lang.ms,
                weight=lang.Constant(
                    weights[name] / (populations[source].size - int(source == target)),
                ),
                initialization_scaling="direct",
                connection="recurrent",
                modulation=lang.StepGain(
                    at=500 * lang.ms,
                    before=0,
                    after=1,
                )
                if cross
                else None,
            )
    bundle = lang.compile(net)
    model = mf.reduce(
        bundle,
        closure=closures.RateConductance(
            gain=closures.NoisyLIF(sigma=4 * lang.mV),
            tau_rate={name: 5 if name.startswith("E") else 2 for name in populations},
        ),
        weights=mf.MeanIncomingWeights(
            weights,
            provenance="Illustrative explicit mean incoming conductances; no spiking fit",
        ),
        applied_current={"E_A": 0.6, "I_A": 0.55, "E_B": 0.62, "I_B": 0.55},
        delays=delays,
        modulation="translate",
    )
    return model, bundle


def run(out_dir=HERE / "results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, bundle = build_model()
    bundle.visualise(out_dir / "network.png")
    initial = dict.fromkeys(model.states, 0.0)
    initial.update({"E_A.rate": 5, "I_A.rate": 10, "E_B.rate": 7, "I_B.rate": 10})
    settings = {
        "duration": 600,
        "initial": initial,
        "history": mf.ConstantHistory(initial),
        "sample_interval": 0.5,
        "method": "DOP853",
        "max_step": 0.1,
        "rtol": 1e-8,
        "atol": 1e-10,
    }
    switched = mf.integrate(model, **settings)
    uncoupled = mf.integrate(
        model,
        controls={name: mf.PiecewiseConstant((), (0,)) for name in model.controls},
        **settings,
    )
    model.write(out_dir / "model.json")
    switched.write(out_dir / "switched.json")
    uncoupled.write(out_dir / "uncoupled.json")
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex="col", layout="constrained")
    for ax, kind in zip(axes.flat, ("E", "E", "I", "I"), strict=True):
        for circuit, color in (("A", "tab:blue"), ("B", "tab:orange")):
            name = f"{kind}_{circuit}.rate"
            ax.plot(
                switched.time_ms,
                switched.state[name],
                color=color,
                label=f"{kind}_{circuit}, onset at 500 ms",
            )
            ax.plot(
                uncoupled.time_ms,
                uncoupled.state[name],
                color=color,
                linestyle="--",
                label=f"{kind}_{circuit}, uncoupled",
            )
        ax.axvline(500, color="grey", linewidth=0.8)
        ax.set_ylabel("Rate (Hz)")
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time (ms)")
    for ax in axes[:, 1]:
        ax.set_xlim(470, 600)
    axes[0, 0].set_title("Full trajectory")
    axes[0, 1].set_title("Coupling onset")
    fig.suptitle("Illustrative delayed coupled E/I mean field")
    fig.savefig(out_dir / "trajectory.png", dpi=160)
    plt.close(fig)
    result = {
        "internal_delay_ms": 0.1,
        "cross_delay_ms": 2,
        "coupling_onset_ms": 500,
        "modulation_timing": "target_delivery",
        "max_rate_change_after_onset_hz": max(
            float(np.max(np.abs(switched.state[n] - uncoupled.state[n])))
            for n in model.states
            if n.endswith(".rate")
        ),
        "scientific_scope": "Illustrative stationary gain/noise closure; no quantitative exp133 reproduction or gamma claim",
    }
    (out_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Results: {out_dir.resolve()}")
    return result, switched, uncoupled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    run(parser.parse_args().out_dir)
