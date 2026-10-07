"""Compare built-in LIF, ALIF and AdEx for CUBA and COBA input families.

Run with: uv run python examples/neurons/neurons.py
Requires Graphviz's dot command for the network diagram.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from snnlab import lang, viz
from snnlab.sim.execution import DenseArrayBinding, ExecutionSpec, simulate

OUTPUT_DIR = Path(__file__).resolve().parent
DT_MS = 0.25
DURATION_MS = 400.0


def compare(input_family):
    current = input_family == "cuba"
    common = dict(
        tau_mem=20 * lang.ms,
        capacitance_nf=1.0,
        refractory_steps=4,
        voltage_grad_dampen=1.0,
    )
    lif, alif, adex = (
        (lang.CUBA_LIF, lang.CUBA_ALIF, lang.CUBA_ADEX)
        if current
        else (lang.COBA_LIF, lang.COBA_ALIF, lang.COBA_ADEX)
    )
    synapse = lang.ExponentialCurrent if current else lang.AMPA
    weight = 0.8 if current else 0.015
    net = lang.Network(f"{input_family}_neuron_comparison", dt=DT_MS * lang.ms)
    events = net.input(
        "events", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    models = {
        "LIF": lif(**common),
        "ALIF": alif(**common, adaptation_increment_mv=4.0),
        "AdEx": adex(**common, a_us=0.01, b_na=0.4),
    }
    for name, neuron in models.items():
        cells = net.population(name, size=1, neuron=neuron)
        net.connect(
            events,
            cells.excitatory,
            name=f"drive_{name}",
            synapse=synapse(tau=5 * lang.ms),
            weight=lang.Constant(weight),
            initialization_scaling="direct",
            constraint=lang.NonNegative(),
        )
        net.output(f"{name}_voltage", cells.voltage)
        net.output(f"{name}_spikes", cells.spikes)
        if name != "LIF":
            net.output(f"{name}_adaptation", cells.state("adaptation"))
    bundle = lang.compile(net, target="tools/snnsim")
    viz.render_diagram(
        lang.diagram(bundle, view="expanded"),
        OUTPUT_DIR / f"network-{input_family}.png",
        scale=2,
        canvas_size=(1920, 1800),
        height_to_width_ratio=None,
    )
    steps = round(DURATION_MS / DT_MS)
    stimulus = torch.zeros(steps, 1, 1)
    stimulus[::4] = 1  # One event each millisecond, identical for every model.
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            graph=bundle.graph,
            device="cpu",
            seed=17,
            input_bindings=(DenseArrayBinding("events", stimulus),),
        )
    ).numpy(batch=0)
    time = (np.arange(steps) + 1) * DT_MS
    colors = {"LIF": "#4c78a8", "ALIF": "#f58518", "AdEx": "#54a24b"}
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), constrained_layout=True)
    for row, name in enumerate(models):
        voltage = result.outputs[f"{name}_voltage"].ravel()
        spikes = result.outputs[f"{name}_spikes"].ravel()
        axes[row, 0].plot(time, voltage, color=colors[name], linewidth=1)
        axes[row, 0].set(xlim=(0, 70), ylabel=f"{name}\nVoltage (mV)")
        axes[row, 0].axhline(-50, color="gray", linestyle=":", linewidth=1)
        if name == "LIF":
            adaptation = np.zeros(steps)
        else:
            adaptation = result.outputs[f"{name}_adaptation"].ravel()
        axes[row, 1].plot(time, adaptation, color=colors[name])
        axes[row, 1].set(
            xlim=(0, DURATION_MS),
            ylabel="Current (nA)" if name == "AdEx" else "Threshold offset (mV)",
        )
        axes[row, 2].eventplot(time[spikes > 0], colors=colors[name], linewidths=1)
        axes[row, 2].set(xlim=(0, DURATION_MS), yticks=[])
        count = int(spikes.sum())
        axes[row, 2].set_ylabel(f"{count} spikes")
        print(f"{input_family.upper()} {name}: {count} spikes in {DURATION_MS:g} ms")
        for ax in axes[row]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel("Time (ms)")
    axes[0, 0].set_title("Membrane voltage: first 70 ms")
    axes[0, 1].set_title("Adaptation: full trial")
    axes[0, 2].set_title("Spike times: full trial")
    drive = "current" if current else "conductance"
    fig.suptitle(
        f"{input_family.upper()}: LIF, ALIF and AdEx under the same {drive} input",
        fontsize=15,
    )
    fig.savefig(OUTPUT_DIR / f"neurons-{input_family}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    for family in ("cuba", "coba"):
        compare(family)
