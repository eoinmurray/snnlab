"""Run a genuine ionic-conductance neuron through snnlab's extension interface.

Run: uv run python examples/hodgkin-huxley/hodgkin_huxley.py
Writes hodgkin-huxley.png and network.png beside this script.
Graphviz's dot command is required for the network diagram.
The sibling hh_neuron.py must remain importable when reloading a saved bundle.
This is a forward-simulation example; binary spike detection has no surrogate
gradient. RK4 substeps are at most 0.025 ms, without clipping channel gates.
"""

from pathlib import Path

import hh_neuron
import matplotlib.pyplot as plt
import numpy as np
import torch

from snnlab import extensions, lang, viz
from snnlab.sim import DenseArrayBinding, Performance, simulate

DT_MS = 0.05
DURATION_MS = 100.0
PULSE_START_MS, PULSE_STOP_MS = 10.0, 70.0
CURRENT_NA = 0.1
OUTPUT_PATH = Path(__file__).resolve().with_name("hodgkin-huxley.png")


def build_example(dt_ms=DT_MS):
    # Call once per process: registration records importable callback references.
    extensions.register_neuron(
        "example.hodgkin_huxley/v1",
        hh_neuron.step,
        initialize=hh_neuron.initial_state,
        input_unit="nA",
        state_units={"m": "1", "h": "1", "n": "1"},
        capture_safe=True,
    )
    extensions.register_synapse(
        "example.injected_current/v1",
        hh_neuron.current_step,
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
    # Values apply over intervals beginning at k*dt, rather than (k+1)*dt.
    edges = np.arange(len(time) + 1) * DT_MS
    axes[2].stairs(outputs["current"], edges, color="#555555")
    axes[2].set(
        ylabel="Injected current (nA)", xlabel="Time (ms)", xlim=(0, DURATION_MS)
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT_PATH, dpi=160)
    plt.close(fig)
    print(
        f"{len(spike_times)} spikes; voltage range {outputs['voltage'].min():.1f} to {outputs['voltage'].max():.1f} mV"
    )
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
