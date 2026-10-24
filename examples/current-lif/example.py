import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snnlab import lang, viz
from snnlab.sim import (
    PoissonInputBinding,
    simulate,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
DT_MS = 0.5
DURATION_MS = 500
INPUT_RATE_HZ = 80
SEED = 17


def prepare_results():
    if OUTPUT_DIR.is_symlink():
        OUTPUT_DIR.unlink()
    elif OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)


def main():
    prepare_results()

    net = lang.Network("current_lif", dt=DT_MS * lang.ms)

    inputs = net.input(
        "inputs", shape=("time", "batch", 4), signal_type="spikes", unit="spike"
    )
    poisson_input = PoissonInputBinding(
        input_id="inputs",
        steps_count=round(DURATION_MS / DT_MS),
        rates_hz=(INPUT_RATE_HZ,),
        seed=SEED,
    )

    cells = net.population(
        "E", size=16, neuron=lang.CUBA_LIF(tau_mem=20 * lang.ms, capacitance_nf=1.0)
    )

    projection = net.connect(
        inputs,
        cells.excitatory,
        name="input_to_E",
        synapse=lang.ExponentialCurrent(tau=5 * lang.ms),
        weight=lang.Uniform(4.0, 6.0),
        constraint=lang.NonNegative(),
    )

    net.output("spikes", cells.spikes)

    net.expose(inputs, name="input_spikes")
    net.expose(cells.voltage, name="e_voltage")
    net.expose(projection.current, name="e_current")

    bundle = lang.compile(net, target="tools/snnsim")
    diagram = lang.diagram(bundle, view="expanded")
    viz.render_diagram(
        diagram,
        OUTPUT_DIR / "network.png",
        scale=2,
        height_to_width_ratio=None,
        canvas_size=(1920, 900),
    )

    result = simulate(
        graph=bundle.graph,
        input_bindings=(poisson_input,),
        seed=SEED,
        device="cpu",
    )

    data = result.numpy(batch=0)
    input_spikes = data.diagnostics["input_spikes"]
    spikes = data.outputs["spikes"]
    voltages = data.diagnostics["e_voltage"]
    currents = data.diagnostics["e_current"]
    time_ms = data.time_ms

    grid = viz.FigureGrid(
        rows=4, columns=1, bounds=(0.14, 0.07, 0.82, 0.86), row_gap=0.07
    )
    grid.place("inputs", row=0, column=0)
    grid.place("spikes", row=1, column=0)
    grid.place("voltage", row=2, column=0)
    grid.place("current", row=3, column=0)
    figure = grid.figure(figsize=(8, 10), dpi=150)
    input_axis = grid.add_axes(figure, "inputs")
    spike_axis = grid.add_axes(figure, "spikes", sharex=input_axis)
    voltage_axis = grid.add_axes(figure, "voltage", sharex=input_axis)

    current_axis = grid.add_axes(figure, "current", sharex=input_axis)

    for axis, values, title, label in (
        (input_axis, input_spikes, "Input spikes (4 channels)", "Input channel"),
        (spike_axis, spikes, "E spikes (16 cells)", "E cell"),
    ):
        steps, cell_ids = np.nonzero(values)
        axis.scatter(time_ms[steps], cell_ids, marker="|", s=24, color="#1a1a1a")
        axis.set_ylim(-0.5, values.shape[1] - 0.5)
        axis.set_yticks(
            np.linspace(0, values.shape[1] - 1, min(values.shape[1], 4), dtype=int)
        )
        axis.set_ylabel(label)
        axis.set_title(title, pad=10)
        axis.tick_params(labelbottom=False)

    voltage_axis.plot(time_ms, voltages[:, 0], color="#1a1a1a", linewidth=1)
    voltage_axis.set_ylabel("Voltage (mV)")
    voltage_axis.set_title("Membrane voltage of E cell 0", pad=10)
    voltage_axis.tick_params(labelbottom=False)
    current_axis.plot(time_ms, currents[:, 0], color="#1a1a1a", linewidth=1)
    current_axis.set_ylabel("Current (nA)")
    current_axis.set_title("Synaptic current into E cell 0", pad=10)
    current_axis.set_xlabel("Time (ms)")
    voltage_axis.set_xlim(0, DURATION_MS)
    path = OUTPUT_DIR / "current-lif.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
