"""Simulate a single excitatory layer and save one figure beside this script."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snnlab import lang, viz
from snnlab.sim.execution import ExecutionSpec, PoissonInputBinding, simulate

OUTPUT_DIR = Path(__file__).resolve().parent
DT_MS = 0.1
DURATION_MS = 300
INPUT_RATE_HZ = 80
SEED = 17


def main():
    # 1. Create the network.
    net = lang.Network("quickstart", dt=DT_MS * lang.ms)
    # 2. Define the input and its stimulus binding.
    inputs = net.input(
        "inputs", shape=("time", "batch", 4), signal_type="spikes", unit="spike"
    )
    poisson_input = PoissonInputBinding(
        input_id="inputs",
        steps_count=round(DURATION_MS / DT_MS),
        rates_hz=(INPUT_RATE_HZ,),
        seed=SEED,
    )
    # 3. Define the excitatory layer and its input projection.
    cells = net.population("E", size=16, neuron=lang.COBA_LIF(tau_mem=20 * lang.ms))
    # Connect all 4 input channels to all 16 E cells using 64 weights.
    net.connect(
        inputs,
        cells.excitatory,
        name="input_to_E",
        synapse=lang.AMPA(tau=2 * lang.ms),
        weight=lang.Uniform(0.1, 0.5),
        constraint=lang.NonNegative(),
    )
    # 4. Choose outputs and exposed diagnostics.
    # The official network output: spikes.
    net.output("spikes", cells.spikes)
    # Expose input spikes and membrane voltage for diagnostics.
    net.expose(inputs, name="input_spikes")
    net.expose(cells.voltage, name="e_voltage")
    # 5. Compile the network into a bundle.
    bundle = lang.compile(net, target="tools/snnsim")
    diagram = lang.diagram(bundle, view="expanded")
    viz.render_diagram(
        diagram, OUTPUT_DIR / "network.png", scale=2, height_to_width_ratio=None
    )

    # 6. Describe the execution and simulate.
    execution = ExecutionSpec(
        kind="simulate",
        graph=bundle.graph,
        input_bindings=(poisson_input,),
        seed=SEED,
        device="cpu",
    )
    result = simulate(execution)
    # 7. Retrieve named results and select the first (only) batch item.
    data = result.numpy(batch=0)
    input_spikes = data.diagnostics["input_spikes"]
    spikes = data.outputs["spikes"]
    voltages = data.diagnostics["e_voltage"]
    time_ms = data.time_ms
    assert time_ms is not None

    # Plot the results with snnlab.viz and Matplotlib.
    grid = viz.FigureGrid(
        rows=3, columns=1, bounds=(0.12, 0.09, 0.85, 0.85), row_gap=0.07
    )
    grid.place("inputs", row=0, column=0)
    grid.place("spikes", row=1, column=0)
    grid.place("voltage", row=2, column=0)
    figure = grid.figure(figsize=(8, 7.5), dpi=150)
    input_axis = grid.add_axes(figure, "inputs")
    spike_axis = grid.add_axes(figure, "spikes", sharex=input_axis)
    voltage_axis = grid.add_axes(figure, "voltage", sharex=input_axis)

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
    voltage_axis.set_xlabel("Time (ms)")
    voltage_axis.set_xlim(0, DURATION_MS)
    path = OUTPUT_DIR / "quickstart.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {path}")

    counts = spikes.sum(axis=0)
    print(f"Total spikes: {int(counts.sum())}")
    print(f"Spikes per cell: {counts.astype(int).tolist()}")


if __name__ == "__main__":
    main()
