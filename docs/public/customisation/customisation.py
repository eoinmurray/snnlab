"""Compare a built-in current LIF with a registered adaptive current neuron.

Run from the repository root: uv run python examples/customisation/customisation.py
Outputs are saved beside this script. Diagram rendering requires Graphviz.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from snnlab import extensions, lang, viz
from snnlab.sim.execution import DenseArrayBinding, ExecutionSpec, simulate

OUTPUT_DIR = Path(__file__).resolve().parent
DT_MS = 0.5
DURATION_MS = 500
SEED = 17


# Register references to functions in the sibling importable module.
def register_definitions():
    extensions.register_neuron(
        "example.adaptive_lif/v1",
        "custom_neurons:adaptive_step",
        initialize="custom_neurons:adaptive_initial_state",
        input_unit="nA",
        state_units={"adaptation": "nA"},
    )
    extensions.register_initializer(
        "example.clipped_normal/v1", "custom_neurons:clipped_normal"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Load the saved bundle in a fresh process; do not build or register definitions.",
    )
    args = parser.parse_args()
    bundle_path = OUTPUT_DIR / "network.bundle"
    # 3. Define a stimulus with custom tensors: quiet, then sustained activity.
    steps = round(DURATION_MS / DT_MS)
    rates = torch.full((steps, 1, 4), 20.0)
    rates[round(100 / DT_MS) : round(400 / DT_MS)] = 120.0
    generator = torch.Generator().manual_seed(SEED)
    input_spikes = (
        torch.rand(rates.shape, generator=generator) < rates * DT_MS / 1000
    ).float()
    binding = DenseArrayBinding("inputs", input_spikes)

    # 4. Compare built-in and custom neuron dynamics with matched input weights.
    if not args.simulate:
        register_definitions()
        net = lang.Network("customisation", dt=DT_MS * lang.ms)
        inputs = net.input(
            "inputs", shape=("time", "batch", 4), signal_type="spikes", unit="spike"
        )
        standard = net.population(
            "standard", size=8, neuron=lang.CUBA_LIF(tau_mem=20 * lang.ms)
        )
        adaptive = net.population(
            "adaptive",
            size=8,
            neuron=lang.CustomNeuron(
                "example.adaptive_lif/v1",
                tau_mem_ms=20,
                tau_adapt_ms=100,
                adaptation_na=0.35,
            ),
        )
        weights = net.parameter(
            "weights",
            shape=(8, 4),
            unit="nA",
            initializer=lang.CustomInitializer(
                "example.clipped_normal/v1", mean=5.0, std=1.0, minimum=0.0
            ),
            constraint=lang.NonNegative(),
        )
        for cells in (standard, adaptive):
            projection = net.connect(
                inputs,
                cells.excitatory,
                name=f"input_to_{cells.id}",
                synapse=lang.ExponentialCurrent(tau=5 * lang.ms),
                weight=weights,
            )
            net.output(f"{cells.id}_spikes", cells.spikes)
            net.expose(cells.voltage, name=f"{cells.id}_voltage")
            net.expose(projection.current, name=f"{cells.id}_current")
        net.expose(inputs, name="input_spikes")
        net.expose(adaptive.state("adaptation"), name="adaptation")

        bundle = lang.compile(net, target="tools/snnsim")
        bundle.write(bundle_path)

    # 5. Compile and execute: the bundle stores names/config, not Python code.
    bundle = lang.load_bundle(bundle_path)
    viz.render_diagram(
        lang.diagram(bundle, view="expanded"),
        OUTPUT_DIR / "network.png",
        scale=2,
        height_to_width_ratio=None,
        canvas_size=(1920, 900),
    )
    execution = ExecutionSpec(
        kind="simulate",
        bundle=bundle_path,
        input_bindings=(binding,),
        seed=SEED,
        device="cpu",
    )
    result = simulate(execution)
    data = result.numpy(batch=0)
    assert data.time_ms is not None

    # 6. Plot the same stimulus, both responses and the custom adaptation state.
    import matplotlib.pyplot as plt

    layout = viz.FigureGrid(
        rows=4, columns=1, row_gap=0.035, bounds=(0.14, 0.07, 0.82, 0.86)
    )
    for index, name in enumerate(("inputs", "spikes", "voltage", "adaptation")):
        layout.place(name, row=index, column=0)
    figure = layout.figure(figsize=(11, 11))
    axes = [layout.add_axes(figure, "inputs")]
    for name in ("spikes", "voltage", "adaptation"):
        axes.append(layout.add_axes(figure, name, sharex=axes[0]))
    for channel in range(4):
        axes[0].plot(
            data.time_ms[data.diagnostics["input_spikes"][:, channel] > 0],
            np.full(int(data.diagnostics["input_spikes"][:, channel].sum()), channel),
            "|",
            color=viz.Theme().ink,
        )
    axes[0].set_ylabel("Input channel")
    for prefix, offset, color in (
        ("standard", 0, viz.Theme().ink),
        ("adaptive", 9, viz.Theme().accent),
    ):
        spikes = data.outputs[f"{prefix}_spikes"]
        for cell in range(8):
            times = data.time_ms[spikes[:, cell] > 0]
            axes[1].plot(times, np.full(len(times), cell + offset), "|", color=color)
        axes[2].plot(
            data.time_ms,
            data.diagnostics[f"{prefix}_voltage"][:, 0],
            color=color,
            label=prefix,
        )
    axes[1].set_ylabel("Cell")
    axes[1].set_yticks([3.5, 12.5], ["Standard", "Adaptive"])
    axes[2].set_ylabel("Voltage (mV)")
    axes[2].legend(loc="upper right")
    axes[3].plot(
        data.time_ms, data.diagnostics["adaptation"][:, 0], color=viz.Theme().accent
    )
    axes[3].set_ylabel("Adaptation (nA)")
    axes[3].set_xlabel("Time (ms)")
    for axis in axes:
        axis.axvspan(100, 400, color=viz.Theme().rule, alpha=0.5, zorder=-1)
        axis.set_xlim(0, DURATION_MS)
    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)
    figure.savefig(OUTPUT_DIR / "customisation.png", dpi=160)
    plt.close(figure)
    for prefix in ("standard", "adaptive"):
        print(f"{prefix}: {int(data.outputs[f'{prefix}_spikes'].sum())} spikes")
    print(f"Saved {OUTPUT_DIR / 'customisation.png'}")


if __name__ == "__main__":
    main()
