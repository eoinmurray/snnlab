import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch

from snnlab import extensions, lang, viz
from snnlab.sim import (
    DenseArrayBinding,
    Performance,
    simulate,
)


def prepare_results():
    if OUTPUT_DIR.is_symlink():
        OUTPUT_DIR.unlink()
    elif OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)


def validate_adaptive(config):
    for name in ("tau_mem_ms", "tau_adapt_ms"):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(config["adaptation_na"]) or config["adaptation_na"] < 0:
        raise ValueError("adaptation_na must be finite and nonnegative")


def adaptive_initial_state(context):
    shape, device, dtype = context.shape, context.device, context.dtype
    return {
        "voltage": torch.full(shape, -65.0, device=device, dtype=dtype),
        "adaptation": torch.zeros(shape, device=device, dtype=dtype),
    }


def adaptive_step(context):
    voltage = context.state["voltage"]
    adaptation = context.state["adaptation"]
    beta = math.exp(-context.dt_ms / context.config["tau_mem_ms"])

    current = (
        context.excitatory.to(voltage.dtype)
        - context.inhibitory.to(voltage.dtype)
        - adaptation
    )
    voltage = (
        -65
        + (voltage + 65) * beta
        + current * context.config["tau_mem_ms"] * (1 - beta)
    )
    spikes = context.spike(voltage + 50)

    voltage = torch.where(spikes.bool(), -65.0, voltage)
    adaptation = adaptation * math.exp(-context.dt_ms / context.config["tau_adapt_ms"])
    adaptation = adaptation + spikes * context.config["adaptation_na"]
    return {**context.state, "voltage": voltage, "adaptation": adaptation}, spikes


def validate_exponential(config):
    if not math.isfinite(config["tau_ms"]) or config["tau_ms"] <= 0:
        raise ValueError("tau_ms must be finite and positive")


def exponential_initial_state(context):
    return {
        "value": torch.zeros(context.shape, device=context.device, dtype=context.dtype)
    }


def exponential_step(context):
    previous = context.state["value"]
    decay = math.exp(-context.dt_ms / context.config["tau_ms"])
    drive = context.drive.to(previous.dtype)
    return {**context.state, "value": previous * decay + drive}


def clipped_normal(shape, config, *, device, dtype):
    values = torch.randn(shape, device=device, dtype=dtype)
    return (values * config["std"] + config["mean"]).clamp(min=config["minimum"])


OUTPUT_DIR = Path(__file__).resolve().parent / "results"
DT_MS = 0.5
DURATION_MS = 500
SEED = 17


def register_definitions():
    extensions.register_neuron(
        "example.adaptive_lif/v1",
        f"{__name__}:adaptive_step",
        initialize=f"{__name__}:adaptive_initial_state",
        input_unit="nA",
        state_units={"adaptation": "nA"},
        validate=f"{__name__}:validate_adaptive",
        capture_safe=True,
    )
    extensions.register_synapse(
        "example.exponential_current/v1",
        f"{__name__}:exponential_step",
        initialize=f"{__name__}:exponential_initial_state",
        output_unit="nA",
        validate=f"{__name__}:validate_exponential",
        capture_safe=True,
    )
    extensions.register_initializer(
        "example.clipped_normal/v1",
        f"{__name__}:clipped_normal",
    )


def main():
    prepare_results()
    bundle_path = OUTPUT_DIR / "network.bundle"

    steps = round(DURATION_MS / DT_MS)
    rates = torch.full((steps, 1, 4), 20.0)
    rates[round(100 / DT_MS) : round(400 / DT_MS)] = 120.0
    generator = torch.Generator().manual_seed(SEED)
    input_spikes = (
        torch.rand(rates.shape, generator=generator) < rates * DT_MS / 1000
    ).float()
    binding = DenseArrayBinding("inputs", input_spikes)

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
            synapse=lang.CustomSynapse("example.exponential_current/v1", tau_ms=5.0),
            weight=weights,
        )
        net.output(f"{cells.id}_spikes", cells.spikes)
        net.expose(cells.voltage, name=f"{cells.id}_voltage")
        net.expose(projection.current, name=f"{cells.id}_current")
    net.expose(inputs, name="input_spikes")
    net.expose(adaptive.state("adaptation"), name="adaptation")

    bundle = lang.compile(net, target="tools/snnsim")
    bundle.write(bundle_path)

    viz.render_diagram(
        lang.diagram(bundle, view="expanded"),
        OUTPUT_DIR / "network.png",
        scale=2,
        height_to_width_ratio=None,
        canvas_size=(1920, 900),
    )
    result = simulate(
        bundle=bundle_path,
        input_bindings=(binding,),
        seed=SEED,
        device="cpu",
        performance=Performance(compile_mode=None),
    )
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, allow_nan=False) + "\n"
    )
    data = result.numpy(batch=0)

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


if __name__ == "__main__":
    import example

    example.main()
