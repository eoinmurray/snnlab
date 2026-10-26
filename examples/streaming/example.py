import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from snnlab import lang, sim, viz

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
DT_MS = 0.1
WINDOW_STEPS = 50
WINDOW_MS = WINDOW_STEPS * DT_MS
WINDOWS = 40
SEED = 17


def prepare_results():
    if OUTPUT_DIR.is_symlink():
        OUTPUT_DIR.unlink()
    elif OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)


def make_network():
    net = lang.Network(
        "streaming",
        dt=DT_MS * lang.ms,
    )
    events = net.input(
        "events",
        shape=("time", "batch", 4),
        signal_type="spikes",
        unit="spike",
    )
    cells = net.population(
        "E",
        size=8,
        neuron=lang.COBA_LIF(tau_mem=20 * lang.ms),
    )
    net.connect(
        events,
        cells.excitatory,
        name="drive",
        synapse=lang.AMPA(tau=2 * lang.ms),
        weight=lang.Uniform(
            0.1,
            0.5,
        ),
        constraint=lang.NonNegative(),
    )
    net.output(
        "spikes",
        cells.spikes,
    )
    net.output(
        "voltage",
        cells.voltage,
    )
    return net


def incoming_windows():
    generator = torch.Generator().manual_seed(SEED + 1)
    for index in range(WINDOWS):
        rate_hz = 160 if WINDOWS // 4 <= index < 3 * WINDOWS // 4 else 40
        yield torch.poisson(
            torch.full(
                (WINDOW_STEPS, 1, 4),
                rate_hz * DT_MS / 1000,
                dtype=torch.float32,
            ),
            generator=generator,
        )


def run_stream(bundle):
    windows = []
    times = []
    spikes = []
    voltages = []
    with sim.prepare_stream(
        bundle,
        window_steps=WINDOW_STEPS,
        batch_size=1,
        seed=SEED,
        device="cpu",
        dtype=torch.float32,
    ) as stream:
        for window in incoming_windows():
            result = stream.advance({"events": window})
            data = result.numpy(batch=0)
            windows.append(window[:, 0].numpy())
            times.append(data.time_ms)
            spikes.append(data.outputs["spikes"])
            voltages.append(data.outputs["voltage"])
        summary = {
            "seed": SEED,
            "input_seed": SEED + 1,
            "dt_ms": DT_MS,
            "window_steps": WINDOW_STEPS,
            "windows": WINDOWS,
            "completed_steps": stream.completed_steps,
            "model_time_ms": stream.completed_steps * DT_MS,
            "stream_info": stream.info,
        }
    arrays = {
        "time_ms": np.concatenate(times),
        "input_spikes": np.concatenate(windows),
        "spikes": np.concatenate(spikes),
        "voltage_mv": np.concatenate(voltages),
    }
    return arrays, summary


def plot_results(data):
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10, 7),
        dpi=150,
        sharex=True,
        layout="constrained",
    )
    for axis, key, label in (
        (axes[0], "input_spikes", "Input channel"),
        (axes[1], "spikes", "E cell"),
    ):
        steps, cells = np.nonzero(data[key])
        axis.scatter(
            data["time_ms"][steps],
            cells,
            marker="|",
            s=30,
            color="#1a1a1a",
        )
        axis.set_ylabel(label)
        axis.set_yticks(range(data[key].shape[1]))
        axis.set_ylim(
            -0.5,
            data[key].shape[1] - 0.5,
        )
    axes[0].set_title("40 Hz/channel → 160 Hz/channel → 40 Hz/channel")
    axes[1].set_title("Eight neurons; state continues across input windows")
    axes[2].plot(
        data["time_ms"],
        data["voltage_mv"][:, 0],
        color="#a74727",
        linewidth=1,
    )
    axes[2].set_title("E cell 0; dotted lines mark 5 ms window boundaries")
    axes[2].set_ylabel("Voltage (mV)")
    axes[2].set_xlabel("Model time (ms)")
    for axis in axes:
        for boundary in np.arange(WINDOW_MS, WINDOWS * WINDOW_MS, WINDOW_MS):
            axis.axvline(
                boundary,
                color="0.8",
                linestyle=":",
                linewidth=0.6,
                zorder=0,
            )
        axis.axvspan(
            WINDOWS * WINDOW_MS / 4,
            3 * WINDOWS * WINDOW_MS / 4,
            color="#a74727",
            alpha=0.07,
            zorder=0,
        )
    axes[2].set_xlim(
        0,
        WINDOWS * WINDOW_MS,
    )
    figure.savefig(OUTPUT_DIR / "streaming.png")
    plt.close(figure)


def main():
    prepare_results()
    bundle = lang.compile(make_network())
    viz.render_diagram(
        lang.diagram(
            bundle,
            view="expanded",
        ),
        OUTPUT_DIR / "network.png",
        scale=2,
        height_to_width_ratio=None,
        canvas_size=(1920, 900),
    )
    data, summary = run_stream(bundle)
    np.savez_compressed(
        OUTPUT_DIR / "streaming.npz",
        **data,
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )
    plot_results(data)


if __name__ == "__main__":
    main()
