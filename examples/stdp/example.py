import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from snnlab import lang, viz
from snnlab.sim import (
    GraphExecutor,
    RecordingSpec,
    SignalRecording,
    plan_graph,
)
from snnlab.sim.plasticity import stdp_simple

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
DT_MS = 0.5
STEPS_PER_SECOND = round(1000 / DT_MS)
LEARNING_SECONDS = 100
PROBE_SECONDS = 5
N_EXC = 1000
N_INH = 200
INH_RATE_HZ = 10.0
W_MAX_US = 0.001
W_INH_US = 0.002
A_PLUS_US = 0.02 * W_MAX_US
A_MINUS_US = 1.05 * A_PLUS_US
TAU_MS = 20.0
SEED = 7

# One trial per experiment: label, excitatory input rate (Hz) and the correlation
# within the first half of the excitatory inputs.
EXPERIMENTS = (
    ("10 Hz, independent", 10.0, 0.0),
    ("10 Hz, half correlated", 10.0, 0.2),
    ("20 Hz, independent", 20.0, 0.0),
)
GROUP = N_EXC // 2
COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
INK = "#1a1a1a"
MUTED = "#8a8985"


def prepare_results():
    if OUTPUT_DIR.is_symlink():
        OUTPUT_DIR.unlink()
    elif OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)


def build_network():
    net = lang.Network(
        "stdp",
        dt=DT_MS * lang.ms,
    )

    exc = net.input(
        "exc",
        shape=("time", "batch", N_EXC),
        signal_type="spikes",
        unit="spike",
    )
    inh = net.input(
        "inh",
        shape=("time", "batch", N_INH),
        signal_type="spikes",
        unit="spike",
    )

    cell = net.population(
        "neuron",
        size=1,
        neuron=lang.COBA_LIF(tau_mem=20 * lang.ms),
    )
    net.connect(
        exc,
        cell.excitatory,
        name="exc_to_neuron",
        synapse=lang.AMPA(tau=5 * lang.ms),
        weight=lang.Uniform(
            0.0,
            W_MAX_US,
        ),
        initialization_scaling="direct",
        plasticity=lang.STDPSimple(
            a_plus=A_PLUS_US,
            a_minus=A_MINUS_US,
            tau_plus=TAU_MS * lang.ms,
            tau_minus=TAU_MS * lang.ms,
            w_max=W_MAX_US,
            pairing="all_to_all",
        ),
    )
    net.connect(
        inh,
        cell.inhibitory,
        name="inh_to_neuron",
        synapse=lang.GABA(tau=5 * lang.ms),
        weight=lang.Constant(W_INH_US),
        initialization_scaling="direct",
    )

    net.output(
        "spikes",
        cell.spikes,
    )
    return net


def input_chunk(rng):
    """One second of Poisson spikes for every experiment, as (time, batch, inputs)."""
    p_inh = INH_RATE_HZ * DT_MS / 1000
    exc = np.empty((STEPS_PER_SECOND, len(EXPERIMENTS), N_EXC), dtype=bool)
    for trial, (_, rate_hz, correlation) in enumerate(EXPERIMENTS):
        p = rate_hz * DT_MS / 1000
        exc[:, trial] = rng.random((STEPS_PER_SECOND, N_EXC)) < p
        if correlation > 0:
            # Each input of the first half keeps its own spikes at rate (1 - c)·r
            # and copies a shared train at rate c·r.
            own = rng.random((STEPS_PER_SECOND, GROUP)) < (1 - correlation) * p
            shared = rng.random((STEPS_PER_SECOND, 1)) < correlation * p
            exc[:, trial, :GROUP] = own | shared
    inh = rng.random((STEPS_PER_SECOND, len(EXPERIMENTS), N_INH)) < p_inh
    return {
        "exc": torch.from_numpy(exc).float(),
        "inh": torch.from_numpy(inh).float(),
    }


def learn(model, rng, seconds):
    recording = RecordingSpec(
        signals=(
            SignalRecording(
                "exc_to_neuron.weight",
                every_steps=STEPS_PER_SECOND,
            ),
        ),
    )
    state = None
    rates, snapshots = [], []
    for _ in range(seconds):
        result = model(
            input_chunk(rng),
            runtime_state=state,
            recording=recording,
            diagnostics=False,
        )
        state = result.runtime_state
        rates.append(result.outputs["spikes"].sum(dim=0)[:, 0])
        snapshots.append(result.recorded_signals["exc_to_neuron.weight"][-1, :, :, 0])
    final = result.parameters["exc_to_neuron.weight"][:, :, 0]
    return state, torch.stack(rates), torch.stack(snapshots), final


def probe(model, rng, state, seconds):
    spikes = 0
    for _ in range(seconds):
        result = model(
            input_chunk(rng),
            runtime_state=state,
            learning=False,
            diagnostics=False,
        )
        state = result.runtime_state
        spikes = spikes + result.outputs["spikes"].sum(dim=0)[:, 0]
    return spikes / seconds


def learning_window(max_lag_ms=100):
    rule = stdp_simple.STDPSimple(
        a_plus=A_PLUS_US,
        a_minus=A_MINUS_US,
        tau_plus_ms=TAU_MS,
        tau_minus_ms=TAU_MS,
        w_max=W_MAX_US,
        pairing="all_to_all",
    )
    max_lag = round(max_lag_ms / DT_MS)
    lags = np.arange(-max_lag, max_lag + 1, 4)
    # One trial per lag: a presynaptic spike at `start` and a postsynaptic spike
    # `lag` steps later.
    start = max_lag + 1
    pre = torch.zeros(2 * max_lag + 3, len(lags), 1, dtype=torch.float64)
    post = torch.zeros_like(pre)
    pre[start, :, 0] = 1
    post[start + lags, np.arange(len(lags)), 0] = 1
    w0 = torch.full((len(lags), 1, 1), W_MAX_US / 2, dtype=torch.float64)
    weights = w0
    state = stdp_simple.init_state(
        rule,
        batch=len(lags),
        n_source=1,
        n_target=1,
    )
    for pre_t, post_t in zip(pre, post, strict=True):
        weights, state = stdp_simple.step(
            rule,
            state,
            weights,
            pre_t,
            post_t,
            dt_ms=DT_MS,
        )
    return lags * DT_MS, (weights - w0).flatten().numpy()


def run_example(learning_seconds=LEARNING_SECONDS, probe_seconds=PROBE_SECONDS):
    bundle = lang.compile(
        build_network(),
        target="tools/snnsim",
    )
    model = GraphExecutor(
        plan_graph(bundle.graph),
        seed=SEED,
    )
    initial = model.parameter_map()["exc_to_neuron.weight"].detach().flatten()
    rng = np.random.default_rng(SEED)

    rate_before = probe(model, rng, None, probe_seconds)
    state, rates, snapshots, final = learn(model, rng, learning_seconds)
    rate_after = probe(model, rng, state, probe_seconds)

    return {
        "bundle": bundle,
        "initial": initial.numpy(),
        "rates": rates.numpy(),
        "snapshots": snapshots.numpy(),
        "final": final.numpy(),
        "rate_before": rate_before.numpy(),
        "rate_after": rate_after.numpy(),
        "window": learning_window(),
    }


def summarise(run):
    final = run["final"] / W_MAX_US
    experiments = []
    for trial, (label, rate_hz, correlation) in enumerate(EXPERIMENTS):
        experiments.append(
            {
                "label": label,
                "input_rate_hz": rate_hz,
                "correlation_first_half": correlation,
                "output_rate_before_hz": round(float(run["rate_before"][trial]), 2),
                "output_rate_after_hz": round(float(run["rate_after"][trial]), 2),
                "mean_weight_first_half": round(float(final[trial, :GROUP].mean()), 4),
                "mean_weight_second_half": round(float(final[trial, GROUP:].mean()), 4),
                "fraction_below_0.1": round(float(np.mean(final[trial] < 0.1)), 4),
                "fraction_above_0.9": round(float(np.mean(final[trial] > 0.9)), 4),
            }
        )
    return {
        "learning_seconds": len(run["rates"]),
        "seed": SEED,
        "weights_relative_to": "w_max",
        "experiments": experiments,
        "final_weights_sha256": hashlib.sha256(run["final"].tobytes()).hexdigest(),
    }


def plot_window(window):
    lags_ms, change = window
    analytic_ms = np.linspace(lags_ms[0], lags_ms[-1], 801)
    analytic = np.where(
        analytic_ms > 0,
        A_PLUS_US * np.exp(-analytic_ms / TAU_MS),
        -A_MINUS_US * np.exp(analytic_ms / TAU_MS),
    )
    figure, axis = plt.subplots(
        figsize=(7, 3.6),
        dpi=150,
        layout="constrained",
    )
    axis.axhline(
        0,
        color=MUTED,
        linewidth=0.8,
    )
    axis.axvline(
        0,
        color=MUTED,
        linewidth=0.8,
    )
    for side in (analytic_ms < 0, analytic_ms > 0):
        axis.plot(
            analytic_ms[side],
            analytic[side] / W_MAX_US,
            color=MUTED,
            linewidth=3,
            label="Formula" if side[-1] else None,
        )
    axis.scatter(
        lags_ms,
        change / W_MAX_US,
        s=12,
        color=COLORS[0],
        zorder=3,
        label="Update step, one spike pair",
    )
    axis.annotate(
        "same step: no pairing",
        xy=(0, 0),
        xytext=(12, -0.008),
        color=INK,
        arrowprops={"arrowstyle": "-", "color": MUTED},
    )
    axis.set_xlabel("Lag t_post − t_pre (ms)")
    axis.set_ylabel("Weight change (× w_max)")
    axis.set_title(
        "Learning window",
        pad=10,
    )
    axis.legend(frameon=False)
    figure.savefig(OUTPUT_DIR / "stdp-window.png")
    plt.close(figure)


def plot_learning(run):
    rates = run["rates"]
    snapshots = run["snapshots"] / W_MAX_US
    final = run["final"] / W_MAX_US
    seconds = np.arange(1, len(rates) + 1)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7.5),
        dpi=150,
        layout="constrained",
    )
    rate_axis, mean_axis, index_axis, histogram_axis = axes.flat

    window = min(5, len(rates))
    kernel = np.ones(window) / window
    for trial, (label, _, _) in enumerate(EXPERIMENTS):
        smooth = np.convolve(rates[:, trial], kernel, mode="valid")
        rate_axis.plot(
            seconds[len(kernel) - 1 :],
            smooth,
            color=COLORS[trial],
            linewidth=2,
            label=label,
        )
    rate_axis.set_xlabel("Time (s)")
    rate_axis.set_ylabel("Output rate (Hz)")
    rate_axis.set_title(
        f"Output rate during learning ({window} s running mean)",
        pad=10,
    )
    rate_axis.set_ylim(bottom=0)
    rate_axis.legend(frameon=False)

    groups = (
        (0, slice(None), "-", "10 Hz, independent: all inputs"),
        (1, slice(0, GROUP), "-", "Half correlated: correlated inputs"),
        (1, slice(GROUP, None), "--", "Half correlated: independent inputs"),
        (2, slice(None), "-", "20 Hz, independent: all inputs"),
    )
    for trial, inputs, style, label in groups:
        mean_axis.plot(
            seconds,
            snapshots[:, trial, inputs].mean(axis=1),
            color=COLORS[trial],
            linestyle=style,
            linewidth=2,
            label=label,
        )
    mean_axis.set_xlabel("Time (s)")
    mean_axis.set_ylabel("Mean weight (× w_max)")
    mean_axis.set_title(
        "Mean weights, recorded every second",
        pad=10,
    )
    mean_axis.set_ylim(
        0,
        1,
    )
    mean_axis.legend(
        frameon=False,
        fontsize=8,
    )

    index_axis.axvspan(
        0,
        GROUP,
        color=COLORS[1],
        alpha=0.08,
        linewidth=0,
    )
    index_axis.scatter(
        np.arange(N_EXC),
        final[1],
        s=4,
        color=COLORS[1],
    )
    index_axis.text(
        GROUP / 2,
        1.08,
        "inputs 0–499: correlated",
        ha="center",
        color=INK,
    )
    index_axis.text(
        GROUP * 1.5,
        1.08,
        "inputs 500–999: independent",
        ha="center",
        color=INK,
    )
    index_axis.set_xlabel("Excitatory input")
    index_axis.set_ylabel("Final weight (× w_max)")
    index_axis.set_title(
        "Half correlated: final weight of each input",
        pad=10,
    )
    index_axis.set_ylim(
        -0.03,
        1.15,
    )
    index_axis.set_yticks(np.linspace(0, 1, 6))
    index_axis.set_xlim(
        0,
        N_EXC,
    )

    bins = np.linspace(0, 1, 21)
    histogram_axis.hist(
        run["initial"] / W_MAX_US,
        bins=bins,
        histtype="step",
        color=MUTED,
        linestyle="--",
        linewidth=1.5,
        label="Initial (all experiments)",
    )
    for trial, (label, _, _) in enumerate(EXPERIMENTS):
        histogram_axis.hist(
            final[trial],
            bins=bins,
            histtype="step",
            color=COLORS[trial],
            linewidth=2,
            label=label,
        )
    histogram_axis.set_xlabel("Final weight (× w_max)")
    histogram_axis.set_ylabel("Inputs")
    histogram_axis.set_title(
        "Final weight distributions",
        pad=10,
    )
    histogram_axis.legend(
        frameon=False,
        fontsize=8,
    )

    figure.savefig(OUTPUT_DIR / "stdp-learning.png")
    plt.close(figure)


def main():
    prepare_results()
    run = run_example()

    diagram = lang.diagram(
        run["bundle"],
        view="expanded",
    )
    viz.render_diagram(
        diagram,
        OUTPUT_DIR / "network.png",
        scale=2,
        height_to_width_ratio=None,
        canvas_size=(1920, 900),
    )
    plot_window(run["window"])
    plot_learning(run)
    summary = summarise(run)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for experiment in summary["experiments"]:
        print(
            f"{experiment['label']}: output rate "
            f"{experiment['output_rate_before_hz']} -> "
            f"{experiment['output_rate_after_hz']} Hz; mean weight "
            f"{experiment['mean_weight_first_half']} (first half), "
            f"{experiment['mean_weight_second_half']} (second half)"
        )


if __name__ == "__main__":
    main()
