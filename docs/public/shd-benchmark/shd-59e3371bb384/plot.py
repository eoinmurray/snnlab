"""Render a run-pinned SHD report from retained baseline and diagnostic artifacts.

A large Kineto trace is streamed once into a compact, reproducible excerpt and
actual-kernel aggregate. Rendering thereafter needs only retained small files.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN_ID = "shd-59e3371bb384"
FIGURES = ("learning", "epoch-timing", "sparsity", "diagnostics")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def trace_events(path):
    """Stream the top-level Kineto traceEvents array with bounded buffering."""
    decoder = json.JSONDecoder()
    with Path(path).open() as source:
        buffer = ""
        while '"traceEvents"' not in buffer:
            chunk = source.read(1024 * 1024)
            if not chunk:
                raise ValueError("missing traceEvents array")
            buffer += chunk
        marker = buffer.index('"traceEvents"')
        position = buffer.index("[", marker) + 1
        while True:
            while position < len(buffer) and buffer[position] in " \t\r\n,":
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                event, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                buffer = buffer[position:]
                position = 0
                chunk = source.read(1024 * 1024)
                if not chunk:
                    raise ValueError("incomplete traceEvents array") from None
                buffer += chunk
                continue
            position = end
            yield event
            if position > 1024 * 1024:
                buffer, position = buffer[position:], 0


def extract_trace(trace, destination):
    """Fixed selection: first 100 ms of the first captured host forward span."""
    kernels = defaultdict(lambda: {"calls": 0, "duration_us": 0.0})
    first_forward = None
    candidates = []
    for event in trace_events(trace):
        if event.get("ph") != "X":
            continue
        cat, name = event.get("cat"), event.get("name", "")
        ts, duration = event.get("ts", 0), event.get("dur", 0)
        if cat == "kernel":
            kernels[name]["calls"] += 1
            kernels[name]["duration_us"] += duration
        if cat == "user_annotation" and name == "snnlab::forward":
            first_forward = ts if first_forward is None else min(first_forward, ts)
    if first_forward is None:
        raise ValueError("trace has no host forward annotation")
    # A second streaming pass avoids depending on event ordering in the export.
    for event in trace_events(trace):
        if event.get("ph") != "X":
            continue
        cat, name = event.get("cat"), event.get("name", "")
        ts, duration = event.get("ts", 0), event.get("dur", 0)
        if ts >= first_forward + 100_000 or ts + duration <= first_forward:
            continue
        lane = None
        if cat == "user_annotation" and name.startswith("snnlab::projection/"):
            lane = "Host: projection calls"
        elif cat == "user_annotation" and name.startswith("snnlab::neuron/"):
            lane = "Host: neuron calls"
        elif cat == "kernel":
            lane = "CUDA kernels"
        elif cat == "gpu_memcpy":
            lane = "CUDA copies"
        if lane:
            candidates.append(
                {"lane": lane, "name": name, "ts_us": ts, "duration_us": duration}
            )
    events = [
        {
            "lane": row["lane"],
            "name": row["name"],
            "start_ms": max(0, (row["ts_us"] - first_forward) / 1000),
            "duration_ms": (
                min(row["ts_us"] + row["duration_us"], first_forward + 100_000)
                - max(row["ts_us"], first_forward)
            )
            / 1000,
        }
        for row in candidates
    ]
    if not any(row["lane"] == "CUDA kernels" for row in events):
        raise ValueError("selected excerpt has no CUDA kernel events")
    result = {
        "run_id": RUN_ID,
        "raw_sha256": digest(trace),
        "raw_bytes": Path(trace).stat().st_size,
        "selection": "first 100 ms after the first captured CPU snnlab::forward start; all kernels aggregated over complete capture",
        "duration_ms": 100,
        "events": events,
        "kernels": [
            {"name": name, **row}
            for name, row in sorted(
                kernels.items(), key=lambda item: item[1]["duration_us"], reverse=True
            )
        ],
    }
    write_json(destination, result)
    return result


def analyze(root):
    root = Path(root)

    def load(name):
        return json.loads((root / name).read_text())

    cloud, cfg, metrics = (
        load("cloud.json"),
        load("configuration.json"),
        load("metrics.json"),
    )
    summary, timing, tracing = (
        load("summary.json"),
        load("timing.json"),
        load("tracing/summary.json"),
    )
    excerpt = load("timeline-excerpt.json")
    if cloud["run_id"] != RUN_ID or root.name != RUN_ID or excerpt["run_id"] != RUN_ID:
        raise ValueError("report must use the explicitly pinned SHD run")
    if (
        cloud["status"] != "complete"
        or summary["completed_updates"] != 290
        or cfg["smoke"]
    ):
        raise ValueError("expected completed, full-data, 290-update run")
    expected = {
        "epochs": 10,
        "batch_size": 256,
        "seed": 42,
        "steps": 1000,
        "dt_ms": 1.0,
        "channels": 700,
        "hidden": 128,
        "classes": 20,
    }
    if any(cfg["configuration"][name] != value for name, value in expected.items()):
        raise ValueError("workload configuration differs from the report protocol")
    if (cfg["dataset"]["train_count"], cfg["dataset"]["validation_count"]) != (
        7340,
        816,
    ) or cfg["dataset"]["official_test_accessed"]:
        raise ValueError("dataset split differs from the report protocol")
    if tracing["status"] != "complete" or len(tracing["captured_updates"]) != 3:
        raise ValueError("expected three complete diagnostic batches")
    if [row["epoch"] for row in metrics["epochs"]] != list(range(11)):
        raise ValueError("missing initial validation or training epochs")
    if (
        sum(row["calls"] for row in excerpt["kernels"])
        != tracing["timeline"]["kernel_count"]
    ):
        raise ValueError("raw kernel events do not match retained tracing summary")
    cpu = defaultdict(lambda: {"calls": 0, "self_cpu_us": 0.0})
    with (root / "tracing/operators.csv").open() as handle:
        for row in csv.DictReader(handle):
            name = row["operator"]
            if (
                name.startswith(("snnlab::", "[memory]"))
                or float(row["self_cpu_us"]) <= 0
            ):
                continue
            if name.startswith("## Call CompiledFxGraph"):
                name = "Compiled graph calls"
            cpu[name]["calls"] += int(row["calls"])
            cpu[name]["self_cpu_us"] += float(row["self_cpu_us"])
    phase = defaultdict(list)
    for row in tracing["phases"]:
        phase[row["region"]].append(row["host_enqueue_s"])
    host_copies = [
        row
        for row in tracing["timeline"]["copies"]
        if "HtoD" in str(row["name"]) and row["bytes"] is not None
    ]
    examples = cfg["dataset"]["train_count"] * cfg["configuration"]["epochs"]
    warm = [row for row in timing["epochs"] if row["epoch"] >= 2]
    result = {
        "run_id": RUN_ID,
        "baseline": {
            "training_s": timing["training_s"],
            "validation_s": timing["validation_s"],
            "native_total_s": timing["total_s"],
            "workflow_wall_s": summary["wall_s"],
            "examples_per_s": examples / timing["training_s"],
            "update_mean_s": timing["training_s"] / 290,
            "warm_training_epoch_mean_s": float(
                np.mean([row["training_s"] for row in warm])
            ),
            "peak_allocated_gib": summary["peak_cuda_allocated_bytes"] / 2**30,
            "peak_reserved_gib": summary["peak_cuda_reserved_bytes"] / 2**30,
            "final_train_accuracy": metrics["epochs"][-1]["train_accuracy"],
            "final_validation_accuracy": metrics["epochs"][-1]["validation_accuracy"],
            "best_validation_accuracy": max(
                row["validation_accuracy"] for row in metrics["epochs"]
            ),
        },
        "diagnostic": {
            "phase_host_medians_s": {
                name: float(np.median(values)) for name, values in phase.items()
            },
            "window_wall_s": tracing["window_wall_s"],
            "cuda_peak_allocated_gib": max(
                row["cuda_peak_allocated_bytes"] for row in tracing["memory"]
            )
            / 2**30,
            "cuda_peak_reserved_gib": max(
                row["cuda_peak_reserved_bytes"] for row in tracing["memory"]
            )
            / 2**30,
            "host_to_device_bytes": sum(row["bytes"] for row in host_copies),
            "host_to_device_copy_s": sum(row["duration_us"] for row in host_copies)
            / 1e6,
            "host_to_device_aggregate_GB_per_s": sum(
                row["bytes"] for row in host_copies
            )
            / (sum(row["duration_us"] for row in host_copies) * 1000),
            "cpu_operators": [
                {"name": name, **row}
                for name, row in sorted(
                    cpu.items(), key=lambda item: item[1]["self_cpu_us"], reverse=True
                )[:5]
            ],
            "cuda_kernels": excerpt["kernels"][:5],
        },
        "source_sha256": {
            str(path.relative_to(root)): digest(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.suffix in {".json", ".csv"}
            and path.name not in {"report-data.json", "presentation.json"}
        },
        "plot_script_sha256": digest(__file__),
    }
    write_json(root / "report-data.json", result)
    return cfg, metrics, timing, tracing, excerpt, result


def render(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    from snnlab.viz import Theme

    root = Path(root)
    cfg, metrics, timing, tracing, excerpt, analysis = analyze(root)
    theme = Theme()
    theme.apply()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "savefig.dpi": 240,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": RUN_ID,
        }
    )

    def setup(fig, axes, title, footer):
        fig.suptitle(title, x=0.075, ha="left", fontsize=13, fontweight="bold")
        for index, ax in enumerate(np.asarray(axes).flat):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(False)
            ax.set_title(
                f"{'ABCD'[index]}  {ax.get_title(loc='left') or ax.get_title()}",
                loc="left",
            )
        fig.text(0.075, 0.015, footer, fontsize=8, color=theme.muted)

    def save(fig, name):
        for suffix in ("png", "svg", "pdf"):
            metadata = (
                {"Date": None}
                if suffix == "svg"
                else {"CreationDate": None, "ModDate": None}
                if suffix == "pdf"
                else None
            )
            fig.savefig(
                root / f"{name}.{suffix}", bbox_inches="tight", metadata=metadata
            )
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"wspace": 0.32})
    fig.subplots_adjust(top=0.76, bottom=0.20, left=0.075, right=0.97)
    for key, label, color, style, rows in (
        ("train", "Train: online", theme.ink, "-", metrics["epochs"][1:]),
        (
            "validation",
            "Validation: fixed weights",
            theme.accent,
            "--",
            metrics["epochs"],
        ),
    ):
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(
            epochs,
            [row[f"{key}_cross_entropies"]["objective[0]"] for row in rows],
            color=color,
            linestyle=style,
            marker="o",
            ms=3,
            label=label,
        )
        axes[1].plot(
            epochs,
            [row[f"{key}_accuracy"] for row in rows],
            color=color,
            linestyle=style,
            marker="o",
            ms=3,
        )
    axes[0].set(
        title="Cross-entropy",
        xlabel="Completed epoch",
        ylabel="Example-mean cross-entropy",
        ylim=(0, 3.3),
    )
    axes[1].set(
        title="Accuracy",
        xlabel="Completed epoch",
        ylabel="Classification accuracy",
        ylim=(0, 0.55),
    )
    for ax in axes:
        ax.set(xticks=range(0, 11, 2), xlim=(0, 10))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.07, 0.91),
        ncol=2,
        frameon=False,
    )
    setup(
        fig,
        axes,
        "SHD / learning over ten epochs",
        "7,340 train / 816 validation  |  seed 42  |  one run; unsmoothed measurements",
    )
    save(fig, "learning")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.09, right=0.97)
    rows = timing["epochs"]
    epochs = np.array([row["epoch"] for row in rows])
    training = np.array([row["training_s"] for row in rows])
    validation = np.array([row["validation_s"] for row in rows])
    ax.bar(epochs, training, color=theme.ink, label="Training updates")
    ax.bar(epochs, validation, bottom=training, color=theme.accent, label="Validation")
    ax.set(
        title="Training and validation phases",
        xlabel="Epoch (0 = initial validation)",
        ylabel="Time (s)",
        xticks=epochs,
        ylim=(0, 90),
    )
    ax.legend(loc="upper right", frameon=False)
    setup(
        fig,
        [ax],
        "SHD / baseline epoch costs",
        "Tracing disabled  |  setup, checkpoint and reporting time excluded from bars",
    )
    save(fig, "epoch-timing")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.7))
    fig.subplots_adjust(
        top=0.84, bottom=0.10, left=0.09, right=0.97, hspace=0.60, wspace=0.34
    )
    activity = {row["signal"]: row for row in tracing["activity"]}
    labels = ["Input", "E1", "E2"]
    signal_ids = ["input/drive", "population/E1", "population/E2"]
    density = [activity[name]["density"] * 100 for name in signal_ids]
    axes[0, 0].bar(labels, density, color=[theme.mid_grey, theme.ink, theme.accent])
    for index, value in enumerate(density):
        axes[0, 0].text(index, value + 0.25, f"{value:.2f}%", ha="center", fontsize=9)
    axes[0, 0].set(
        title="Spike density",
        ylabel="Nonzero spike bins (%)",
        ylim=(0, 11),
    )
    rates = [activity[name]["per_neuron_rate_hz"] for name in signal_ids[1:]]
    bins = np.linspace(0, np.ceil(max(map(max, rates)) / 20) * 20, 21)
    for values, label, color in zip(rates, labels[1:], [theme.ink, theme.accent]):
        axes[0, 1].hist(
            values, bins=bins, histtype="step", linewidth=1.8, label=label, color=color
        )
    axes[0, 1].set(
        title="Per-neuron rates",
        xlabel="Rate over 768 utterances (Hz)",
        ylabel="Neurons per bin",
    )
    axes[0, 1].legend(frameon=False)
    names = ["input_to_E1.weight", "E1_to_E2.weight", "readout_projection.weight"]
    for phase, offset, color, label in (
        ("before_window", -0.19, theme.ink, "After 5 updates"),
        ("after_window", 0.19, theme.accent, "After 8 updates"),
    ):
        rows = {
            row["parameter"]: row for row in tracing["weights"] if row["phase"] == phase
        }
        axes[1, 0].bar(
            np.arange(3) + offset,
            [rows[name]["zero_fraction"] * 100 for name in names],
            width=0.36,
            color=color,
            label=label,
        )
    axes[1, 0].set(
        title="Numerical weight zeros",
        ylabel="Exactly zero weights (%)",
        xticks=range(3),
        xticklabels=["Input→E1", "E1→E2", "E2→out"],
        ylim=(0, 55),
    )
    axes[1, 0].legend(frameon=False)
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Block and gradient structure", loc="left")
    table_rows = []
    rows = {
        row["parameter"]: row
        for row in tracing["weights"]
        if row["phase"] == "after_window"
    }
    for label, name in zip(["Input→E1", "E1→E2", "E2→out"], names):
        row = rows[name]
        table_rows.append(
            [label, f"{row['blocks'] - row['nonzero_blocks']}/{row['blocks']}"]
        )
    table = axes[1, 1].table(
        cellText=table_rows,
        colLabels=["Projection", "Zero 16×16 tiles"],
        loc="upper center",
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0.37, 1, 0.48],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for cell in table.get_celld().values():
        cell.set_edgecolor(theme.rule)
    axes[1, 1].text(
        0,
        0.10,
        "100% of sampled gradient entries\nnonzero in all three learned matrices\n(three batches; before clipping).",
        fontsize=9,
        va="bottom",
    )
    setup(
        fig,
        axes,
        "SHD / sparse spikes, dense sampled gradients",
        "Diagnostic batches 6–8 of a fresh run  |  early training, not the final checkpoint",
    )
    save(fig, "sparsity")

    fig, axes = plt.subplots(
        2, 1, figsize=(9, 6.8), gridspec_kw={"height_ratios": [1.1, 1]}
    )
    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.25, right=0.97, hspace=0.62)
    phase_names = [
        "batch_prepare_wait",
        "forward",
        "backward",
        "loss",
        "optimizer_constraints",
    ]
    phase_labels = [
        "Batch preparation",
        "Forward",
        "Backward",
        "Loss",
        "Adam + constraints",
    ]
    for index, name in enumerate(phase_names):
        values = [
            row["host_enqueue_s"] * 1000
            for row in tracing["phases"]
            if row["region"] == name
        ]
        axes[0].scatter(
            values, np.array([-0.13, 0, 0.13]) + index, s=24, color=theme.ink
        )
        axes[0].scatter(
            [np.median(values)],
            [index],
            s=180,
            color=theme.accent,
            marker="|",
            zorder=3,
        )
    axes[0].set(
        title="Instrumented phase times",
        yticks=range(5),
        yticklabels=phase_labels,
        xlabel="Host enqueue duration (ms; log scale)",
        xscale="log",
        xlim=(0.3, 3000),
    )
    axes[0].invert_yaxis()
    lanes = [
        "Host: projection calls",
        "Host: neuron calls",
        "CUDA kernels",
        "CUDA copies",
    ]
    colors = [theme.accent, theme.mid_grey, theme.ink, theme.amber]
    for index, (lane, color) in enumerate(zip(lanes, colors)):
        spans = [
            (event["start_ms"], event["duration_ms"])
            for event in excerpt["events"]
            if event["lane"] == lane
        ]
        axes[1].broken_barh(
            spans, (index - 0.25, 0.5), facecolors=color, edgecolors="none"
        )
    if not any(event["lane"] == "CUDA copies" for event in excerpt["events"]):
        axes[1].text(
            50,
            3,
            "No copy events in this interval",
            color=theme.muted,
            fontsize=8,
            ha="center",
            va="center",
        )
    axes[1].set(
        title="First 100 ms of the first captured forward",
        yticks=range(4),
        yticklabels=lanes,
        xlabel="Time from host forward start (ms)",
        xlim=(0, 100),
        ylim=(-0.7, 3.7),
    )
    axes[1].invert_yaxis()
    setup(
        fig,
        axes,
        "SHD / diagnostic execution, not baseline timing",
        "Dots: three batches; red ticks: medians  |  activity monitoring and detailed profiler labels enabled",
    )
    save(fig, "diagnostics")
    outputs = {
        f"{name}.{suffix}": digest(root / f"{name}.{suffix}")
        for name in FIGURES
        for suffix in ("png", "pdf", "svg")
    }
    write_json(
        root / "presentation.json",
        {
            "run_id": RUN_ID,
            "source_sha256": analysis["source_sha256"],
            "renderer_sha256": digest(__file__),
            "figures": outputs,
        },
    )
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--trace", type=Path, help="large raw trace; optional after initial extraction"
    )
    args = parser.parse_args()
    if args.trace:
        extract_trace(args.trace, args.root / "timeline-excerpt.json")
    print(json.dumps(render(args.root)["baseline"], indent=2))
