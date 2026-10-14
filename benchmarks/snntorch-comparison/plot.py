"""Plot retained metrics; refuse to mix source snapshots, protocols or subsets."""

import argparse
import hashlib
import json
from pathlib import Path


def make_plots(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    from snnlab.viz import FigureGrid, Theme

    root = Path(root)
    report = json.loads((root / "results.json").read_text())
    results = report["training"]
    left, right = results["snntorch"], results["snnlab"]
    for key in (
        "config",
        "provenance",
        "source_manifest",
        "source_identity",
        "snnlab_execution",
        "synthetic",
    ):
        if left[key] != right[key] or (
            key in report["protocol"] and left[key] != report["protocol"][key]
        ):
            raise ValueError(f"comparison mismatch: {key}")
    if not all(
        result["complete"] and result["verification"]["passed"]
        for result in results.values()
    ):
        raise ValueError("both verified training jobs must complete")
    if len(left["epochs"]) != len(right["epochs"]):
        raise ValueError("epoch count mismatch")
    # Use SNNLab's shared Pinglab theme, with the canonical screen typography
    # and export settings from experiments/helpers/theme.py. No Pinglab checkout
    # dependency is needed on Modal.
    with plt.rc_context():
        theme = Theme()
        theme.apply()
        plt.rcParams.update(
            {
                "axes.titlesize": 11,
                "axes.titleweight": "bold",
                "axes.titlelocation": "left",
                "axes.titlepad": 10,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "xtick.major.width": 1,
                "ytick.major.width": 1,
                "xtick.major.size": 4,
                "ytick.major.size": 4,
                "legend.fontsize": 9,
                "legend.edgecolor": theme.ink,
                "legend.facecolor": theme.background,
                "lines.linewidth": 2,
                "savefig.dpi": 240,
                "savefig.bbox": "tight",
                "pdf.fonttype": 42,
                "svg.fonttype": "none",
            }
        )
        grid = FigureGrid(
            rows=1,
            columns=2,
            column_gap=0.13,
            bounds=(0.09, 0.19, 0.88, 0.61),
            theme=theme,
        )
        grid.place("loss", row=0, column=0)
        grid.place("accuracy", row=0, column=1)
        figure = grid.figure(figsize=(8, 3.7))
        axes = {name: grid.add_axes(figure, name) for name in ("loss", "accuracy")}
        handles = []
        for name, label, color, style in (
            ("snntorch", "snnTorch", theme.ink, "-"),
            ("snnlab", "SNNLab", theme.accent, "--"),
        ):
            rows = results[name]["epochs"]
            epochs = [row["epoch"] for row in rows]
            (line,) = axes["loss"].plot(
                epochs,
                [row["train_loss"] for row in rows],
                color=color,
                linestyle=style,
                label=label,
            )
            handles.append(line)
            axes["accuracy"].plot(
                epochs,
                [row["test_accuracy"] for row in rows],
                color=color,
                linestyle=style,
            )
        for label, (name, axis) in zip(("A", "B"), axes.items()):
            axis.set(
                xlabel="Epoch",
                title="Training loss" if name == "loss" else "Test accuracy",
            )
            axis.set_xlim(1, len(left["epochs"]))
            count = len(left["epochs"])
            axis.set_xticks(sorted({1, count, *range(5, count + 1, 5)}))
            axis.grid(False)
            axis.text(
                -0.12,
                1.04,
                label,
                transform=axis.transAxes,
                fontsize=13,
                fontweight="bold",
                ha="right",
                va="bottom",
            )
        axes["loss"].set(ylabel="Cross-entropy", ylim=(0, None))
        axes["accuracy"].set(ylabel="Accuracy", ylim=(0, 1))
        axes["accuracy"].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        figure.text(
            0.03,
            0.97,
            "Synthetic smoke test — not MNIST"
            if left["synthetic"]
            else "MNIST / 10% training subset",
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        figure.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.995),
            ncol=2,
            frameon=False,
        )
        config = left["config"]
        figure.text(
            0.09,
            0.035,
            f"{config['train_size']:,} train / {config['test_size']:,} test  |  seed {config['seed']}  |  loss summed over {config['steps']} steps",
            fontsize=8,
            color=theme.muted,
        )
        for suffix in ("png", "pdf"):
            figure.savefig(root / f"training-curves.{suffix}")
        plt.close(figure)
    report["presentation"] = {
        "style": "Pinglab house style",
        "panels": ["train_loss", "test_accuracy"],
        "smoothing": "none",
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return root / "training-curves.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    make_plots(parser.parse_args().root)
