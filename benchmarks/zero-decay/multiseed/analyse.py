"""Summarize paired accuracy repetition; concurrent timings are excluded."""

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    report = json.loads((HERE / "results.json").read_text())
    if not report["complete"]:
        raise ValueError("Wait for every preselected seed and backend before analysis")
    seeds, backends = report["seeds"], report["backends"]
    rows = {(row["seed"], row["backend"]): row for row in report["trials"]}
    if len(rows) != len(report["trials"]) or set(rows) != {
        (seed, backend) for seed in seeds for backend in backends
    }:
        raise ValueError("Missing or duplicated trials")
    paired = []
    for seed in seeds:
        group = [rows[seed, backend] for backend in backends]
        if len({row["initial_weights_sha256"] for row in group}) != 1:
            raise ValueError(f"Unpaired initialization for seed {seed}")
        for row in group:
            metrics = row["metrics"]
            if (
                metrics["epochs"],
                metrics["train_samples"],
                metrics["test_samples"],
                metrics["batch_size"],
            ) != (2, 60000, 10000, 128):
                raise ValueError("Incomplete training/test protocol")
        snnlab = group[1:]
        same_metrics = all(
            all(
                row["metrics"][key] == snnlab[0]["metrics"][key]
                for key in ("epoch_losses", "test_loss", "test_accuracy")
            )
            for row in snnlab[1:]
        )
        same_weights = len({row["final_weights_sha256"] for row in snnlab}) == 1
        accuracies = {
            row["backend"]: row["metrics"]["test_accuracy"] * 100 for row in group
        }
        paired.append(
            {
                "seed": seed,
                "accuracy_percent": accuracies,
                "snntorch_minus_strict_pp": accuracies["snntorch"]
                - accuracies["strict_full"],
                "snntorch_minus_strict_test_loss": group[0]["metrics"]["test_loss"]
                - group[1]["metrics"]["test_loss"],
                "snnlab_modes_exact_metrics": same_metrics,
                "snnlab_modes_exact_final_weights": same_weights,
            }
        )
    differences = np.array([row["snntorch_minus_strict_pp"] for row in paired])
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))
    margin = float(t.ppf(0.975, len(seeds) - 1) * sd / np.sqrt(len(seeds)))
    summary = {
        "seed_count": len(seeds),
        "paired": paired,
        "mean_accuracy_percent": {
            backend: float(
                np.mean([row["accuracy_percent"][backend] for row in paired])
            )
            for backend in backends
        },
        "sd_accuracy_percent": {
            backend: float(
                np.std([row["accuracy_percent"][backend] for row in paired], ddof=1)
            )
            for backend in backends
        },
        "paired_mean_snntorch_minus_strict_pp": mean,
        "paired_sd_snntorch_minus_strict_pp": sd,
        "paired_mean_approximate_95pct_t_interval_pp": [mean - margin, mean + margin],
        "paired_range_pp": [float(differences.min()), float(differences.max())],
        "snntorch_wins": int((differences > 0).sum()),
        "snnlab_wins": int((differences < 0).sum()),
        "ties": int((differences == 0).sum()),
        "limitations": "Five preselected seeds, two epochs, one architecture, one dtype/device family. The t interval assumes approximately normal paired seed differences and is exploratory. Each seed varies initialization, training order and encoding jointly; test encoding is paired within seed but differs between seeds. Concurrent timings are not speed measurements.",
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for index, row in enumerate(paired):
        color = plt.get_cmap("tab10")(index)
        values = [row["accuracy_percent"][name] for name in ("snntorch", "strict_full")]
        axes[0].plot(
            [0, 1], values, "o-", color=color, label=f"Seed {row['seed']}", alpha=0.85
        )
        axes[1].scatter(index, row["snntorch_minus_strict_pp"], color=color, s=50)
    axes[0].set_xticks([0, 1], ["snnTorch", "SNNLab strict"])
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Paired training results")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(range(len(seeds)), [str(seed) for seed in seeds])
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("snnTorch − SNNLab accuracy (percentage points)")
    axes[1].set_title("Direction and size of the gap")
    figure.suptitle("Original arithmetic: two MNIST epochs, five paired seeds")
    figure.savefig(HERE / "accuracy.png", dpi=180)
    figure.savefig(HERE / "accuracy.svg")
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
