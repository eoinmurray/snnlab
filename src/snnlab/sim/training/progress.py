"""Opt-in formatting of finalized epoch reports."""

import sys
from dataclasses import dataclass
from typing import Mapping, TextIO

from snnlab.sim.training.history import metric_value
from snnlab.sim.training.policies import EpochReport


def format_epoch(
    report: EpochReport,
    *,
    metrics=("loss", "accuracy"),
    splits=("train", "validation"),
    labels: Mapping[str, str] | None = None,
    precision=4,
    timing=False,
    objective=None,
):
    """Format available report fields without observations or device reads."""
    if not isinstance(report, EpochReport):
        raise TypeError("report must be an EpochReport")
    if type(precision) is not int or not 0 <= precision <= 16:
        raise ValueError("precision must be an integer in [0, 16]")
    parts = [f"epoch {report.epoch}/{report.total_epochs}"]
    for split in splits:
        if split not in {"train", "validation"}:
            raise ValueError("progress split must be train or validation")
        values = getattr(report, split)
        if values is None:
            continue
        fields = []
        for metric in metrics:
            value = metric_value(values, metric, objective)
            if value is not None:
                fields.append(f"{metric}={value:.{precision}f}")
        if fields:
            semantic = {
                "online_pre_update": "online",
                "fixed_weight_evaluation": "fixed",
            }.get(values.get("measurement"), "unknown")
            parts.append(
                f"{(labels or {}).get(split, split)}[{semantic}] " + " ".join(fields)
            )
    if (
        timing
        and report.timing is not None
        and report.timing.get("total_s") is not None
    ):
        parts.append(f"time={report.timing['total_s']:.{precision}f}s")
    return " | ".join(parts)


@dataclass
class EpochLogger:
    metrics: tuple[str, ...] = ("loss", "accuracy")
    splits: tuple[str, ...] = ("train", "validation")
    labels: Mapping[str, str] | None = None
    precision: int = 4
    timing: bool = False
    stream: TextIO | None = None
    objective: int | None = None

    def __call__(self, report):
        print(
            format_epoch(
                report,
                metrics=self.metrics,
                splits=self.splits,
                labels=self.labels,
                precision=self.precision,
                timing=self.timing,
                objective=self.objective,
            ),
            file=self.stream if self.stream is not None else sys.stdout,
            flush=True,
        )
