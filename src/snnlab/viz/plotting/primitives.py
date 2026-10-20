"""Small Matplotlib primitives over public saved-evidence contracts."""

from __future__ import annotations

import numpy as np


def plot_raster(
    axis, recording, *, signal, batch=None, offset=0, color=None, label=None, **kwargs
):
    """Draw recorded spikes using actual times and original cell identities.

    Return a PathCollection. Dense recordings require explicit batch selection
    when a batch axis remains. No figure, legend, or files are created.
    """
    from snnlab.sim.artifacts.recordings import RecordingData

    if hasattr(recording, "recording"):
        recording = recording.recording
    if not isinstance(recording, RecordingData):
        raise TypeError("recording must be RecordingData or NumpyExecutionResult")
    recording.validate()
    if signal not in recording.signals:
        raise ValueError(f"recording signal is unavailable: {signal!r}")
    if batch is not None:
        recording = recording.select_batch(batch)
    values, info = recording.signals[signal], recording.info[signal]
    if info.unit != "spike":
        raise ValueError("raster requires a signal explicitly measured in spikes")
    if info.dt_ms is None:
        raise ValueError("raster requires a recorded timestep")
    if info.kind == "spike_events":
        if info.selected_batch is None:
            raise ValueError("raster requires explicit batch selection")
        times, cells = values[:, 0] * info.dt_ms, values[:, 2]
    else:
        if "batch" in info.axes:
            raise ValueError("raster requires explicit batch selection")
        if (
            "time" not in info.axes
            or values.ndim != 2
            or info.time_ms is None
            or info.cell_ids is None
        ):
            raise ValueError("raster requires recorded time and cell coordinates")
        time_major = np.moveaxis(values, info.axes.index("time"), 0)
        if not np.isfinite(time_major).all() or np.any(time_major < 0):
            raise ValueError("raster spikes must be finite and nonnegative")
        rows, columns = np.nonzero(time_major)
        times, cells = info.time_ms[rows], np.asarray(info.cell_ids)[columns]
    return axis.scatter(times, cells + offset, color=color, label=label, **kwargs)


def plot_learning_curves(
    axes,
    history,
    *,
    metrics=("loss", "accuracy"),
    splits=("train", "validation"),
    include_test=False,
    objective=None,
    colors=None,
    labels=None,
    **kwargs,
):
    """Return artists keyed by (split, metric), preserving gaps and semantics.

    ``axes`` maps metric names to caller-owned Matplotlib axes. Test markers
    stand alone at the epoch of the checkpoint actually evaluated.
    """
    from snnlab.sim.training.history import TrainingHistory, metric_value

    if not isinstance(history, TrainingHistory):
        raise TypeError("history must be TrainingHistory")
    if any(metric not in axes for metric in metrics):
        raise ValueError("axes must provide an axis for every requested metric")
    artists = {}
    semantics = {
        "online_pre_update": "online",
        "fixed_weight_evaluation": "fixed",
        None: "unknown",
    }
    for metric in metrics:
        axis = axes[metric]
        for split in splits:
            curve = history.curve(metric=metric, split=split, objective=objective)
            groups, active = [], []
            for index, available in enumerate(curve.available):
                if active and (
                    not available
                    or curve.epochs[index] != curve.epochs[active[-1]] + 1
                    or curve.measurement[index] != curve.measurement[active[-1]]
                ):
                    groups.append(active)
                    active = []
                if available:
                    active.append(index)
            if active:
                groups.append(active)
            drawn, seen = [], set()
            color = (colors or {}).get(split)
            for indices in groups:
                semantic = curve.measurement[indices[0]]
                label = (labels or {}).get(split, split) + f"[{semantics[semantic]}]"
                (line,) = axis.plot(
                    curve.epochs[indices],
                    curve.values[indices],
                    color=color,
                    label=label if semantic not in seen else "_nolegend_",
                    **kwargs,
                )
                color = line.get_color()
                seen.add(semantic)
                drawn.append(line)
            artists[split, metric] = tuple(drawn)
        if include_test and history.test is not None:
            test = history.test
            value = metric_value(test, metric, objective)
            if value is not None:
                artist = axis.scatter(
                    [test.epoch],
                    [value],
                    color=(colors or {}).get("test"),
                    marker="x",
                    label=(labels or {}).get("test", "test")
                    + f"[{semantics[test.get('measurement')]}, {test.checkpoint}]",
                )
                artists["test", metric] = (artist,)
    return artists
