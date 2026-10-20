"""Axes primitives use observed coordinates and leave missing evidence absent."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from snnlab import sim, viz


def test_raster_uses_strided_time_and_original_cell_ids():
    recording = sim.RecordingData(
        {"E.spikes": np.asarray([[[0, 1]], [[1, 0]], [[0, 0]]])},
        {
            "E.spikes": sim.SignalInfo(
                ("time", "batch", "cell"),
                unit="spike",
                cell_ids=(8, 3),
                step_indices=(10, 14, 18),
                dt_ms=0.2,
                batch_size=1,
            )
        },
    )
    fig, ax = plt.subplots()
    artist = viz.plot_raster(
        ax,
        recording,
        signal="E.spikes",
        batch=0,
        offset=20,
        color="red",
        label="population",
    )
    np.testing.assert_allclose(artist.get_offsets(), [[2.0, 23], [2.8, 28]])
    assert artist.get_label() == "population"
    assert len(fig.axes) == 1
    with pytest.raises(ValueError, match="explicit batch"):
        viz.plot_raster(ax, recording, signal="E.spikes")
    plt.close(fig)


def test_sparse_raster_preserves_event_time_and_batch_identity():
    recording = sim.RecordingData(
        {"I.spikes": np.asarray([[5, 0, 9], [7, 1, 4]])},
        {
            "I.spikes": sim.SignalInfo(
                ("event", "coordinate"),
                kind="spike_events",
                unit="spike",
                cell_ids=(9, 4),
                batch_size=2,
                dt_ms=0.1,
            )
        },
    )
    fig, ax = plt.subplots()
    artist = viz.plot_raster(ax, recording, signal="I.spikes", batch=1)
    np.testing.assert_allclose(artist.get_offsets(), [[0.7, 4]])
    plt.close(fig)


def test_learning_curves_break_gaps_and_keep_test_a_separate_marker():
    history = sim.TrainingHistory.from_metrics(
        {
            "epochs": [
                {
                    "epoch": 0,
                    "validation_loss": 3.0,
                    "validation_measurement": "fixed_weight_evaluation",
                },
                {
                    "epoch": 1,
                    "train_loss": 2.0,
                    "train_measurement": "online_pre_update",
                    "validation_loss": 2.0,
                    "validation_measurement": "fixed_weight_evaluation",
                },
                {
                    "epoch": 3,
                    "train_loss": 1.0,
                    "train_measurement": "online_pre_update",
                    "validation_loss": 1.0,
                    "validation_measurement": "fixed_weight_evaluation",
                },
            ],
            "test": {
                "epoch": 3,
                "update": 6,
                "checkpoint": "final",
                "loss": 1.5,
                "measurement": "fixed_weight_evaluation",
            },
        }
    )
    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2)
    artists = viz.plot_learning_curves(
        {"loss": loss_ax, "accuracy": accuracy_ax}, history, include_test=True
    )
    assert len(artists["train", "loss"]) == 2
    assert len(artists["validation", "loss"]) == 2
    assert artists["train", "accuracy"] == ()
    assert artists["validation", "accuracy"] == ()
    np.testing.assert_array_equal(artists["train", "loss"][0].get_xdata(), [1])
    np.testing.assert_array_equal(artists["validation", "loss"][0].get_xdata(), [0, 1])
    np.testing.assert_allclose(artists["test", "loss"][0].get_offsets(), [[3, 1.5]])
    assert len(accuracy_ax.lines) == 0
    plt.close(fig)
