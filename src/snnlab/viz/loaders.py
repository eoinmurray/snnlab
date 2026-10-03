"""Adapters from native tool artifacts into renderer-neutral recordings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contracts import Recording, RecordingError


def load_snnsim_recording(run_dir: str | Path) -> Recording:
    """Load a canonical snnsim snapshot and preserve unknown signal fields."""

    root = Path(run_dir)
    snapshot = root / "recording.npz"
    if not snapshot.is_file():
        raise RecordingError(f"missing snnsim snapshot: {snapshot}")
    with np.load(snapshot) as payload:
        if "dt" not in payload:
            raise RecordingError(f"snapshot has no dt field: {snapshot}")
        dt_ms = float(payload["dt"])
        time_anchor = next(
            (
                np.asarray(payload[name])
                for name in ("spk_e", "v_e_1", "input_spikes")
                if name in payload and np.asarray(payload[name]).ndim
            ),
            None,
        )
        if time_anchor is None:
            raise RecordingError(f"snapshot has no time-series anchor: {snapshot}")
        steps = time_anchor.shape[0]
        signals = {
            name: np.asarray(payload[name])
            for name in payload.files
            if name not in {"dt", "n_e", "n_i", "label"}
            and np.asarray(payload[name]).ndim
            and np.asarray(payload[name]).shape[0] == steps
        }
        metadata = {
            name: np.asarray(payload[name]).item()
            for name in ("n_e", "n_i", "label")
            if name in payload and np.asarray(payload[name]).size == 1
        }
        metadata["retained_static"] = {
            name: np.asarray(payload[name])
            for name in payload.files
            if name not in {"dt", "n_e", "n_i", "label"}
            and np.asarray(payload[name]).ndim
            and np.asarray(payload[name]).shape[0] != steps
        }
    config = root / "config.json"
    if config.is_file():
        metadata["config"] = json.loads(config.read_text())
    return Recording(
        dt_ms=dt_ms, signals=signals, metadata=metadata, source=snapshot
    )
