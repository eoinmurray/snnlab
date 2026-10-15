"""Checksum-verified SHD acquisition, utterance split and native encoder hook."""

import gzip
import hashlib
import shutil
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import torch

from snnlab import extensions

URL = "https://zenkelab.org/datasets/shd/shd_train.h5.gz"
# Published by the dataset authors in /datasets/md5sums.txt.
ARCHIVE_MD5 = "d47c9825dee33347913e8ce0f2be08b0"


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def acquire(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "shd_train.h5.gz"
    if not archive.exists():
        temporary = cache / "shd_train.download"
        with (
            urllib.request.urlopen(URL, timeout=60) as source,
            temporary.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)
        if digest(temporary, "md5") != ARCHIVE_MD5:
            temporary.unlink()
            raise ValueError("SHD download checksum mismatch")
        temporary.replace(archive)
    if digest(archive, "md5") != ARCHIVE_MD5:
        raise ValueError("cached SHD checksum mismatch")
    path = cache / "shd_train.h5"
    temporary = cache / "shd_train.unpack"
    with gzip.open(archive, "rb") as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target)
    temporary.replace(path)
    return path, {
        "url": URL,
        "archive_md5": ARCHIVE_MD5,
        "archive_sha256": digest(archive),
        "hdf5_sha256": digest(path),
    }


def select_indices(labels, cfg):
    """Match the source's seed-42 per-class 10% holdout, retaining every item."""
    labels = np.asarray(labels)
    if (
        labels.ndim != 1
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any((labels < 0) | (labels >= cfg.classes))
    ):
        raise ValueError("invalid SHD labels")
    rng = np.random.default_rng(cfg.seed)
    train, validation = [], []
    for label in range(cfg.classes):
        available = np.flatnonzero(labels == label)
        selected = rng.permutation(available)
        held_out = max(1, round(len(available) * cfg.validation_fraction))
        if held_out >= len(available):
            raise ValueError("each class needs training and validation examples")
        train.extend(selected[:-held_out])
        validation.extend(selected[-held_out:])
    return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def prepare(path, output, cfg):
    """Save ragged events; dense minibatches are created only by the trainer."""
    with h5py.File(path, "r") as handle:
        labels = np.asarray(handle["labels"], dtype=np.int64)
        train, validation = select_indices(labels, cfg)
        indices = np.concatenate((train, validation))
        times, channels, offsets = [], [], [0]
        dropped = 0
        for index in indices:
            t = np.asarray(handle["spikes/times"][index], dtype=np.float64) * 1000
            ch = np.asarray(handle["spikes/units"][index], dtype=np.int64)
            if (
                t.shape != ch.shape
                or not np.isfinite(t).all()
                or np.any(t < 0)
                or np.any((ch < 0) | (ch >= cfg.channels))
            ):
                raise ValueError("invalid SHD events")
            keep = t < cfg.steps * cfg.dt_ms
            dropped += int((~keep).sum())
            times.append(t[keep])
            channels.append(ch[keep])
            offsets.append(offsets[-1] + int(keep.sum()))
    np.savez_compressed(
        output,
        event_offsets=np.asarray(offsets, dtype=np.int64),
        event_time_ms=np.concatenate(times),
        event_channel=np.concatenate(channels),
        train_labels=labels[train],
        validation_labels=labels[validation],
        official_train_indices=indices,
    )
    return {
        "kind": "SHD",
        "train_count": len(train),
        "validation_count": len(validation),
        "train_class_counts": np.bincount(
            labels[train], minlength=cfg.classes
        ).tolist(),
        "validation_class_counts": np.bincount(
            labels[validation], minlength=cfg.classes
        ).tolist(),
        "selection_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "snapshot_sha256": digest(output),
        "dropped_events": dropped,
        "official_test_accessed": False,
    }


def synthetic_snapshot(output, cfg):
    """Small download-free plumbing check, explicitly not SHD performance data."""
    rng = np.random.default_rng(cfg.seed)
    train_labels = np.tile(np.arange(cfg.classes), 2)
    validation_labels = np.arange(cfg.classes)
    labels = np.concatenate((train_labels, validation_labels))
    times, channels, offsets = [], [], [0]
    for label in labels:
        count = max(1, round(cfg.channels * cfg.steps * 0.15))
        t = rng.integers(cfg.steps, size=count) * cfg.dt_ms
        ch = rng.integers(cfg.channels, size=count)
        ch[: count // 4] = label * (cfg.channels // cfg.classes)
        times.append(t)
        channels.append(ch)
        offsets.append(offsets[-1] + count)
    np.savez_compressed(
        output,
        event_offsets=np.asarray(offsets, dtype=np.int64),
        event_time_ms=np.concatenate(times),
        event_channel=np.concatenate(channels),
        train_labels=train_labels,
        validation_labels=validation_labels,
    )
    return {
        "kind": "synthetic-smoke",
        "train_count": len(train_labels),
        "validation_count": len(validation_labels),
        "snapshot_sha256": digest(output),
        "official_test_accessed": False,
    }


def encode_events(arrays, selected, *, dt_ms, channels, config, seed):
    """Match source binning and insertion/deletion, using native indexed seeds."""
    spikes = np.zeros((config["steps"], len(selected), channels), dtype=np.float32)
    for column, local_index in enumerate(selected):
        index = int(local_index) + config["sample_offset"]
        start, end = arrays["event_offsets"][index : index + 2]
        steps = np.floor(arrays["event_time_ms"][start:end] / dt_ms).astype(np.int64)
        keep = (steps >= 0) & (steps < len(spikes))
        spikes[steps[keep], column, arrays["event_channel"][start:end][keep]] = 1
    noise = np.random.default_rng(seed).random(spikes.shape, dtype=np.float32)
    occupied = spikes != 0
    spikes[occupied & (noise < config["spike_deletion_probability"])] = 0
    spikes[~occupied & (noise < config["input_noise_hz"] * dt_ms / 1000)] = 1
    return torch.from_numpy(spikes)


extensions.register_encoder("benchmarks.shd.events/v1", encode_events)
