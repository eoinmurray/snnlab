"""Dataset loaders — MNIST (static images) and SHD (spiking audio events).

Pulled out of cli.py. The single
canonical entry point is `load_dataset(name, ...)`. Image-mode notebooks
(snapshot rendering) use `_load_dataset_image()` for a single sample.

MNIST returns dense (N, 784) pixel rows in [0, 1] that the train/infer paths
Poisson-encode per batch. SHD is event data — each utterance is a list of
(spike_time_seconds, unit) pairs over 700 channels — so it returns an OBJECT
array of per-sample event tuples instead of a dense block; the caller bins each
sample to spikes lazily (see ShdBinnedDataset in train.py). Densifying the whole
set at the model's native dt would be tens of GB, hence the lazy contract.
"""

from __future__ import annotations

import gzip
import os
import shutil
import urllib.request

import numpy as np

# ── SHD (Spiking Heidelberg Digits) constants ─────────────────────────────
# 700 input channels (cochlea model), 20 classes (spoken digits 0-9 in German
# + English). These are fixed properties of the dataset, so the loader reports
# them rather than the caller guessing.
SHD_N_IN = 700
SHD_N_CLASSES = 20
_SHD_DIR = "/tmp/shd"
_SHD_URLS = {
    # Canonical host (Zenke lab). Files are gzip-compressed HDF5.
    "train": "https://zenkelab.org/datasets/shd_train.h5.gz",
    "test": "https://zenkelab.org/datasets/shd_test.h5.gz",
}

# Smart per-dataset hidden-size defaults (used by build_net auto-config).
DATASET_N_HIDDEN_DEFAULTS = {
    "mnist": 1024,  # n_in = 784, next pow2
    "shd": 256,  # n_in = 700; Cramer et al. RSNNs use ~128-256 recurrent units
}

MNIST_SPLIT_SEED = 42
MNIST_VALIDATION_FRACTION = 0.1


def _download_shd(split):
    """Fetch + gunzip one SHD split to _SHD_DIR, returning the local .h5 path.

    Downloads shd_<split>.h5.gz once and decompresses it; subsequent calls reuse
    the cached .h5. No checksum verification yet — see the .md5 files hosted
    alongside the archives if this ever needs hardening.
    """
    os.makedirs(_SHD_DIR, exist_ok=True)
    h5_path = os.path.join(_SHD_DIR, f"shd_{split}.h5")
    if os.path.exists(h5_path):
        return h5_path
    gz_path = h5_path + ".gz"
    if not os.path.exists(gz_path):
        urllib.request.urlretrieve(_SHD_URLS[split], gz_path)
    with gzip.open(gz_path, "rb") as f_in, open(h5_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return h5_path


def _read_shd_split(split):
    """Read one SHD .h5 split into (events, labels).

    Returns:
        events: object ndarray, events[i] = (units_int16, times_float32) — the
                spike unit indices and their times (seconds) for utterance i.
        labels: int64 ndarray of class ids in [0, SHD_N_CLASSES).
    """
    import h5py

    with h5py.File(_download_shd(split), "r") as f:
        times = f["spikes"]["times"]
        units = f["spikes"]["units"]
        labels = np.asarray(f["labels"], dtype=np.int64)
        events = np.empty(len(labels), dtype=object)
        for i in range(len(labels)):
            events[i] = (
                np.asarray(units[i], dtype=np.int16),
                np.asarray(times[i], dtype=np.float32),
            )
    return events, labels


def _subsample(events, labels, max_samples):
    """Deterministic (seed 42) subset of an event/label pair for smoke runs."""
    if max_samples is None or max_samples >= len(labels):
        return events, labels
    idx = np.random.RandomState(42).choice(len(labels), max_samples, replace=False)
    return events[idx], labels[idx]


def load_dataset(
    name,
    max_samples=None,
    split=False,
    dt_ms=None,
    t_ms=None,
    evaluation_split="validation",
    evaluation_only=False,
):
    """Load full dataset as (X, y) numpy arrays in [0, 1] / int64.

    Args:
        name: "mnist" (dense pixel rows) or "shd" (object array of events)
        max_samples: optional cap on the official training partition; the
               official MNIST test partition is never subsampled through this
               argument
        split: if True, return (X_train, X_eval, y_train, y_eval)
        evaluation_split: for MNIST, select either the deterministic validation
               split used during training or the untouched official test split.
               SHD retains its official train/test contract.
        dt_ms, t_ms: currently unused; kept for call-site compatibility
        evaluation_only: skip the unused MNIST training partition; requires
               split=True and evaluation_split='test'. Training outputs are None.

    Single canonical loader used by train, infer, and image paths so
    "first digit-0 sample" means the same physical sample everywhere.
    """
    if evaluation_only and (not split or evaluation_split != "test"):
        raise ValueError("evaluation_only requires split=True and evaluation_split='test'")
    if name == "mnist":
        from torchvision import datasets, transforms

        mnist_test = datasets.MNIST(
            root="/tmp/mnist",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        X_test = mnist_test.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_test = mnist_test.targets.numpy().astype(np.int64)
        if evaluation_only:
            return None, X_test, None, y_test
        mnist_train = datasets.MNIST(
            root="/tmp/mnist",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        X = mnist_train.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y = mnist_train.targets.numpy().astype(np.int64)
    elif name == "shd":
        # Event data. Keep SHD's OFFICIAL train/test split — it holds out two
        # speakers unseen in training, and every published SHD number is reported
        # on that test set, so resplitting would break
        # comparability. Returns object arrays of per-sample (units, times); the
        # caller bins them to spikes lazily at the run's dt.
        X_tr, y_tr = _read_shd_split("train")
        X_te, y_te = _read_shd_split("test")
        X_tr, y_tr = _subsample(X_tr, y_tr, max_samples)
        X_te, y_te = _subsample(X_te, y_te, max_samples)
        if split:
            return X_tr, X_te, y_tr, y_te
        return np.concatenate([X_tr, X_te]), np.concatenate([y_tr, y_te])
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if max_samples is not None and max_samples < len(X):
        idx = np.random.RandomState(MNIST_SPLIT_SEED).choice(
            len(X), max_samples, replace=False
        )
        X, y = X[idx], y[idx]

    if split:
        if evaluation_split == "test":
            return X, X_test, y, y_test
        from sklearn.model_selection import train_test_split

        validation_size = max(
            int(np.ceil(len(y) * MNIST_VALIDATION_FRACTION)),
            len(np.unique(y)),
        )
        X_train, X_validation, y_train, y_validation = train_test_split(
            X,
            y,
            test_size=validation_size,
            random_state=MNIST_SPLIT_SEED,
            stratify=y,
        )
        if evaluation_split == "validation":
            return X_train, X_validation, y_train, y_validation
        raise ValueError(
            "MNIST evaluation_split must be 'validation' or 'test', "
            f"got {evaluation_split!r}"
        )
    return X, y


def _load_dataset_image(dataset="mnist", digit_class=0, sample_idx=0):
    """Load a single image from a dataset. Returns (pixel_vec, digit_image)."""
    if dataset == "mnist":
        from torchvision import datasets, transforms

        mnist = datasets.MNIST(
            root="/tmp/mnist",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        data = mnist.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        targets = mnist.targets.numpy()
        images = mnist.data.numpy()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    idx = np.where(targets == digit_class)[0][sample_idx]
    return data[idx], images[idx]
