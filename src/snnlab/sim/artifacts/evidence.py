"""Small, explicit persistence helpers for portable evidence (never pickle)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

METRICS_SCHEMA = "snnlab.metrics/v1"


def json_value(value):
    """Convert supported JSON/NumPy scalar containers; reject opaque values."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("evidence mapping keys must be strings")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported evidence value: {type(value).__name__}")


def publish_file(path, write, *, overwrite=False):
    """Stage a file beside its destination and publish it atomically."""
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be boolean")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(descriptor)
    pending = Path(temporary)
    try:
        write(pending)
        if overwrite:
            os.replace(pending, path)
        else:
            os.link(pending, path)
    finally:
        pending.unlink(missing_ok=True)
    return path


def write_metrics(path, metrics, *, overwrite=False):
    from snnlab import __version__

    payload = (
        json.dumps(
            {
                "schema": METRICS_SCHEMA,
                "snnlab_version": __version__,
                "metrics": json_value(metrics),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    return publish_file(
        path, lambda pending: pending.write_text(payload), overwrite=overwrite
    )


def read_metrics(path, *, legacy=False):
    """Read validated native metrics; opt into historical bare JSON mappings."""

    def invalid_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    payload = json.loads(Path(path).read_text(), parse_constant=invalid_constant)
    if not isinstance(payload, dict):
        raise ValueError("metrics must be a JSON mapping")
    if payload.get("schema") == METRICS_SCHEMA:
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("metrics envelope is missing a metrics mapping")
        # Also rejects overflowed JSON numbers such as 1e999.
        json.dumps(metrics, allow_nan=False)
        return metrics
    if "schema" in payload or not legacy:
        raise ValueError(
            "unsupported metrics schema; bare mappings require legacy=True"
        )
    json.dumps(payload, allow_nan=False)
    return payload


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def member_path(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("artifact file reference must be relative")
    root = Path(root).resolve()
    destination = (root / name).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("artifact file reference escapes its directory")
    return destination


def publish_directory(path, write, *, overwrite=False):
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be boolean")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.parent / f".{path.name}.export-lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        with tempfile.TemporaryDirectory(
            prefix=f".{path.name}-", dir=path.parent
        ) as temporary:
            pending = Path(temporary) / "pending"
            pending.mkdir()
            write(pending)
            previous = Path(temporary) / "previous"
            if path.exists():
                path.rename(previous)
            try:
                pending.rename(path)
            except BaseException:
                if previous.exists():
                    previous.rename(path)
                raise
            if previous.is_dir():
                shutil.rmtree(previous)
    finally:
        lock.unlink()
    return path
