"""Authenticated bundle metadata with external extension resolution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snnlab import extensions as E


class BundleCompatibilityError(ValueError):
    """The bundle is valid data, but outside this backend adapter's subset."""


@dataclass(frozen=True)
class BackendCapability:
    """Versioned capability requirement attached to one graph element."""

    schema: str
    element: str
    feature: str


def required_capabilities_v1(graph: dict[str, Any]) -> tuple[BackendCapability, ...]:
    """Describe bundle requirements without importing the authoring package."""
    rows: list[BackendCapability] = []
    for pop in graph.get("populations", []):
        rows.append(
            BackendCapability(
                "tools/snnsim.capability/v1",
                pop["id"],
                f"neuron:{pop['neuron']['kind']}",
            )
        )
    for projection in graph.get("projections", []):
        rows.extend(
            (
                BackendCapability(
                    "tools/snnsim.capability/v1",
                    projection["id"],
                    f"synapse:{projection['synapse']['kind']}",
                ),
                BackendCapability(
                    "tools/snnsim.capability/v1",
                    projection["id"],
                    f"connection:{projection['connection']}",
                ),
                BackendCapability(
                    "tools/snnsim.capability/v1",
                    projection["id"],
                    "delay:integer_steps",
                ),
            )
        )
    for operation in graph.get("operations", []):
        rows.append(
            BackendCapability(
                "tools/snnsim.capability/v1",
                operation["id"],
                f"operation:{operation['kind']}",
            )
        )
    for observable in graph.get("observables", []):
        rows.append(
            BackendCapability(
                "tools/snnsim.capability/v1",
                observable["id"],
                f"recording:{observable['signal'].partition('.')[2]}",
            )
        )
    return tuple(rows)


def _canonical_json(data: Any) -> bytes:
    return (
        json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def _digest(data: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(data)).hexdigest()


def load_graph_bundle(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and authenticate manifest.json + graph.json from a bundle directory."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"bundle not found: {root}")
    if root.is_file():
        if root.name != "manifest.json":
            raise BundleCompatibilityError(
                "--bundle must name a bundle directory or its manifest.json"
            )
        root = root.parent
    manifest_path = root / "manifest.json"
    graph_path = root / "graph.json"
    if not manifest_path.is_file() or not graph_path.is_file():
        raise BundleCompatibilityError(
            f"{root} is not a bundle: manifest.json and graph.json are required"
        )
    manifest = json.loads(manifest_path.read_text())
    graph = json.loads(graph_path.read_text())
    if manifest.get("schema") != "snnlang.bundle/v1":
        raise BundleCompatibilityError(
            f"unsupported bundle schema: {manifest.get('schema')!r}"
        )
    if graph.get("schema") != "snnlang.graph/v1":
        raise BundleCompatibilityError(
            f"unsupported graph schema: {graph.get('schema')!r}"
        )
    actual = _digest(graph)
    if actual != manifest.get("graph_digest"):
        raise BundleCompatibilityError("graph.json digest does not match manifest.json")
    E.restore(graph)
    return manifest, graph


def load_training_recipe(
    path: str | Path, manifest: dict[str, Any], graph: dict[str, Any]
) -> dict[str, Any]:
    """Load and authenticate the optional training recipe required by train."""
    root = Path(path)
    if root.is_file():
        root = root.parent
    training_path = root / "training.json"
    if not training_path.is_file():
        raise BundleCompatibilityError("bundle-driven training requires training.json")
    training = json.loads(training_path.read_text())
    if training.get("schema") != "snnlang.training/v1":
        raise BundleCompatibilityError(
            f"unsupported training schema: {training.get('schema')!r}"
        )
    if training.get("graph_digest") != manifest.get("graph_digest"):
        raise BundleCompatibilityError(
            "training.json graph digest does not match graph.json"
        )
    declared = {row.get("path"): row.get("digest") for row in manifest.get("files", [])}
    if declared.get("training.json") != _digest(training):
        raise BundleCompatibilityError(
            "training.json digest does not match manifest.json"
        )
    if training["graph_digest"] != _digest(graph):
        raise BundleCompatibilityError(
            "training.json graph digest does not authenticate this graph"
        )
    E.restore(training)
    return training
