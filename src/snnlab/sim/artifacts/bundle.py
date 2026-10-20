"""Authenticated bundle metadata with external extension resolution."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snnlab import extensions as E
from snnlab.sim.artifacts.evidence import member_path, sha256_file


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


def load_graph_bundle(path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and authenticate manifest.json + graph.json from a bundle directory."""
    from snnlab.lang.compilation.bundles import Bundle

    if isinstance(path, Bundle):
        from snnlab.lang.compilation.graph_validation import validate_graph

        graph, manifest = copy.deepcopy(path.graph), copy.deepcopy(path.manifest)
        if (
            graph.get("schema") != "snnlang.graph/v1"
            or manifest.get("schema") != "snnlang.bundle/v1"
        ):
            raise BundleCompatibilityError("unsupported in-memory bundle schema")
        validate_graph(graph).raise_for_errors()
        declared_assets = {asset["id"] for asset in graph.get("assets", ())}
        manifest_assets = manifest.get("assets", ())
        if (
            len(manifest_assets) != len(declared_assets)
            or {asset["id"] for asset in manifest_assets} != declared_assets
        ):
            raise BundleCompatibilityError(
                "bundle asset declarations disagree with manifest"
            )
        for asset in manifest.get("assets", ()):
            source = path.asset_sources.get(Path(asset["path"]).name)
            if source is None or "sha256:" + sha256_file(source) != asset["digest"]:
                raise BundleCompatibilityError(
                    "bundle asset digest mismatch or missing asset"
                )
        # In-memory objects are structurally validated, not authenticated as a
        # historical disk image. Recipe graph references must still agree.
        manifest["graph_digest"] = _digest(graph)
        return manifest, graph
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
    for asset in manifest.get("assets", ()):
        if "sha256:" + sha256_file(member_path(root, asset["path"])) != asset["digest"]:
            raise BundleCompatibilityError("bundle asset digest mismatch")
    return manifest, graph


def load_training_recipe(
    path, manifest: dict[str, Any], graph: dict[str, Any]
) -> dict[str, Any]:
    """Load and authenticate the optional training recipe required by train."""
    from snnlab.lang.compilation.bundles import Bundle

    if isinstance(path, Bundle):
        from snnlab.lang.compilation.training_validation import validate_training

        if path.training is None:
            raise BundleCompatibilityError(
                "bundle-driven training requires a training recipe"
            )
        training = copy.deepcopy(path.training)
        if training.get("schema") != "snnlang.training/v1" or training.get(
            "graph_digest"
        ) != _digest(graph):
            raise BundleCompatibilityError(
                "in-memory training recipe targets a different graph or schema"
            )
        validate_training(graph, training).raise_for_errors()
        return training
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
