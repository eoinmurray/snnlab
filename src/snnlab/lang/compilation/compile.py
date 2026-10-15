"""Compile authoring declarations into validated portable bundles."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from snnlab import extensions as E
from snnlab._shared import neurons as N
from snnlab.lang._version import __version__
from snnlab.lang.compilation.bundles import BUNDLE_SCHEMA, Bundle
from snnlab.lang.compilation.capabilities import (
    capability_report,
    capability_requirements,
)
from snnlab.lang.compilation.diagnostics import ValidationResult
from snnlab.lang.compilation.graph_validation import validate_graph
from snnlab.lang.compilation.serialization import _training_dict, digest, graph_dict
from snnlab.lang.compilation.training_validation import validate_training
from snnlab.lang.network import Network
from snnlab.lang.training import TrainSpec


def compile(
    network: Network,
    *,
    training: TrainSpec | None = None,
    target: str | None = None,
    assets: Mapping[str, str | Path] | None = None,
    refractory_timing: str = N.REFRACTORY_TIMING,
) -> Bundle:
    """Compile and authenticate a graph with an explicit refractory schedule.

    ``legacy_coba`` preserves historical COBA LIF counter scheduling for fresh
    duration-based replay graphs. Other built-in neurons retain following-step
    scheduling. The default remains ``following_steps``.
    """
    if refractory_timing not in (N.REFRACTORY_TIMING, N.LEGACY_REFRACTORY_TIMING):
        raise ValueError("refractory_timing must be 'following_steps' or 'legacy_coba'")
    graph = E.with_requirements(graph_dict(network))
    graph["refractory_timing"] = refractory_timing
    graph_validation = validate_graph(graph)
    graph_validation.raise_for_errors()
    graph_digest = digest(graph)
    training_data = (
        E.with_requirements(_training_dict(training, graph_digest, graph))
        if training
        else None
    )
    training_validation = (
        validate_training(graph, training_data) if training_data else ValidationResult()
    )
    training_validation.raise_for_errors()
    diagnostics = (
        graph_validation.diagnostics
        + training_validation.diagnostics
        + capability_report(graph, target)
    )
    asset_sources: dict[str, Path] = {}
    manifest_assets = []
    declarations = {a["id"]: a for a in graph["assets"]}
    for logical, source_value in sorted((assets or {}).items()):
        if logical not in declarations:
            raise ValueError(
                f"physical asset supplied for undeclared logical asset: {logical}"
            )
        source = Path(source_value)
        if not source.is_file():
            raise ValueError(f"asset does not exist: {source}")
        suffix = source.suffix
        bundled_name = logical + suffix
        content_digest = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
        asset_sources[bundled_name] = source
        manifest_assets.append(
            {"id": logical, "path": f"assets/{bundled_name}", "digest": content_digest}
        )
    missing = set(declarations) - set(assets or {})
    if missing:
        raise ValueError(
            f"no physical path supplied for assets: {', '.join(sorted(missing))}"
        )
    files = [{"path": "graph.json", "digest": graph_digest}]
    if training_data:
        files.append({"path": "training.json", "digest": digest(training_data)})
    files.extend({"path": x["path"], "digest": x["digest"]} for x in manifest_assets)
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "compiler": {"name": "snnlang", "version": __version__},
        "graph_digest": graph_digest,
        "target": target,
        "required_capabilities": capability_requirements(graph),
        "files": files,
        "assets": manifest_assets,
    }
    requirements = E.required(graph, training_data)
    if requirements:
        manifest["extensions"] = requirements
    return Bundle(
        graph,
        training_data,
        manifest,
        diagnostics,
        asset_sources,
    )
