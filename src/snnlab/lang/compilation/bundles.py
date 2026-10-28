"""Portable architecture bundles and authenticated loading."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from snnlab import extensions as E
from snnlab.lang.compilation.diagnostics import Diagnostic
from snnlab.lang.compilation.graph_validation import validate_graph
from snnlab.lang.compilation.reports import text_report
from snnlab.lang.compilation.serialization import digest
from snnlab.lang.compilation.training_validation import validate_training
from snnlab.viz.diagrams.model import DiagramTheme
from snnlab.viz.diagrams.spec import DiagramLayout

BUNDLE_SCHEMA = "snnlang.bundle/v1"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@dataclass
class Bundle:
    graph: dict[str, Any]
    training: dict[str, Any] | None
    manifest: dict[str, Any]
    diagnostics: list[Diagnostic]
    asset_sources: dict[str, Path] = field(default_factory=dict)

    def write(self, path: str | Path, *, visualise: bool = False) -> Path:
        E.require_portable(self.graph, self.training)
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "graph.json", self.graph)
        if self.training:
            _write_json(root / "training.json", self.training)
        assets_dir = root / "assets"
        for name, source in sorted(self.asset_sources.items()):
            assets_dir.mkdir(exist_ok=True)
            shutil.copyfile(source, assets_dir / name)
        _write_json(root / "manifest.json", self.manifest)
        reports = root / "reports"
        reports.mkdir(exist_ok=True)
        (reports / "summary.md").write_text(
            text_report(self.graph, self.training, self.diagnostics)
        )
        if visualise:
            from snnlab.lang.diagrams import diagram
            from snnlab.viz import render_diagram

            for view in ("circuit", "training", "expanded"):
                visual = diagram(self, view=view)
                render_diagram(visual, reports / f"{view}.svg")
                render_diagram(visual, reports / f"{view}.png", scale=2)
        return root

    def visualise(
        self,
        path: str | Path,
        *,
        view: str = "circuit",
        scale: int = 1,
        connection_labels: Mapping[str, str] | None = None,
        layout: DiagramLayout | dict | None = None,
        theme: DiagramTheme | None = None,
    ) -> Path:
        from snnlab.lang.diagrams import diagram
        from snnlab.viz import render_diagram
        from snnlab.viz.diagrams.spec import check_legacy_labels

        layout = check_legacy_labels(layout, connection_labels)

        return render_diagram(
            diagram(self, view=view, connection_labels=connection_labels),
            Path(path),
            scale=scale,
            layout=layout,
            **({"theme": theme} if theme is not None else {}),
        )

    def excalidraw_export(
        self,
        path: str | Path,
        *,
        view: str = "circuit",
        connection_labels: Mapping[str, str] | None = None,
        overwrite: bool = False,
        layout: DiagramLayout | dict | None = None,
        theme: DiagramTheme | None = None,
    ) -> Path:
        """Export an editable drawing without changing the bundle or running it."""
        from snnlab.lang.diagrams import diagram
        from snnlab.viz.diagrams.excalidraw import export_excalidraw
        from snnlab.viz.diagrams.spec import check_legacy_labels

        layout = check_legacy_labels(layout, connection_labels)

        visual = diagram(self, view=view, connection_labels=connection_labels)
        return export_excalidraw(
            visual,
            path,
            graph_digest=self.manifest["graph_digest"],
            overwrite=overwrite,
            layout=layout,
            **({"theme": theme} if theme is not None else {}),
        )


def load_bundle(path: str | Path) -> Bundle:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"bundle not found: {root}")
    if root.is_file() and root.name == "manifest.json":
        root = root.parent
    if not (root / "manifest.json").is_file() or not (root / "graph.json").is_file():
        raise ValueError(
            f"{root} is not a bundle: manifest.json and graph.json are required"
        )
    graph = json.loads((root / "graph.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    training_path = root / "training.json"
    training = json.loads(training_path.read_text()) if training_path.exists() else None
    if digest(graph) != manifest["graph_digest"]:
        raise ValueError("graph digest does not match manifest")
    asset_sources: dict[str, Path] = {}
    for file_entry in manifest.get("files", []):
        file_path = root / file_entry["path"]
        if not file_path.is_file():
            raise ValueError(f"bundle file is missing: {file_entry['path']}")
        if file_entry["path"].endswith(".json"):
            actual_digest = digest(json.loads(file_path.read_text()))
        else:
            actual_digest = (
                "sha256:" + hashlib.sha256(file_path.read_bytes()).hexdigest()
            )
        if actual_digest != file_entry["digest"]:
            raise ValueError(f"bundle file digest mismatch: {file_entry['path']}")
    for asset in manifest.get("assets", []):
        asset_sources[Path(asset["path"]).name] = root / asset["path"]
    validation = validate_graph(graph)
    if training:
        if training["graph_digest"] != manifest["graph_digest"]:
            raise ValueError("training specification targets a different graph")
        validation.diagnostics.extend(validate_training(graph, training).diagnostics)
    validation.raise_for_errors()
    return Bundle(
        graph,
        training,
        manifest,
        validation.diagnostics,
        asset_sources,
    )
