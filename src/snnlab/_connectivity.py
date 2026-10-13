"""Structural connectivity contracts shared by authoring and execution."""

from __future__ import annotations

from typing import Any, Mapping


def resolve_autapses(graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve policies without rewriting historical graph data."""
    populations = {row["id"] for row in graph.get("populations", [])}
    parameters = {row["id"]: row for row in graph.get("parameters", [])}
    resolved = {}
    masks: dict[str, bool] = {}
    owners: dict[str, str] = {}
    for row in graph.get("projections", []):
        requested = row.get("allow_autapses")
        allowed = row.get("allow_autapses", True)
        if not isinstance(allowed, bool):
            raise ValueError(f"{row['id']}: allow_autapses must be boolean")
        source = row["source"].partition(".")[0]
        target = row["target"].partition(".")[0]
        same_population = source in populations and source == target
        excluded = same_population and not allowed
        parameter_ids = row.get("parameters", [])
        if not parameter_ids or parameter_ids[0] not in parameters:
            continue  # Ordinary graph validation reports unresolved parameters.
        shape = parameters[parameter_ids[0]].get("shape", [])
        if len(shape) != 2 or not isinstance(shape[1], int):
            continue  # Ordinary graph validation reports invalid shapes.
        if excluded and shape[0] != shape[1]:
            raise ValueError(
                f"{row['id']}: autapse exclusion requires matching neuron axes"
            )
        resolved[row["id"]] = {
            "requested": requested,
            "allow_autapses": allowed,
            "effective": "excluded"
            if excluded
            else "allowed"
            if same_population
            else "not_applicable",
            "origin": "explicit" if "allow_autapses" in row else "legacy_default",
            "eligible_fan_in": shape[1] - int(excluded),
        }
        for parameter in parameter_ids:
            if parameter in masks and masks[parameter] != excluded:
                raise ValueError(
                    f"{parameter}: incompatible autapse masks for shared parameter "
                    f"used by {owners[parameter]} and {row['id']}"
                )
            masks[parameter] = excluded
            owners[parameter] = row["id"]
    for row in graph.get("operations", []):
        for parameter in row.get("parameters", []):
            if masks.get(parameter, False):
                raise ValueError(
                    f"{parameter}: incompatible autapse masks for shared parameter "
                    f"used by {owners[parameter]} and operation {row['id']}"
                )
    return resolved
