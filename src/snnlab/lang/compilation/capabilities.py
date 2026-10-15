"""Backend capability diagnostics and requirements."""

from __future__ import annotations

from typing import Any, Mapping

from snnlab._shared import neurons as N
from snnlab.lang.compilation.diagnostics import Diagnostic

CAPABILITY_SCHEMA = "snnlang.capabilities/v1"


def capability_report(graph: Mapping[str, Any], target: str | None) -> list[Diagnostic]:
    if target is None:
        return []
    supported = {
        "linear",
        "reduce_mean",
        "reduce_sum",
        "select_final",
        "duration_normalise",
        "cumulative_sum",
        "divide",
        "custom_operation",
    }
    diagnostics = []
    vocabulary = "snnlang.capabilities/v1"
    neuron_support = {
        "coba_lif",
        "cuba_lif",
        "leaky_integrator",
        "custom_neuron",
    } | N.ADAPTIVE_KINDS
    synapse_support = {
        "ampa",
        "gaba",
        "leaky_integrator",
        "exponential_current",
        "custom_synapse",
    }
    connection_support = {"feedforward", "recurrent", "feedback"}
    for population in graph["populations"]:
        kind = population["neuron"]["kind"]
        if kind not in neuron_support:
            diagnostics.append(
                Diagnostic(
                    "warning",
                    "C102",
                    f"{vocabulary}: {target} lacks neuron:{kind}",
                    population["id"],
                )
            )
    for projection in graph["projections"]:
        synapse = projection["synapse"]["kind"]
        connection = projection["connection"]
        if synapse not in synapse_support:
            diagnostics.append(
                Diagnostic(
                    "warning",
                    "C103",
                    f"{vocabulary}: {target} lacks synapse:{synapse}",
                    projection["id"],
                )
            )
        if connection not in connection_support:
            diagnostics.append(
                Diagnostic(
                    "warning",
                    "C104",
                    f"{vocabulary}: {target} lacks connection:{connection}",
                    projection["id"],
                )
            )
    for op in graph["operations"]:
        if op["kind"] not in supported:
            diagnostics.append(
                Diagnostic(
                    "warning",
                    "C101",
                    f"{target} capability for operation is unknown",
                    op["id"],
                )
            )
    return diagnostics


def capability_requirements(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical element-level requirements archived with every bundle."""
    elements = []
    for population in graph["populations"]:
        elements.append(
            {
                "element": population["id"],
                "features": [f"neuron:{population['neuron']['kind']}"],
            }
        )
    for projection in graph["projections"]:
        delay = projection.get("delay")
        elements.append(
            {
                "element": projection["id"],
                "features": [
                    f"synapse:{projection['synapse']['kind']}",
                    f"connection:{projection['connection']}",
                    "delay:none" if delay is None else "delay:explicit",
                ],
            }
        )
    for operation in graph["operations"]:
        elements.append(
            {"element": operation["id"], "features": [f"operation:{operation['kind']}"]}
        )
    for observable in graph["observables"]:
        elements.append(
            {
                "element": observable["id"],
                "features": [f"recording:{observable['signal'].partition('.')[2]}"],
            }
        )
    return {
        "schema": CAPABILITY_SCHEMA,
        "elements": sorted(elements, key=lambda row: row["element"]),
    }
