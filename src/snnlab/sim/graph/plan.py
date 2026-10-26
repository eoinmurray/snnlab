"""Graph capabilities, topology lowering and prepared execution plans."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from snnlab import extensions as E
from snnlab._shared import neurons as N
from snnlab._shared.connectivity import resolve_autapses
from snnlab._shared.gains import validate as validate_gain
from snnlab._shared.projections import continuous_source, delay_steps, validate_source
from snnlab._shared.spatial import source_root, validate_spatial


def _signal_axes(graph: Mapping[str, Any]) -> dict[str, tuple[int | str, ...]]:
    axes = {
        f"{row['id']}.value": tuple(row["shape"])
        for row in (*graph.get("inputs", []), *graph.get("operations", []))
    }
    populations = {row["id"]: row for row in graph["populations"]}
    for name, population in populations.items():
        shape = ("time", "batch", population["size"])
        axes[f"{name}.voltage"] = shape
        if population["neuron"]["kind"] == "leaky_integrator":
            axes[f"{name}.pre_reset_voltage"] = shape
        if population["spiking"]:
            axes[f"{name}.spikes"] = shape
    for row in graph.get("projections", []):
        target = row["target"].partition(".")[0]
        axes[f"{row['id']}.{E.projection_port(row['synapse'])}"] = (
            "time",
            "batch",
            populations[target]["size"],
        )
    for row in graph.get("populations", []):
        for port in E.state_units("neuron", row["neuron"]):
            axes[f"{row['id']}.{port}"] = ("time", "batch", row["size"])
    for row in graph.get("projections", []):
        if row["synapse"]["kind"] == "custom_synapse":
            size = populations[row["target"].partition(".")[0]]["size"]
            for port in E.resolve("synapse", row["synapse"]).state_units:
                axes[f"{row['id']}.{port}"] = ("time", "batch", size)
    return axes


def _validate_interchange_parameters(
    graph: Mapping[str, Any], parameters: Mapping[str, torch.Tensor]
) -> None:
    rows = {row["id"]: row for row in graph.get("parameters", [])}
    if set(parameters) != set(rows):
        raise ValueError(
            f"graph parameter interchange must be complete; missing={sorted(set(rows) - set(parameters))}, extra={sorted(set(parameters) - set(rows))}"
        )
    for name, value in parameters.items():
        runtime_shape = tuple(reversed(rows[name]["shape"]))
        if tuple(value.shape) != runtime_shape:
            raise ValueError(
                f"graph parameter {name} expected runtime shape {runtime_shape}, got {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise ValueError(f"graph parameter {name} must use a floating dtype")


@dataclass(frozen=True)
class CapabilityIssue:
    element: str
    capability: str
    message: str


GRAPH_CAPABILITIES_V1 = {
    "schema": "tools/snnsim.capabilities/v1",
    "connectivity": {"dense", "conv2d"},
    "neurons": {"coba_lif", "cuba_lif", "leaky_integrator", "custom_neuron"}
    | N.ADAPTIVE_KINDS,
    "synapses": {
        "ampa",
        "gaba",
        "leaky_integrator",
        "exponential_current",
        "custom_synapse",
    },
    "operations": {
        "pool2d",
        "linear",
        "reduce_mean",
        "reduce_sum",
        "select_final",
        "duration_normalise",
        "cumulative_sum",
        "custom_operation",
        "divide",
    },
    "connections": {"feedforward", "recurrent", "feedback"},
    "recordings": {"spikes", "voltage"},
    "delays": "integer_steps",
    "training": {
        "objectives": {"cross_entropy", "custom_objective"},
        "regularizers": {"spike_budget", "custom_regularizer"},
        "optimizers": {"adamw", "custom_optimizer"},
        "parameter_groups": "named_trainable_and_frozen",
        "updates": "deterministic_epochs_and_minibatches",
        "targets": "named_classification_or_regression_arrays",
    },
}


def graph_capability_issues(graph: Mapping[str, Any]) -> list[CapabilityIssue]:
    """Return precise graph-executor capability failures."""
    issues: list[CapabilityIssue] = []
    neuron_capabilities: set[str] = {
        "coba_lif",
        "cuba_lif",
        "leaky_integrator",
        "custom_neuron",
    } | N.ADAPTIVE_KINDS
    synapse_capabilities: set[str] = {
        "ampa",
        "gaba",
        "leaky_integrator",
        "exponential_current",
        "custom_synapse",
    }
    operation_capabilities: set[str] = {
        "pool2d",
        "linear",
        "reduce_mean",
        "reduce_sum",
        "select_final",
        "duration_normalise",
        "cumulative_sum",
        "custom_operation",
        "divide",
    }
    connection_capabilities: set[str] = {"feedforward", "recurrent", "feedback"}
    for pop in graph.get("populations", []):
        if pop.get("neuron", {}).get("kind") == "custom_neuron":
            E.resolve("neuron", pop["neuron"])
        kind = pop.get("neuron", {}).get("kind")
        if kind not in neuron_capabilities:
            issues.append(
                CapabilityIssue(pop["id"], f"neuron:{kind}", "unsupported neuron kind")
            )
    for projection in graph.get("projections", []):
        if projection.get("synapse", {}).get("kind") == "custom_synapse":
            E.resolve("synapse", projection["synapse"])
        synapse = projection.get("synapse", {}).get("kind")
        if synapse not in synapse_capabilities:
            issues.append(
                CapabilityIssue(
                    projection["id"], f"synapse:{synapse}", "unsupported synapse kind"
                )
            )
        connection = projection.get("connection")
        if connection not in connection_capabilities:
            issues.append(
                CapabilityIssue(
                    projection["id"],
                    f"connection:{connection}",
                    "unsupported connection kind",
                )
            )
    for operation in graph.get("operations", []):
        if operation.get("kind") == "custom_operation":
            E.resolve(
                "operation",
                {
                    "definition": operation["config"]["definition"],
                    "config": operation["config"].get("settings", {}),
                },
            )
        kind = operation.get("kind")
        if kind not in operation_capabilities:
            issues.append(
                CapabilityIssue(
                    operation["id"], f"operation:{kind}", "unsupported operation kind"
                )
            )
    return issues


@dataclass(frozen=True)
class PlannedProjection:
    id: str
    source: str
    target: str
    polarity: str
    decay: float
    delay_steps: int
    parameter: str
    enabled: bool
    root_source: str = ""
    continuous: bool = False


@dataclass(frozen=True)
class GraphPlan:
    graph: Mapping[str, Any]
    dt_ms: float
    populations: tuple[Mapping[str, Any], ...]
    projections: tuple[PlannedProjection, ...]
    observables: tuple[Mapping[str, Any], ...]
    outputs: tuple[Mapping[str, Any], ...]
    refractory: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    autapses: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def _plan_projections(graph, dt, parameter_rows):
    planned = []
    for row in graph.get("projections", []):
        validate_source(graph, row)
        expected_unit = E.synapse_unit(row["synapse"])
        if "modulation" in row:
            validate_gain(row["modulation"])
        target_neuron = next(
            p["neuron"]
            for p in graph["populations"]
            if p["id"] == row["target"].partition(".")[0]
        )
        if expected_unit != E.neuron_unit(target_neuron):
            raise ValueError(
                f"{row['id']}: synapse unit {expected_unit} incompatible with neuron input unit {E.neuron_unit(target_neuron)}"
            )
        for parameter_id in row.get("parameters", []):
            unit = parameter_rows.get(parameter_id, {}).get("unit")
            weight_unit = E.synapse_weight_unit(row["synapse"])
            if unit != weight_unit:
                raise ValueError(
                    f"{row['id']}: projection parameter {parameter_id} requires unit {weight_unit}, got {unit}"
                )
        steps = delay_steps(row, dt)
        root = source_root(graph, row["source"])
        continuous = continuous_source(graph, row)
        source_owner = root.partition(".")[0]
        # Spike recurrent/feedback edges are causal at zero declared delay.
        # Continuous population edges always read beginning-of-step snapshots;
        # feedforward spike edges can consume current-step emissions.
        if (
            source_owner in {p["id"] for p in graph.get("populations", [])}
            and row.get("connection") != "feedforward"
            and not continuous
        ):
            steps = max(1, steps)
        tau = float(row["synapse"].get("tau", {"value": 1.0})["value"])
        if tau <= 0 or not math.isfinite(tau):
            raise ValueError(f"{row['id']}: synapse tau must be finite and positive")
        decay = (
            0.0 if row["synapse"]["kind"] == "leaky_integrator" else math.exp(-dt / tau)
        )
        planned.append(
            PlannedProjection(
                id=row["id"],
                source=row["source"],
                target=row["target"],
                polarity=row["polarity"],
                decay=decay,
                delay_steps=steps,
                parameter=row["parameters"][0],
                enabled=row.get("enabled", True),
                root_source=root,
                continuous=continuous,
            )
        )
    return planned


def plan_graph(graph: Mapping[str, Any]) -> GraphPlan:
    validate_spatial(graph)
    E.restore(graph)
    autapses = resolve_autapses(graph)
    if "voltage_sampling" in graph and graph["voltage_sampling"] != "explicit":
        raise ValueError("unsupported voltage_sampling contract")
    if graph.get("refractory_timing") not in (
        None,
        N.REFRACTORY_TIMING,
        N.LEGACY_REFRACTORY_TIMING,
    ):
        raise ValueError("unsupported refractory_timing contract")
    legacy_refractory = graph.get("refractory_timing") != N.REFRACTORY_TIMING
    populations = {p["id"]: p for p in graph.get("populations", [])}
    signal_references = [
        r["signal"] for r in (*graph.get("outputs", []), *graph.get("observables", []))
    ]
    signal_references.extend(
        source for op in graph.get("operations", []) for source in op["sources"]
    )
    for signal in signal_references:
        owner, _, port = signal.partition(".")
        if (
            port == "pre_reset_voltage"
            and populations.get(owner, {}).get("neuron", {}).get("kind")
            != "leaky_integrator"
        ):
            raise ValueError(f"{signal}: pre_reset_voltage requires a leaky integrator")
    for population in populations.values():
        N.validate(population["neuron"], legacy=legacy_refractory)
    issues = graph_capability_issues(graph)
    if issues:
        detail = "; ".join(
            f"{x.element} requires {x.capability}: {x.message}" for x in issues
        )
        raise ValueError(f"graph executor capability failure: {detail}")
    dt = float(graph["timebase"]["dt"]["value"])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("graph timebase dt must be finite and positive")
    refractory = {}
    for population in populations.values():
        if population["neuron"]["kind"] in N.SPIKING_KINDS:
            try:
                refractory[population["id"]] = N.resolve_refractory(
                    population["neuron"], dt, legacy=legacy_refractory
                )
            except ValueError as error:
                raise ValueError(f"{population['id']}: {error}") from error
    parameter_rows = {row["id"]: row for row in graph.get("parameters", [])}
    planned = _plan_projections(graph=graph, dt=dt, parameter_rows=parameter_rows)

    populations = list(graph.get("populations", []))
    population_ids = {p["id"] for p in populations}
    zero_edges = [
        (p.root_source.partition(".")[0], p.target.partition(".")[0])
        for p in planned
        if p.enabled
        and p.delay_steps == 0
        and not p.continuous
        and p.root_source.partition(".")[0] in population_ids
    ]
    ordered: list[Mapping[str, Any]] = []
    remaining = {p["id"]: p for p in populations}
    while remaining:
        ready = sorted(
            name
            for name in remaining
            if not any(dst == name and src in remaining for src, dst in zero_edges)
        )
        if not ready:
            raise ValueError(
                "zero-delay population projections form an algebraic cycle"
            )
        for name in ready:
            ordered.append(remaining.pop(name))
    return GraphPlan(
        graph=graph,
        dt_ms=dt,
        populations=tuple(ordered),
        projections=tuple(planned),
        observables=tuple(graph.get("observables", [])),
        outputs=tuple(graph.get("outputs", [])),
        refractory=refractory,
        autapses=autapses,
    )
