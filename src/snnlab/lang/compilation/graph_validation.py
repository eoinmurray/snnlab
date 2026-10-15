"""Structural and physical validation of graph specifications."""

from __future__ import annotations

import math
from typing import Any, Mapping

from snnlab import extensions as E
from snnlab._shared import neurons as N
from snnlab._shared.connectivity import resolve_autapses
from snnlab._shared.gains import validate as validate_gain
from snnlab.lang.compilation.diagnostics import Diagnostic, ValidationResult


def _check_extension(out, category, spec, subject=None):
    try:
        E.resolve(category, spec)
    except (ValueError, TypeError) as error:
        out.diagnostics.append(Diagnostic("error", "E500", str(error), subject))


def _validate_neuron(neuron, graph):
    legacy = graph.get("refractory_timing") != N.REFRACTORY_TIMING
    N.validate(neuron, legacy=legacy)
    if neuron.get("kind") in N.SPIKING_KINDS:
        N.resolve_refractory(
            neuron, float(graph["timebase"]["dt"]["value"]), legacy=legacy
        )


def validate_graph(graph: Mapping[str, Any]) -> ValidationResult:
    out = ValidationResult()
    try:
        E.restore(graph)
    except (ValueError, TypeError) as error:
        out.diagnostics.append(Diagnostic("error", "E500", str(error)))
        return out
    if "voltage_sampling" in graph and graph["voltage_sampling"] != "explicit":
        out.diagnostics.append(
            Diagnostic("error", "E115", "unsupported voltage_sampling contract")
        )
    if graph.get("refractory_timing") not in (
        None,
        N.REFRACTORY_TIMING,
        N.LEGACY_REFRACTORY_TIMING,
    ):
        out.diagnostics.append(
            Diagnostic("error", "E116", "unsupported refractory_timing contract")
        )
    collections = (
        "inputs",
        "populations",
        "projections",
        "operations",
        "parameters",
        "constants",
        "outputs",
        "observables",
        "assets",
        "groups",
    )
    seen: dict[str, str] = {}
    for collection in collections:
        for row in graph.get(collection, []):
            name = row.get("id")
            if not name:
                out.diagnostics.append(
                    Diagnostic("error", "E001", "missing identifier", collection)
                )
            elif name in seen:
                out.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E002",
                        f"duplicate identifier; first used by {seen[name]}",
                        name,
                    )
                )
            else:
                seen[name] = collection

    signals: dict[str, dict[str, Any]] = {}
    for row in graph.get("inputs", []):
        signals[f"{row['id']}.value"] = row
    for row in graph.get("populations", []):
        neuron = row.get("neuron", {})
        if neuron.get("kind") == "custom_neuron":
            _check_extension(out, "neuron", neuron, row["id"])
        try:
            for port, unit in E.state_units("neuron", neuron).items():
                signals[f"{row['id']}.{port}"] = {
                    "shape": ["time", "batch", row["size"]],
                    "unit": unit,
                }
        except (ValueError, TypeError):
            pass
        try:
            _validate_neuron(neuron, graph)
        except ValueError as error:
            out.diagnostics.append(Diagnostic("error", "E501", str(error), row["id"]))
        dampening = row.get("neuron", {}).get("voltage_grad_dampen", 1.0)
        if (
            not isinstance(dampening, (int, float))
            or isinstance(dampening, bool)
            or not math.isfinite(dampening)
            or dampening <= 0
        ):
            out.diagnostics.append(
                Diagnostic(
                    "error",
                    "E113",
                    "voltage_grad_dampen must be a positive finite factor",
                    row["id"],
                )
            )
        signals[f"{row['id']}.voltage"] = {
            "shape": ["time", "batch", row["size"]],
            "unit": "mV",
        }
        if neuron.get("kind") == "leaky_integrator":
            signals[f"{row['id']}.pre_reset_voltage"] = {
                "shape": ["time", "batch", row["size"]],
                "unit": "mV",
            }
        if row["spiking"]:
            signals[f"{row['id']}.spikes"] = {
                "shape": ["time", "batch", row["size"]],
                "unit": "spike",
            }
    population_sizes = {row["id"]: row["size"] for row in graph.get("populations", [])}
    for row in graph.get("projections", []):
        target = row["target"].partition(".")[0]
        if "modulation" in row:
            try:
                validate_gain(row["modulation"])
            except (ValueError, TypeError) as error:
                out.diagnostics.append(
                    Diagnostic("error", "E118", str(error), row["id"])
                )
        if target in population_sizes:
            try:
                synapse = row["synapse"]
                unit = E.synapse_unit(synapse)
                port = E.projection_port(synapse)
                signals[f"{row['id']}.{port}"] = {
                    "shape": ["time", "batch", population_sizes[target]],
                    "unit": unit,
                }
                if synapse["kind"] == "custom_synapse":
                    for state_port, state_unit in E.resolve(
                        "synapse", synapse
                    ).state_units.items():
                        signals[f"{row['id']}.{state_port}"] = {
                            "shape": ["time", "batch", population_sizes[target]],
                            "unit": state_unit,
                        }
            except (ValueError, TypeError) as error:
                out.diagnostics.append(
                    Diagnostic("error", "E500", str(error), row["id"])
                )
    for row in graph.get("operations", []):
        signals[f"{row['id']}.value"] = row

    parameter_rows = {p["id"]: p for p in graph.get("parameters", [])}
    initializer_fields = {
        "normal": {"mean", "std"},
        "lower_clamped_normal": {"mean", "std", "initial_zero_fraction", "zeroing"},
        "signed_normal": {"mean", "std"},
        "uniform": {"low", "high"},
        "constant": {"value"},
        "zeros": set(),
    }
    for row in parameter_rows.values():
        if "initialization_scaling" in row and row["initialization_scaling"] not in (
            "direct",
            "fan_in_normalized",
        ):
            out.diagnostics.append(
                Diagnostic("error", "E114", "invalid initialization_scaling", row["id"])
            )
        if not row.get("unit"):
            out.diagnostics.append(
                Diagnostic(
                    "error", "E109", "parameter requires an explicit unit", row["id"]
                )
            )
        initializer = row.get("initializer", {})
        kind = initializer.get("kind")
        if kind == "custom_initializer":
            _check_extension(out, "initializer", initializer, row["id"])
        elif kind not in initializer_fields:
            out.diagnostics.append(
                Diagnostic(
                    "error", "E110", f"unsupported initializer {kind}", row["id"]
                )
            )
            continue
        missing = initializer_fields.get(kind, set()) - set(initializer)
        if missing:
            out.diagnostics.append(
                Diagnostic(
                    "error",
                    "E111",
                    f"initializer missing fields {sorted(missing)}",
                    row["id"],
                )
            )
        constraint = row.get("constraint")
        if constraint is not None and constraint.get("kind") == "custom_constraint":
            _check_extension(out, "constraint", constraint, row["id"])
        elif constraint is not None and constraint.get("kind") != "non_negative":
            out.diagnostics.append(
                Diagnostic(
                    "error",
                    "E112",
                    f"unsupported constraint {constraint.get('kind')}",
                    row["id"],
                )
            )
    parameter_ids = set(parameter_rows)
    try:
        resolve_autapses(graph)
    except ValueError as error:
        out.diagnostics.append(Diagnostic("error", "E117", str(error)))
    population_ids = {p["id"] for p in graph.get("populations", [])}
    consumers: set[str] = set()
    adjacency: dict[str, set[str]] = {p: set() for p in population_ids}
    for row in graph.get("projections", []):
        try:
            synapse = row["synapse"]
            unit = E.synapse_unit(synapse)
            target_neuron = next(
                (
                    p["neuron"]
                    for p in graph["populations"]
                    if p["id"] == row["target"].partition(".")[0]
                ),
                None,
            )
            if target_neuron is not None and unit != E.neuron_unit(target_neuron):
                out.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E502",
                        f"synapse unit {unit} incompatible with neuron input unit {E.neuron_unit(target_neuron)}",
                        row["id"],
                    )
                )
            if synapse.get("kind") == "exponential_current":
                tau = synapse.get("tau", {})
                if (
                    tau.get("unit") != "ms"
                    or not isinstance(tau.get("value"), (int, float))
                    or not math.isfinite(tau["value"])
                    or tau["value"] <= 0
                ):
                    raise ValueError(
                        "ExponentialCurrent tau must be positive finite ms"
                    )
        except (ValueError, TypeError) as error:
            unit = "uS"
            out.diagnostics.append(Diagnostic("error", "E500", str(error), row["id"]))
        if not isinstance(row.get("enabled", True), bool):
            out.diagnostics.append(
                Diagnostic(
                    "error", "E108", "projection enabled must be boolean", row["id"]
                )
            )
        source = row["source"]
        target_pop, _, target_port = row["target"].partition(".")
        if source not in signals:
            out.diagnostics.append(
                Diagnostic("error", "E101", f"unresolved source {source}", row["id"])
            )
        if target_pop not in population_ids:
            out.diagnostics.append(
                Diagnostic(
                    "error",
                    "E102",
                    f"unresolved target population {target_pop}",
                    row["id"],
                )
            )
        if target_port not in {"excitatory", "inhibitory", "modulatory"}:
            out.diagnostics.append(
                Diagnostic(
                    "error",
                    "E103",
                    f"incompatible target port {target_port}",
                    row["id"],
                )
            )
        expected = {
            "excitatory": "excitatory",
            "inhibitory": "inhibitory",
            "modulatory": "modulatory",
        }
        if expected.get(target_port) != row.get("polarity"):
            out.diagnostics.append(
                Diagnostic(
                    "error",
                    "E104",
                    "projection polarity and target port disagree",
                    row["id"],
                )
            )
        delay = row.get("delay")
        if row.get("connection") == "feedback" and (
            not delay or not isinstance(delay, dict) or delay.get("value", 0) <= 0
        ):
            out.diagnostics.append(
                Diagnostic(
                    "error",
                    "E105",
                    "feedback requires an explicit non-zero delay",
                    row["id"],
                )
            )
        for pid in row.get("parameters", []):
            if pid not in parameter_ids:
                out.diagnostics.append(
                    Diagnostic(
                        "error", "E106", f"unresolved parameter {pid}", row["id"]
                    )
                )
            elif parameter_rows[pid].get("unit") != unit:
                out.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E113",
                        f"projection parameter requires unit {unit}, got {parameter_rows[pid].get('unit')}",
                        row["id"],
                    )
                )
            elif source in signals and target_pop in population_ids:
                expected_shape = [
                    next(
                        p["size"] for p in graph["populations"] if p["id"] == target_pop
                    ),
                    signals[source]["shape"][-1],
                ]
                if parameter_rows[pid]["shape"] != expected_shape:
                    out.diagnostics.append(
                        Diagnostic(
                            "error",
                            "E107",
                            f"projection parameter shape {parameter_rows[pid]['shape']} does not match {expected_shape}",
                            row["id"],
                        )
                    )
        consumers.add(source)
        source_owner = source.partition(".")[0]
        if (
            row.get("enabled", True)
            and source_owner in adjacency
            and target_pop in adjacency
        ):
            adjacency[source_owner].add(target_pop)

    for row in graph.get("operations", []):
        if row.get("kind") == "custom_operation":
            spec = {
                "definition": row.get("config", {}).get("definition"),
                "config": row.get("config", {}).get("settings", {}),
            }
            _check_extension(out, "operation", spec, row["id"])
        for source in row["sources"]:
            if source not in signals:
                out.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E201",
                        f"unresolved operation source {source}",
                        row["id"],
                    )
                )
            consumers.add(source)
        for pid in row.get("parameters", []):
            if pid not in parameter_ids:
                out.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E202",
                        f"unresolved operation parameter {pid}",
                        row["id"],
                    )
                )
        if (
            row["kind"] == "duration_normalise"
            and not row["config"].get("duration")
            and not row["config"].get("mask")
        ):
            out.diagnostics.append(
                Diagnostic(
                    "error", "E203", "spike-rate duration is ambiguous", row["id"]
                )
            )
        if (
            row["kind"] == "duration_normalise"
            and row["config"].get("duration") is not None
        ):
            try:
                duration = float(row["config"]["duration"])
            except (TypeError, ValueError):
                duration = -1
            if duration <= 0:
                out.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E208",
                        "spike-rate duration must be positive seconds",
                        row["id"],
                    )
                )
        if not row.get("shape") or not row.get("unit"):
            out.diagnostics.append(
                Diagnostic(
                    "error",
                    "E204",
                    "operation requires explicit shape and unit",
                    row["id"],
                )
            )
        primary = signals.get(row["sources"][0]) if row.get("sources") else None
        if primary:
            primary_shape = list(primary.get("shape", []))
            expected_shape: list[Any] | None = None
            if row["kind"] == "linear":
                expected_shape = [
                    *primary_shape[:-1],
                    row.get("config", {}).get("size"),
                ]
                for pid in row.get("parameters", []):
                    if pid in parameter_rows:
                        expected_parameter = [
                            row.get("config", {}).get("size"),
                            primary_shape[-1],
                        ]
                        if parameter_rows[pid]["shape"] != expected_parameter:
                            out.diagnostics.append(
                                Diagnostic(
                                    "error",
                                    "E209",
                                    f"linear parameter shape {parameter_rows[pid]['shape']} does not match {expected_parameter}",
                                    row["id"],
                                )
                            )
            elif row["kind"] in {"reduce_mean", "reduce_sum"}:
                if row.get("config", {}).get("window", "full") != "full":
                    out.diagnostics.append(
                        Diagnostic(
                            "error",
                            "E210",
                            "only full-window reductions are supported",
                            row["id"],
                        )
                    )
                if "time" not in primary_shape:
                    out.diagnostics.append(
                        Diagnostic(
                            "error",
                            "E211",
                            "time reduction requires a time axis",
                            row["id"],
                        )
                    )
                else:
                    time_axis = primary_shape.index("time")
                    expected_shape = [
                        dimension
                        for index, dimension in enumerate(primary_shape)
                        if index != time_axis
                    ]
            elif row["kind"] == "select_final":
                if not primary_shape or primary_shape[0] != "time":
                    out.diagnostics.append(
                        Diagnostic(
                            "error",
                            "E212",
                            "final selection requires a leading time axis",
                            row["id"],
                        )
                    )
                else:
                    expected_shape = primary_shape[1:]
            elif row["kind"] == "cumulative_sum":
                expected_shape = primary_shape
            elif row["kind"] == "duration_normalise":
                expected_shape = primary_shape
            if expected_shape is not None and row.get("shape") != expected_shape:
                out.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E213",
                        f"operation shape {row.get('shape')} does not match inferred {expected_shape}",
                        row["id"],
                    )
                )
        if (
            primary
            and row["kind"]
            in {"linear", "reduce_mean", "reduce_sum", "select_final", "cumulative_sum"}
            and row["unit"] != primary.get("unit")
        ):
            out.diagnostics.append(
                Diagnostic(
                    "error",
                    "E205",
                    f"operation unit {row['unit']} is incompatible with source unit {primary.get('unit')}",
                    row["id"],
                )
            )
        if row["kind"] == "duration_normalise":
            mask_id = row["config"].get("mask")
            if primary and primary.get("unit") != "spike":
                out.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E206",
                        "spike-rate numerator must have unit spike",
                        row["id"],
                    )
                )
            if mask_id:
                mask = signals.get(mask_id)
                if mask and (
                    mask.get("signal_type") != "mask"
                    or mask.get("shape", []) != ["time", "batch"]
                ):
                    out.diagnostics.append(
                        Diagnostic(
                            "error",
                            "E207",
                            "valid-duration mask must have type mask and shape (time, batch)",
                            row["id"],
                        )
                    )

    roots = {s.partition(".")[0] for s in consumers}
    output_signals = {o["signal"] for o in graph.get("outputs", [])}
    observable_signals = {o["signal"] for o in graph.get("observables", [])}
    for row in graph.get("outputs", []) + graph.get("observables", []):
        if row["signal"] not in signals:
            out.diagnostics.append(
                Diagnostic(
                    "error", "E301", f"unresolved signal {row['signal']}", row["id"]
                )
            )
    for pop in population_ids:
        incoming = any(pop in targets for targets in adjacency.values())
        outgoing = bool(adjacency[pop]) or pop in roots
        if not incoming and not outgoing:
            out.diagnostics.append(
                Diagnostic("warning", "W101", "disconnected population", pop)
            )
    for row in graph.get("projections", []):
        target = row["target"].partition(".")[0]
        target_used = (
            target in roots
            or bool(adjacency.get(target))
            or any(
                s.startswith(target + ".") for s in output_signals | observable_signals
            )
        )
        if not target_used:
            out.diagnostics.append(
                Diagnostic(
                    "warning",
                    "W102",
                    "projection target has no downstream consumer or observation",
                    row["id"],
                )
            )
    return out


def migrate_refractory(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Return a duration-only graph copy, preserving historical scheduling.

    Authenticate saved bundles with load_bundle first. The returned graph has
    a new digest; source manifests, checkpoints and runtime state are unchanged.
    """
    validate_graph(graph).raise_for_errors()
    migrated = N.migrate_refractory(graph)
    validate_graph(migrated).raise_for_errors()
    return migrated
