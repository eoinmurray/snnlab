"""Validation, canonical serialisation, bundle I/O, and reports."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from snnlab import extensions as E

from ._version import __version__
from .core import Network
from .simulation import SimulationSpec, simulation_dict, validate_simulation
from .training import TrainSpec

SCHEMA = "snnlang.graph/v1"
BUNDLE_SCHEMA = "snnlang.bundle/v1"
TRAINING_SCHEMA = "snnlang.training/v1"
CAPABILITY_SCHEMA = "snnlang.capabilities/v1"


def canonical_json(data: Any) -> bytes:
    return (
        json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def digest(data: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(data)).hexdigest()


@dataclass(frozen=True)
class Diagnostic:
    severity: str
    code: str
    message: str
    subject: str | None = None

    def line(self) -> str:
        where = f" [{self.subject}]" if self.subject else ""
        return f"{self.severity.upper()} {self.code}{where}: {self.message}"


@dataclass
class ValidationResult:
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "warning"]

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("\n".join(d.line() for d in self.errors))


def graph_dict(net: Network) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "voltage_sampling": "explicit",
        "name": net.name,
        "timebase": {"dt": net.dt.json()},
        "inputs": sorted(net.inputs, key=lambda x: x["id"]),
        "populations": sorted(net.populations, key=lambda x: x["id"]),
        "projections": sorted(net.projections, key=lambda x: x["id"]),
        "operations": sorted(net.operations, key=lambda x: x["id"]),
        "parameters": sorted(net.parameters, key=lambda x: x["id"]),
        "constants": sorted(net.constants, key=lambda x: x["id"]),
        "outputs": sorted(net.outputs, key=lambda x: x["id"]),
        "observables": sorted(net.observables, key=lambda x: x["id"]),
        "assets": sorted(net.assets, key=lambda x: x["id"]),
        "groups": [
            {"id": g.name, "members": sorted(g.members), "parent": g.parent}
            for g in sorted(net.groups.values(), key=lambda x: x.name)
        ],
    }


def _training_dict(
    spec: TrainSpec, graph_digest: str, graph: Mapping[str, Any]
) -> dict[str, Any]:
    groups = [
        {
            "id": g.name,
            "parameters": sorted(g.ids()),
            "lr": g.lr,
            "frozen": g.frozen,
        }
        for g in spec.parameter_groups
    ]
    return {
        "schema": TRAINING_SCHEMA,
        "graph_digest": graph_digest,
        "objectives": [
            {
                "kind": o.kind,
                "prediction": o.prediction,
                "target": o.target,
                "weight": o.weight,
                **(
                    {"definition": o.definition, "config": o.config}
                    if o.definition
                    else {}
                ),
            }
            for o in spec.objectives
        ],
        "parameter_groups": groups,
        "resolved_parameters": {
            "trainable": sorted(
                parameter
                for group in groups
                if not group["frozen"]
                for parameter in group["parameters"]
            ),
            "frozen": sorted(
                parameter
                for group in groups
                if group["frozen"]
                for parameter in group["parameters"]
            ),
            "learning_rates": {
                parameter: group["lr"]
                for group in sorted(groups, key=lambda row: row["id"])
                if not group["frozen"]
                for parameter in group["parameters"]
            },
        },
        "regularizers": [
            {
                "kind": r.kind,
                "signals": sorted(r.signals),
                "strength": r.strength,
                "config": r.config,
                **({"definition": r.definition} if r.definition else {}),
            }
            for r in spec.regularizers
        ],
        "stop_gradients": sorted(s.signal for s in spec.stop_gradients),
        "optimizer": {
            "kind": spec.optimizer.kind,
            "config": spec.optimizer.config,
            **(
                {"definition": spec.optimizer.definition}
                if spec.optimizer.definition
                else {}
            ),
        },
        "epochs": spec.epochs,
        "gradient_clip": spec.gradient_clip,
        "surrogate": spec.surrogate.json() if spec.surrogate else None,
        "presentation_duration": (
            spec.presentation_duration.json() if spec.presentation_duration else None
        ),
        "resolved_gradients": {
            "surrogate": spec.surrogate.json() if spec.surrogate else None,
            "voltage_gradient_dampening": {
                population["id"]: population["neuron"].get("voltage_grad_dampen", 1.0)
                for population in graph.get("populations", [])
                if population.get("spiking")
            },
        },
    }


def _check_extension(out, category, spec, subject=None):
    try:
        E.resolve(category, spec)
    except (ValueError, TypeError) as error:
        out.diagnostics.append(Diagnostic("error", "E500", str(error), subject))


def _validate_neuron(neuron):
    if neuron.get("kind") != "cuba_lif":
        return
    tau = neuron.get("tau_mem", {})
    if (
        tau.get("unit") != "ms"
        or not isinstance(tau.get("value"), (int, float))
        or not math.isfinite(tau["value"])
        or tau["value"] <= 0
    ):
        raise ValueError("CUBA_LIF tau_mem must be positive finite ms")
    for key in ("capacitance_nf", "voltage_grad_dampen"):
        value = neuron.get(key, 1.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"CUBA_LIF {key} must be positive and finite")
    for key in ("resting_mv", "threshold_mv", "reset_mv", "initial_voltage_mv"):
        value = neuron.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"CUBA_LIF {key} must be finite")
    count = neuron.get("refractory_steps", 0)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("CUBA_LIF refractory_steps must be a non-negative integer")
    if neuron["reset_mv"] >= neuron["threshold_mv"]:
        raise ValueError("CUBA_LIF reset_mv must be below threshold_mv")


def validate_graph(graph: Mapping[str, Any]) -> ValidationResult:
    out = ValidationResult()
    if "voltage_sampling" in graph and graph["voltage_sampling"] != "explicit":
        out.diagnostics.append(
            Diagnostic("error", "E115", "unsupported voltage_sampling contract")
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
                for port, unit in E.resolve("neuron", neuron).state_units.items():
                    signals[f"{row['id']}.{port}"] = {
                        "shape": ["time", "batch", row["size"]],
                        "unit": unit,
                    }
            except (ValueError, TypeError):
                pass
        try:
            _validate_neuron(neuron)
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


def validate_training(
    graph: Mapping[str, Any], training: Mapping[str, Any]
) -> ValidationResult:
    result = ValidationResult()
    parameters = {p["id"] for p in graph["parameters"]}
    output_signals = {o["signal"] for o in graph["outputs"]}
    signals = {f"{x['id']}.value" for x in graph["operations"]} | output_signals
    selected: dict[str, str] = {}
    group_names: set[str] = set()
    for group in training.get("parameter_groups", []):
        group_id = group.get("id")
        if not group_id or group_id in group_names:
            result.diagnostics.append(
                Diagnostic(
                    "error",
                    "E407",
                    "parameter group id must be non-empty and unique",
                    group_id,
                )
            )
        group_names.add(group_id)
        members = group.get("parameters", [])
        if not members:
            result.diagnostics.append(
                Diagnostic(
                    "error", "E408", "parameter group must not be empty", group_id
                )
            )
        lr = group.get("lr")
        if (
            not isinstance(lr, (int, float))
            or isinstance(lr, bool)
            or not math.isfinite(lr)
        ):
            result.diagnostics.append(
                Diagnostic(
                    "error",
                    "E409",
                    "parameter group learning rate must be finite",
                    group_id,
                )
            )
        elif group.get("frozen") and lr != 0:
            result.diagnostics.append(
                Diagnostic(
                    "error",
                    "E410",
                    "frozen parameter group learning rate must be zero",
                    group_id,
                )
            )
        elif not group.get("frozen") and lr <= 0:
            result.diagnostics.append(
                Diagnostic(
                    "error",
                    "E411",
                    "trainable parameter group learning rate must be positive",
                    group_id,
                )
            )
        for pid in members:
            if pid not in parameters:
                result.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E401",
                        f"unknown training parameter {pid}",
                        group["id"],
                    )
                )
            if pid in selected:
                result.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E402",
                        f"parameter already selected by {selected[pid]}",
                        pid,
                    )
                )
            selected[pid] = group["id"]
    omitted = sorted(parameters - set(selected))
    if omitted:
        result.diagnostics.append(
            Diagnostic(
                "error",
                "E412",
                f"parameters must be assigned to exactly one trainable or frozen group; omitted={omitted}",
            )
        )
    resolved = training.get("resolved_parameters", {})
    expected_trainable = sorted(
        pid
        for group in training.get("parameter_groups", [])
        if not group.get("frozen")
        for pid in group.get("parameters", [])
    )
    expected_frozen = sorted(
        pid
        for group in training.get("parameter_groups", [])
        if group.get("frozen")
        for pid in group.get("parameters", [])
    )
    if (
        resolved.get("trainable") != expected_trainable
        or resolved.get("frozen") != expected_frozen
    ):
        result.diagnostics.append(
            Diagnostic(
                "error",
                "E413",
                "resolved trainable/frozen parameter sets do not match groups",
            )
        )
    expected_rates = {
        pid: group["lr"]
        for group in training.get("parameter_groups", [])
        if not group.get("frozen")
        for pid in group.get("parameters", [])
    }
    if resolved.get("learning_rates") != expected_rates:
        result.diagnostics.append(
            Diagnostic(
                "error", "E414", "resolved parameter learning rates do not match groups"
            )
        )
    optimizer = training.get("optimizer", {})
    if optimizer.get("kind") == "custom_optimizer":
        _check_extension(result, "optimizer", optimizer)
    surrogate = training.get("surrogate")
    if surrogate is not None:
        slope = surrogate.get("slope")
        if surrogate.get("kind") == "custom_surrogate":
            _check_extension(result, "surrogate", surrogate)
        elif surrogate.get("kind") != "fast_sigmoid":
            result.diagnostics.append(
                Diagnostic(
                    "error", "E415", f"unsupported surrogate {surrogate.get('kind')}"
                )
            )
        elif (
            not isinstance(slope, (int, float))
            or isinstance(slope, bool)
            or not math.isfinite(slope)
            or slope <= 0
        ):
            result.diagnostics.append(
                Diagnostic(
                    "error",
                    "E416",
                    "fast-sigmoid surrogate slope must be positive and finite",
                )
            )
    expected_gradients = {
        "surrogate": surrogate,
        "voltage_gradient_dampening": {
            population["id"]: population["neuron"].get("voltage_grad_dampen", 1.0)
            for population in graph.get("populations", [])
            if population.get("spiking")
        },
    }
    if training.get("resolved_gradients") != expected_gradients:
        result.diagnostics.append(
            Diagnostic(
                "error",
                "E417",
                "resolved gradient contract does not match graph and training recipe",
            )
        )
    for objective in training.get("objectives", []):
        if objective.get("kind") == "custom_objective":
            _check_extension(result, "objective", objective)
        elif objective.get("kind") != "cross_entropy":
            result.diagnostics.append(
                Diagnostic(
                    "error", "E424", f"unsupported objective {objective.get('kind')}"
                )
            )
        weight = objective.get("weight")
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(weight)
            or weight <= 0
        ):
            result.diagnostics.append(
                Diagnostic(
                    "error", "E425", "objective weight must be positive and finite"
                )
            )
        if objective["prediction"] not in output_signals:
            result.diagnostics.append(
                Diagnostic(
                    "error",
                    "E403",
                    f"objective prediction is not a named reachable output: {objective['prediction']}",
                )
            )
        if not objective.get("target"):
            result.diagnostics.append(
                Diagnostic("error", "E404", "objective target is empty")
            )
    for regularizer in training.get("regularizers", []):
        regularizer_signals = regularizer.get("signals", [])
        if not regularizer_signals:
            result.diagnostics.append(
                Diagnostic("error", "E405", "regularizer requires at least one signal")
            )
        for signal in regularizer_signals:
            if signal not in signals and not any(
                signal.startswith(p["id"] + ".") for p in graph["populations"]
            ):
                result.diagnostics.append(
                    Diagnostic(
                        "error", "E405", f"regularizer signal is unresolved: {signal}"
                    )
                )
        if regularizer.get("kind") == "custom_regularizer":
            _check_extension(result, "regularizer", regularizer)
            strength = regularizer.get("strength")
            if (
                not isinstance(strength, (int, float))
                or not math.isfinite(strength)
                or strength < 0
            ):
                result.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E420",
                        "regularizer strength must be finite and non-negative",
                    )
                )
        elif regularizer.get("kind") != "spike_budget":
            result.diagnostics.append(
                Diagnostic(
                    "error",
                    "E418",
                    f"unsupported regularizer {regularizer.get('kind')}",
                )
            )
        else:
            ceiling = regularizer.get("config", {}).get("ceiling", {})
            strength = regularizer.get("strength")
            if (
                ceiling.get("unit") != "Hz"
                or not isinstance(ceiling.get("value"), (int, float))
                or ceiling.get("value") < 0
            ):
                result.diagnostics.append(
                    Diagnostic(
                        "error", "E419", "spike-budget ceiling must be non-negative Hz"
                    )
                )
            if (
                not isinstance(strength, (int, float))
                or isinstance(strength, bool)
                or not math.isfinite(strength)
                or strength < 0
            ):
                result.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E420",
                        "spike-budget strength must be non-negative and finite",
                    )
                )
            expected_config = {
                "ceiling": ceiling,
                "penalty": "squared_hinge",
                "aggregation": "mean_presentations_then_layers_of_population_mean_rate",
            }
            if regularizer.get("config") != expected_config:
                result.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E421",
                        "spike-budget aggregation contract is unsupported",
                    )
                )
    duration = training.get("presentation_duration")
    if duration is not None:
        value = duration.get("value")
        if (
            duration.get("unit") != "ms"
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            result.diagnostics.append(
                Diagnostic(
                    "error",
                    "E422",
                    "presentation duration must be positive finite milliseconds",
                )
            )
        else:
            dt_ms = float(graph["timebase"]["dt"]["value"])
            steps = value / dt_ms
            if not math.isclose(steps, round(steps), abs_tol=1e-9):
                result.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E423",
                        "presentation duration must be an integer number of graph timesteps",
                    )
                )
    for signal in training.get("stop_gradients", []):
        if signal not in signals and not any(
            signal.startswith(p["id"] + ".") for p in graph["populations"]
        ):
            result.diagnostics.append(
                Diagnostic(
                    "error", "E406", f"stop-gradient signal is unresolved: {signal}"
                )
            )

    trainable = set(training.get("resolved_parameters", {}).get("trainable", []))
    operations = {
        f"{operation['id']}.value": operation
        for operation in graph.get("operations", [])
    }
    populations = {population["id"] for population in graph.get("populations", [])}
    incoming: dict[str, list[Mapping[str, Any]]] = {
        population: [] for population in populations
    }
    for projection in graph.get("projections", []):
        if projection.get("enabled", True):
            incoming.setdefault(projection["target"].partition(".")[0], []).append(
                projection
            )
    barriers = set(training.get("stop_gradients", []))

    def upstream_parameters(signal: str, visiting: set[str] | None = None) -> set[str]:
        if signal in barriers:
            return set()
        visiting = set() if visiting is None else visiting
        if signal in visiting:
            return set()
        visiting = {*visiting, signal}
        if signal in operations:
            operation = operations[signal]
            found = set(operation.get("parameters", []))
            for source in operation.get("sources", []):
                found.update(upstream_parameters(source, visiting))
            return found
        owner = signal.partition(".")[0]
        if owner in populations:
            found = set()
            for projection in incoming.get(owner, []):
                found.update(projection.get("parameters", []))
                found.update(upstream_parameters(projection["source"], visiting))
            return found
        return set()

    for index, objective in enumerate(training.get("objectives", [])):
        reachable = upstream_parameters(objective.get("prediction", ""))
        if not reachable & trainable:
            result.diagnostics.append(
                Diagnostic(
                    "error",
                    "E426",
                    "objective has no differentiable route to a trainable parameter; "
                    f"reachable={sorted(reachable)}, trainable={sorted(trainable)}",
                    f"objective[{index}]",
                )
            )
    for index, regularizer in enumerate(training.get("regularizers", [])):
        reachable = set()
        for signal in regularizer.get("signals", []):
            reachable.update(upstream_parameters(signal))
        if not reachable & trainable:
            result.diagnostics.append(
                Diagnostic(
                    "error",
                    "E427",
                    "regularizer has no differentiable route to a trainable parameter; "
                    f"reachable={sorted(reachable)}, trainable={sorted(trainable)}",
                    f"regularizer[{index}]",
                )
            )
    return result


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
    neuron_support = {"coba_lif", "cuba_lif", "leaky_integrator", "custom_neuron"}
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


def text_report(
    graph: Mapping[str, Any],
    training: Mapping[str, Any] | None,
    diagnostics: list[Diagnostic],
) -> str:
    params = graph["parameters"]
    count = sum(_shape_product(p["shape"]) for p in params)
    state_scalars = sum(p["size"] for p in graph["populations"])
    projection_edges = sum(
        _shape_product(
            next(p["shape"] for p in params if p["id"] == projection["parameters"][0])
        )
        for projection in graph["projections"]
    )
    selected = {
        p
        for group in (training or {}).get("parameter_groups", [])
        if not group["frozen"]
        for p in group["parameters"]
    }
    recurrent = sorted(
        p["id"]
        for p in graph["projections"]
        if p["connection"] in {"recurrent", "feedback"}
    )
    lines = [
        f"# snnlang report — {graph['name']}",
        "",
        f"Populations: {len(graph['populations'])} ({sum(p['size'] for p in graph['populations']):,} units)",
        f"Projections: {len(graph['projections'])}",
        f"Operations: {len(graph['operations'])}",
        f"Parameters: {len(params)} tensors / {count:,} scalars",
        f"Estimated state: {state_scalars:,} scalars per sample and timestep",
        f"Estimated dense projection edges: {projection_edges:,}",
        f"Trainable this recipe: {len(selected)} tensors",
        f"Outputs: {', '.join(o['id'] for o in graph['outputs']) or 'none'}",
        f"Recurrent paths: {', '.join(recurrent) or 'none'}",
        f"Diagnostics: {sum(d.severity == 'error' for d in diagnostics)} errors, {sum(d.severity == 'warning' for d in diagnostics)} warnings",
        "",
        "## Populations",
    ]
    lines += [
        f"- {p['id']}: {p['size']} × {p['neuron']['kind']} ({'spiking' if p['spiking'] else 'non-spiking'})"
        for p in graph["populations"]
    ]
    lines += ["", "## Projections"]
    lines += [
        f"- {p['id']}: {p['source']} → {p['target']} [{p['connection']}, {p['polarity']}]"
        for p in graph["projections"]
    ]
    lines += ["", "## Parameters"]
    lines += [
        f"- {p['id']}: {p['shape']} {p['unit']} ({'selected' if p['id'] in selected else 'frozen/unselected'})"
        for p in params
    ]
    if diagnostics:
        lines += ["", "## Diagnostics"] + [f"- {d.line()}" for d in diagnostics]
    return "\n".join(lines) + "\n"


def _shape_product(shape: list[Any]) -> int:
    result = 1
    for value in shape:
        if isinstance(value, int):
            result *= value
    return result


@dataclass
class Bundle:
    graph: dict[str, Any]
    training: dict[str, Any] | None
    manifest: dict[str, Any]
    diagnostics: list[Diagnostic]
    asset_sources: dict[str, Path] = field(default_factory=dict)
    simulation: dict[str, Any] | None = None

    def write(self, path: str | Path, *, visualise: bool = False) -> Path:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "graph.json").write_bytes(canonical_json(self.graph))
        if self.training:
            (root / "training.json").write_bytes(canonical_json(self.training))
        if self.simulation:
            (root / "simulation.json").write_bytes(canonical_json(self.simulation))
        assets_dir = root / "assets"
        for name, source in sorted(self.asset_sources.items()):
            assets_dir.mkdir(exist_ok=True)
            shutil.copyfile(source, assets_dir / name)
        (root / "manifest.json").write_bytes(canonical_json(self.manifest))
        reports = root / "reports"
        reports.mkdir(exist_ok=True)
        (reports / "summary.md").write_text(
            text_report(self.graph, self.training, self.diagnostics)
        )
        if visualise:
            from snnlab.viz import render_diagram

            from .diagram import diagram

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
        expand_groups: Collection[str] = (),
    ) -> Path:
        from snnlab.viz import render_diagram

        from .diagram import diagram

        return render_diagram(
            diagram(self, view=view, expand_groups=expand_groups),
            Path(path),
            scale=scale,
        )


def compile(
    network: Network,
    *,
    training: TrainSpec | None = None,
    simulation: SimulationSpec | None = None,
    target: str | None = None,
    assets: Mapping[str, str | Path] | None = None,
) -> Bundle:
    graph = graph_dict(network)
    graph_validation = validate_graph(graph)
    graph_validation.raise_for_errors()
    graph_digest = digest(graph)
    training_data = _training_dict(training, graph_digest, graph) if training else None
    training_validation = (
        validate_training(graph, training_data) if training_data else ValidationResult()
    )
    training_validation.raise_for_errors()
    simulation_data = simulation_dict(simulation, graph_digest) if simulation else None
    if simulation_data:
        validate_simulation(graph, simulation_data)
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
    if simulation_data:
        files.append({"path": "simulation.json", "digest": digest(simulation_data)})
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
        simulation=simulation_data,
    )


def load_bundle(path: str | Path) -> Bundle:
    root = Path(path)
    graph = json.loads((root / "graph.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    training_path = root / "training.json"
    training = json.loads(training_path.read_text()) if training_path.exists() else None
    simulation_path = root / "simulation.json"
    simulation = (
        json.loads(simulation_path.read_text()) if simulation_path.exists() else None
    )
    declared_paths = {row.get("path") for row in manifest.get("files", [])}
    if simulation and "simulation.json" not in declared_paths:
        raise ValueError("simulation.json is not authenticated by manifest")
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
    if simulation:
        if simulation.get("graph_digest") != manifest["graph_digest"]:
            raise ValueError("simulation specification targets a different graph")
        validate_simulation(graph, simulation)
    validation.raise_for_errors()
    return Bundle(
        graph,
        training,
        manifest,
        validation.diagnostics,
        asset_sources,
        simulation=simulation,
    )
