"""Validation of compiled training recipes against their graphs."""

from __future__ import annotations

import math
from typing import Any, Mapping

from snnlab import extensions as E
from snnlab.lang.compilation.diagnostics import Diagnostic, ValidationResult
from snnlab.lang.compilation.graph_validation import _check_extension


def validate_training(
    graph: Mapping[str, Any], training: Mapping[str, Any]
) -> ValidationResult:
    result = ValidationResult()
    try:
        E.restore(training)
    except (ValueError, TypeError) as error:
        result.diagnostics.append(Diagnostic("error", "E500", str(error)))
        return result
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
        if surrogate.get("kind") == "fast_sigmoid":
            scale = surrogate.get("gradient_scale", 1.0)
            if (
                not isinstance(scale, (int, float))
                or isinstance(scale, bool)
                or not math.isfinite(scale)
                or scale < 0
            ):
                result.diagnostics.append(
                    Diagnostic("error", "E416", "fast-sigmoid gradient_scale must be nonnegative and finite")
                )
            if not isinstance(surrogate.get("threshold_inclusive", True), bool):
                result.diagnostics.append(
                    Diagnostic("error", "E416", "fast-sigmoid threshold_inclusive must be boolean")
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
