"""Named Python extensions shared by authoring and graph execution.

Bundles retain versioned names and JSON configuration, never Python callables.
Import the module registering a definition before compiling or executing it.
This module deliberately does not import PyTorch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class Definition:
    name: str
    function: Callable[..., Any]
    initialize: Callable[..., Any] | None = None
    validate: Callable[[Mapping[str, Any]], None] | None = None
    unit: str = "1"
    state_units: Mapping[str, str] = field(default_factory=dict)
    classification: bool = False


@dataclass(frozen=True)
class StateContext:
    """Initial state context; shape is (batch, cells), config is serialized."""

    shape: tuple[int, int]
    device: Any
    dtype: Any
    dt_ms: float
    config: Mapping[str, Any]


@dataclass(frozen=True)
class NeuronContext:
    state: Mapping[str, Any]
    excitatory: Any
    inhibitory: Any
    dt_ms: float
    config: Mapping[str, Any]
    spike: Callable[[Any], Any]


@dataclass(frozen=True)
class SynapseContext:
    state: Mapping[str, Any]
    drive: Any
    dt_ms: float
    config: Mapping[str, Any]


_REGISTRY: dict[tuple[str, str], Definition] = {}
_CATEGORIES = {
    "neuron",
    "synapse",
    "initializer",
    "constraint",
    "operation",
    "objective",
    "regularizer",
    "optimizer",
    "surrogate",
    "encoder",
}


def register(
    category: str,
    name: str,
    function: Callable[..., Any],
    *,
    initialize=None,
    validate=None,
    unit="1",
    state_units=None,
    classification=False,
) -> Definition:
    """Register a definition once; identifiers look like ``my_lab.model/v1``."""
    if category not in _CATEGORIES:
        raise ValueError(f"unknown extension category {category!r}")
    if not isinstance(name, str) or not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_.-]*/v[1-9][0-9]*", name
    ):
        raise ValueError(
            "extension names must be qualified and versioned, e.g. my_lab.model/v1"
        )
    if (
        not callable(function)
        or (initialize is not None and not callable(initialize))
        or (validate is not None and not callable(validate))
    ):
        raise TypeError("extension callbacks must be callable")
    if not isinstance(unit, str) or not unit:
        raise ValueError("extension unit must be non-empty")
    if category in {"neuron", "synapse"} and unit not in {"nA", "uS"}:
        raise ValueError("neuron input and synapse output units must be nA or uS")
    if category == "neuron" and initialize is None:
        raise ValueError("custom neurons require an initializer for their tensor state")
    units = dict(state_units or {})
    if any(
        not isinstance(k, str) or not k or "." in k or not isinstance(v, str) or not v
        for k, v in units.items()
    ):
        raise ValueError("state ports require simple names and non-empty units")
    if category == "neuron" and set(units) & {"voltage", "spikes", "refractory"}:
        raise ValueError("standard neuron ports cannot be redeclared")
    if category == "synapse" and set(units) & {"value", "current", "conductance"}:
        raise ValueError("standard synapse ports cannot be redeclared")
    key = category, name
    if key in _REGISTRY:
        raise ValueError(f"extension already registered: {category}:{name}")
    definition = Definition(
        name,
        function,
        initialize,
        validate,
        unit,
        MappingProxyType(units),
        bool(classification),
    )
    _REGISTRY[key] = definition
    return definition


def get(category: str, name: str) -> Definition:
    try:
        return _REGISTRY[category, name]
    except KeyError:
        raise ValueError(
            f"unregistered {category} extension {name!r}; import its registration module first"
        ) from None


def resolve(category: str, spec: Mapping[str, Any]) -> Definition:
    definition = get(category, spec.get("definition", ""))
    if definition.validate is not None:
        definition.validate(spec.get("config", {}))
    return definition


def is_custom(spec: Mapping[str, Any]) -> bool:
    return str(spec.get("kind", "")).startswith("custom_")


def synapse_unit(spec: Mapping[str, Any]) -> str:
    if spec.get("kind") == "exponential_current":
        return "nA"
    if spec.get("kind") == "custom_synapse":
        return resolve("synapse", spec).unit
    return "uS"


def neuron_unit(spec: Mapping[str, Any]) -> str:
    if spec.get("kind") in {"cuba_lif", "lif"}:
        return "nA"
    if spec.get("kind") == "custom_neuron":
        return resolve("neuron", spec).unit
    return "uS"


def projection_port(spec: Mapping[str, Any]) -> str:
    return "current" if synapse_unit(spec) == "nA" else "conductance"


def register_neuron(
    name, step, *, initialize, input_unit="nA", state_units=None, validate=None
):
    return register(
        "neuron",
        name,
        step,
        initialize=initialize,
        unit=input_unit,
        state_units=state_units,
        validate=validate,
    )


def register_synapse(
    name, step, *, initialize=None, output_unit="nA", state_units=None, validate=None
):
    return register(
        "synapse",
        name,
        step,
        initialize=initialize,
        unit=output_unit,
        state_units=state_units,
        validate=validate,
    )


def register_initializer(name, function, *, validate=None):
    return register("initializer", name, function, validate=validate)


def register_constraint(name, function, *, validate=None):
    return register("constraint", name, function, validate=validate)


def register_operation(name, function, *, validate=None):
    return register("operation", name, function, validate=validate)


def register_objective(name, function, *, classification=False, validate=None):
    return register(
        "objective", name, function, classification=classification, validate=validate
    )


def register_regularizer(name, function, *, validate=None):
    return register("regularizer", name, function, validate=validate)


def register_optimizer(name, factory, *, validate=None):
    return register("optimizer", name, factory, validate=validate)


def register_surrogate(name, derivative, *, validate=None):
    return register("surrogate", name, derivative, validate=validate)


def register_encoder(name, function, *, validate=None):
    return register("encoder", name, function, validate=validate)


def definitions(category: str | None = None) -> tuple[Definition, ...]:
    """Inspect registered definitions without exposing mutable registry storage."""
    if category is not None and category not in _CATEGORIES:
        raise ValueError(f"unknown extension category {category!r}")
    return tuple(
        value
        for (kind, _), value in sorted(_REGISTRY.items())
        if category is None or kind == category
    )


def required(graph, training=None):
    """List explicitly named dependencies; no automatic imports or code loading."""
    found = set()

    def visit(value):
        if isinstance(value, Mapping):
            kind = value.get("kind", "")
            if isinstance(kind, str) and kind.startswith("custom_"):
                category = kind.removeprefix("custom_")
                name = value.get("definition") or value.get("config", {}).get(
                    "definition"
                )
                if name:
                    found.add((category, name))
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(graph)
    visit(training)
    return [
        {"category": category, "definition": name} for category, name in sorted(found)
    ]
