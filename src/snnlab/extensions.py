"""Named Python extensions shared by authoring and graph execution.

Bundles retain versioned names, import references and JSON configuration.
Referenced modules must be available in each process; no code is bundled.
This module deliberately does not import PyTorch.
"""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass, field
from importlib import metadata
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
    references: Mapping[str, str] = field(default_factory=dict)
    package: str | None = None
    package_version: str | None = None


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


def _import_callback(reference: str) -> Callable[..., Any]:
    module, separator, attribute = reference.partition(":")
    if (
        not separator
        or module == "__main__"
        or not all(part.isidentifier() for part in module.split("."))
        or not all(part.isidentifier() for part in attribute.split("."))
    ):
        raise ValueError(
            f"invalid extension callback reference {reference!r}; use module:function"
        )
    try:
        value = importlib.import_module(module)
        for part in attribute.split("."):
            value = getattr(value, part)
    except (ImportError, AttributeError) as error:
        raise ValueError(
            f"cannot load extension callback {reference!r}: {error}; make its module available in this environment"
        ) from error
    if not callable(value):
        raise TypeError(f"extension callback {reference!r} is not callable")
    return value


def _callback_reference(function) -> str | None:
    module = getattr(function, "__module__", "")
    attribute = getattr(function, "__qualname__", "")
    if module == "__main__" or not module or not attribute:
        return None
    if not all(part.isidentifier() for part in attribute.split(".")):
        return None
    value = sys.modules.get(module)
    for part in attribute.split("."):
        value = getattr(value, part, None)
    return f"{module}:{attribute}" if value is function else None


def _check_package(package, package_version):
    if package is None:
        if package_version is not None:
            raise ValueError("package_version requires a package name")
        return None
    if not isinstance(package, str) or not package:
        raise ValueError("extension package must be a non-empty distribution name")
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        raise ValueError(
            f"required extension package {package!r} is not installed"
        ) from None
    if package_version is not None and installed != package_version:
        raise ValueError(
            f"extension package {package!r} requires version {package_version}, installed {installed}"
        )
    return installed


def register(
    category: str,
    name: str,
    function: Callable[..., Any] | str,
    *,
    initialize=None,
    validate=None,
    unit="1",
    state_units=None,
    classification=False,
    package=None,
    package_version=None,
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
    package_version = _check_package(package, package_version)
    references = {}
    callbacks = {}
    for role, callback in (
        ("function", function),
        ("initialize", initialize),
        ("validate", validate),
    ):
        if isinstance(callback, str):
            references[role] = callback
            callback = _import_callback(callback)
        elif callback is not None:
            reference = _callback_reference(callback)
            if reference is not None:
                references[role] = reference
        callbacks[role] = callback
    function, initialize, validate = (
        callbacks[role] for role in ("function", "initialize", "validate")
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
        MappingProxyType(references),
        package,
        package_version,
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
    name,
    step,
    *,
    initialize,
    input_unit="nA",
    state_units=None,
    validate=None,
    package=None,
    package_version=None,
):
    return register(
        "neuron",
        name,
        step,
        initialize=initialize,
        unit=input_unit,
        state_units=state_units,
        validate=validate,
        package=package,
        package_version=package_version,
    )


def register_synapse(
    name,
    step,
    *,
    initialize=None,
    output_unit="nA",
    state_units=None,
    validate=None,
    package=None,
    package_version=None,
):
    return register(
        "synapse",
        name,
        step,
        initialize=initialize,
        unit=output_unit,
        state_units=state_units,
        validate=validate,
        package=package,
        package_version=package_version,
    )


def register_objective(
    name,
    function,
    *,
    classification=False,
    validate=None,
    package=None,
    package_version=None,
):
    return register(
        "objective",
        name,
        function,
        classification=classification,
        validate=validate,
        package=package,
        package_version=package_version,
    )


def register_initializer(
    name, function, *, validate=None, package=None, package_version=None
):
    return register(
        "initializer",
        name,
        function,
        validate=validate,
        package=package,
        package_version=package_version,
    )


def register_constraint(
    name, function, *, validate=None, package=None, package_version=None
):
    return register(
        "constraint",
        name,
        function,
        validate=validate,
        package=package,
        package_version=package_version,
    )


def register_operation(
    name, function, *, validate=None, package=None, package_version=None
):
    return register(
        "operation",
        name,
        function,
        validate=validate,
        package=package,
        package_version=package_version,
    )


def register_regularizer(
    name, function, *, validate=None, package=None, package_version=None
):
    return register(
        "regularizer",
        name,
        function,
        validate=validate,
        package=package,
        package_version=package_version,
    )


def register_optimizer(
    name, factory, *, validate=None, package=None, package_version=None
):
    return register(
        "optimizer",
        name,
        factory,
        validate=validate,
        package=package,
        package_version=package_version,
    )


def register_surrogate(
    name, derivative, *, validate=None, package=None, package_version=None
):
    return register(
        "surrogate",
        name,
        derivative,
        validate=validate,
        package=package,
        package_version=package_version,
    )


def register_encoder(
    name, function, *, validate=None, package=None, package_version=None
):
    return register(
        "encoder",
        name,
        function,
        validate=validate,
        package=package,
        package_version=package_version,
    )


def definitions(category: str | None = None) -> tuple[Definition, ...]:
    """Inspect registered definitions without exposing mutable registry storage."""
    if category is not None and category not in _CATEGORIES:
        raise ValueError(f"unknown extension category {category!r}")
    return tuple(
        value
        for (kind, _), value in sorted(_REGISTRY.items())
        if category is None or kind == category
    )


def _descriptor(category, definition):
    roles = {"function"}
    if definition.initialize is not None:
        roles.add("initialize")
    if definition.validate is not None:
        roles.add("validate")
    row = {
        "category": category,
        "definition": definition.name,
        "unit": definition.unit,
        "state_units": dict(definition.state_units),
        "classification": definition.classification,
    }
    if roles <= definition.references.keys():
        row["implementation"] = dict(definition.references)
    if definition.package is not None:
        row.update(
            package=definition.package, package_version=definition.package_version
        )
    return row


def required(graph, training=None):
    """List used definitions with persistent callback references when available."""
    found = set()
    saved = {}
    for data in (graph, training):
        for row in (data or {}).get("extensions", []):
            key = row["category"], row["definition"]
            if key in saved and saved[key] != row:
                raise ValueError(f"conflicting extension descriptors for {key}")
            saved[key] = row

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
        saved[(category, name)]
        if (category, name) in saved
        else _descriptor(category, get(category, name))
        for category, name in sorted(found)
    ]


def with_requirements(data):
    """Attach import metadata to the authenticated graph or training recipe."""
    rows = required(data)
    return {**data, "extensions": rows} if rows else data


def require_portable(graph, training=None):
    """Reject saving definitions whose callbacks cannot be imported elsewhere."""
    for row in required(graph, training):
        if "implementation" not in row:
            raise ValueError(
                f"cannot save {row['category']} extension {row['definition']!r}: callbacks must be importable module functions; move lambdas, closures and __main__ functions into a module"
            )


def restore(data):
    """Automatically resolve descriptors in an authenticated graph or recipe."""
    rows = data.get("extensions", [])
    if not isinstance(rows, (list, tuple)):
        raise ValueError("extension descriptors must be a list")
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("extension descriptors must be mappings")
        category, name = row.get("category"), row.get("definition")
        if not isinstance(category, str) or category not in _CATEGORIES:
            raise ValueError(f"unknown extension category {category!r}")
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_.-]*/v[1-9][0-9]*", name
        ):
            raise ValueError(f"invalid saved extension name {name!r}")
        if (category, name) in seen:
            raise ValueError(f"duplicate saved extension {category}:{name}")
        seen.add((category, name))
        implementation = row.get("implementation")
        if implementation is None:
            get(category, name)  # Older bundles require explicit registration.
            continue
        if not isinstance(implementation, Mapping) or "function" not in implementation:
            raise ValueError(f"invalid implementation descriptor for {category}:{name}")
        if set(implementation) - {"function", "initialize", "validate"} or any(
            not isinstance(reference, str) for reference in implementation.values()
        ):
            raise ValueError(f"invalid callback references for {category}:{name}")
        package, version = row.get("package"), row.get("package_version")
        if package is not None and (not isinstance(version, str) or not version):
            raise ValueError(
                f"extension package version is missing for {category}:{name}"
            )
        _check_package(package, version)
        existing = _REGISTRY.get((category, name))
        if existing is not None:
            if _descriptor(category, existing) != row:
                raise ValueError(
                    f"extension implementation conflicts with saved definition {category}:{name}"
                )
            continue
        # Imports can populate the registry when a module also registers definitions.
        for reference in implementation.values():
            _import_callback(reference)
        existing = _REGISTRY.get((category, name))
        if existing is not None:
            if _descriptor(category, existing) != row:
                raise ValueError(
                    f"extension implementation conflicts with saved definition {category}:{name}"
                )
            continue
        register(
            category,
            name,
            implementation["function"],
            initialize=implementation.get("initialize"),
            validate=implementation.get("validate"),
            unit=row.get("unit", "1"),
            state_units=row.get("state_units", {}),
            classification=row.get("classification", False),
            package=package,
            package_version=version,
        )
