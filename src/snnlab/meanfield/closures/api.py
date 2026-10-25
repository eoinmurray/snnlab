"""Public symbolic construction contract shared by built-in and custom closures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import sympy as sp

from snnlab.meanfield.model import Model, magnitude


class Closure(Protocol):
    def build(self, ctx: BuildContext) -> Model:
        """Construct a model using the resolved graph, inputs and weight summaries."""
        ...


@dataclass
class BuildContext:
    """Per-reduction builder. Names and declared units survive into the model.

    Numeric equations use ms, Hz, uS, mV and nA. Units are metadata; custom
    closures are responsible for dimensional correctness of their expressions.
    """

    graph: dict
    weights: dict
    inputs: dict
    applied_current: dict
    provenance: dict
    external_filters: str
    delays: str
    states: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    defaults: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    functions: dict = field(default_factory=dict)
    derivatives: dict = field(default_factory=dict)
    bounds: dict = field(default_factory=dict)
    parameter_bounds: dict = field(default_factory=dict)
    assumptions: list = field(default_factory=list)
    filters: list = field(default_factory=list)
    notation: dict = field(default_factory=dict)
    definitions: dict = field(default_factory=dict)
    display_derivatives: dict = field(default_factory=dict)
    function_definitions: dict = field(default_factory=dict)
    symbol_metadata: dict = field(default_factory=dict)

    @property
    def populations(self):
        return tuple(self.graph["populations"])

    @property
    def projections(self):
        return tuple(p for p in self.graph["projections"] if p.get("enabled", True))

    def _symbol(self, name, unit):
        if name in self.units:
            raise ValueError(f"duplicate symbol: {name}")
        self.units[name] = unit
        return sp.Symbol(name, real=True)

    def state(self, name, *, unit, bounds=(-float("inf"), float("inf"))):
        symbol = self._symbol(name, unit)
        self.states[name] = symbol
        self.bounds[name] = bounds
        return symbol

    def parameter(self, name, default, *, unit, bounds=(-float("inf"), float("inf"))):
        symbol = self._symbol(name, unit)
        self.parameters[name] = symbol
        self.defaults[name] = magnitude(default, unit)
        self.parameter_bounds[name] = bounds
        return symbol

    def function(self, name, function, *, derivative, definition=None):
        """Bind a scalar function and analytic first derivative for lambdify."""
        slope = sp.Function(f"{name}_slope")

        def fdiff(self, argindex=1):
            if argindex != 1:
                raise ValueError("scalar functions take one argument")
            return slope(self.args[0])

        symbolic = type(name, (sp.Function,), {"nargs": 1, "fdiff": fdiff})
        self.functions[name] = function
        self.functions[f"{name}_slope"] = derivative
        self.function_definitions[name] = (
            {
                "kind": "custom",
                "python_module": getattr(function, "__module__", None),
                "python_name": getattr(function, "__qualname__", None),
            }
            if definition is None
            else dict(definition)
        )
        self.function_definitions[f"{name}_slope"] = {
            "kind": "derivative",
            "of": name,
            "argument": 1,
        }
        return symbolic

    def derivative(self, state, expression):
        if state not in self.states.values() or state in self.derivatives:
            raise ValueError("derivative must be assigned once to a declared state")
        self.derivatives[state] = sp.sympify(expression)

    def describe(self, symbol, *, description, **metadata):
        """Record a symbol's meaning and optional inspectable shorthand definition."""
        if symbol not in {
            *self.states.values(),
            *self.parameters.values(),
            *self.definitions,
        }:
            raise ValueError(
                "describe requires a declared state, parameter or definition"
            )
        self.symbol_metadata[str(symbol)] = {"description": description, **metadata}

    def display_definition(
        self, name, expression, *, latex, description=None, unit=None
    ):
        """Name an exact expression for presentation, without changing dynamics."""
        symbol = sp.Symbol(name, real=True)
        if symbol in self.definitions or name in self.units:
            raise ValueError(f"duplicate display definition: {name}")
        self.definitions[symbol] = sp.sympify(expression)
        self.notation[symbol] = latex
        if description is not None or unit is not None:
            self.describe(
                symbol,
                description=description or "Derived model quantity",
                **({"unit": unit} if unit is not None else {}),
            )
        return symbol

    def build(self):
        if set(self.states.values()) != set(self.derivatives):
            raise ValueError("every state requires a derivative")
        return Model(
            self.states,
            tuple(self.derivatives[s] for s in self.states.values()),
            self.parameters,
            self.defaults,
            self.units,
            self.functions,
            tuple(self.assumptions),
            self.provenance,
            tuple(self.filters),
            bounds=self.bounds,
            parameter_bounds=self.parameter_bounds,
            notation=self.notation,
            definitions=self.definitions,
            display_rhs=tuple(
                self.display_derivatives.get(s, self.derivatives[s])
                for s in self.states.values()
            ),
            function_definitions=self.function_definitions,
            symbol_metadata=self.symbol_metadata,
        )
