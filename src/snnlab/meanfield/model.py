"""Symbolic population models, with ms, Hz, uS, mV and nA conventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
import sympy as sp

from snnlab.lang.units import Quantity


def magnitude(value, unit):
    if isinstance(value, Quantity):
        if value.unit != unit:
            raise ValueError(f"expected {unit}, received {value.unit}")
        value = value.value
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("values must be finite")
    return value


@dataclass(frozen=True)
class Filter:
    state: sp.Symbol
    source: str
    tau: sp.Symbol
    coupling: sp.Expr


@dataclass(frozen=True)
class Delayed:
    source: sp.Symbol
    delay_ms: float

    def __post_init__(self):
        delay = magnitude(self.delay_ms, "ms")
        if delay <= 0:
            raise ValueError("delayed dependencies require a positive delay")
        object.__setattr__(self, "delay_ms", delay)


@dataclass(frozen=True)
class Model:
    """Named equations and explicit approximation metadata; use module operations."""

    states: Mapping[str, sp.Symbol]
    rhs: tuple[sp.Expr, ...]
    parameters: Mapping[str, sp.Symbol]
    defaults: Mapping[str, float]
    units: Mapping[str, str]
    functions: Mapping[str, Callable] = field(default_factory=dict, repr=False)
    assumptions: tuple[str, ...] = ()
    provenance: Mapping[str, str] = field(default_factory=dict)
    filters: tuple[Filter, ...] = ()
    reconstructions: Mapping[str, sp.Expr] = field(default_factory=dict)
    constraints: tuple[sp.Expr, ...] = ()
    bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    parameter_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    notation: Mapping[sp.Symbol, str] = field(default_factory=dict)
    definitions: Mapping[sp.Symbol, sp.Expr] = field(default_factory=dict)
    display_rhs: tuple[sp.Expr, ...] = ()
    function_definitions: Mapping[str, dict] = field(default_factory=dict)
    symbol_metadata: Mapping[str, dict] = field(default_factory=dict)
    delayed: Mapping[sp.Symbol, Delayed] = field(default_factory=dict)
    projection_delays: Mapping[str, float] = field(default_factory=dict)
    controls: Mapping[str, Callable] = field(default_factory=dict)
    delay_mode: str = "reject"

    def __post_init__(self):
        for name in (
            "states",
            "parameters",
            "defaults",
            "units",
            "functions",
            "provenance",
            "reconstructions",
            "bounds",
            "parameter_bounds",
            "notation",
            "definitions",
            "function_definitions",
            "symbol_metadata",
            "delayed",
            "projection_delays",
            "controls",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        if not self.states or len(self.states) != len(self.rhs):
            raise ValueError("each state requires exactly one derivative")
        if self.display_rhs and len(self.display_rhs) != len(self.rhs):
            raise ValueError("display equations must match the state count")
        if set(self.parameters) != set(self.defaults):
            raise ValueError("each parameter requires a default")
        known = set(self.states.values()) | set(self.parameters.values())
        if any(item.source not in known for item in self.delayed.values()):
            raise ValueError("delayed source must be a declared state or parameter")
        if known.intersection(self.delayed):
            raise ValueError(
                "delayed symbols must be distinct from states and parameters"
            )
        known |= set(self.delayed)
        if set(self.states) & set(self.parameters):
            raise ValueError("state and parameter names must be distinct")
        if any(expression.free_symbols - known for expression in self.rhs):
            raise ValueError("equations contain undeclared symbols")
        if (set(self.states) | set(self.parameters)) - set(self.units):
            raise ValueError("every state and parameter requires a declared unit")
        self.values()

    def write(self, path):
        """Write an inspectable JSON model definition and individually named equations.

        The document includes parameters, units, assumptions and mathematical
        presentation. Python callbacks remain runtime implementations.
        """
        from .serialization import write_model

        return write_model(self, path)

    def values(self, overrides=None):
        values = dict(self.defaults)
        for name, value in (overrides or {}).items():
            if name not in self.parameters:
                raise ValueError(f"unknown parameter: {name}")
            values[name] = magnitude(value, self.units[name])
        for name, value in values.items():
            lower, upper = self.parameter_bounds.get(name, (-np.inf, np.inf))
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"parameter outside its bounds: {name}")
        substitutions = {
            symbol: values[name] for name, symbol in self.parameters.items()
        }
        for constraint in self.constraints:
            if abs(float(constraint.subs(substitutions))) > 1e-12:
                raise ValueError("matched-decay parameter constraint violated")
        for expression in self.reconstructions.values():
            denominator = sp.denom(expression)
            if float(denominator.subs(substitutions)) == 0:
                raise ValueError("matched-decay basis coupling must remain nonzero")
        return tuple(values[name] for name in self.parameters)

    @cached_property
    def numerical_rhs(self):
        return sp.lambdify(
            (*self.states.values(), *self.parameters.values(), *self.delayed),
            self.rhs,
            modules=[dict(self.functions), "numpy"],
            cse=True,
        )

    @cached_property
    def numerical_jacobian(self):
        self.require_ode_analysis()
        return sp.lambdify(
            (*self.states.values(), *self.parameters.values()),
            sp.Matrix(self.rhs).jacobian(list(self.states.values())),
            modules=[dict(self.functions), "numpy"],
            cse=True,
        )

    def require_ode_analysis(self):
        if self.delayed:
            raise ValueError(
                "delayed-system stability requires characteristic-root analysis; ODE Jacobians, continuation stability and stability crossings are unsupported"
            )

    @cached_property
    def stationary_rhs(self):
        return tuple(
            expr.xreplace(
                {symbol: item.source for symbol, item in self.delayed.items()}
            )
            for expr in self.rhs
        )

    @cached_property
    def numerical_stationary_rhs(self):
        return sp.lambdify(
            (*self.states.values(), *self.parameters.values()),
            self.stationary_rhs,
            modules=[dict(self.functions), "numpy"],
            cse=True,
        )

    @cached_property
    def numerical_stationary_jacobian(self):
        return sp.lambdify(
            (*self.states.values(), *self.parameters.values()),
            sp.Matrix(self.stationary_rhs).jacobian(list(self.states.values())),
            modules=[dict(self.functions), "numpy"],
            cse=True,
        )

    def vector(self, state):
        if isinstance(state, Mapping):
            if set(state) != set(self.states):
                raise ValueError(
                    "initial state must name every retained state exactly once"
                )
            state = [magnitude(state[name], self.units[name]) for name in self.states]
        result = np.asarray(state, dtype=float)
        if result.shape != (len(self.states),) or not np.isfinite(result).all():
            raise ValueError("state has incorrect shape or nonfinite values")
        return result
