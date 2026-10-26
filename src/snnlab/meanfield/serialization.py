"""Inspectable model documents; symbolic strings are data, never evaluated here."""

import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import sympy as sp

from .controls import control_spec
from .symbolic import _latex_equations, latex, summary


def _bounds(values):
    return [float(value) if math.isfinite(value) else None for value in values]


def write_model(model, path):
    """Atomically replace a JSON file after successful strict serialization."""
    states = {
        name: {
            **model.symbol_metadata.get(name, {}),
            "symbol": str(symbol),
            "latex": sp.latex(symbol, symbol_names=dict(model.notation)),
            "unit": model.units[name],
            "bounds": _bounds(model.bounds.get(name, (-math.inf, math.inf))),
        }
        for name, symbol in model.states.items()
    }
    parameters = {
        name: {
            **model.symbol_metadata.get(name, {}),
            "symbol": str(symbol),
            "latex": sp.latex(symbol, symbol_names=dict(model.notation)),
            "unit": model.units[name],
            "value": model.defaults[name],
            "bounds": _bounds(model.parameter_bounds.get(name, (-math.inf, math.inf))),
        }
        for name, symbol in model.parameters.items()
    }
    equation_records = _latex_equations(model)
    display_rhs = dict(zip(model.states, model.display_rhs or model.rhs, strict=True))
    definitions = {
        str(symbol): expression for symbol, expression in model.definitions.items()
    }
    for name, record in equation_records.items():
        expression = display_rhs[name] if name in display_rhs else definitions[name]
        record["dependencies"] = sorted(
            str(symbol) for symbol in expression.free_symbols
        )
    symbols = {
        **{name: {**record, "kind": "state"} for name, record in states.items()},
        **{
            name: {**record, "kind": "parameter"} for name, record in parameters.items()
        },
    }
    for symbol, expression in model.definitions.items():
        name = str(symbol)
        left = sp.latex(symbol, symbol_names=dict(model.notation))
        right = sp.latex(expression, symbol_names=dict(model.notation))
        symbols[name] = {
            **model.symbol_metadata.get(name, {}),
            "kind": "derived",
            "symbol": name,
            "latex": left,
            "definition_latex": left + " = " + right,
            "expression": str(expression),
            "unit": model.units.get(
                name, model.symbol_metadata.get(name, {}).get("unit")
            ),
            "dependencies": sorted(str(item) for item in expression.free_symbols),
        }
    for name in model.functions:
        definition = model.function_definitions.get(name, {})
        symbols[name] = {
            "kind": "function",
            "symbol": name,
            "latex": sp.latex(sp.Symbol(name)),
            "specification": dict(definition),
        }
    for symbol, item in model.delayed.items():
        symbols[str(symbol)] = {
            "kind": "delayed_dependency",
            "source": str(item.source),
            "delay_ms": item.delay_ms,
            "unit": model.units[str(item.source)],
            "latex": sp.latex(symbol, symbol_names=dict(model.notation)),
        }
    document = {
        "schema": "snnlab.meanfield.model/v1",
        "time_unit": "ms",
        "delays": {
            "mode": model.delay_mode,
            "projections_ms": dict(model.projection_delays),
            "dependencies": {
                str(s): {"source": str(d.source), "delay_ms": d.delay_ms}
                for s, d in model.delayed.items()
            },
            "required_history_interval_ms": [
                -max((d.delay_ms for d in model.delayed.values()), default=0),
                0,
            ],
            "history_contract": "History or ConstantHistory; every named state in its declared unit; continuous piecewise smooth history agrees with initial at zero; actual history and solver settings are exported by Trajectory.write",
            "modulation_timing": "target_delivery",
        },
        "controls": {
            name: control_spec(control) for name, control in model.controls.items()
        },
        "state_order": list(model.states),
        "parameter_order": list(model.parameters),
        "states": states,
        "parameters": parameters,
        "symbols": symbols,
        "model_definition": {
            "expression_format": "sympy-str",
            "rhs": {
                name: str(expression)
                for name, expression in zip(model.states, model.rhs, strict=True)
            },
            "display_rhs": {
                name: str(expression)
                for name, expression in zip(
                    model.states, model.display_rhs or model.rhs, strict=True
                )
            },
            "definitions": {
                str(symbol): str(expression)
                for symbol, expression in model.definitions.items()
            },
            "constraints": [str(expression) for expression in model.constraints],
            "reconstructions": {
                name: str(expression)
                for name, expression in model.reconstructions.items()
            },
            "filters": [
                {
                    "state": str(f.state),
                    "source": f.source,
                    "tau": str(f.tau),
                    "coupling": str(f.coupling),
                }
                for f in model.filters
            ],
            "functions": {
                name: dict(
                    model.function_definitions.get(name, {"kind": "runtime_callback"})
                )
                for name in model.functions
            },
        },
        "equation_order": list(equation_records),
        "equations": equation_records,
        "latex": latex(model),
        "assumptions": list(model.assumptions),
        "provenance": dict(model.provenance),
        "summary": summary(model),
    }
    # Validate before creating a directory or touching an existing export.
    text = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
