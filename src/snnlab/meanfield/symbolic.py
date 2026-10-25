"""Symbolic inspection and human-readable rendering."""

import sympy as sp


def equations(model):
    """Map state names to SymPy derivative equations (time measured in ms)."""
    time = sp.Symbol("t", real=True)
    return {
        name: sp.Eq(
            sp.Derivative(symbol, time, evaluate=False), expression, evaluate=False
        )
        for (name, symbol), expression in zip(model.states.items(), model.rhs)
    }


def jacobian(model):
    return sp.Matrix(model.rhs).jacobian(list(model.states.values()))


def _latex_equations(model, *, expanded=False):
    """Named equation parts shared by the TeX and structured JSON exporters."""

    def render(expression):
        return sp.latex(expression, symbol_names=dict(model.notation))

    rhs = model.rhs if expanded else (model.display_rhs or model.rhs)
    rows = {}
    for (name, symbol), expression in zip(model.states.items(), rhs):
        left = rf"\dot{{{render(symbol)}}}"
        tau = (
            model.parameters.get(name.removesuffix(".rate") + ".tau_rate")
            if name.endswith(".rate")
            else None
        )
        if tau is not None and sp.denom(expression) == tau:
            left = render(tau) + r"\," + left
            expression = sp.expand_mul(expression * tau)
        rows[name] = {
            "kind": "differential",
            "lhs_latex": left,
            "rhs_latex": render(expression),
        }

    if not expanded:
        used = set().union(*(expression.free_symbols for expression in rhs))
        definitions = []
        pending = [symbol for symbol in model.definitions if symbol in used]
        while pending:
            symbol = pending.pop(0)
            if symbol in definitions:
                continue
            definitions.append(symbol)
            pending.extend(
                dependency
                for dependency in model.definitions
                if dependency in model.definitions[symbol].free_symbols
                and dependency not in definitions
            )
        # Keep the current/filter/reconstruction relations before constants.
        definitions.sort(key=lambda symbol: not model.definitions[symbol].free_symbols)
        for symbol in definitions:
            if str(symbol) in rows:
                raise ValueError(f"duplicate equation ID: {symbol}")
            rows[str(symbol)] = {
                "kind": "definition",
                "lhs_latex": render(symbol),
                "rhs_latex": render(model.definitions[symbol]),
            }
    for row in rows.values():
        row["latex"] = row["lhs_latex"] + " = " + row["rhs_latex"]
    return rows


def latex(model, *, expanded=False):
    """Render mathematical labels and exact auxiliary definitions.

    ``expanded=True`` inlines all auxiliary expressions. Numerical dynamics and
    ``equations(model)`` always retain the fully expanded computational model.
    """
    rows = _latex_equations(model, expanded=expanded)
    return (
        "\\begin{aligned}\n"
        + " \\\\\n".join(
            row["lhs_latex"] + " &= " + row["rhs_latex"] for row in rows.values()
        )
        + "\n\\end{aligned}"
    )


def summary(model):
    lines = [
        f"{len(model.states)} states; {len(model.parameters)} parameters",
        "States:",
    ]
    lines.extend(
        f"{i}. {name} [{model.units[name]}]" for i, name in enumerate(model.states, 1)
    )
    lines.append("Assumptions:")
    lines.extend(f"{i}. {text}" for i, text in enumerate(model.assumptions, 1))
    if model.reconstructions:
        lines.append("Required initial-state reconstructions:")
        lines.extend(
            f"{i}. {name} = {expression}"
            for i, (name, expression) in enumerate(model.reconstructions.items(), 1)
        )
    lines.append(f"Provenance: {dict(model.provenance)}")
    return "\n".join(lines)
