"""Explicit, non-mutating transformations of symbolic population models."""

from dataclasses import replace


def substitute(model, parameters):
    """Return a model with changed parameter defaults, preserving symbolic names."""
    values = model.values(parameters)
    return replace(model, defaults=dict(zip(model.parameters, values)))


def reduce_states(model, *, method="matched_decay", retain=None):
    """Merge outgoing filters of each source on a proportional-state manifold.

    Initial conditions of removed filters MUST satisfy ``reconstructions``.
    Equality of decay times is checked again on every numerical evaluation.
    """
    if method != "matched_decay":
        raise ValueError("only matched_decay is supported")
    if model.delayed:
        raise ValueError(
            "matched-decay reduction of delayed histories is unsupported; retain distinct pathway filters"
        )
    if model.reconstructions:
        raise ValueError("model has already undergone matched-decay reduction")
    groups = {}
    for filt in model.filters:
        groups.setdefault(filt.source, []).append(filt)
    retain = set(retain or ())
    if retain - {str(f.state) for f in model.filters}:
        raise ValueError("retain contains an unknown recurrent conductance")
    substitutions, constraints = {}, list(model.constraints)
    kept = []
    defaults = {
        symbol: model.defaults[name] for name, symbol in model.parameters.items()
    }
    for filters in groups.values():
        selected = [f for f in filters if str(f.state) in retain]
        if len(selected) > 1:
            raise ValueError("retain at most one filter per source population")
        # Use the first positive pathway when no basis was requested.
        base = (
            selected[0]
            if selected
            else next(
                (f for f in filters if float(f.coupling.subs(defaults)) > 0), filters[0]
            )
        )
        if float(base.coupling.subs(defaults)) <= 0:
            raise ValueError("matched-decay basis requires positive coupling")
        kept.append(base)
        for filt in filters:
            if filt == base:
                continue
            if abs(float((filt.tau - base.tau).subs(defaults))) > 1e-12:
                raise ValueError("outgoing filters must have matching decay times")
            constraints.append(filt.tau - base.tau)
            substitutions[filt.state] = filt.coupling / base.coupling * base.state
    states = {
        name: symbol
        for name, symbol in model.states.items()
        if symbol not in substitutions
    }
    rhs = tuple(
        expression.subs(substitutions, simultaneous=True)
        for symbol, expression in zip(model.states.values(), model.rhs)
        if symbol not in substitutions
    )
    reconstruction = {
        str(symbol): expression for symbol, expression in substitutions.items()
    }
    display_rhs = tuple(
        expression
        for symbol, expression in zip(
            model.states.values(), model.display_rhs or model.rhs
        )
        if symbol not in substitutions
    )
    assumptions = model.assumptions
    if substitutions:
        assumptions += (
            "Matched-decay reduction: removed initial conductances must equal the reconstruction expressions; time-varying reconstruction ratios are unsupported",
        )
    return replace(
        model,
        states=states,
        rhs=rhs,
        filters=tuple(kept),
        constraints=tuple(constraints),
        reconstructions=reconstruction,
        assumptions=assumptions,
        bounds={name: model.bounds[name] for name in states},
        display_rhs=display_rhs,
        definitions={**model.definitions, **substitutions},
    )
