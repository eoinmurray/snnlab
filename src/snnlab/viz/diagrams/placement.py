"""Measured box placement with explicit required and weighted constraints."""

from __future__ import annotations

import itertools
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .geometry import Box
from .spec import GroupLayout, NodeLayout, merged


def overlaps(a, b, tolerance=1e-5):
    return (
        a.x < b.right - tolerance
        and a.right > b.x + tolerance
        and a.y < b.bottom - tolerance
        and a.bottom > b.y + tolerance
    )


def group_specs(diagram, spec):
    groups = {
        g.id: merged(GroupLayout(members=g.members, label=g.label), spec.defaults.group)
        for g in diagram.groups
    }
    if len(groups) != len(diagram.groups):
        raise ValueError("diagram group IDs must be unique")
    for identifier, override in spec.groups.items():
        if identifier not in groups and override.members is None:
            raise ValueError(f"new visual group {identifier!r} needs members")
        groups[identifier] = merged(
            spec.defaults.group, groups.get(identifier, GroupLayout()), override
        )
    groups = {k: merged(spec.defaults.group, v) for k, v in groups.items()}
    known = {n.id for n in diagram.nodes}
    if known & groups.keys():
        raise ValueError("node and group IDs must be distinct")
    for identifier, group in groups.items():
        if not group.members:
            raise ValueError(f"group {identifier!r} has no members")
        if len(set(group.members)) != len(group.members) or set(group.members) - known:
            raise ValueError(
                f"group {identifier!r} has duplicate or unknown node members"
            )
        if group.parent is not None and group.parent not in groups:
            raise ValueError(f"unknown parent group {group.parent!r}")
        visited = {identifier}
        parent = group.parent
        while parent is not None:
            if parent in visited:
                raise ValueError(f"cyclic group nesting at {parent!r}")
            visited.add(parent)
            parent = groups[parent].parent
        if group.parent and not set(group.members) <= set(groups[group.parent].members):
            raise ValueError(
                f"group {identifier!r} members must be included in its parent"
            )
        if group.node_defaults and (
            group.node_defaults.position is not None
            or group.node_defaults.relative_to is not None
        ):
            raise ValueError("group node_defaults cannot specify positions")
    for a, b in itertools.combinations(groups, 2):
        shared = set(groups[a].members) & set(groups[b].members)
        if shared and a not in ancestors(b, groups) and b not in ancestors(a, groups):
            raise ValueError(
                f"overlapping groups {a!r} and {b!r} need explicit parent nesting"
            )
    return groups


def ancestors(identifier, groups):
    result = []
    parent = groups[identifier].parent
    while parent is not None:
        result.append(parent)
        parent = groups[parent].parent
    return result


def node_specs(diagram, spec, groups):
    result = {}
    for node in diagram.nodes:
        memberships = sorted(
            (k for k, g in groups.items() if node.id in g.members),
            key=lambda k: len(ancestors(k, groups)),
        )
        settings = [spec.defaults.node]
        settings.extend(
            groups[k].node_defaults
            for k in memberships
            if groups[k].node_defaults is not None
        )
        settings.append(spec.nodes.get(node.id, NodeLayout()))
        value = merged(*settings)
        if node.id in spec.positions:
            if value.relative_to is not None or value.anchor not in (None, "center"):
                raise ValueError(
                    "positions shorthand pins centres; use nodes[id].position for relative/corner pins"
                )
            if value.position is not None and tuple(value.position) != tuple(
                spec.positions[node.id]
            ):
                raise ValueError(f"conflicting positions for node {node.id!r}")
            from dataclasses import replace

            value = replace(value, position=spec.positions[node.id])
        if value.relative_to is not None and (
            value.relative_to not in groups
            or node.id not in groups[value.relative_to].members
        ):
            raise ValueError(
                f"node {node.id!r} relative_to must name a containing group"
            )
        result[node.id] = value
    return result


def _plus(*terms):
    result = {}
    for expression, factor in terms:
        for index, coefficient in expression.items():
            result[index] = result.get(index, 0) + coefficient * factor
    return {k: v for k, v in result.items() if v}


class _Problem:
    def __init__(self, bound):
        self.bound = bound
        self.cost, self.lower, self.upper, self.integer = [], [], [], []
        self.rows, self.labels = [], []
        self.preferences = []

    def variable(self, cost=0, *, low=None, high=None, integer=0):
        index = len(self.cost)
        self.cost.append(cost)
        self.lower.append(-self.bound if low is None else low)
        self.upper.append(self.bound if high is None else high)
        self.integer.append(integer)
        return {index: 1}

    def constraint(
        self, expr, low=-math.inf, high=math.inf, name="layout constraint", weight=None
    ):
        original = dict(expr)
        if weight is not None:
            slack = self.variable(weight, low=0)
            if math.isfinite(high):
                self.rows.append((_plus((expr, 1), (slack, -1)), -math.inf, high))
            if math.isfinite(low):
                self.rows.append((_plus((expr, 1), (slack, 1)), low, math.inf))
            self.preferences.append((name, original, low, high))
        else:
            self.rows.append((expr, low, high))
            self.labels.append(name)

    def preference(self, expr, target, weight=1):
        self.constraint(expr, target, target, weight=weight)

    def solve(self):
        ri, ci, data = [], [], []
        for row_index, (expr, _, _) in enumerate(self.rows):
            for column, value in expr.items():
                ri.append(row_index)
                ci.append(column)
                data.append(value)
        matrix = coo_matrix(
            (data, (ri, ci)), shape=(len(self.rows), len(self.cost))
        ).tocsc()
        matrix.indices = matrix.indices.astype(np.int32)
        matrix.indptr = matrix.indptr.astype(np.int32)
        result = self._optimize(matrix)
        if not result.success:
            detail = "; ".join(dict.fromkeys(self.labels))
            raise ValueError(
                f"layout could not satisfy required constraints ({result.message}): {detail}"
            )
        diagnostics = []
        for name, expr, low, high in self.preferences:
            value = sum(result.x[k] * v for k, v in expr.items())
            violation = max(low - value, value - high, 0)
            if violation > 1e-5 and name != "layout constraint":
                diagnostics.append(
                    {
                        "kind": "unmet_preference",
                        "constraint": name,
                        "violation": float(violation),
                    }
                )
        return result.x, diagnostics

    def _optimize(self, matrix):
        started = time.monotonic()
        constraints = LinearConstraint(
            matrix, [r[1] for r in self.rows], [r[2] for r in self.rows]
        )
        arguments = dict(
            c=np.array(self.cost),
            integrality=np.array(self.integer),
            bounds=Bounds(self.lower, self.upper),
            constraints=constraints,
        )
        result = milp(**arguments, options={"time_limit": 30})
        if result.status == 4:
            # HiGHS can reject its own presolved solution at the feasibility
            # tolerance. Retry the original model, without changing constraints.
            result = milp(
                **arguments,
                options={
                    "time_limit": max(0.01, 30 - (time.monotonic() - started)),
                    "presolve": False,
                },
            )
        if result.success:
            lower, upper = np.array(self.lower), np.array(self.upper)
            for index, integral in enumerate(self.integer):
                if integral:
                    lower[index] = upper[index] = round(result.x[index])
            # Fix the discrete separation choices and solve the continuous
            # geometry again. This removes big-M/integer-tolerance drift.
            arguments.update(
                integrality=np.zeros(len(self.integer)), bounds=Bounds(lower, upper)
            )
            result = milp(
                **arguments,
                options={
                    "time_limit": max(0.01, 30 - (time.monotonic() - started)),
                    "presolve": False,
                },
            )
        return result


class _Entity:
    def __init__(self, problem, size=None):
        self.l, self.t, self.r, self.b = (problem.variable() for _ in range(4))
        self.cx = _plus((self.l, 0.5), (self.r, 0.5))
        self.cy = _plus((self.t, 0.5), (self.b, 0.5))
        self.w = _plus((self.r, 1), (self.l, -1))
        self.h = _plus((self.b, 1), (self.t, -1))
        if size:
            problem.constraint(self.w, size[0], size[0])
            problem.constraint(self.h, size[1], size[1])
        else:
            problem.constraint(self.w, 1)
            problem.constraint(self.h, 1)
            for expression in (self.w, self.h):
                for k, v in expression.items():
                    problem.cost[k] += 0.01 * v


def _separate(problem, a, b, gap, name):
    directions = [(a.r, b.l), (b.r, a.l), (a.b, b.t), (b.b, a.t)]
    choices = []
    for end, start in directions:
        choice = problem.variable(low=0, high=1, integer=1)
        choices.append(choice)
        problem.constraint(
            _plus((end, 1), (start, -1), (choice, 4 * problem.bound)),
            high=4 * problem.bound - gap,
            name=name,
        )
    problem.constraint(_plus(*[(c, 1) for c in choices]), low=1, name=name)


def _apply_constraint(problem, entities, constraint, default_gap):
    c = constraint
    es = [entities[n] for n in c.nodes]
    name = c.name or f"{c.kind}({', '.join(c.nodes)})"
    weight = None if c.required else c.priority
    gap = default_gap if c.gap is None else c.gap
    if c.kind == "non_overlap":
        if not c.required:
            raise ValueError(
                "non_overlap is a required constraint; soft disjunctions are unsupported"
            )
        for a, b in itertools.combinations(es, 2):
            _separate(problem, a, b, gap, name)
    elif c.kind.startswith("align_"):
        axis = "cx" if c.kind.endswith("x") else "cy"
        for e in es[1:]:
            problem.constraint(
                _plus((getattr(es[0], axis), 1), (getattr(e, axis), -1)),
                0,
                0,
                name,
                weight,
            )
    elif c.kind.startswith("equal_spacing"):
        axis = "cx" if c.kind.endswith("x") else "cy"
        for a, b, d in zip(es, es[1:], es[2:], strict=False):
            problem.constraint(
                _plus(
                    (getattr(a, axis), 1), (getattr(b, axis), -2), (getattr(d, axis), 1)
                ),
                0,
                0,
                name,
                weight,
            )
    else:
        horizontal = c.kind in {"left_of", "right_of", "order_x"}
        if c.kind in {"below", "right_of"}:
            es.reverse()
        end, start = ("r", "l") if horizontal else ("b", "t")
        for a, b in itertools.pairwise(es):
            problem.constraint(
                _plus((getattr(b, start), 1), (getattr(a, end), -1)),
                low=gap,
                name=name,
                weight=weight,
            )


def _group_arrangement(problem, entities, identifier, group, groups, gap):
    children = [k for k, g in groups.items() if g.parent == identifier]
    nested = {n for k in children for n in groups[k].members}
    children.extend(n for n in group.members if n not in nested)
    order = list(group.order or children)
    if set(order) != set(children) or len(order) != len(children):
        raise ValueError(
            f"group {identifier!r} order must include each immediate child exactly once"
        )
    mode = group.arrangement or "auto"
    if mode == "auto":
        if group.order is not None or group.columns is not None:
            raise ValueError(
                f"group {identifier!r}: order/columns require row, column or grid arrangement"
            )
        return
    if group.columns is not None and mode != "grid":
        raise ValueError("columns requires grid arrangement")
    from .spec import LayoutConstraint

    spacing = gap if group.gap is None else group.gap
    alignment = group.alignment or "center"
    axis = "cy" if mode == "row" else "cx"
    start, end = ("t", "b") if mode == "row" else ("l", "r")
    if alignment != "center":
        axis = start if alignment == "start" else end
    columns = group.columns or max(1, math.ceil(math.sqrt(len(order))))
    rows = (
        [order]
        if mode != "grid"
        else [order[i : i + columns] for i in range(0, len(order), columns)]
    )
    for row in rows:
        if len(row) > 1:
            _apply_constraint(
                problem,
                entities,
                LayoutConstraint(
                    "order_x" if mode in {"row", "grid"} else "order_y",
                    tuple(row),
                    gap=spacing,
                    name=f"group {identifier} {mode}",
                ),
                gap,
            )
            for a, b in itertools.pairwise(row):
                coordinate = (
                    {"start": "t", "center": "cy", "end": "b"}[alignment]
                    if mode == "grid"
                    else axis
                )
                problem.constraint(
                    _plus(
                        (getattr(entities[a], coordinate), 1),
                        (getattr(entities[b], coordinate), -1),
                    ),
                    0,
                    0,
                    f"group {identifier} alignment",
                )
    if mode == "grid":
        for upper, lower in itertools.pairwise(rows):
            for a in upper:
                for b in lower:
                    problem.constraint(
                        _plus((entities[b].t, 1), (entities[a].b, -1)),
                        low=spacing,
                        name=f"group {identifier} grid rows",
                    )
            for a, b in zip(upper, lower, strict=False):
                problem.constraint(
                    _plus(
                        (
                            getattr(
                                entities[a],
                                {"start": "l", "center": "cx", "end": "r"}[alignment],
                            ),
                            1,
                        ),
                        (
                            getattr(
                                entities[b],
                                {"start": "l", "center": "cx", "end": "r"}[alignment],
                            ),
                            -1,
                        ),
                    ),
                    0,
                    0,
                    f"group {identifier} grid columns",
                )


def place(diagram, spec, groups, nodes, sizes, initial, title_sizes):
    gap = spec.defaults.gap if spec.gap is None else spec.gap
    coordinates = [v for p in initial.values() for v in p]
    coordinates.extend(v for n in nodes.values() if n.position for v in n.position)
    coordinates.extend(v for g in groups.values() if g.position for v in g.position)
    for group in groups.values():
        coordinates.extend(group.min_size or ())
        coordinates.extend(group.max_size or ())
        coordinates.extend(v for v in (group.gap, group.padding) if v is not None)
    coordinates.extend(c.gap for c in spec.all_constraints() if c.gap is not None)
    bound = (
        max(
            10000,
            sum(sum(s) for s in sizes.values())
            + max(map(abs, coordinates), default=0) * (len(sizes) + len(groups) + 1)
            + gap * (len(sizes) + len(groups) + 1),
        )
        * 8
    )
    problem = _Problem(bound)
    entities = {k: _Entity(problem, size) for k, size in sizes.items()}
    entities.update({k: _Entity(problem) for k in groups})
    for identifier, e in entities.items():
        if identifier in initial:
            for expr, target in zip((e.cx, e.cy), initial[identifier], strict=True):
                problem.preference(expr, target)
    _pin_nodes(problem, entities, nodes)
    _place_groups(problem, entities, groups, title_sizes, gap)
    constraints = spec.all_constraints()
    for constraint in constraints:
        if unknown := set(constraint.nodes) - entities.keys():
            raise ValueError(f"unknown IDs in {constraint.kind}: {sorted(unknown)}")
        if constraint.kind in {
            "non_overlap",
            "above",
            "below",
            "left_of",
            "right_of",
            "order_x",
            "order_y",
        }:
            for a, b in itertools.combinations(constraint.nodes, 2):
                if (
                    a in groups
                    and (
                        b in groups[a].members
                        or b in groups
                        and a in ancestors(b, groups)
                    )
                    or b in groups
                    and (
                        a in groups[b].members
                        or a in groups
                        and b in ancestors(a, groups)
                    )
                ):
                    raise ValueError(
                        f"cannot separate a group from its contents: {a}, {b}"
                    )
        _apply_constraint(problem, entities, constraint, gap)
    if spec.non_overlap:
        _non_overlap(problem, entities, sizes, nodes, groups, gap)
    solution, diagnostics = problem.solve()

    def value(expression):
        return float(sum(solution[k] * v for k, v in expression.items()))

    boxes = {
        k: Box(value(e.l), value(e.t), value(e.w), value(e.h))
        for k, e in entities.items()
    }
    return boxes, diagnostics


def _pin_nodes(problem, entities, nodes):
    for identifier, settings in nodes.items():
        if settings.position is None:
            continue
        e = entities[identifier]
        anchor = settings.anchor or "center"
        xexpr = e.l if "left" in anchor else e.r if "right" in anchor else e.cx
        yexpr = e.t if "top" in anchor else e.b if "bottom" in anchor else e.cy
        for expr, coordinate, axis in zip(
            (xexpr, yexpr), settings.position, ("l", "t"), strict=True
        ):
            if settings.relative_to:
                expr = _plus(
                    (expr, 1), (getattr(entities[settings.relative_to], axis), -1)
                )
            problem.constraint(
                expr, coordinate, coordinate, f"pin {identifier} {anchor}"
            )


def _place_groups(problem, entities, groups, title_sizes, gap):
    for identifier, group in groups.items():
        e = entities[identifier]
        padding = 24 if group.padding is None else group.padding
        title_height = title_sizes[identifier][1]
        for member in group.members:
            child = entities[member]
            for inner, outer, amount in (
                (child.l, e.l, padding),
                (
                    child.t,
                    e.t,
                    padding
                    + (title_height + 12 if group.title_position != "bottom" else 0),
                ),
                (e.r, child.r, padding),
                (
                    e.b,
                    child.b,
                    padding
                    + (title_height + 12 if group.title_position == "bottom" else 0),
                ),
            ):
                problem.constraint(
                    _plus((inner, 1), (outer, -1)),
                    low=amount,
                    name=f"contain {member} in {identifier}",
                )
        problem.constraint(
            e.w,
            low=title_sizes[identifier][0] + padding * 2,
            name=f"group {identifier} title width",
        )
        for size, lower in ((group.min_size, True), (group.max_size, False)):
            if size:
                for expression, amount in zip((e.w, e.h), size, strict=True):
                    problem.constraint(
                        expression,
                        low=amount if lower else -math.inf,
                        high=math.inf if lower else amount,
                        name=f"group {identifier} size",
                    )
        if group.parent:
            parent = entities[group.parent]
            parent_spec = groups[group.parent]
            pad = 24 if parent_spec.padding is None else parent_spec.padding
            title_h = title_sizes[group.parent][1] + 12
            for inner, outer, amount in (
                (e.l, parent.l, pad),
                (
                    e.t,
                    parent.t,
                    pad + (title_h if parent_spec.title_position != "bottom" else 0),
                ),
                (parent.r, e.r, pad),
                (
                    parent.b,
                    e.b,
                    pad + (title_h if parent_spec.title_position == "bottom" else 0),
                ),
            ):
                problem.constraint(
                    _plus((inner, 1), (outer, -1)),
                    low=amount,
                    name=f"nested group {identifier}",
                )
        if group.position:
            for expr, coordinate in zip((e.cx, e.cy), group.position, strict=True):
                problem.constraint(
                    expr, coordinate, coordinate, f"pin group {identifier}"
                )
        _group_arrangement(problem, entities, identifier, group, groups, gap)


def _non_overlap(problem, entities, sizes, nodes, groups, gap):
    for a, b in itertools.combinations(sizes, 2):
        shared = sorted(
            (k for k, g in groups.items() if a in g.members and b in g.members),
            key=lambda k: len(ancestors(k, groups)),
            reverse=True,
        )
        pair_gap = next(
            (groups[k].gap for k in shared if groups[k].gap is not None), gap
        )
        clearance = max(pair_gap, nodes[a].clearance or 0, nodes[b].clearance or 0)
        _separate(problem, entities[a], entities[b], clearance, f"non-overlap {a}, {b}")
    for group_id, group in groups.items():
        for node_id in sizes:
            if node_id not in group.members:
                _separate(
                    problem,
                    entities[group_id],
                    entities[node_id],
                    0,
                    f"non-overlap group {group_id}, {node_id}",
                )
    for a, b in itertools.combinations(groups, 2):
        if a not in ancestors(b, groups) and b not in ancestors(a, groups):
            _separate(
                problem, entities[a], entities[b], 0, f"non-overlap groups {a}, {b}"
            )
