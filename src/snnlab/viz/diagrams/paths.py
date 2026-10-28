"""Obstacle-aware routes in the same logical coordinate system as node boxes."""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import replace

from .geometry import Box

SIDES = {"left": (0, 0.5), "right": (1, 0.5), "top": (0.5, 0), "bottom": (0.5, 1)}
VECTORS = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}


def inside(point, box):
    return (
        box.x + 1e-6 < point[0] < box.right - 1e-6
        and box.y + 1e-6 < point[1] < box.bottom - 1e-6
    )


def crosses(a, b, box):
    """Open segment/rectangle intersection; running along an edge is permitted."""
    if inside(a, box) or inside(b, box):
        return True
    lower, upper = 0.0, 1.0
    for p, delta, low, high in (
        (a[0], b[0] - a[0], box.x + 1e-6, box.right - 1e-6),
        (a[1], b[1] - a[1], box.y + 1e-6, box.bottom - 1e-6),
    ):
        if abs(delta) < 1e-10:
            if not low < p < high:
                return False
        else:
            t0, t1 = sorted(((low - p) / delta, (high - p) / delta))
            lower, upper = max(lower, t0), min(upper, t1)
            if lower >= upper:
                return False
    return lower < upper


def _grid_path(start, end, obstacles):
    if start == end:
        return [start]
    if any(inside(start, box) or inside(end, box) for box in obstacles):
        raise ValueError(
            f"route endpoint or waypoint is inside an obstacle: {start}, {end}"
        )
    xs = sorted({start[0], end[0], *(x for b in obstacles for x in (b.x, b.right))})
    ys = sorted({start[1], end[1], *(y for b in obstacles for y in (b.y, b.bottom))})
    start_index = (xs.index(start[0]), ys.index(start[1]))
    end_index = (xs.index(end[0]), ys.index(end[1]))
    queue = [(0.0, *start_index, -1)]
    distance = {(*start_index, -1): 0.0}
    previous = {}
    while queue:
        cost, xi, yi, direction = heapq.heappop(queue)
        state = (xi, yi, direction)
        if cost > distance[state] + 1e-6:
            continue
        if (xi, yi) == end_index:
            result = []
            while state is not None:
                result.append((xs[state[0]], ys[state[1]]))
                state = previous.get(state)
            return list(reversed(result))
        a = (xs[xi], ys[yi])
        for nx, ny, nd in (
            (xi - 1, yi, 0),
            (xi + 1, yi, 0),
            (xi, yi - 1, 1),
            (xi, yi + 1, 1),
        ):
            if not (0 <= nx < len(xs) and 0 <= ny < len(ys)):
                continue
            b = (xs[nx], ys[ny])
            if any(crosses(a, b, box) for box in obstacles):
                continue
            new_cost = (
                cost
                + abs(a[0] - b[0])
                + abs(a[1] - b[1])
                + (12 if direction not in {-1, nd} else 0)
            )
            new_state = nx, ny, nd
            if new_cost < distance.get(new_state, math.inf) - 1e-6:
                distance[new_state] = new_cost
                previous[new_state] = (xi, yi, direction)
                heapq.heappush(queue, (new_cost, *new_state))
    raise ValueError(f"no obstacle-free route from {start} to {end}")


def simplify(points, protected=()):
    result = []
    for p in points:
        p = tuple(p)
        if result and result[-1] == p:
            continue
        while len(result) >= 2 and result[-1] not in protected:
            a, b = result[-2:]
            if (
                abs((b[0] - a[0]) * (p[1] - b[1]) - (b[1] - a[1]) * (p[0] - b[0]))
                > 1e-6
            ):
                break
            if (b[0] - a[0]) * (p[0] - b[0]) + (b[1] - a[1]) * (p[1] - b[1]) < 0:
                break
            result.pop()
        result.append(p)
    return result


def _port(box, name, ports, other, shift):
    if name in (None, "auto"):
        dx, dy = other.center[0] - box.center[0], other.center[1] - box.center[1]
        name = (
            ("right" if dx >= 0 else "left")
            if abs(dx) >= abs(dy)
            else ("bottom" if dy >= 0 else "top")
        )
    if name in SIDES:
        u, v = SIDES[name]
        if name in {"left", "right"}:
            v += shift / box.height
        else:
            u += shift / box.width
        if not 0.05 <= (v if name in {"left", "right"} else u) <= 0.95:
            raise ValueError(
                "parallel connections need more space on their attachment side"
            )
        side = name
    else:
        if name not in (ports or {}):
            raise ValueError(f"unknown node port {name!r}")
        u, v = ports[name]
        side = (
            "left" if u == 0 else "right" if u == 1 else "top" if v == 0 else "bottom"
        )
    return (box.x + u * box.width, box.y + v * box.height), side


def _route(edge, settings, boxes, nodes, *, shift=0, extra_obstacles=()):
    source, target = boxes[edge.source], boxes[edge.target]
    source_port, target_port = settings.source_port, settings.target_port
    if edge.source == edge.target:
        source_port, target_port = source_port or "right", target_port or "top"
    a, a_side = _port(source, source_port, nodes[edge.source].ports, target, shift)
    b, b_side = _port(target, target_port, nodes[edge.target].ports, source, shift)
    clearance = 12 if settings.clearance is None else settings.clearance
    if settings.routing == "curved":
        clearance = max(clearance, 6)
    # Arrowheads and corner rounding stay inside this clearance corridor.
    lead = (
        max(
            clearance,
            nodes[edge.source].clearance or 0,
            nodes[edge.target].clearance or 0,
            12,
        )
        + abs(shift) * 0.25
    )
    start = tuple(c + d * lead for c, d in zip(a, VECTORS[a_side], strict=True))
    end = tuple(c + d * lead for c, d in zip(b, VECTORS[b_side], strict=True))
    obstacles = [
        boxes[k].expanded(max(clearance, nodes[k].clearance or 0)) for k in nodes
    ]
    obstacles.extend(extra_obstacles)
    waypoints = list(settings.waypoints or ())
    if settings.lane is not None:
        if waypoints:
            raise ValueError("specify either an external lane or waypoints, not both")
        side = settings.lane_side or "right"
        bounds = union(boxes[k] for k in nodes).expanded(clearance + settings.lane + 12)
        if side in {"left", "right"}:
            x = bounds.x if side == "left" else bounds.right
            waypoints = [(x, start[1]), (x, end[1])]
        else:
            y = bounds.y if side == "top" else bounds.bottom
            waypoints = [(start[0], y), (end[0], y)]
    points = [a, start]
    stops = [start, *waypoints, end]
    mode = settings.routing or "orthogonal"
    for p, q in itertools.pairwise(stops):
        if mode == "polyline" and settings.waypoints:
            if any(crosses(p, q, box) for box in obstacles):
                raise ValueError(f"explicit polyline crosses an obstacle: {p}, {q}")
            points.append(q)
        else:
            points.extend(_grid_path(p, q, obstacles)[1:])
    points.append(b)
    for p, q, owner in ((a, start, edge.source), (end, b, edge.target)):
        blocked = any(
            crosses(p, q, boxes[k].expanded(max(clearance, nodes[k].clearance or 0)))
            for k in nodes
            if k != owner
        ) or any(crosses(p, q, box) for box in extra_obstacles)
        if blocked:
            raise ValueError(
                f"attachment corridor blocked for edge {edge.id or edge.source + ' -> ' + edge.target}"
            )
    return simplify(points, waypoints), a_side, b_side


def union(boxes):
    boxes = list(boxes)
    if not boxes:
        return Box(0, 0, 1, 1)
    left, top = min(b.x for b in boxes), min(b.y for b in boxes)
    return Box(
        left,
        top,
        max(b.right for b in boxes) - left,
        max(b.bottom for b in boxes) - top,
    )


def along(points, fraction):
    lengths = [math.dist(a, b) for a, b in itertools.pairwise(points)]
    remaining = sum(lengths) * fraction
    for (a, b), length in zip(itertools.pairwise(points), lengths, strict=True):
        if remaining <= length and length:
            return tuple(a[i] + (b[i] - a[i]) * remaining / length for i in (0, 1))
        remaining -= length
    return points[-1]


def route(edge, settings, boxes, nodes, *, shift=0, extra_obstacles=()):
    """Try alternate automatic attachment sides when the preferred side is blocked."""
    try:
        return _route(
            edge, settings, boxes, nodes, shift=shift, extra_obstacles=extra_obstacles
        )
    except ValueError as error:
        if not any(word in str(error) for word in ("obstacle", "corridor")):
            raise
        original = error
    sources = (
        tuple(SIDES)
        if settings.source_port in (None, "auto")
        else (settings.source_port,)
    )
    targets = (
        tuple(SIDES)
        if settings.target_port in (None, "auto")
        else (settings.target_port,)
    )
    candidates = []
    for source, target in itertools.product(sources, targets):
        try:
            candidates.append(
                _route(
                    edge,
                    replace(settings, source_port=source, target_port=target),
                    boxes,
                    nodes,
                    shift=shift,
                    extra_obstacles=extra_obstacles,
                )
            )
        except ValueError:
            continue
    if not candidates:
        raise original
    return min(
        candidates,
        key=lambda candidate: sum(
            math.dist(a, b) for a, b in itertools.pairwise(candidate[0])
        ),
    )
