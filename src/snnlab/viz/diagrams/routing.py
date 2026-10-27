"""Place wire labels in clear space without changing orthogonal routes."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from itertools import pairwise

MIN_WIRE_SPACING = 18.0
FEEDBACK_LANE_SPACING = 32.0


def _arrowhead_replacer(replacements):
    def replace_edge(match):
        attributes = match[2]
        identifier = re.search(r'\bid=(?:"([^"]*)"|(\w+))', attributes)
        if identifier and (
            position := replacements.get(identifier[1] or identifier[2])
        ):
            attributes = re.sub(
                r'\bpos="[^"]*"', lambda _: f'pos="{position}"', attributes
            )
        return match[1] + attributes + match[3]

    return replace_edge


def expose_arrowheads(layout: dict, positioned_dot: str) -> tuple[dict, str]:
    """Stop wires outside the first target boundary, rather than under its fill."""
    layout = deepcopy(layout)
    nodes = {node["_gvid"]: node for node in layout.get("objects", []) if "pos" in node}
    replacements = {}
    for edge in layout.get("edges", []):
        if edge.get("style") == "invis" or not edge.get("id"):
            continue
        node = nodes[edge["head"]]
        x, y = map(float, node["pos"].split(","))
        w, h = float(node["width"]) * 72, float(node["height"]) * 72
        left, bottom, right, top = (
            x - w / 2 - 3,
            y - h / 2 - 3,
            x + w / 2 + 3,
            y + h / 2 + 3,
        )
        points = []
        for command in edge.get("_draw_", []):
            if command["op"] == "b":
                for point in command["points"]:
                    if not points or tuple(point) != points[-1]:
                        points.append(tuple(point))
        old_tip = re.match(r"e,([^ ]+)", edge.get("pos", ""))
        if not points or not old_tip:
            continue
        points.append(tuple(map(float, old_tip[1].split(","))))
        for index, (a, b) in enumerate(pairwise(points)):
            if left <= a[0] <= right and bottom <= a[1] <= top:
                # A self-loop starts inside its own expanded target rectangle;
                # trim the return, not the outgoing segment.
                continue
            dx, dy = b[0] - a[0], b[1] - a[1]
            if abs(dy) < 0.01 and bottom <= a[1] <= top and abs(dx) > 0.01:
                tip = (left if dx > 0 else right, a[1])
                fraction = (tip[0] - a[0]) / dx
                direction = (1 if dx > 0 else -1, 0)
            elif abs(dx) < 0.01 and left <= a[0] <= right and abs(dy) > 0.01:
                tip = (a[0], bottom if dy > 0 else top)
                fraction = (tip[1] - a[1]) / dy
                direction = (0, 1 if dy > 0 else -1)
            else:
                continue
            if not 0 <= fraction <= 1:
                continue
            length = (
                float(edge.get("arrowsize", "0.65"))
                * (20 if "tee" in edge.get("arrowhead", "") else 10)
                + 2
            )
            base = (tip[0] - direction[0] * length, tip[1] - direction[1] * length)
            route = points[: index + 1] + [base]
            controls = [route[0]]
            for start, end in pairwise(route):
                controls.extend((start, end, end))
            edge["pos"] = f"e,{tip[0]:.4f},{tip[1]:.4f} " + " ".join(
                f"{px:.4f},{py:.4f}" for px, py in controls
            )
            edge["_draw_"] = [{"op": "b", "points": controls}]
            side = (-direction[1] * 4, direction[0] * 4)
            edge["_hdraw_"] = [
                {
                    "op": "P",
                    "points": [
                        tip,
                        (base[0] + side[0], base[1] + side[1]),
                        (base[0] - side[0], base[1] - side[1]),
                    ],
                }
            ]
            replacements[edge["id"]] = edge["pos"]
            break

    replace_edge = _arrowhead_replacer(replacements=replacements)

    return layout, re.sub(
        r"(^\s*[^\n]* -> [^\n]*\[)(.*?)(\];)",
        replace_edge,
        positioned_dot,
        flags=re.MULTILINE | re.DOTALL,
    )


def _feedback_anchors(
    *, edge, index, boxes, incident, lane_y, allocated_top, fixed_vertical, group_labels
):
    anchors = []
    for end, other_end in (("tail", "head"), ("head", "tail")):
        identifier = edge[end]
        box = boxes[identifier]
        center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        ports = incident[identifier]
        offset = (ports.index((index, end)) - (len(ports) - 1) / 2) * MIN_WIRE_SPACING
        if abs(offset) + 8 > (box[2] - box[0]) / 2:
            raise ValueError("feedback node is too narrow for separated wire ports")
        preferred = center_x + offset
        options = [preferred] + [
            center_x + step * MIN_WIRE_SPACING for step in range(-10, 11)
        ]
        x = None
        for candidate in sorted(set(options), key=lambda value: abs(value - preferred)):
            if not box[0] + 8 <= candidate <= box[2] - 8:
                continue
            if any(
                abs(candidate - taken) < MIN_WIRE_SPACING - 0.01
                for taken in allocated_top.get(identifier, [])
            ):
                continue
            if any(
                abs(candidate - vx) < MIN_WIRE_SPACING - 0.01
                and low < lane_y
                and high > box[3]
                for vx, low, high in fixed_vertical
            ):
                continue
            if any(
                label[0] - 4 < candidate < label[2] + 4
                and label[3] > box[3]
                and label[1] < lane_y
                for label in group_labels
            ):
                continue
            x = candidate
            break
        clear_above = not any(
            other != identifier
            and x is not None
            and obstacle[0] - 4 < x < obstacle[2] + 4
            and obstacle[3] > box[3]
            and obstacle[1] < lane_y
            for other, obstacle in boxes.items()
        )
        if clear_above and x is not None:
            allocated_top.setdefault(identifier, []).append(x)
            anchors.append(((x, box[3]), (x, lane_y), "top"))
        else:
            other_box = boxes[edge[other_end]]
            direction = 1 if other_box[0] > box[0] else -1
            port = (box[2] if direction > 0 else box[0], center_y + offset)
            escape_x = port[0] + direction * MIN_WIRE_SPACING * (index + 1)
            for _attempt in range(40):
                if not any(
                    other != identifier
                    and obstacle[0] - 6 < escape_x < obstacle[2] + 6
                    and obstacle[3] > port[1]
                    and obstacle[1] < lane_y
                    for other, obstacle in boxes.items()
                ):
                    break
                escape_x += direction * MIN_WIRE_SPACING
            else:
                raise ValueError("no clear vertical escape for feedback lane")
            anchors.append((port, (escape_x, lane_y), direction))
    return anchors


def _route_feedback_edges(
    *,
    edges,
    boxes,
    incident,
    top,
    allocated_top,
    fixed_vertical,
    group_labels,
    replacements,
    label_positions,
):
    for index, edge in enumerate(edges):
        lane_y = top + FEEDBACK_LANE_SPACING * (index + 1)
        anchors = _feedback_anchors(
            edge=edge,
            index=index,
            boxes=boxes,
            incident=incident,
            lane_y=lane_y,
            allocated_top=allocated_top,
            fixed_vertical=fixed_vertical,
            group_labels=group_labels,
        )

        start, first, source_side = anchors[0]
        tip, last, target_side = anchors[1]
        points = [start]
        if source_side != "top":
            points.append((first[0], start[1]))
        points.extend((first, last))
        arrow_length = float(edge.get("arrowsize", "0.65")) * 10 + 2
        if target_side == "top":
            points.append((tip[0], tip[1] + arrow_length))
        else:
            points.extend(
                ((last[0], tip[1]), (tip[0] + target_side * arrow_length, tip[1]))
            )
        controls = [points[0]]
        for a, b in pairwise(points):
            controls.extend((a, b, b))
        edge["pos"] = f"e,{tip[0]:.4f},{tip[1]:.4f} " + " ".join(
            f"{x:.4f},{y:.4f}" for x, y in controls
        )
        edge["_draw_"] = [{"op": "b", "points": controls}]
        edge["_hdraw_"] = [
            {"op": "P", "points": [(tip[0] - 4, tip[1] - 4), (tip[0] + 4, tip[1] + 4)]}
        ]
        replacements[edge["id"]] = edge["pos"]
        if "xlp" in edge:
            edge["xlp"] = f"{(first[0] + last[0]) / 2:.4f},{lane_y + 12:.4f}"
            label_positions[edge["id"]] = edge["xlp"]


def _feedback_edge_replacer(replacements, label_positions):
    def replace_edge(match):
        attributes = match[2]
        identifier = re.search(r'\bid=(?:"([^"]*)"|(\w+))', attributes)
        if identifier:
            new_pos = replacements.get(identifier[1] or identifier[2])
            if new_pos:
                attributes = re.sub(
                    r'\bpos="[^"]*"', lambda _: f'pos="{new_pos}"', attributes
                )
                new_label = label_positions.get(identifier[1] or identifier[2])
                if new_label:
                    attributes = re.sub(
                        r'\bxlp="[^"]*"', lambda _: f'xlp="{new_label}"', attributes
                    )
        return match[1] + attributes + match[3]

    return replace_edge


def route_feedback_lanes(layout: dict, positioned_dot: str) -> tuple[dict, str]:
    """Give long-range feedback separate overhead lanes and separated node ports."""
    layout = deepcopy(layout)
    nodes = {node["_gvid"]: node for node in layout.get("objects", []) if "pos" in node}
    boxes = {}
    for identifier, node in nodes.items():
        x, y = map(float, node["pos"].split(","))
        w, h = float(node["width"]) * 72, float(node["height"]) * 72
        boxes[identifier] = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
    edges = [
        edge
        for edge in layout.get("edges", [])
        if "feedback" in edge.get("class", "").split()
        and edge["tail"] != edge["head"]
        and abs(
            float(nodes[edge["tail"]]["pos"].split(",")[0])
            - float(nodes[edge["head"]]["pos"].split(",")[0])
        )
        > 1
    ]
    if not edges:
        return layout, positioned_dot
    routed_ids = {edge["id"] for edge in edges}
    fixed_vertical = []
    for edge in layout.get("edges", []):
        if edge.get("style") == "invis" or edge.get("id") in routed_ids:
            continue
        for command in edge.get("_draw_", []):
            if command["op"] == "b":
                for a, b in pairwise(command["points"]):
                    if abs(a[0] - b[0]) < 0.01 and abs(a[1] - b[1]) > 0.01:
                        fixed_vertical.append((a[0], min(a[1], b[1]), max(a[1], b[1])))
    allocated_top = {}
    group_labels = [
        box
        for node in layout.get("objects", [])
        if "pos" not in node
        for box in _text_boxes(node.get("_ldraw_", []))
    ]
    incident = {}
    for index, edge in enumerate(edges):
        for end in ("tail", "head"):
            incident.setdefault(edge[end], []).append((index, end))
    top = max(box[3] for box in boxes.values())
    replacements = {}
    label_positions = {}
    _route_feedback_edges(
        edges=edges,
        boxes=boxes,
        incident=incident,
        top=top,
        allocated_top=allocated_top,
        fixed_vertical=fixed_vertical,
        group_labels=group_labels,
        replacements=replacements,
        label_positions=label_positions,
    )

    replace_edge = _feedback_edge_replacer(
        replacements=replacements, label_positions=label_positions
    )

    positioned_dot = re.sub(
        r"(^\s*[^\n]* -> [^\n]*\[)(.*?)(\];)",
        replace_edge,
        positioned_dot,
        flags=re.MULTILINE | re.DOTALL,
    )
    if "lp" in layout:
        old_x, old_y = map(float, layout["lp"].split(","))
        title_y = top + FEEDBACK_LANE_SPACING * (len(edges) + 1) + 20
        layout["lp"] = f"{old_x:.4f},{title_y:.4f}"
        positioned_dot = re.sub(
            r'\blp="[^"]*"', lambda _: f'lp="{layout["lp"]}"', positioned_dot, count=1
        )
        for command in layout.get("_ldraw_", []):
            if "pt" in command:
                command["pt"][1] += title_y - old_y
    return layout, positioned_dot


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _bounds(points, padding=0):
    xs, ys = zip(*points, strict=True)
    return min(xs) - padding, min(ys) - padding, max(xs) + padding, max(ys) + padding


def _wire_obstacles(*, layout, wires, routes, obstacles):
    for index, edge in enumerate(layout.get("edges", [])):
        if edge.get("style") == "invis":
            continue
        segments = []
        for command in edge.get("_draw_", []):
            if command["op"] not in {"b", "B", "L"}:
                continue
            points = command["points"]
            for start, end in pairwise(points):
                if start != end:
                    segments.append((start, end))
                    wires.append(_bounds((start, end), 3))
        routes[index] = segments
        for command in edge.get("_hdraw_", []):
            if "points" in command:
                obstacles.append(_bounds(command["points"], 3))


def position_wire_labels(layout: dict, positioned_dot: str) -> str:
    """Move xlabels into vacant space near their own wire, retaining node positions."""
    obstacles = []
    for node in layout.get("objects", []):
        if "pos" in node:
            x, y = map(float, node["pos"].split(","))
            w, h = float(node["width"]) * 72, float(node["height"]) * 72
            obstacles.append(
                (x - w / 2 - 3, y - h / 2 - 3, x + w / 2 + 3, y + h / 2 + 3)
            )
        else:
            obstacles.extend(_text_boxes(node.get("_ldraw_", [])))
    obstacles.extend(_text_boxes(layout.get("_ldraw_", [])))
    wires = []
    routes = {}
    _wire_obstacles(layout=layout, wires=wires, routes=routes, obstacles=obstacles)

    occupied = [*obstacles, *wires]
    replacements = {}
    for index, edge in enumerate(layout.get("edges", [])):
        if "xlp" not in edge:
            continue
        text_boxes = _text_boxes(edge.get("_ldraw_", []))
        if not text_boxes:
            continue
        bounds = _bounds(
            [(box[0], box[1]) for box in text_boxes]
            + [(box[2], box[3]) for box in text_boxes]
        )
        width, height = bounds[2] - bounds[0] + 10, bounds[3] - bounds[1] + 10
        original = tuple(map(float, edge["xlp"].split(",")))
        candidates = []
        for start, end in routes.get(index, []):
            horizontal = abs(start[1] - end[1]) < 0.01
            for fraction in (0.5, 0.25, 0.75, 0.1, 0.9):
                x = start[0] + fraction * (end[0] - start[0])
                y = start[1] + fraction * (end[1] - start[1])
                for extra in (6, 14, 26, 42, 64, 96, 160, 256):
                    offset = (height if horizontal else width) / 2 + extra
                    for sign in (-1, 1):
                        cx, cy = (
                            (x, y + sign * offset)
                            if horizontal
                            else (x + sign * offset, y)
                        )
                        box = (
                            cx - width / 2,
                            cy - height / 2,
                            cx + width / 2,
                            cy + height / 2,
                        )
                        if not any(_overlaps(box, obstacle) for obstacle in occupied):
                            candidates.append(
                                (
                                    math.hypot(cx - original[0], cy - original[1])
                                    + extra,
                                    cx,
                                    cy,
                                    box,
                                )
                            )
        if not candidates:
            raise ValueError(
                f"diagram wire label has no clear placement: {edge.get('xlabel')}; route {routes.get(index)}"
            )
        _, x, y, box = min(candidates)
        occupied.append(box)
        replacements.setdefault(original, []).append((x, y))

    def replace(match):
        point = tuple(map(float, match[1].split(",")))
        if point not in replacements:
            return match[0]
        x, y = replacements[point].pop(0)
        return f'xlp="{x:.4f},{y:.4f}"'

    return re.sub(r'\bxlp="([^"]+)"', replace, positioned_dot)


def _text_boxes(commands):
    size = 10
    boxes = []
    for command in commands:
        if command.get("op") == "F":
            size = command["size"]
        elif command.get("op") == "T":
            x, y = command["pt"]
            width = command["width"]
            x -= {"l": 0, "c": width / 2, "r": width}[command["align"]]
            boxes.append((x, y - size * 0.3, x + width, y + size))
        elif command.get("op") in {"p", "P"}:
            boxes.append(_bounds(command["points"]))
    return boxes
