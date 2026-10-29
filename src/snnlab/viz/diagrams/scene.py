"""Internal drawing primitives for resolved static diagrams."""

from __future__ import annotations

import hashlib


def _id(kind: str, identifier: str) -> str:
    digest = hashlib.sha256(identifier.encode()).hexdigest()[:24]
    return f"{kind}_{digest}"


def _element(kind, identifier, *, x, y, width, height, groups=(), data=None, **extra):
    return {
        "id": identifier,
        "type": kind,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "strokeColor": "#24292F",
        "backgroundColor": "transparent",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "groups": list(groups),
        "data": data or {},
        **extra,
    }
