"""Native Excalidraw element identities, geometry and bindings."""

from __future__ import annotations

import hashlib


def _id(kind: str, identifier: str) -> str:
    digest = hashlib.sha256(identifier.encode()).hexdigest()[:24]
    return f"{kind}_{digest}"


def _element(kind, identifier, *, x, y, width, height, groups=(), data=None, **extra):
    seed = int(hashlib.sha256(identifier.encode()).hexdigest()[:7], 16)
    return {
        "id": identifier,
        "type": kind,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": "#24292F",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": list(groups),
        "frameId": None,
        "roundness": None,
        "seed": seed,
        "version": 1,
        "versionNonce": seed,
        "isDeleted": False,
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
        "customData": {"snnlab": data or {}},
        **extra,
    }


def _binding(box, point):
    return {
        "elementId": box["id"],
        "focus": 0,
        "gap": 1,
        "fixedPoint": [
            max(0, min(1, (point[0] - box["x"]) / box["width"])),
            max(0, min(1, (point[1] - box["y"]) / box["height"])),
        ],
    }
