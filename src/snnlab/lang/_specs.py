"""Private, data-only specification and JSON-value primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from snnlab.lang.units import Quantity


def _value(value: Any) -> Any:
    if isinstance(value, Quantity):
        return value.json()
    if hasattr(value, "json"):
        return value.json()
    if isinstance(value, (tuple, list)):
        return [_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class Spec:
    kind: str
    values: dict[str, Any] = field(default_factory=dict)

    def json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            **{k: _value(v) for k, v in sorted(self.values.items())},
        }


def _custom(category: str, definition: str, config: dict[str, Any]) -> Spec:
    from snnlab.extensions import get

    get(category, definition)
    return Spec(f"custom_{category}", {"definition": definition, "config": config})
