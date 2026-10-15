"""Physical quantities and authoring unit constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str

    def json(self) -> dict[str, Any]:
        return {"value": self.value, "unit": self.unit}


class Unit:
    def __init__(self, symbol: str):
        self.symbol = symbol

    def __rmul__(self, value: float) -> Quantity:
        return Quantity(float(value), self.symbol)


ms = Unit("ms")


mV = Unit("mV")


nS = Unit("nS")


uS = Unit("uS")


Hz = Unit("Hz")


nA = Unit("nA")
