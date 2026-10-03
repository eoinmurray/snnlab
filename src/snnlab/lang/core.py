"""Graph-shaped authoring model for portable spiking-network descriptions."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, Sequence

Shape = tuple[int | str, ...]


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


class SignalLike(Protocol):
    id: str


ms = Unit("ms")
mV = Unit("mV")
nS = Unit("nS")
uS = Unit("uS")
Hz = Unit("Hz")


def _value(value: Any) -> Any:
    if isinstance(value, Quantity):
        return value.json()
    if hasattr(value, "json"):
        return value.json()
    if isinstance(value, tuple):
        return list(value)
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


def COBA_LIF(**values: Any) -> Spec:
    return Spec("coba_lif", values)


def LIF(**values: Any) -> Spec:
    return Spec("lif", values)


def LeakyIntegrator(**values: Any) -> Spec:
    return Spec("leaky_integrator", values)


def AMPA(**values: Any) -> Spec:
    return Spec("ampa", values)


def GABA(**values: Any) -> Spec:
    return Spec("gaba", values)


def Modulatory(**values: Any) -> Spec:
    return Spec("modulatory", values)


def Normal(mean: float, std: float) -> Spec:
    """Compatibility name for the collection's lower-clamped normal law."""
    return LowerClampedNormal(mean, std)


def LowerClampedNormal(
    mean: float,
    std: float,
    *,
    initial_zero_fraction: float = 0.0,
    zeroing: str = "bernoulli",
) -> Spec:
    return Spec(
        "lower_clamped_normal",
        {
            "mean": mean,
            "std": std,
            "initial_zero_fraction": initial_zero_fraction,
            "zeroing": zeroing,
        },
    )


def SignedNormal(mean: float, std: float) -> Spec:
    return Spec("signed_normal", {"mean": mean, "std": std})


def Uniform(low: float, high: float) -> Spec:
    return Spec("uniform", {"low": low, "high": high})


def Zeros() -> Spec:
    return Spec("zeros")


def Constant(value: float) -> Spec:
    return Spec("constant", {"value": value})


def NonNegative() -> Spec:
    return Spec("non_negative")


@dataclass(frozen=True)
class Signal:
    network: "Network" = field(compare=False, repr=False)
    id: str
    shape: Shape
    unit: str
    signal_type: str
    owner: str
    port: str

    def json_ref(self) -> str:
        return self.id


@dataclass(frozen=True)
class ParameterRef:
    network: "Network" = field(compare=False, repr=False)
    id: str


@dataclass
class Population:
    network: "Network" = field(repr=False)
    id: str
    size: int
    neuron: Spec
    spiking: bool
    group: str | None

    @property
    def spikes(self) -> Signal:
        if not self.spiking:
            raise AttributeError(f"{self.id!r} is non-spiking and has no spikes port")
        return self.network._signal(f"{self.id}.spikes")

    @property
    def voltage(self) -> Signal:
        return self.network._signal(f"{self.id}.voltage")

    @property
    def excitatory(self) -> str:
        return f"{self.id}.excitatory"

    @property
    def inhibitory(self) -> str:
        return f"{self.id}.inhibitory"

    @property
    def modulatory(self) -> str:
        return f"{self.id}.modulatory"


@dataclass
class Projection:
    network: "Network" = field(repr=False)
    id: str
    source: str
    target: str
    synapse: Spec
    connection: str
    delay: Quantity | None
    parameter_ids: tuple[str, ...]
    group: str | None
    enabled: bool

    @property
    def weight(self) -> ParameterRef:
        return ParameterRef(self.network, self.parameter_ids[0])


@dataclass
class Component:
    name: str
    members: list[str] = field(default_factory=list)
    parent: str | None = None


class Network:
    """Mutable Python authoring surface; compilation produces immutable data."""

    def __init__(self, name: str, *, dt: Quantity = 0.1 * ms):
        self.name = name
        self.dt = dt
        self.inputs: list[dict[str, Any]] = []
        self.populations: list[dict[str, Any]] = []
        self.projections: list[dict[str, Any]] = []
        self.operations: list[dict[str, Any]] = []
        self.parameters: list[dict[str, Any]] = []
        self.constants: list[dict[str, Any]] = []
        self.outputs: list[dict[str, Any]] = []
        self.observables: list[dict[str, Any]] = []
        self.assets: list[dict[str, Any]] = []
        self.groups: dict[str, Component] = {}
        self._signals: dict[str, Signal] = {}
        self._names: set[str] = set()
        self._group_stack: list[str] = []

    @property
    def current_group(self) -> str | None:
        return self._group_stack[-1] if self._group_stack else None

    def _claim(self, name: str) -> None:
        if not name or any(c.isspace() for c in name):
            raise ValueError(
                f"name must be non-empty and contain no whitespace: {name!r}"
            )
        if name in self._names:
            raise ValueError(f"duplicate name: {name}")
        self._names.add(name)
        if self.current_group:
            self.groups[self.current_group].members.append(name)

    def _signal(self, signal_id: str) -> Signal:
        return self._signals[signal_id]

    def input(
        self, name: str, *, shape: Shape, signal_type: str, unit: str = "1"
    ) -> Signal:
        self._claim(name)
        signal = Signal(
            self, f"{name}.value", tuple(shape), unit, signal_type, name, "value"
        )
        self._signals[signal.id] = signal
        self.inputs.append(
            {"id": name, "shape": list(shape), "signal_type": signal_type, "unit": unit}
        )
        return signal

    def population(
        self, name: str, *, size: int, neuron: Spec, spiking: bool = True
    ) -> Population:
        self._claim(name)
        if size <= 0:
            raise ValueError("population size must be positive")
        group = self.current_group
        self.populations.append(
            {
                "id": name,
                "size": size,
                "neuron": neuron.json(),
                "spiking": spiking,
                "group": group,
            }
        )
        if spiking:
            self._signals[f"{name}.spikes"] = Signal(
                self,
                f"{name}.spikes",
                ("time", "batch", size),
                "spike",
                "spikes",
                name,
                "spikes",
            )
        self._signals[f"{name}.voltage"] = Signal(
            self,
            f"{name}.voltage",
            ("time", "batch", size),
            "mV",
            "voltage",
            name,
            "voltage",
        )
        return Population(self, name, size, neuron, spiking, group)

    def parameter(
        self,
        name: str,
        *,
        shape: Shape,
        initializer: Spec,
        unit: str = "1",
        constraint: Spec | None = None,
    ) -> ParameterRef:
        self._claim(name)
        self.parameters.append(
            {
                "id": name,
                "shape": list(shape),
                "unit": unit,
                "initializer": initializer.json(),
                "constraint": constraint.json() if constraint else None,
                "group": self.current_group,
            }
        )
        return ParameterRef(self, name)

    def constant(self, name: str, value: Any, *, unit: str = "1") -> str:
        self._claim(name)
        self.constants.append({"id": name, "value": _value(value), "unit": unit})
        return name

    def connect(
        self,
        source: Signal,
        target: str,
        *,
        name: str,
        synapse: Spec,
        weight: Spec | ParameterRef = Constant(1.0),
        constraint: Spec | None = None,
        connection: str = "feedforward",
        delay: Quantity | None = None,
        enabled: bool = True,
    ) -> Projection:
        self._claim(name)
        target_pop, _, target_port = target.partition(".")
        populations = {p["id"]: p for p in self.populations}
        if source.network is not self:
            raise ValueError("source belongs to another network")
        if target_pop not in populations or target_port not in {
            "excitatory",
            "inhibitory",
            "modulatory",
        }:
            raise ValueError(f"invalid target port: {target}")
        if connection not in {"feedforward", "recurrent", "feedback", "modulatory"}:
            raise ValueError(f"invalid connection kind: {connection}")
        if not isinstance(enabled, bool):
            raise TypeError("projection enabled must be boolean")
        if isinstance(weight, ParameterRef):
            parameter_id = weight.id
        else:
            parameter_id = f"{name}.weight"
            self.parameters.append(
                {
                    "id": parameter_id,
                    "shape": [populations[target_pop]["size"], source.shape[-1]],
                    "unit": "uS",
                    "initializer": weight.json(),
                    "constraint": constraint.json() if constraint else None,
                    "group": self.current_group,
                }
            )
        row = {
            "id": name,
            "source": source.id,
            "target": target,
            "synapse": synapse.json(),
            "connection": connection,
            "polarity": target_port,
            "delay": _value(delay),
            "parameters": [parameter_id],
            "group": self.current_group,
        }
        if not enabled:
            row["enabled"] = False
        self.projections.append(row)
        return Projection(
            self,
            name,
            source.id,
            target,
            synapse,
            connection,
            delay,
            (parameter_id,),
            self.current_group,
            enabled,
        )

    def operation(
        self,
        kind: str,
        sources: Signal | Sequence[Signal],
        *,
        name: str,
        shape: Shape,
        unit: str,
        signal_type: str = "continuous",
        parameters: Sequence[ParameterRef] = (),
        **config: Any,
    ) -> Signal:
        self._claim(name)
        source_list = [sources] if isinstance(sources, Signal) else list(sources)
        if any(s.network is not self for s in source_list):
            raise ValueError("operation source belongs to another network")
        signal = Signal(
            self, f"{name}.value", tuple(shape), unit, signal_type, name, "value"
        )
        self._signals[signal.id] = signal
        self.operations.append(
            {
                "id": name,
                "kind": kind,
                "sources": [s.id for s in source_list],
                "shape": list(shape),
                "unit": unit,
                "signal_type": signal_type,
                "parameters": [p.id for p in parameters],
                "config": {k: _value(v) for k, v in sorted(config.items())},
                "group": self.current_group,
            }
        )
        return signal

    def output(self, name: str, signal: SignalLike) -> SignalLike:
        self._claim(name)
        self.outputs.append({"id": name, "signal": signal.id})
        return signal

    def expose(self, *signals: Signal, name: str | None = None) -> None:
        for index, signal in enumerate(signals):
            obs_name = (
                name if name and len(signals) == 1 else f"{signal.owner}_{signal.port}"
            )
            if len(signals) > 1 and name:
                obs_name = f"{name}_{index}"
            self._claim(obs_name)
            self.observables.append({"id": obs_name, "signal": signal.id})

    def asset(self, name: str, *, media_type: str, description: str = "") -> str:
        self._claim(name)
        self.assets.append(
            {"id": name, "media_type": media_type, "description": description}
        )
        return name

    @contextmanager
    def group(self, name: str, *, parent: str | None = None) -> Iterator[Component]:
        self._claim(name)
        if parent and parent not in self.groups:
            raise ValueError(f"unknown parent group: {parent}")
        component = Component(name, parent=parent)
        self.groups[name] = component
        self._group_stack.append(name)
        try:
            yield component
        finally:
            self._group_stack.pop()
