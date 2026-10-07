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
    @property
    def id(self) -> str: ...


ms = Unit("ms")
mV = Unit("mV")
nS = Unit("nS")
uS = Unit("uS")
Hz = Unit("Hz")
nA = Unit("nA")


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


def COBA_LIF(**values: Any) -> Spec:
    return Spec("coba_lif", values)


def LIF(**values: Any) -> Spec:
    return CUBA_LIF(**values)


def CUBA_LIF(
    *,
    tau_mem=20 * ms,
    capacitance_nf=1.0,
    resting_mv=-65.0,
    threshold_mv=-50.0,
    reset_mv=-65.0,
    refractory_steps=0,
    initial_voltage_mv=None,
    voltage_grad_dampen=1.0,
) -> Spec:
    return Spec(
        "cuba_lif",
        dict(
            tau_mem=tau_mem,
            capacitance_nf=capacitance_nf,
            resting_mv=resting_mv,
            threshold_mv=threshold_mv,
            reset_mv=reset_mv,
            refractory_steps=refractory_steps,
            initial_voltage_mv=resting_mv
            if initial_voltage_mv is None
            else initial_voltage_mv,
            voltage_grad_dampen=voltage_grad_dampen,
        ),
    )


def ExponentialCurrent(*, tau=5 * ms) -> Spec:
    return Spec("exponential_current", {"tau": tau})


def ALIF(**values: Any) -> Spec:
    """Current-based LIF with a spike-triggered adapting threshold."""
    return CUBA_ALIF(**values)


def CUBA_ALIF(
    *,
    tau_adaptation=200 * ms,
    adaptation_increment_mv=1.0,
    initial_adaptation_mv=0.0,
    **lif_values: Any,
) -> Spec:
    return Spec(
        "cuba_alif",
        dict(
            CUBA_LIF(**lif_values).values,
            tau_adaptation=tau_adaptation,
            adaptation_increment_mv=adaptation_increment_mv,
            initial_adaptation_mv=initial_adaptation_mv,
        ),
    )


def COBA_ALIF(
    *, excitatory_reversal_mv=0.0, inhibitory_reversal_mv=-80.0, **values: Any
) -> Spec:
    return Spec(
        "coba_alif",
        dict(
            CUBA_ALIF(**values).values,
            excitatory_reversal_mv=excitatory_reversal_mv,
            inhibitory_reversal_mv=inhibitory_reversal_mv,
        ),
    )


def ADEX(**values: Any) -> Spec:
    """Current-based adaptive exponential integrate-and-fire neuron."""
    return CUBA_ADEX(**values)


def CUBA_ADEX(
    *,
    tau_adaptation=200 * ms,
    delta_t_mv=2.0,
    spike_mv=-30.0,
    a_us=0.002,
    b_na=0.05,
    initial_adaptation_na=0.0,
    **lif_values: Any,
) -> Spec:
    return Spec(
        "cuba_adex",
        dict(
            CUBA_LIF(**lif_values).values,
            tau_adaptation=tau_adaptation,
            delta_t_mv=delta_t_mv,
            spike_mv=spike_mv,
            a_us=a_us,
            b_na=b_na,
            initial_adaptation_na=initial_adaptation_na,
        ),
    )


def COBA_ADEX(
    *, excitatory_reversal_mv=0.0, inhibitory_reversal_mv=-80.0, **values: Any
) -> Spec:
    return Spec(
        "coba_adex",
        dict(
            CUBA_ADEX(**values).values,
            excitatory_reversal_mv=excitatory_reversal_mv,
            inhibitory_reversal_mv=inhibitory_reversal_mv,
        ),
    )


def _custom(category: str, definition: str, config: dict[str, Any]) -> Spec:
    from snnlab.extensions import get

    get(category, definition)
    return Spec(f"custom_{category}", {"definition": definition, "config": config})


def CustomNeuron(definition: str, **config: Any) -> Spec:
    return _custom("neuron", definition, config)


def CustomSynapse(definition: str, **config: Any) -> Spec:
    return _custom("synapse", definition, config)


def CustomInitializer(definition: str, **config: Any) -> Spec:
    return _custom("initializer", definition, config)


def CustomConstraint(definition: str, **config: Any) -> Spec:
    return _custom("constraint", definition, config)


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
    def pre_reset_voltage(self) -> Signal:
        if self.neuron.kind != "leaky_integrator":
            raise AttributeError(
                "pre_reset_voltage is supported only for leaky integrators"
            )
        return self.network._signal(f"{self.id}.pre_reset_voltage")

    @property
    def excitatory(self) -> str:
        return f"{self.id}.excitatory"

    @property
    def inhibitory(self) -> str:
        return f"{self.id}.inhibitory"

    @property
    def modulatory(self) -> str:
        return f"{self.id}.modulatory"

    def state(self, name: str) -> Signal:
        return self.network._signal(f"{self.id}.{name}")


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
    def conductance(self) -> Signal:
        return self.network._signal(f"{self.id}.conductance")

    @property
    def current(self) -> Signal:
        return self.network._signal(f"{self.id}.current")

    def state(self, name: str) -> Signal:
        return self.network._signal(f"{self.id}.{name}")

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
        if neuron.kind == "leaky_integrator":
            self._signals[f"{name}.pre_reset_voltage"] = Signal(
                self,
                f"{name}.pre_reset_voltage",
                ("time", "batch", size),
                "mV",
                "voltage",
                name,
                "pre_reset_voltage",
            )
        from snnlab.extensions import state_units

        for port, unit in state_units("neuron", neuron.json()).items():
            self._signals[f"{name}.{port}"] = Signal(
                self,
                f"{name}.{port}",
                ("time", "batch", size),
                unit,
                "continuous",
                name,
                port,
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
        initialization_scaling: str | None = None,
    ) -> ParameterRef:
        if initialization_scaling is not None and initialization_scaling not in (
            "direct",
            "fan_in_normalized",
        ):
            raise ValueError("invalid initialization_scaling")
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
        if initialization_scaling is not None:
            self.parameters[-1]["initialization_scaling"] = initialization_scaling
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
        initialization_scaling: str | None = None,
    ) -> Projection:
        if initialization_scaling is not None and initialization_scaling not in (
            "direct",
            "fan_in_normalized",
        ):
            raise ValueError("invalid initialization_scaling")
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
            if weight.network is not self:
                raise ValueError("weight belongs to another network")
            parameter_id = weight.id
            parameter = next(p for p in self.parameters if p["id"] == parameter_id)
            existing = parameter.get("initialization_scaling")
            if existing is not None and initialization_scaling not in (None, existing):
                raise ValueError(
                    "conflicting initialization_scaling for shared parameter"
                )
            parameter["initialization_scaling"] = (
                existing or initialization_scaling or "fan_in_normalized"
            )
        else:
            from snnlab.extensions import synapse_unit

            parameter_id = f"{name}.weight"
            self.parameters.append(
                {
                    "id": parameter_id,
                    "shape": [populations[target_pop]["size"], source.shape[-1]],
                    "unit": synapse_unit(synapse.json()),
                    "initializer": weight.json(),
                    "initialization_scaling": initialization_scaling
                    or "fan_in_normalized",
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
        from snnlab.extensions import projection_port, resolve, synapse_unit

        port = projection_port(synapse.json())
        self._signals[f"{name}.{port}"] = Signal(
            self,
            f"{name}.{port}",
            ("time", "batch", populations[target_pop]["size"]),
            synapse_unit(synapse.json()),
            "continuous",
            name,
            port,
        )
        if synapse.kind == "custom_synapse":
            for state_port, unit in resolve(
                "synapse", synapse.json()
            ).state_units.items():
                self._signals[f"{name}.{state_port}"] = Signal(
                    self,
                    f"{name}.{state_port}",
                    ("time", "batch", populations[target_pop]["size"]),
                    unit,
                    "continuous",
                    name,
                    state_port,
                )
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
