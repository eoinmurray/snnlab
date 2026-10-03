"""Concise readouts expanded into ordinary graph operations."""

from __future__ import annotations

from dataclasses import dataclass

from . import ops
from .core import LeakyIntegrator, NonNegative, Normal, Signal, Spec, ms


@dataclass(frozen=True)
class Readout:
    signal: Signal
    parameters: tuple = ()

    def __getattr__(self, item):
        return getattr(self.signal, item)

    @property
    def id(self) -> str:
        return self.signal.id


def MeanVoltage(
    *,
    source: Signal,
    classes: int,
    name: str,
    tau=2 * ms,
    weight: Spec = Normal(1.0, 0.1),
) -> Readout:
    net = source.network
    with net.group(name):
        layer = net.population(
            f"{name}_integrator",
            size=classes,
            neuron=LeakyIntegrator(
                tau=tau,
                soft_reset_threshold=1.0,
                surrogate_slope=5.0,
                initial_voltage=0.0,
            ),
            spiking=False,
        )
        projection = net.connect(
            source,
            layer.excitatory,
            name=f"{name}_projection",
            synapse=LeakyIntegrator(tau=tau),
            weight=weight,
            constraint=NonNegative(),
            connection="feedforward",
        )
        mean = ops.reduce(
            layer.voltage, operation="mean", over="time", name=f"{name}_mean"
        )
    return Readout(mean, projection.parameter_ids)


def FinalVoltage(*, source: Signal, classes: int, name: str) -> Readout:
    with source.network.group(name):
        projected = ops.linear(source, size=classes, name=f"{name}_projection")
        signal = source.network.operation(
            "select_final",
            projected,
            name=f"{name}_final",
            shape=tuple(
                dimension for dimension in projected.shape if dimension != "time"
            ),
            unit=projected.unit,
        )
    return Readout(signal)


def SpikeCount(*, source: Signal, classes: int, name: str) -> Readout:
    with source.network.group(name):
        projected = ops.linear(source, size=classes, name=f"{name}_projection")
        result = ops.reduce(
            projected, operation="sum", over="time", name=f"{name}_count"
        )
    return Readout(result)


def SpikeRate(
    *,
    source: Signal,
    classes: int,
    name: str,
    duration: float | None = None,
    mask: Signal | None = None,
    window: str = "full",
) -> Readout:
    if duration is None and mask is None:
        raise ValueError("SpikeRate requires a duration or valid-time mask")
    with source.network.group(name):
        projected = ops.linear(source, size=classes, name=f"{name}_projection")
        count = ops.reduce(
            projected,
            operation="sum",
            over="time",
            name=f"{name}_count",
            window=window,
            mask=mask,
        )
        signal = source.network.operation(
            "duration_normalise",
            [count] + ([mask] if mask else []),
            name=f"{name}_rate",
            shape=count.shape,
            unit="Hz",
            duration=duration,
            mask=mask.id if mask else None,
            window=window,
        )
    return Readout(signal)


def CumulativePotential(*, source: Signal, classes: int, name: str) -> Readout:
    with source.network.group(name):
        projected = ops.linear(source, size=classes, name=f"{name}_projection")
        signal = source.network.operation(
            "cumulative_sum",
            projected,
            name=f"{name}_cumulative",
            shape=projected.shape,
            unit=projected.unit,
        )
    return Readout(signal)
