"""Data-only declarations for reproducible simulation inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .core import Population, Signal, Spec


def _id(value: Population | Signal | str) -> str:
    return value if isinstance(value, str) else value.id


@dataclass(frozen=True)
class ShotNoise:
    rate_hz: float
    amplitude: float
    tau_ms: float

    def json(self) -> dict[str, Any]:
        return {
            "kind": "shot_noise",
            "rate_hz": float(self.rate_hz),
            "amplitude": float(self.amplitude),
            "tau_ms": float(self.tau_ms),
        }


@dataclass(frozen=True)
class GlobalShotNoise(ShotNoise):
    def json(self) -> dict[str, Any]:
        return {**super().json(), "kind": "global_shot_noise"}


@dataclass(frozen=True)
class GroupedShotNoise(ShotNoise):
    """Shot noise shared within contiguous local cell groups."""

    group_size: int = 16

    def json(self) -> dict[str, Any]:
        return {
            **super().json(),
            "kind": "grouped_shot_noise",
            "group_size": int(self.group_size),
        }


@dataclass(frozen=True)
class CellDistribution:
    """Reproducible multiplicative cell heterogeneity laws."""

    rate: Spec = field(default_factory=lambda: Spec("constant", {"value": 1.0}))
    amplitude: Spec = field(default_factory=lambda: Spec("constant", {"value": 1.0}))

    def json(self) -> dict[str, Any]:
        return {"rate": self.rate.json(), "amplitude": self.amplitude.json()}


@dataclass(frozen=True)
class BackgroundChannel:
    private: ShotNoise
    shared: GlobalShotNoise | GroupedShotNoise
    heterogeneity: CellDistribution = field(default_factory=CellDistribution)

    def json(self) -> dict[str, Any]:
        return {
            "private": self.private.json(),
            "shared": self.shared.json(),
            "heterogeneity": self.heterogeneity.json(),
        }


@dataclass(frozen=True)
class ConductanceBackground:
    target: Population | str
    excitatory: BackgroundChannel
    inhibitory: BackgroundChannel

    def json(self) -> dict[str, Any]:
        return {
            "target": _id(self.target),
            "excitatory": self.excitatory.json(),
            "inhibitory": self.inhibitory.json(),
        }


@dataclass(frozen=True)
class StructuredPoisson:
    input: Signal | str
    rate_hz: float

    def json(self) -> dict[str, Any]:
        return {
            "kind": "structured_poisson",
            "input": _id(self.input),
            "rate_hz": float(self.rate_hz),
        }


@dataclass(frozen=True)
class CorrelatedPoissonAfferents:
    """Excitatory afferents with shared and population-private components."""

    input_e: Signal | str
    input_i: Signal | str
    shared_rate_hz: float
    e_private_rate_hz: float
    i_private_rate_hz: float

    def json(self) -> dict[str, Any]:
        return {
            "kind": "correlated_poisson_afferents",
            "input_e": _id(self.input_e),
            "input_i": _id(self.input_i),
            "shared_rate_hz": float(self.shared_rate_hz),
            "e_private_rate_hz": float(self.e_private_rate_hz),
            "i_private_rate_hz": float(self.i_private_rate_hz),
        }


@dataclass(frozen=True)
class StationaryRateWeather:
    """Positive stationary slow modulation shared by external input channels."""

    tau_ms: float
    std_fraction: float

    def json(self) -> dict[str, Any]:
        return {
            "kind": "stationary_lognormal",
            "tau_ms": float(self.tau_ms),
            "std_fraction": float(self.std_fraction),
        }


@dataclass(frozen=True)
class TransientAfferentWave:
    """Smooth finite rate multiplier applied to explicit afferent sources."""

    onset_ms: float
    peak_ms: float
    offset_ms: float
    peak_scale: float
    shared_peak_scale: float | None = None
    plateau_end_ms: float | None = None

    def json(self) -> dict[str, Any]:
        return {
            "kind": "smooth_transient",
            "onset_ms": float(self.onset_ms),
            "peak_ms": float(self.peak_ms),
            "plateau_end_ms": float(
                self.peak_ms if self.plateau_end_ms is None else self.plateau_end_ms
            ),
            "offset_ms": float(self.offset_ms),
            "baseline_scale": 1.0,
            "peak_scale": float(self.peak_scale),
            "shared_peak_scale": float(
                self.peak_scale
                if self.shared_peak_scale is None
                else self.shared_peak_scale
            ),
        }


@dataclass(frozen=True)
class ConductanceSchedule:
    targets: Sequence[Population | str]
    start_ms: float
    end_ms: float
    start_scale: float = 1.0
    end_scale: float = 1.0
    shape: str = "smoothstep"

    def json(self) -> dict[str, Any]:
        return {
            "kind": "conductance_schedule",
            "targets": [_id(target) for target in self.targets],
            "shape": self.shape,
            "start_ms": float(self.start_ms),
            "end_ms": float(self.end_ms),
            "start_scale": float(self.start_scale),
            "end_scale": float(self.end_scale),
        }


@dataclass(frozen=True)
class SimulationSpec:
    spike_sources: Sequence[StructuredPoisson | CorrelatedPoissonAfferents] = ()
    backgrounds: Sequence[ConductanceBackground] = ()
    modulation: Sequence[ConductanceSchedule] = ()
    weather: StationaryRateWeather | None = None
    afferent_wave: TransientAfferentWave | None = None


def simulation_dict(spec: SimulationSpec, graph_digest: str) -> dict[str, Any]:
    return {
        "schema": "snnlang.simulation/v1",
        "graph_digest": graph_digest,
        "seed_derivation": "request_seed+stable_channel_index",
        "spike_sources": [source.json() for source in spec.spike_sources],
        "backgrounds": [background.json() for background in spec.backgrounds],
        "modulation": [schedule.json() for schedule in spec.modulation],
        "weather": spec.weather.json() if spec.weather is not None else None,
        "afferent_wave": (
            spec.afferent_wave.json() if spec.afferent_wave is not None else None
        ),
    }


def validate_simulation(graph: dict[str, Any], recipe: dict[str, Any]) -> None:
    if recipe.get("schema") != "snnlang.simulation/v1":
        raise ValueError("unsupported simulation schema")
    input_ids = {row["id"] + ".value" for row in graph.get("inputs", [])}
    population_ids = {
        row["id"] for row in graph.get("populations", []) if row.get("spiking")
    }
    for source in recipe.get("spike_sources", []):
        kind = source.get("kind")
        if kind == "structured_poisson" and (
            source.get("input") not in input_ids or float(source.get("rate_hz", 0)) < 0
        ):
            raise ValueError(
                "simulation spike source must reference an input and use rate_hz >= 0"
            )
        if kind == "correlated_poisson_afferents" and (
            source.get("input_e") not in input_ids
            or source.get("input_i") not in input_ids
            or source.get("input_e") == source.get("input_i")
            or any(
                float(source.get(key, -1)) < 0
                for key in (
                    "shared_rate_hz",
                    "e_private_rate_hz",
                    "i_private_rate_hz",
                )
            )
        ):
            raise ValueError(
                "correlated afferents require distinct E/I inputs and non-negative rates"
            )
        if kind not in {"structured_poisson", "correlated_poisson_afferents"}:
            raise ValueError(f"unsupported simulation spike source: {kind!r}")
    targets = []
    for background in recipe.get("backgrounds", []):
        target = background.get("target")
        targets.append(target)
        if target not in population_ids:
            raise ValueError("conductance background must target a spiking population")
        for polarity in ("excitatory", "inhibitory"):
            channel = background.get(polarity, {})
            for ownership, expected in (
                ("private", {"shot_noise"}),
                ("shared", {"global_shot_noise", "grouped_shot_noise"}),
            ):
                noise = channel.get(ownership, {})
                if noise.get("kind") not in expected:
                    raise ValueError(
                        f"{polarity} background requires one valid {ownership} stream"
                    )
                if (
                    any(
                        float(noise.get(key, 0)) < 0 for key in ("rate_hz", "amplitude")
                    )
                    or float(noise.get("tau_ms", 0)) <= 0
                ):
                    raise ValueError(
                        "background rates/amplitudes must be non-negative and tau_ms positive"
                    )
                if (
                    noise.get("kind") == "grouped_shot_noise"
                    and int(noise.get("group_size", 0)) <= 0
                ):
                    raise ValueError("grouped shot noise requires group_size > 0")
    if len(targets) != len(set(targets)):
        raise ValueError("simulation may declare only one background per population")
    for schedule in recipe.get("modulation", []):
        if (
            schedule.get("shape") != "smoothstep"
            or float(schedule.get("start_ms", -1)) < 0
            or float(schedule.get("end_ms", 0)) <= float(schedule.get("start_ms", -1))
        ):
            raise ValueError(
                "conductance schedule requires smoothstep and 0 <= start_ms < end_ms"
            )
        if not set(schedule.get("targets", ())) <= population_ids:
            raise ValueError("conductance schedule references an unknown population")
    weather = recipe.get("weather")
    if weather is not None and (
        weather.get("kind") != "stationary_lognormal"
        or float(weather.get("tau_ms", 0)) <= 0
        or float(weather.get("std_fraction", -1)) < 0
    ):
        raise ValueError("stationary weather requires tau_ms > 0 and std_fraction >= 0")
    wave = recipe.get("afferent_wave")
    if wave is not None and (
        wave.get("kind") != "smooth_transient"
        or float(wave.get("onset_ms", -1)) < 0
        or not float(wave.get("onset_ms", -1))
        < float(wave.get("peak_ms", -1))
        <= float(wave.get("plateau_end_ms", wave.get("peak_ms", -1)))
        < float(wave.get("offset_ms", -1))
        or float(wave.get("baseline_scale", 0)) != 1.0
        or float(wave.get("peak_scale", 0)) < 1.0
        or float(wave.get("shared_peak_scale", wave.get("peak_scale", 0))) < 1.0
    ):
        raise ValueError(
            "afferent wave requires 0 <= onset < peak <= plateau_end < offset "
            "and peak scales >= 1"
        )
