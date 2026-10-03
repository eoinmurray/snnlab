"""Deterministically realize authenticated SNNLang simulation recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class RealizedInputs:
    input_spikes: torch.Tensor | None
    input_spikes_i: torch.Tensor | None
    excitatory_e: torch.Tensor | None
    inhibitory_e: torch.Tensor | None
    excitatory_i: torch.Tensor | None
    inhibitory_i: torch.Tensor | None
    retained: dict[str, np.ndarray]


def _distribution(
    spec: dict[str, Any], size: int, generator: torch.Generator
) -> torch.Tensor:
    kind = spec.get("kind")
    if kind == "constant":
        values = torch.full((size,), float(spec.get("value", 1.0)))
    elif kind in {"normal", "lower_clamped_normal"}:
        values = torch.normal(
            float(spec["mean"]), float(spec["std"]), (size,), generator=generator
        )
        if kind == "lower_clamped_normal":
            values.clamp_(min=0)
    elif kind == "uniform":
        values = torch.empty(size).uniform_(
            float(spec["low"]), float(spec["high"]), generator=generator
        )
    else:
        raise ValueError(f"unsupported cell distribution: {kind!r}")
    return values.clamp(min=0)


def _smoothstep(
    recipe: dict[str, Any], target: str, t_steps: int, dt: float
) -> torch.Tensor:
    scale = torch.ones(t_steps)
    for schedule in recipe.get("modulation", []):
        if target not in schedule.get("targets", []):
            continue
        time = torch.arange(t_steps, dtype=torch.float32) * float(dt)
        u = (
            (time - float(schedule["start_ms"]))
            / (float(schedule["end_ms"]) - float(schedule["start_ms"]))
        ).clamp(0, 1)
        alpha = u * u * (3 - 2 * u)
        scale *= float(schedule.get("start_scale", 1.0)) + alpha * (
            float(schedule.get("end_scale", 1.0))
            - float(schedule.get("start_scale", 1.0))
        )
    return scale


def _stationary_weather(
    recipe: dict[str, Any], *, seed: int, dt: float, t_steps: int
) -> torch.Tensor:
    spec = recipe.get("weather")
    if spec is None or float(spec.get("std_fraction", 0)) == 0:
        return torch.ones(t_steps)
    tau_ms = float(spec["tau_ms"])
    sigma = float(spec["std_fraction"])
    alpha = float(np.exp(-float(dt) / tau_ms))
    innovation = float(np.sqrt(1.0 - alpha * alpha))
    generator = torch.Generator().manual_seed(int(seed) + 10_000)
    state = torch.randn(1, generator=generator)[0]
    values = torch.empty(t_steps)
    for step in range(t_steps):
        if step:
            state = alpha * state + innovation * torch.randn(1, generator=generator)[0]
        values[step] = torch.exp(sigma * state - 0.5 * sigma * sigma)
    return values


def _afferent_wave(
    recipe: dict[str, Any], *, dt: float, t_steps: int, scale_key: str = "peak_scale"
) -> torch.Tensor:
    spec = recipe.get("afferent_wave")
    if spec is None:
        return torch.ones(t_steps)
    time = torch.arange(t_steps, dtype=torch.float32) * float(dt)
    onset = float(spec["onset_ms"])
    peak = float(spec["peak_ms"])
    plateau_end = float(spec.get("plateau_end_ms", peak))
    offset = float(spec["offset_ms"])
    peak_scale = float(spec.get(scale_key, spec["peak_scale"]))
    rise_u = ((time - onset) / (peak - onset)).clamp(0, 1)
    fall_u = ((time - plateau_end) / (offset - plateau_end)).clamp(0, 1)
    rise = rise_u * rise_u * (3 - 2 * rise_u)
    fall = fall_u * fall_u * (3 - 2 * fall_u)
    scale = 1 + (peak_scale - 1) * rise
    scale[time >= peak] = peak_scale
    scale *= 1 - (1 - 1 / peak_scale) * fall
    scale[time >= offset] = 1
    return scale


def _poisson_events(
    rate_hz: float,
    *,
    weather: torch.Tensor,
    dt: float,
    size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    probability = (float(rate_hz) * float(dt) / 1000.0 * weather).clamp(max=1)
    return (
        torch.rand(len(weather), size, generator=generator) < probability.unsqueeze(1)
    ).float()


def realize_simulation_inputs(
    recipe: dict[str, Any],
    *,
    seed: int,
    dt: float,
    t_steps: int,
    e_id: str,
    i_id: str,
    n_e: int,
    n_i: int,
    input_size: int | None = None,
) -> RealizedInputs:
    """Generate mutually independent private/shared streams with stable seeds."""
    retained: dict[str, np.ndarray] = {}
    weather = _stationary_weather(recipe, seed=seed, dt=dt, t_steps=t_steps)
    retained["input_weather_scale"] = weather.numpy()
    afferent_wave = _afferent_wave(recipe, dt=dt, t_steps=t_steps)
    shared_afferent_wave = _afferent_wave(
        recipe, dt=dt, t_steps=t_steps, scale_key="shared_peak_scale"
    )
    retained["input_afferent_scale"] = afferent_wave.numpy()
    retained["input_afferent_shared_scale"] = shared_afferent_wave.numpy()
    afferent_weather = weather * afferent_wave
    shared_afferent_weather = weather * shared_afferent_wave
    spike_sources = recipe.get("spike_sources", [])
    if len(spike_sources) > 1:
        raise ValueError("legacy execution supports one structured spike source")
    input_spikes = None
    input_spikes_i = None
    if spike_sources:
        source = spike_sources[0]
        source_size = n_e if input_size is None else int(input_size)
        if source.get("kind") == "correlated_poisson_afferents":
            shared = _poisson_events(
                source["shared_rate_hz"],
                weather=shared_afferent_weather,
                dt=dt,
                size=source_size,
                generator=torch.Generator().manual_seed(int(seed)),
            )
            e_private = _poisson_events(
                source["e_private_rate_hz"],
                weather=afferent_weather,
                dt=dt,
                size=source_size,
                generator=torch.Generator().manual_seed(int(seed) + 1),
            )
            i_private = _poisson_events(
                source["i_private_rate_hz"],
                weather=afferent_weather,
                dt=dt,
                size=source_size,
                generator=torch.Generator().manual_seed(int(seed) + 2),
            )
            input_spikes = torch.maximum(shared, e_private)
            input_spikes_i = torch.maximum(shared, i_private)
            retained["input_afferent_shared"] = shared.numpy()
            retained["input_afferent_e_private"] = e_private.numpy()
            retained["input_afferent_i_private"] = i_private.numpy()
            retained["input_structured_spikes_e"] = input_spikes.numpy()
            retained["input_structured_spikes_i"] = input_spikes_i.numpy()
        else:
            input_spikes = _poisson_events(
                source["rate_hz"],
                weather=afferent_weather,
                dt=dt,
                size=source_size,
                generator=torch.Generator().manual_seed(int(seed)),
            )
            retained["input_structured_spikes"] = input_spikes.numpy()

    resolved: dict[tuple[str, str], torch.Tensor] = {}
    channel_index = 100
    sizes = {e_id: n_e, i_id: n_i}
    names = {e_id: "e", i_id: "i"}
    for background in recipe.get("backgrounds", []):
        target = background["target"]
        if target not in sizes:
            raise ValueError(
                f"simulation background targets unsupported population {target!r}"
            )
        size = sizes[target]
        modulation = _smoothstep(recipe, target, t_steps, dt)
        retained[f"input_modulation_{names[target]}"] = modulation.numpy()
        for polarity in ("excitatory", "inhibitory"):
            channel = background[polarity]
            heterogeneity = channel["heterogeneity"]
            private_gen = torch.Generator().manual_seed(int(seed) + channel_index)
            channel_index += 1
            shared_gen = torch.Generator().manual_seed(int(seed) + channel_index)
            channel_index += 1
            rate_scale = _distribution(heterogeneity["rate"], size, private_gen)
            amplitude_scale = _distribution(
                heterogeneity["amplitude"], size, private_gen
            )
            private = channel["private"]
            private_p = (
                float(private["rate_hz"])
                * float(dt)
                / 1000.0
                * weather.unsqueeze(1)
                * rate_scale.unsqueeze(0)
            ).clamp(max=1)
            private_events = (
                torch.rand(t_steps, size, generator=private_gen) < private_p
            ).float() * (float(private["amplitude"]) * amplitude_scale).unsqueeze(0)
            shared = channel["shared"]
            shared_p = (float(shared["rate_hz"]) * float(dt) / 1000.0 * weather).clamp(
                max=1
            )
            if shared.get("kind") == "grouped_shot_noise":
                group_size = int(shared["group_size"])
                groups = (size + group_size - 1) // group_size
                group_events = (
                    torch.rand(t_steps, groups, generator=shared_gen)
                    < shared_p.unsqueeze(1)
                ).float() * float(shared["amplitude"])
                shared_cells = group_events.repeat_interleave(group_size, dim=1)[
                    :, :size
                ]
            else:
                shared_events = (
                    torch.rand(t_steps, 1, generator=shared_gen) < shared_p.unsqueeze(1)
                ).float() * float(shared["amplitude"])
                shared_cells = shared_events.expand(-1, size)
            total = private_events + shared_cells
            executed = total * modulation.unsqueeze(1)
            stem = f"input_{polarity}_{names[target]}"
            retained[f"{stem}_private"] = private_events.numpy()
            retained[f"{stem}_shared"] = shared_cells.numpy()
            retained[f"{stem}_total"] = total.numpy()
            retained[f"{stem}_executed"] = executed.numpy()
            retained[f"{stem}_rate_scale"] = rate_scale.numpy()
            retained[f"{stem}_amplitude_scale"] = amplitude_scale.numpy()
            resolved[(target, polarity)] = executed

    return RealizedInputs(
        input_spikes=input_spikes,
        input_spikes_i=input_spikes_i,
        excitatory_e=resolved.get((e_id, "excitatory")),
        inhibitory_e=resolved.get((e_id, "inhibitory")),
        excitatory_i=resolved.get((i_id, "excitatory")),
        inhibitory_i=resolved.get((i_id, "inhibitory")),
        retained=retained,
    )
