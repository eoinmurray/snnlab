"""Compact, named training observations and isolated evaluation streams."""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from .execution import InputBinding


@dataclass(frozen=True)
class ObservationProbe:
    """One fixed, separately named evaluation draw; no optimizer or targets."""

    input_bindings: Sequence[InputBinding] = field(default_factory=tuple)
    seed: int = 0
    split: Literal["reference", "validation"] = "reference"
    protocol: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class EpochObservations:
    population_rates: Sequence[str] = field(default_factory=tuple)
    parameter_norms: Sequence[str] = field(default_factory=tuple)
    output_activity: Sequence[str] = field(default_factory=tuple)
    gradient_norms: Sequence[str] = field(default_factory=tuple)
    probes: Mapping[str, ObservationProbe] = field(default_factory=dict)

    def validate(self, graph, parameters):
        populations = {row["id"]: row for row in graph["populations"]}
        for field_name in (
            "population_rates",
            "output_activity",
            "parameter_norms",
            "gradient_norms",
        ):
            ids = getattr(self, field_name)
            if (
                isinstance(ids, str)
                or any(not isinstance(name, str) for name in ids)
                or len(ids) != len(set(ids))
            ):
                raise ValueError(f"{field_name} requires unique graph IDs")
            for name in ids:
                if field_name in {"population_rates", "output_activity"}:
                    if name not in populations or not populations[name]["spiking"]:
                        raise ValueError(
                            f"{field_name}: {name!r} must name a spiking population"
                        )
                elif name not in parameters:
                    raise ValueError(f"{field_name}: unknown parameter {name!r}")
                elif (
                    field_name == "gradient_norms"
                    and not parameters[name].requires_grad
                ):
                    raise ValueError(f"gradient_norms: {name!r} is not trainable")
        for name, probe in self.probes.items():
            if (
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(probe, ObservationProbe)
            ):
                raise ValueError(
                    "probes require nonempty names and ObservationProbe values"
                )
            if (
                probe.split not in {"reference", "validation"}
                or isinstance(probe.seed, bool)
                or not isinstance(probe.seed, int)
                or probe.seed < 0
            ):
                raise ValueError(f"{name}: invalid probe split or seed")


@contextmanager
def evaluation_rng(seed=None):
    """Restore Python, NumPy, CPU and available accelerator RNGs on every exit."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def finite_norm(value):
    result = float(torch.linalg.vector_norm(value.detach()))
    if not math.isfinite(result):
        raise ValueError("epoch observation norm is non-finite")
    return result


class ActivityMeasurements:
    def __init__(self, spec, populations, duration_s, sample_count):
        self.spec, self.populations = spec, populations
        self.duration_s, self.sample_count = duration_s, sample_count
        self.counts = {}
        self.silent = {}

    def add(self, counts):
        for name, value in counts.items():
            self.counts[name] = (
                self.counts.get(name, torch.zeros(value.shape[1], dtype=torch.float64))
                + value.sum(dim=0).double().cpu()
            )
            self.silent[name] = self.silent.get(name, 0) + int(
                (value.sum(dim=1) == 0).sum()
            )

    def finish(self):
        rates = {
            name: float(self.counts[name].sum())
            / (self.sample_count * self.populations[name]["size"] * self.duration_s)
            for name in self.spec.population_rates
        }
        activity = {}
        for name in self.spec.output_activity:
            counts = self.counts[name]
            total = float(counts.sum())
            activity[name] = {
                "spike_totals": [int(value) for value in counts.tolist()],
                "silent_sample_fraction": self.silent[name] / self.sample_count,
                "class_spike_shares": (counts / total).tolist() if total else None,
                "status": "ok" if total else "no_spikes",
                "class_axis": "population_cell_index",
            }
        return {
            "population_rates_hz": rates,
            "output_activity": activity,
            "duration_s": self.duration_s,
            "sample_count": self.sample_count,
            "population_sizes": {
                name: self.populations[name]["size"] for name in self.counts
            },
            "aggregation": "total_spikes_per_sample_cell_second",
            "window": "full_presentation",
        }


class GradientMeasurements:
    def __init__(self, names, state=None):
        self.names = tuple(names)
        self.state = state or {
            "updates": 0,
            "counts": {name: 0 for name in names},
            "pre": {name: 0.0 for name in names},
            "post": {name: 0.0 for name in names},
        }

    def add(self, before, parameters):
        self.state["updates"] += 1
        for name in self.names:
            if name in before:
                self.state["counts"][name] += 1
                self.state["pre"][name] += finite_norm(before[name])
                self.state["post"][name] += finite_norm(parameters[name].grad)

    def finish(self):
        return {
            "pre_clip_l2_norm": {
                name: self.state["pre"][name] / self.state["counts"][name]
                if self.state["counts"][name]
                else None
                for name in self.names
            },
            "post_clip_l2_norm": {
                name: self.state["post"][name] / self.state["counts"][name]
                if self.state["counts"][name]
                else None
                for name in self.names
            },
            "missing_updates": {
                name: self.state["updates"] - self.state["counts"][name]
                for name in self.names
            },
            "updates": self.state["updates"],
            "aggregation": "mean_over_epoch_updates_with_gradient",
            "status": "ok" if self.state["updates"] else "no_updates",
        }


def validate_observation_state(state, contract, data_state, completed_updates):
    """Reject incomplete or inconsistent retained audit coordinates."""
    if not isinstance(state, dict) or set(state) != {
        "contract",
        "history",
        "gradients",
    }:
        raise ValueError("malformed checkpoint epoch observation state")
    if state["contract"] != contract:
        raise ValueError(
            "checkpoint epoch observation configuration does not match request"
        )
    history = state["history"]
    epoch, batch = data_state["epoch"], data_state["batch"]
    if not isinstance(history, list) or [row.get("epoch") for row in history] != list(
        range(epoch + 1)
    ):
        raise ValueError("checkpoint epoch observation history is not contiguous")
    updates = [row.get("update") for row in history]
    if (
        updates[0] != 0
        or any(
            not isinstance(value, int) or value < 0 or value > completed_updates
            for value in updates
        )
        or any(a >= b for a, b in zip(updates, updates[1:]))
    ):
        raise ValueError("checkpoint epoch observation update coordinates are invalid")
    gradients = state["gradients"]
    names = set(contract["gradient_norms"])
    if (
        not isinstance(gradients, dict)
        or set(gradients) != {"updates", "counts", "pre", "post"}
        or gradients["updates"] != batch
    ):
        raise ValueError("checkpoint partial-epoch gradient coordinates are invalid")
    for key in ("counts", "pre", "post"):
        if not isinstance(gradients[key], dict) or set(gradients[key]) != names:
            raise ValueError("checkpoint gradient observation coverage is invalid")
    for name in names:
        count = gradients["counts"][name]
        if (
            not isinstance(count, int)
            or not 0 <= count <= batch
            or any(
                not isinstance(gradients[key][name], (int, float))
                or not math.isfinite(gradients[key][name])
                or gradients[key][name] < 0
                for key in ("pre", "post")
            )
        ):
            raise ValueError("checkpoint gradient observation values are invalid")
