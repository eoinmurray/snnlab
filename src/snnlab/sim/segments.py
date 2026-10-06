"""Absolute boundary schedules, selective voltage resets and online decisions."""

import math
from dataclasses import dataclass
from typing import Sequence

import torch


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class Boundary:
    step: int
    batches: tuple[int, ...] | None = None


@dataclass(frozen=True)
class BoundarySchedule:
    steps: tuple[int, ...] = ()
    batches: tuple[int, ...] | None = None
    events: tuple[Boundary, ...] = ()

    def resolved(self, batch_size):
        if self.events and (self.steps or self.batches is not None):
            raise ValueError("schedule uses either steps/batches or events")
        rows = self.events or tuple(Boundary(step, self.batches) for step in self.steps)
        if not rows:
            raise ValueError("boundary schedule must not be empty")
        result = [[] for _ in range(batch_size)]
        for row in rows:
            if not isinstance(row, Boundary):
                raise TypeError("events must contain Boundary objects")
            step = _integer(row.step, "boundary step")
            indices = (
                tuple(range(batch_size)) if row.batches is None else tuple(row.batches)
            )
            if not indices or len(set(indices)) != len(indices):
                raise ValueError("boundary batches must be non-empty and unique")
            for index in indices:
                _integer(index, "batch index")
                if index >= batch_size:
                    raise ValueError(
                        "boundary batch index exceeds execution batch size"
                    )
                if result[index] and step <= result[index][-1]:
                    raise ValueError(
                        "boundaries must be strictly increasing for each batch"
                    )
                result[index].append(step)
        return result

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {"steps", "batches", "events"}:
            raise ValueError("invalid boundary schedule fields")
        events = []
        for row in value.get("events", ()):
            if (
                not isinstance(row, dict)
                or set(row) - {"step", "batches"}
                or "step" not in row
            ):
                raise ValueError("invalid boundary event")
            events.append(Boundary(row["step"], row.get("batches")))
        return cls(value.get("steps", ()), value.get("batches"), tuple(events))


@dataclass(frozen=True)
class ResetVoltage:
    population_id: str
    boundaries: BoundarySchedule
    value_mv: float = 0.0

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {
            "population_id",
            "boundaries",
            "value_mv",
        }:
            raise ValueError("invalid reset fields")
        return cls(
            value["population_id"],
            BoundarySchedule.from_dict(value["boundaries"]),
            value.get("value_mv", 0.0),
        )


@dataclass(frozen=True)
class DecisionSegments:
    boundaries: BoundarySchedule
    end_step: int
    population_totals: tuple[str, ...] = ()
    output_spike_counts: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {
            "boundaries",
            "end_step",
            "population_totals",
            "output_spike_counts",
        }:
            raise ValueError("invalid decision fields")
        return cls(
            BoundarySchedule.from_dict(value["boundaries"]),
            value["end_step"],
            tuple(value.get("population_totals", ())),
            tuple(value.get("output_spike_counts", ())),
        )


def segment_identity(
    resets: Sequence[ResetVoltage],
    decisions: DecisionSegments | None,
    *,
    graph,
    batch_size,
):
    """Validate and resolve the full policy before execution or cache reuse."""
    _integer(batch_size, "batch_size")
    if batch_size == 0:
        raise ValueError("batch_size must be positive")
    populations = {row["id"]: row for row in graph["populations"]}
    resolved = []
    occupied = set()
    for reset in resets:
        if not isinstance(reset, ResetVoltage):
            raise TypeError("resets must contain ResetVoltage objects")
        if (
            reset.population_id not in populations
            or populations[reset.population_id]["neuron"]["kind"] != "leaky_integrator"
        ):
            raise ValueError("ResetVoltage requires a leaky-integrator population")
        if isinstance(reset.value_mv, bool) or not math.isfinite(float(reset.value_mv)):
            raise ValueError("reset value_mv must be finite")
        if not isinstance(reset.boundaries, BoundarySchedule):
            raise TypeError("reset boundaries must be BoundarySchedule")
        rows = reset.boundaries.resolved(batch_size)
        for index, steps in enumerate(rows):
            for step in steps:
                key = (reset.population_id, index, step)
                if key in occupied:
                    raise ValueError("overlapping voltage resets are ambiguous")
                occupied.add(key)
        resolved.append(
            {
                "population_id": reset.population_id,
                "value_mv": float(reset.value_mv),
                "steps_by_batch": rows,
            }
        )
    policy = None
    if decisions is not None:
        if not isinstance(decisions, DecisionSegments):
            raise TypeError("decisions must be DecisionSegments")
        end = _integer(decisions.end_step, "end_step")
        if not isinstance(decisions.boundaries, BoundarySchedule):
            raise TypeError("decision boundaries must be BoundarySchedule")
        rows = decisions.boundaries.resolved(batch_size)
        if any(not steps or steps[-1] >= end for steps in rows):
            raise ValueError(
                "each decision batch needs a start boundary before end_step"
            )
        for names in (decisions.population_totals, decisions.output_spike_counts):
            if len(set(names)) != len(names):
                raise ValueError("decision population names must be unique")
            for name in names:
                if name not in populations or not populations[name].get(
                    "spiking", False
                ):
                    raise ValueError(
                        f"decision counts require a spiking population: {name}"
                    )
        if not decisions.population_totals and not decisions.output_spike_counts:
            raise ValueError("decisions require at least one count population")
        policy = {
            "steps_by_batch": rows,
            "end_step": end,
            "population_totals": list(decisions.population_totals),
            "output_spike_counts": list(decisions.output_spike_counts),
        }
    if not resets and decisions is None:
        return None
    return {
        "schema": "snnlab.decision-segments/v1",
        "timing": "close-before-reset-before-update",
        "batch_size": batch_size,
        "resets": resolved,
        "decisions": policy,
    }


class SegmentAccumulator:
    def __init__(self, identity, populations, batch, device, saved, counts, start_step):
        self.identity = identity
        self.policy = identity["decisions"] if identity else None
        self.boundary_batches = {}
        self.reset_events = {}
        if self.policy:
            for index, steps in enumerate(self.policy["steps_by_batch"]):
                for step in (*steps, self.policy["end_step"]):
                    self.boundary_batches.setdefault(step, []).append(index)
        for reset in identity["resets"] if identity else ():
            by_step = {}
            for index, steps in enumerate(reset["steps_by_batch"]):
                for step in steps:
                    by_step.setdefault(step, []).append(index)
            for step, selected in by_step.items():
                self.reset_events.setdefault(step, []).append(
                    (reset["population_id"], reset["value_mv"], selected)
                )
        self.records = []
        self.counts = {}
        self.starts = [None] * batch
        self.last_boundary = -1
        if saved:
            if saved.get("identity") != identity:
                raise ValueError("runtime reset/decision policy does not match request")
            self.starts = list(saved["starts"])
            self.last_boundary = saved["last_boundary"]
            if (
                len(self.starts) != batch
                or type(self.last_boundary) is not int
                or self.last_boundary > start_step
            ):
                raise ValueError("invalid runtime decision cursor")
        elif identity and start_step:
            raise ValueError(
                "cannot attach reset/decision policy to an unaudited continuation"
            )
        if self.policy:
            if saved:
                expected_starts = [
                    max((step for step in steps if step <= start_step), default=None)
                    if start_step < self.policy["end_step"]
                    else None
                    for steps in self.policy["steps_by_batch"]
                ]
                if self.last_boundary != start_step or self.starts != expected_starts:
                    raise ValueError("invalid runtime decision cursor")
            names = set(
                self.policy["population_totals"] + self.policy["output_spike_counts"]
            )
            if saved and set(counts) != names:
                raise ValueError("runtime decision count population mismatch")
            for name in names:
                shape = (batch, populations[name]["size"])
                value = (
                    counts[name]
                    if saved
                    else torch.zeros(shape, dtype=torch.int64, device=device)
                )
                if (
                    tuple(value.shape) != shape
                    or value.dtype != torch.int64
                    or value.device != device
                    or bool((value < 0).any())
                ):
                    raise ValueError("invalid runtime decision counts")
                if saved and any(
                    start is None and bool(value[index].any())
                    for index, start in enumerate(self.starts)
                ):
                    raise ValueError("inactive runtime decision counts must be zero")
                self.counts[name] = value.detach().clone()

    def boundary(self, step, dt_ms):
        if not self.policy or step <= self.last_boundary:
            return
        for batch in self.boundary_batches.get(step, ()):
            start = self.starts[batch]
            if start is not None:
                self.records.append(
                    {
                        "batch": batch,
                        "start_step": start,
                        "end_step": step,
                        "duration_ms": (step - start) * dt_ms,
                        "population_totals": {
                            name: int(self.counts[name][batch].sum())
                            for name in self.policy["population_totals"]
                        },
                        "output_spike_counts": {
                            name: self.counts[name][batch].tolist()
                            for name in self.policy["output_spike_counts"]
                        },
                    }
                )
            for value in self.counts.values():
                value[batch].zero_()
            self.starts[batch] = None if step == self.policy["end_step"] else step
        self.last_boundary = step

    def reset(self, step, voltage):
        for name, value_mv, selected in self.reset_events.get(step, ()):
            value = voltage[name].clone()
            value[selected] = value_mv
            voltage[name] = value

    def observe(self, spikes):
        for name, counts in self.counts.items():
            for batch, start in enumerate(self.starts):
                if start is not None:
                    counts[batch] += spikes[name][batch].detach().to(torch.int64)

    def state(self):
        return (
            {
                "identity": self.identity,
                "starts": self.starts,
                "last_boundary": self.last_boundary,
            }
            if self.identity
            else {}
        )
