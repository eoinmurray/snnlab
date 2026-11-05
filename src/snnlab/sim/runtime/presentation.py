"""Partial synaptic initialization for fresh, scored presentations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch

from snnlab.sim.graph.validation import check_condition
from snnlab.sim.runtime.state import (
    GraphRuntimeState,
    load_runtime_state,
    save_runtime_state,
)


@dataclass(frozen=True)
class PresentationState:
    """Selected projection states and chronological pending input queues.

    Projection tensors use (batch, target cells); input queues use
    (delay steps, batch, source channels), oldest pending event first.
    Unspecified state starts fresh. No neuron, time or reduction state carries.
    Use the executor's ``runtime_state_signature`` for ``signature``.
    """

    signature: str
    conductances: Mapping[str, torch.Tensor] = field(default_factory=dict)
    currents: Mapping[str, torch.Tensor] = field(default_factory=dict)
    input_histories: Mapping[str, torch.Tensor] = field(default_factory=dict)
    provenance: Mapping = field(default_factory=dict)

    def to(self, device):
        return PresentationState(
            self.signature,
            **{
                group: {
                    name: value.detach().to(device).clone()
                    for name, value in getattr(self, group).items()
                }
                for group in ("conductances", "currents", "input_histories")
            },
            provenance=dict(self.provenance),
        )

    def save(self, path: str | Path) -> Path:
        """Retain representative synaptic history using authenticated native tensors."""
        state = GraphRuntimeState(
            signature=self.signature,
            compatibility={
                "presentation_state": 1,
                "provenance": dict(self.provenance),
            },
            completed_steps=0,
            voltages={},
            refractory={},
            population_histories={},
            conductances=dict(self.conductances),
            currents=dict(self.currents),
            input_histories=dict(self.input_histories),
        )
        return save_runtime_state(path, state)

    @classmethod
    def load(cls, path: str | Path, *, device="cpu"):
        state = load_runtime_state(path, device=device)
        if state.compatibility.get("presentation_state") != 1 or state.completed_steps:
            raise ValueError("artifact is not a fresh presentation state")
        return cls(
            state.signature,
            state.conductances,
            state.currents,
            state.input_histories,
            state.compatibility.get("provenance", {}),
        )


def apply_presentation_state(forward, runtime, state):
    """Validate all partial tensors before installing detached private copies."""
    from snnlab import extensions as E

    if not isinstance(state, PresentationState):
        raise TypeError("initial_state must be PresentationState")
    if forward.options.runtime_state is not None:
        raise ValueError("initial_state cannot be combined with runtime continuation")
    if state.signature != forward.expected_signature:
        raise ValueError("presentation state graph signature mismatch")
    rows = (
        forward.projection_rows
        if hasattr(forward, "projection_rows")
        else {row["id"]: row for row in forward.executor.plan.graph["projections"]}
    )
    updates = []
    for group, unit in (("conductances", "uS"), ("currents", "nA")):
        expected = {
            p.id: (p.id, p.polarity)
            for p in forward.executor.plan.projections
            if E.synapse_unit(rows[p.id]["synapse"]) == unit
        }
        for name, value in getattr(state, group).items():
            if name not in expected:
                raise ValueError(f"unknown presentation {group} projection {name}")
            if rows[name]["synapse"]["kind"] == "custom_synapse":
                raise ValueError(
                    "presentation initialization of custom synapses is unsupported"
                )
            key = expected[name]
            updates.append(
                (
                    runtime.conductance,
                    key,
                    value,
                    runtime.conductance[key],
                    group == "conductances",
                )
            )
    for name, value in state.input_histories.items():
        if name not in runtime.input_histories:
            raise ValueError(f"unknown or undelayed presentation input {name}")
        updates.append(
            (runtime.input_histories, name, value, runtime.input_histories[name], True)
        )
    _validate_values(updates)
    input_rows = {row["id"]: row for row in forward.executor.plan.graph["inputs"]}
    for name, value in state.input_histories.items():
        if input_rows[name].get("signal_type") == "spikes":
            check_condition(
                ((value == 0) | (value == 1)).all(),
                f"presentation input queue {name} requires binary spikes",
            )
    for destination, key, value, _, _ in updates:
        destination[key] = value.detach().clone()


def _validate_values(updates):
    for _, key, value, prototype, nonnegative in updates:
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != prototype.shape
            or value.dtype != prototype.dtype
            or value.device != prototype.device
        ):
            raise ValueError(f"presentation state {key} shape/dtype/device mismatch")
        valid = torch.isfinite(value).all()
        if nonnegative:
            valid = valid & (value >= 0).all()
        check_condition(
            valid, f"presentation state {key} must be finite and physically valid"
        )
