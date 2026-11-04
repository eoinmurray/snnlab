"""Coordinated dataset preparation and identity-based random streams."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from snnlab.sim.runtime.presentation import PresentationState


@dataclass(frozen=True)
class BatchContext:
    """Sample identities and explicit, independent preparation RNG domains.

    Evaluation always uses epoch zero. ``fixed`` ignores phase, epoch and draw;
    ``presentation`` changes each training epoch and fixes evaluation draws.
    Streams never consume process-global RNG state.
    """

    sample_indices: tuple[int, ...]
    sample_ids: tuple[int | str, ...]
    phase: str = "evaluation"
    epoch: int = 0
    draw_seed: int = 0
    root_seed: int = 0
    _streams: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def rng_streams(self):
        return tuple(dict(value) for value in self._streams.values())

    def seed(self, sample_id, stream, *, policy="presentation"):
        if sample_id not in self.sample_ids:
            raise ValueError("RNG sample identity is not in this batch")
        if not isinstance(stream, str) or not stream:
            raise ValueError("RNG stream must be a nonempty name")
        if policy not in {"fixed", "presentation"}:
            raise ValueError("RNG policy must be fixed or presentation")
        coordinate = (
            ("fixed", 0, 0)
            if policy == "fixed"
            else (
                self.phase,
                self.epoch if self.phase == "train" else 0,
                self.draw_seed,
            )
        )
        payload = json.dumps(
            ["snnlab.sample-rng/v1", self.root_seed, sample_id, stream, *coordinate],
            separators=(",", ":"),
        ).encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)
        self._streams[(sample_id, stream, policy)] = {
            "sample_id": sample_id,
            "domain": stream,
            "policy": policy,
            "seed": seed,
        }
        return seed

    def generator(self, sample_id, stream, *, policy="presentation", device="cpu"):
        return torch.Generator(device=device).manual_seed(
            self.seed(sample_id, stream, policy=policy)
        )


@dataclass(frozen=True)
class PreparedBatch:
    """A registered encoder's named inputs and optional fresh synaptic state."""

    inputs: Mapping[str, torch.Tensor]
    initial_state: PresentationState | None = None


class BatchInputs(dict):
    """Private tensor mapping retaining batch-local preparation through transfers."""

    def __init__(self, tensors, initial_state=None):
        super().__init__(tensors)
        self.initial_state = initial_state

    def state_tensors(self):
        if self.initial_state is not None:
            for group in ("conductances", "currents", "input_histories"):
                yield from getattr(self.initial_state, group).values()

    def moved(self, device, *, non_blocking=False):
        return BatchInputs(
            {
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.items()
            },
            self.initial_state.to(device) if self.initial_state is not None else None,
        )
