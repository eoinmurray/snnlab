"""Explicit weights-only transfer, independent of optimizer resume."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from snnlab.sim.training.checkpoints import TrainingCheckpoint, load_training_checkpoint

if TYPE_CHECKING:
    from snnlab.sim.artifacts.network_snapshot import NetworkSnapshot


@dataclass(frozen=True)
class WeightInitialization:
    """Copy effective weights in runtime (fan-in, target) orientation.

    ``mapping`` maps destination names to source names. Unmapped same-name
    parameters transfer automatically; ``new_parameters`` explicitly keeps fresh
    graph initialization. Extra source parameters require ``unused='ignore'``.
    Native checkpoint directories and NetworkSnapshot objects are supported.
    """

    source: (
        Mapping[str, torch.Tensor] | TrainingCheckpoint | str | Path | NetworkSnapshot
    )
    mapping: Mapping[str, str] = field(default_factory=dict)
    new_parameters: tuple[str, ...] = ()
    unused: str = "error"


def initialize_weights(model, policy):
    from snnlab.sim.artifacts.network_snapshot import NetworkSnapshot

    if not isinstance(policy, WeightInitialization):
        raise TypeError("weights must be WeightInitialization")
    if policy.unused not in {"error", "ignore"}:
        raise ValueError("unused weights policy must be error or ignore")
    source = policy.source
    if isinstance(source, (str, Path)):
        source = load_training_checkpoint(source)
    if isinstance(source, TrainingCheckpoint):
        source = source.parameters
    elif isinstance(source, NetworkSnapshot):
        source = source.load_parameters()
    if not isinstance(source, Mapping):
        raise TypeError(
            "weights source must be named tensors, checkpoint or NetworkSnapshot"
        )
    parameters = model.parameter_map()
    names, fresh = _weight_names(parameters, policy, source)
    copied, digests = {}, {}
    for name, source_name in names.items():
        value, parameter = source[source_name], parameters[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != parameter.shape
            or value.dtype != parameter.dtype
        ):
            raise ValueError(
                f"weight {name} shape/dtype mismatch; runtime orientation is (fan-in, target)"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"weight {name} must be finite")
        copied[name] = value.detach().to(parameter.device).clone()
        digests[source_name] = hashlib.sha256(
            value.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
    previous = {name: value.detach().clone() for name, value in parameters.items()}
    try:
        with torch.no_grad():
            for name, value in copied.items():
                parameters[name].copy_(value)
        model.enforce_constraints()
        if any(
            not torch.equal(parameters[name], value) for name, value in copied.items()
        ):
            raise ValueError(
                "transferred weights violate graph constraints or connectivity"
            )
    except Exception:
        with torch.no_grad():
            for name, value in previous.items():
                parameters[name].copy_(value)
        raise
    return {
        "schema": "snnlab.weight-initialization/v1",
        "mapping": names,
        "new_parameters": sorted(fresh),
        "unused": policy.unused,
        "source_tensor_sha256": digests,
        "orientation": "fan_in_target",
        "scaling": "effective_no_rescaling",
    }


def _weight_names(parameters, policy, source):
    fresh = set(policy.new_parameters)
    if len(fresh) != len(policy.new_parameters) or fresh - parameters.keys():
        raise ValueError("new_parameters must name unique graph parameters")
    if set(policy.mapping) - parameters.keys() or fresh & policy.mapping.keys():
        raise ValueError("invalid or ambiguous weight mapping destinations")
    names = {
        name: policy.mapping.get(name, name) for name in parameters if name not in fresh
    }
    if len(set(names.values())) != len(names):
        raise ValueError("ambiguous weight mapping sources")
    missing = set(names.values()) - source.keys()
    extra = source.keys() - set(names.values())
    if missing or (extra and policy.unused == "error"):
        raise ValueError(
            f"weight source names mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return names, fresh
