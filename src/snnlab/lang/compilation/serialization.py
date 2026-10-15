"""Canonical graph and training serialization and digests."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from snnlab._shared import neurons as N
from snnlab.lang.network import Network
from snnlab.lang.training import TrainSpec

SCHEMA = "snnlang.graph/v1"


TRAINING_SCHEMA = "snnlang.training/v1"


def canonical_json(data: Any) -> bytes:
    return (
        json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def digest(data: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(data)).hexdigest()


def graph_dict(net: Network) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "voltage_sampling": "explicit",
        "refractory_timing": N.REFRACTORY_TIMING,
        "name": net.name,
        "timebase": {"dt": net.dt.json()},
        "inputs": sorted(net.inputs, key=lambda x: x["id"]),
        "populations": sorted(net.populations, key=lambda x: x["id"]),
        "projections": sorted(net.projections, key=lambda x: x["id"]),
        "operations": sorted(net.operations, key=lambda x: x["id"]),
        "parameters": sorted(net.parameters, key=lambda x: x["id"]),
        "constants": sorted(net.constants, key=lambda x: x["id"]),
        "outputs": sorted(net.outputs, key=lambda x: x["id"]),
        "observables": sorted(net.observables, key=lambda x: x["id"]),
        "assets": sorted(net.assets, key=lambda x: x["id"]),
        "groups": [
            {"id": g.name, "members": sorted(g.members), "parent": g.parent}
            for g in sorted(net.groups.values(), key=lambda x: x.name)
        ],
    }


def _training_dict(
    spec: TrainSpec, graph_digest: str, graph: Mapping[str, Any]
) -> dict[str, Any]:
    groups = [
        {
            "id": g.name,
            "parameters": sorted(g.ids()),
            "lr": g.lr,
            "frozen": g.frozen,
        }
        for g in spec.parameter_groups
    ]
    return {
        "schema": TRAINING_SCHEMA,
        "graph_digest": graph_digest,
        "objectives": [
            {
                "kind": o.kind,
                "prediction": o.prediction,
                "target": o.target,
                "weight": o.weight,
                **(
                    {"definition": o.definition, "config": o.config}
                    if o.definition
                    else {}
                ),
            }
            for o in spec.objectives
        ],
        "parameter_groups": groups,
        "resolved_parameters": {
            "trainable": sorted(
                parameter
                for group in groups
                if not group["frozen"]
                for parameter in group["parameters"]
            ),
            "frozen": sorted(
                parameter
                for group in groups
                if group["frozen"]
                for parameter in group["parameters"]
            ),
            "learning_rates": {
                parameter: group["lr"]
                for group in sorted(groups, key=lambda row: row["id"])
                if not group["frozen"]
                for parameter in group["parameters"]
            },
        },
        "regularizers": [
            {
                "kind": r.kind,
                "signals": sorted(r.signals),
                "strength": r.strength,
                "config": r.config,
                **({"definition": r.definition} if r.definition else {}),
            }
            for r in spec.regularizers
        ],
        "stop_gradients": sorted(s.signal for s in spec.stop_gradients),
        "optimizer": {
            "kind": spec.optimizer.kind,
            "config": spec.optimizer.config,
            **(
                {"definition": spec.optimizer.definition}
                if spec.optimizer.definition
                else {}
            ),
        },
        "epochs": spec.epochs,
        "gradient_clip": spec.gradient_clip,
        "surrogate": spec.surrogate.json() if spec.surrogate else None,
        "presentation_duration": (
            spec.presentation_duration.json() if spec.presentation_duration else None
        ),
        "resolved_gradients": {
            "surrogate": spec.surrogate.json() if spec.surrogate else None,
            "voltage_gradient_dampening": {
                population["id"]: population["neuron"].get("voltage_grad_dampen", 1.0)
                for population in graph.get("populations", [])
                if population.get("spiking")
            },
        },
    }
