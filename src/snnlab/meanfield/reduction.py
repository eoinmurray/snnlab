"""Translate authored networks into a closure's symbolic construction context."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from snnlab.lang.compilation.compile import compile
from snnlab.lang.compilation.serialization import digest
from snnlab.lang.network import Network
from snnlab.meanfield.closures.api import BuildContext
from snnlab.meanfield.model import magnitude


@dataclass(frozen=True)
class StationaryRate:
    """Mean source activity; no Poisson or independence assumption is implied."""

    rate: object

    def __post_init__(self):
        if magnitude(self.rate, "Hz") < 0:
            raise ValueError("source rates must be nonnegative")


@dataclass(frozen=True)
class MeanIncomingWeights:
    """Per-projection sum(W)/N_target, including absent edges and autapse zeros.

    These summaries must already include simulator scaling and constraints.
    """

    values: Mapping
    provenance: str

    def resolve(self, graph):
        required = {p["id"] for p in graph["projections"] if p.get("enabled", True)}
        if set(self.values) != required:
            raise ValueError(
                "weight summaries must name every enabled projection exactly once"
            )
        result = {name: magnitude(value, "uS") for name, value in self.values.items()}
        if not self.provenance.strip() or any(value < 0 for value in result.values()):
            raise ValueError(
                "conductance summaries require provenance and nonnegative weights"
            )
        return result


@dataclass(frozen=True)
class RealisedWeights:
    """Effective dense matrices, after simulator preprocessing.

    Native simulator parameter tensors have layout ``source_target``. Use
    ``target_source`` explicitly for matrices stored in authoring orientation.
    """

    values: Mapping
    provenance: str
    layout: str = "source_target"

    def resolve(self, graph):
        if self.layout not in ("source_target", "target_source"):
            raise ValueError("layout must be source_target or target_source")
        sizes = {p["id"]: p["size"] for p in graph["populations"]}
        sizes.update({i["id"]: i["shape"][-1] for i in graph["inputs"]})
        summaries = {}
        required = set()
        for p in graph["projections"]:
            if not p.get("enabled", True):
                continue
            parameter = p["parameters"][0]
            required.add(parameter)
            matrix = np.asarray(self.values[parameter], dtype=float)
            target = p["target"].rsplit(".", 1)[0]
            source = p["source"].rsplit(".", 1)[0]
            shape = (
                (sizes[source], sizes[target])
                if self.layout == "source_target"
                else (sizes[target], sizes[source])
            )
            if (
                matrix.shape != shape
                or not np.isfinite(matrix).all()
                or (matrix < 0).any()
            ):
                raise ValueError(
                    f"{p['id']}: expected a finite nonnegative effective dense matrix"
                )
            if (
                source == target
                and not p.get("allow_autapses", False)
                and np.any(np.diag(matrix))
            ):
                raise ValueError(
                    f"{p['id']}: effective matrix must already exclude autapses"
                )
            summaries[p["id"]] = matrix.sum(dtype=np.float64) / sizes[target]
        if set(self.values) != required:
            raise ValueError(
                "realised weights must name every used parameter exactly once"
            )
        return MeanIncomingWeights(summaries, self.provenance).resolve(graph)


def reduce(
    network,
    *,
    closure,
    weights,
    inputs=None,
    applied_current=None,
    external_filters="dynamic",
    delays="reject",
):
    """Build a qualitative model without sampling weights or simulating spikes.

    Supply a Network or compiled Bundle. Graph schedules must be expressed as
    explicit integration controls; a schedule is never silently discarded.
    """
    bundle = compile(network) if isinstance(network, Network) else network
    graph = deepcopy(bundle.graph)
    if external_filters not in ("dynamic", "quasistatic"):
        raise ValueError("external_filters must be dynamic or quasistatic")
    if delays not in ("reject", "omit"):
        raise ValueError("delays must be reject or omit")
    if delays == "reject" and any(
        p.get("delay", {}).get("value", 0)
        for p in graph["projections"]
        if p.get("enabled", True) and p.get("delay")
    ):
        raise ValueError(
            "transmission delays require explicit delays='omit' for this ODE model"
        )
    inputs = dict(inputs or {})
    required = {
        p["source"].rsplit(".", 1)[0]
        for p in graph["projections"]
        if p.get("enabled", True)
    } & {i["id"] for i in graph["inputs"]}
    if set(inputs) != required or not all(
        isinstance(v, StationaryRate) for v in inputs.values()
    ):
        raise ValueError("provide one StationaryRate for each connected external input")
    if any(
        i["signal_type"] != "spikes" for i in graph["inputs"] if i["id"] in required
    ):
        raise ValueError("StationaryRate requires a spike input")
    return closure.build(
        BuildContext(
            graph=graph,
            weights=weights.resolve(graph),
            inputs=inputs,
            applied_current=dict(applied_current or {}),
            provenance={"graph_digest": digest(graph), "weights": weights.provenance},
            external_filters=external_filters,
            delays=delays,
        )
    )
