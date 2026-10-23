"""Source units and beginning-of-step sampling for graded projections."""

import math

from snnlab import extensions as E
from snnlab._shared.spatial import source_root


def delay_steps(row, dt):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("graph timebase dt must be finite and positive")
    delay = row.get("delay")
    value = 0.0 if delay is None else float(delay["value"])
    if (
        not math.isfinite(value)
        or value < 0
        or (delay is not None and delay.get("unit") != "ms")
    ):
        raise ValueError(f"{row['id']}: delay must be finite non-negative ms")
    raw = value / dt
    steps = int(round(raw))
    if not math.isclose(raw, steps, abs_tol=1e-9):
        raise ValueError(
            f"{row['id']}: delay {value} ms is not an integer number of dt={dt} ms steps"
        )
    return steps


def continuous_source(graph, row):
    root = source_root(graph, row["source"])
    owner, _, port = root.partition(".")
    population = next(
        (p for p in graph.get("populations", []) if p["id"] == owner), None
    )
    return population is not None and port != "spikes"


def validate_source(graph, row):
    """Explicit-drive synapses use dimensionless, pre-gate projection weights."""
    source = row["source"]
    root = source_root(graph, source)
    owner, _, port = root.partition(".")
    population = next(
        (p for p in graph.get("populations", []) if p["id"] == owner), None
    )
    expected = E.synapse_input_unit(row["synapse"])
    if population is not None:
        units = {"voltage": "mV", **E.state_units("neuron", population["neuron"])}
        if port == "spikes":
            if not population["spiking"]:
                raise ValueError(f"{source}: population has no spikes port")
            unit = "spike"
        else:
            if port not in units or source != root:
                raise ValueError(f"{source}: unsupported continuous population source")
            if expected is None:
                raise ValueError(
                    "population projection sources must use the spikes port unless "
                    "a custom synapse declares input_unit"
                )
            unit = units[port]
    else:
        signal = next(
            (
                r
                for r in (*graph.get("inputs", []), *graph.get("operations", []))
                if f"{r['id']}.value" == source
            ),
            None,
        )
        if signal is None:
            raise ValueError(f"{source}: unsupported projection source")
        unit = signal["unit"]
    if expected is not None and unit != expected:
        raise ValueError(
            f"{source}: synapse input unit {expected} incompatible with source unit {unit}"
        )
