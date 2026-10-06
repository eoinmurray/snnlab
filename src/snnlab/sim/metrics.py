"""Compatibility imports for analysis functions formerly housed in sim.

New analysis code should import directly from ``snnlab.analysis``.
``compute_metrics`` retains its historical specialised reporting contract.
"""

from snnlab.analysis import (
    conductance_loop_score,
    iei_histogram,
    population_event_times,
    rhythmicity_metrics,
    rhythmicity_scalars,
    rolling_conductance_loop_score,
    spike_autocorrelogram,
)
from snnlab.analysis._legacy import compute_metrics

__all__ = [
    "compute_metrics",
    "conductance_loop_score",
    "iei_histogram",
    "population_event_times",
    "rhythmicity_metrics",
    "rhythmicity_scalars",
    "rolling_conductance_loop_score",
    "spike_autocorrelogram",
]
