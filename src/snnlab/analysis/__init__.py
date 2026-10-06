"""Analyse recorded SNN data with NumPy, independently of the simulator.

Public functions are available directly as ``snnlab.analysis.<name>``.
Spike functions expect one presentation's dense ``(time, cells)`` raster.
Timestep, interval, lag and window arguments use milliseconds.
"""

from ._activity import (
    binned_spike_counts,
    inter_spike_intervals,
    isi_cv,
    per_neuron_firing_rates,
    population_rate_trace,
    spike_count_correlations,
)
from ._conductance import conductance_loop_score, rolling_conductance_loop_score
from ._cycles import (
    cycle_boundaries,
    cycle_occupancy_distribution,
    cycle_spike_counts,
    detect_population_bursts,
)
from ._rhythmicity import (
    iei_histogram,
    population_event_times,
    rhythmicity_metrics,
    rhythmicity_scalars,
    spike_autocorrelogram,
)
from ._spectra import band_power, band_power_fraction, power_spectrum, spectral_peak
from ._spikes import active_fraction, firing_rate, population_spike_count_cv

__all__ = [
    "per_neuron_firing_rates",
    "binned_spike_counts",
    "population_rate_trace",
    "power_spectrum",
    "spectral_peak",
    "band_power",
    "band_power_fraction",
    "inter_spike_intervals",
    "isi_cv",
    "spike_count_correlations",
    "detect_population_bursts",
    "cycle_boundaries",
    "cycle_spike_counts",
    "cycle_occupancy_distribution",
    "firing_rate",
    "population_spike_count_cv",
    "active_fraction",
    "population_event_times",
    "iei_histogram",
    "spike_autocorrelogram",
    "rhythmicity_scalars",
    "rhythmicity_metrics",
    "conductance_loop_score",
    "rolling_conductance_loop_score",
]
