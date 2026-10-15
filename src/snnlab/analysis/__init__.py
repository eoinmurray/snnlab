"""Analyse recorded SNN data with NumPy, independently of the simulator.

Public functions are available directly as ``snnlab.analysis.<name>``.
Spike functions expect one presentation's dense ``(time, cells)`` raster.
Timestep, interval, lag and window arguments use milliseconds.

Implementation modules are private; public exports are resolved on demand.
"""

from importlib import import_module

_EXPORTS = {
    "per_neuron_firing_rates": "activity",
    "binned_spike_counts": "activity",
    "population_rate_trace": "activity",
    "power_spectrum": "spectra",
    "spectral_peak": "spectra",
    "band_power": "spectra",
    "band_power_fraction": "spectra",
    "inter_spike_intervals": "intervals",
    "isi_cv": "intervals",
    "spike_count_correlations": "correlations",
    "detect_population_bursts": "cycles",
    "cycle_boundaries": "cycles",
    "cycle_spike_counts": "cycles",
    "cycle_occupancy_distribution": "cycles",
    "firing_rate": "activity",
    "population_spike_count_cv": "activity",
    "active_fraction": "activity",
    "population_event_times": "rhythmicity",
    "iei_histogram": "rhythmicity",
    "spike_autocorrelogram": "rhythmicity",
    "rhythmicity_scalars": "rhythmicity",
    "rhythmicity_metrics": "rhythmicity",
    "conductance_loop_score": "conductance",
    "rolling_conductance_loop_score": "conductance",
}
__all__ = [*_EXPORTS]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
