"""Analyse a saved demonstration raster without constructing or running a model."""

import json
from pathlib import Path

import numpy as np

from snnlab import analysis

root = Path("analysis-example")
root.mkdir(exist_ok=True)
# Stand-in for activity saved by a simulation or experimental recording.
e = np.zeros((2000, 8), dtype=bool)
i = np.zeros((2000, 2), dtype=bool)
i[25::50] = True
e[26::50, :6] = True
e[27::50, 6] = True
np.savez_compressed(root / "activity.npz", e_spikes=e, i_spikes=i, dt_ms=1.0)

with np.load(root / "activity.npz", allow_pickle=False) as saved:
    e, i, dt = saved["e_spikes"], saved["i_spikes"], float(saved["dt_ms"])

rates = analysis.per_neuron_firing_rates(e, dt)
trace = analysis.population_rate_trace(e, dt, bin_ms=5, rounding="exact")
# The spectrum's timestep is the binned trace's timestep, not the raw raster's.
spectrum = analysis.power_spectrum(
    trace["rate_hz"],
    trace["metadata"]["realized_bin_ms"],
    method="welch",
    nperseg=200,
    noverlap=100,
)
peak = analysis.spectral_peak(
    spectrum["frequencies_hz"],
    spectrum["power"],
    (10, 30),
    interpolation="parabolic",
)
fraction = analysis.band_power_fraction(
    spectrum["frequencies_hz"],
    spectrum["power"],
    (10, 30),
    reference_band_hz=(1, 95),
)
irregularity = analysis.isi_cv(e, dt, min_spikes=3, aggregation="mean")
correlations = analysis.spike_count_correlations(
    e,
    dt,
    bin_ms=5,
    constant="omit",
    aggregation="mean",
)
bursts = analysis.detect_population_bursts(i, dt, min_separation_ms=25)
cycles = analysis.cycle_boundaries(bursts["peak_steps"], len(e), policy="midpoints")
counts = analysis.cycle_spike_counts(e, cycles["edges"])
occupancy = analysis.cycle_occupancy_distribution(counts["counts"], max_count=2)

np.testing.assert_equal(rates["rates_hz"], [20] * 7 + [0])
assert cycles["cycle_count"] == 40
np.testing.assert_equal(occupancy["fractions"], [0.125, 0.875, 0])
assert abs(peak["frequency_hz"] - 20) < 1
assert np.isnan(irregularity["cv"][7])
print("Rates (Hz):", rates["rates_hz"])
print("Band peak (Hz):", peak["frequency_hz"])
print("ISI CV:", irregularity["cv"])
print("Mean count correlation:", correlations["aggregate"])
print("Cycles:", cycles["cycle_count"])
print("Occupancy fractions:", occupancy["fractions"])

results = dict(
    rates=rates,
    trace=trace,
    spectrum=spectrum,
    peak=peak,
    fraction=fraction,
    irregularity=irregularity,
    correlations=correlations,
    bursts=bursts,
    cycles=cycles,
    counts=counts,
    occupancy=occupancy,
)
(root / "settings.json").write_text(
    json.dumps(
        {name: result["metadata"] for name, result in results.items()},
        indent=2,
        allow_nan=False,
    )
    + "\n"
)
