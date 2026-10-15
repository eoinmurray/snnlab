# snnlab.analysis

Measure saved NumPy-compatible activity independently of simulation and plotting.
Use the central public API; no implementation-module imports are needed:

```python
import snnlab as snn

rate_hz = snn.analysis.firing_rate(spikes, dt)
counts = snn.analysis.binned_spike_counts(spikes, dt, bin_ms=5)
```

Spike inputs describe one presentation's binary `(time, cells)` raster. Times
and timestep arguments use milliseconds, rates use Hz, and spectral frequencies
use Hz. Select batches and independent trials explicitly; do not concatenate
presentations for measurements that must not cross their boundaries.

| File | Responsibility |
| --- | --- |
| `__init__.py` | Sole public export map, resolving all 24 functions lazily. |
| `activity.py` | Rates, counts, rate traces, active fraction and count CV. |
| `intervals.py` | Per-neuron intervals and ISI CV within presentations. |
| `correlations.py` | Binned spike-count correlations. |
| `spectra.py` | Power spectra, peaks, band power and fractions. |
| `rhythmicity.py` | Pooled events, IEIs, autocorrelograms and rhythm descriptors. |
| `cycles.py` | Burst detection, cycle construction, counts and occupancy. |
| `conductance.py` | Conductance-loop scores and rolling summaries. |
| `_validation.py` | Private raster, timing, integer and denominator validation. |
| `_sampling.py` | Private windowing, binning and smoothing. |

Implementation modules are private. Importing the package alone loads neither
NumPy nor SciPy. Basic activity, interval, correlation, rhythmicity and conductance
helpers use NumPy without SciPy. Smoothing loads SciPy when requested; spectra
and cycle helpers load it on access. Nothing here imports simulator implementations,
PyTorch or Matplotlib.

The API reference and Analysis guide document the same seven measurement groups.
Existing public signatures, rounding, smoothing, trial boundaries, empty/silent
behaviour, status values, metadata and return structures are preserved. Historical
rhythmicity/conductance helpers retain their own validation and return conventions;
the newer measurement mappings are not a universal result wrapper. Private helpers
are reused only where they were already part of a function's behaviour.

The former private `_activity`, `_spikes`, `_spectra`, `_rhythmicity`, `_cycles`
and `_conductance` modules are retired. Replace internal-path imports with
`from snnlab.analysis import ...`. The former simulator metrics path and aggregate
`compute_metrics` report remain retired.
