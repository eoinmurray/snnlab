"""Rhythmicity metric primitives (nb054): IEI histogram, spike-time
autocorrelogram, and the candidate scalars.

Ground-truth checks on synthetic rasters: a flat (asynchronous) train sits at
the baseline value 1.0; a periodic volley train has a normalised autocorrelogram
peak at the period and a lobe-to-trough ratio well above 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from snnlab.analysis import (  # noqa: E402
    conductance_loop_score,
    population_event_times,
    rhythmicity_metrics,
    rolling_conductance_loop_score,
    spike_autocorrelogram,
)


def _poisson_raster(rate_hz, dt, n_neurons, t_ms, rng):
    """Homogeneous Poisson raster [T, N] at a flat per-neuron rate."""
    T = int(t_ms / dt)
    p = rate_hz * dt / 1000.0
    return (rng.random((T, n_neurons)) < p).astype(np.int8)


def _volley_raster(period_ms, dt, n_neurons, t_ms, jitter_ms, rng):
    """Rhythmic raster: every neuron fires once per cycle, jittered around the
    volley time — a clean periodic population rhythm."""
    T = int(t_ms / dt)
    raster = np.zeros((T, n_neurons), dtype=np.int8)
    volley_times = np.arange(period_ms / 2, t_ms, period_ms)
    for vt in volley_times:
        steps = ((vt + rng.normal(0, jitter_ms, n_neurons)) / dt).astype(int)
        steps = steps[(steps >= 0) & (steps < T)]
        raster[steps, np.arange(steps.size)] = 1
    return raster


def test_population_event_times_count_and_sorted():
    raster = np.zeros((10, 3), dtype=np.int8)
    raster[2, 0] = 1
    raster[2, 1] = 1
    raster[5, 2] = 1
    ev = population_event_times(raster, dt=1.0)
    assert ev.size == 3
    assert np.all(np.diff(ev) >= 0)
    assert ev.tolist() == [2.0, 2.0, 5.0]


def test_autocorrelogram_peaks_at_period():
    rng = np.random.default_rng(0)
    dt, period = 0.25, 25.0  # 40 Hz
    raster = _volley_raster(period, dt, n_neurons=200, t_ms=2000.0, jitter_ms=1.0, rng=rng)
    lags, ac = spike_autocorrelogram(raster, dt, max_lag_ms=80.0, bin_ms=1.0)
    # The autocorrelogram should peak near one period (and not at, say, 1.5×).
    band = (lags > period * 0.6) & (lags < period * 1.4)
    peak_lag = lags[band][np.nanargmax(ac[band])]
    assert abs(peak_lag - period) < 4.0, peak_lag
    # Secondary peak is well above the unity floor.
    assert np.nanmax(ac[band]) > 1.5


@pytest.mark.parametrize("dt", [0.1, 0.3, 0.6])
def test_nonintegral_bins_keep_physical_lags_and_biophysical_lookup(dt):
    raster = _volley_raster(
        30.0, dt, n_neurons=10, t_ms=1800.0,
        jitter_ms=0, rng=np.random.default_rng(0),
    )
    result = rhythmicity_metrics(raster, dt, max_lag_ms=60, bio_lag_ms=30)
    lags, ac = result["ac_lags"], result["ac"]
    window = (lags >= 20) & (lags <= 40)
    peak_lag = lags[window][np.argmax(ac[window])]
    assert peak_lag == pytest.approx(30, abs=1.2)
    assert result["biophysical"] > 5


def test_flat_poisson_sits_at_baseline():
    rng = np.random.default_rng(1)
    raster = _poisson_raster(rate_hz=20.0, dt=0.25, n_neurons=400, t_ms=3000.0, rng=rng)
    m = rhythmicity_metrics(raster, dt=0.25, max_lag_ms=100.0, bin_ms=1.0)
    # No rhythm: autocorrelogram hugs 1.0, lobe-to-trough barely above 1.
    assert abs(m["iei_anchored"] - 1.0) < 0.25, m["iei_anchored"]
    assert m["lobe_to_trough"] < 1.5, m["lobe_to_trough"]


def test_rhythm_separates_from_poisson():
    rng = np.random.default_rng(2)
    flat = _poisson_raster(20.0, 0.25, 400, 3000.0, rng)
    volley = _volley_raster(25.0, 0.25, 400, 3000.0, jitter_ms=1.5, rng=rng)
    m_flat = rhythmicity_metrics(flat, dt=0.25)
    m_rhythm = rhythmicity_metrics(volley, dt=0.25)
    # The rhythm must score strictly higher on both data-driven scalars.
    assert m_rhythm["lobe_to_trough"] > 3.0 * m_flat["lobe_to_trough"]
    assert m_rhythm["iei_anchored"] > m_flat["iei_anchored"]


def test_trough_located_near_half_period():
    # The autocorrelogram's first minimum (the Mexican-hat trough) sits near
    # the half-period, between the central lobe and the period-lag secondary
    # peak — located directly from the autocorrelogram, not the IEI.
    rng = np.random.default_rng(3)
    period = 20.0
    volley = _volley_raster(period, 0.25, 300, 2000.0, jitter_ms=1.0, rng=rng)
    m = rhythmicity_metrics(volley, dt=0.25, max_lag_ms=80.0, bin_ms=1.0)
    assert m["trough_lag"] is not None and m["lobe_lag"] is not None
    assert abs(m["trough_lag"] - period / 2) < 5.0, m["trough_lag"]
    assert m["lobe_lag"] < m["trough_lag"]
    assert m["lobe_to_trough"] > 2.0


def test_contrast_bounded_and_separates():
    rng = np.random.default_rng(7)
    flat = _poisson_raster(20.0, 0.25, 400, 3000.0, rng)
    volley = _volley_raster(25.0, 0.25, 400, 3000.0, jitter_ms=1.5, rng=rng)
    cf = rhythmicity_metrics(flat, dt=0.25)["contrast"]
    cv = rhythmicity_metrics(volley, dt=0.25)["contrast"]
    assert 0.0 <= cf < 0.2, cf            # flat ≈ 0
    assert 0.0 <= cv < 1.0, cv            # bounded above by 1
    assert cv > cf + 0.3                  # rhythm clearly higher


def test_conductance_loop_score_is_bounded_scale_invariant_and_directional():
    phase = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    loop = conductance_loop_score(np.cos(phase), np.sin(phase))
    scaled = conductance_loop_score(7 * np.cos(phase), 7 * np.sin(phase))
    cancelling = conductance_loop_score(
        np.r_[np.cos(phase), np.cos(phase[::-1])],
        np.r_[np.sin(phase), np.sin(phase[::-1])],
    )
    assert 0.9 < loop <= 1.0
    assert scaled == pytest.approx(loop)
    assert cancelling < 0.05


def test_rolling_conductance_loop_score_has_run_relative_normalized_view():
    phase = np.linspace(0, 6 * np.pi, 600)
    result = rolling_conductance_loop_score(
        np.cos(phase), np.sin(phase), dt=1.0, window_ms=100, stride_ms=10
    )
    assert result["times_ms"].shape == result["raw"].shape
    assert np.all((result["normalized"] >= 0) & (result["normalized"] <= 1))
