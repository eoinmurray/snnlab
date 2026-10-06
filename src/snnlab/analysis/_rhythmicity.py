"""Population event intervals and autocorrelation-based descriptors."""

from __future__ import annotations

import numpy as np


def population_event_times(spikes, dt):
    """Pooled population spike-event times in ms from a [T_steps, N] raster.

    Every (timestep, neuron) spike contributes one event at time step*dt;
    simultaneous spikes across neurons give repeated times. Returned sorted.
    """
    spikes = np.asarray(spikes)
    step_idx = np.nonzero(spikes)[0]
    return np.sort(step_idx).astype(float) * dt


def iei_histogram(event_times_ms, max_lag_ms=100.0, bin_ms=1.0):
    """Inter-event-interval histogram of a pooled event train.

    Intervals between consecutive (sorted) events. Returns (lag_centers_ms,
    counts). The first mode is the short-interval pile-up; a rhythm shows a
    second mode near the period.
    """
    t = np.sort(np.asarray(event_times_ms, dtype=float))
    edges = np.arange(0.0, max_lag_ms + bin_ms, bin_ms)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if t.size < 2:
        return centers, np.zeros(centers.size)
    counts, _ = np.histogram(np.diff(t), bins=edges)
    return centers, counts.astype(float)


def spike_autocorrelogram(spikes, dt, max_lag_ms=100.0, bin_ms=1.0):
    """Normalised population spike-time autocorrelogram.

    Bins the population count r(t) at bin_ms, forms the pair-count
    autocorrelation Σ_t r(t)·r(t+ℓ), divides by the per-lag overlap count and
    by mean(r)² so the asymptotic floor is 1.0 (rate-matched independence).
    The zero-lag bin is dominated by self-pairs and returned as NaN.

    Returns (lags_ms, ac). A rhythm shows ac > 1 in a central lobe near ℓ = 0,
    a trough (ac < 1) around the half-period, and a secondary peak at the
    period — the "Mexican hat". Flat (asynchronous) firing gives ac ≈ 1.
    """
    spikes = np.asarray(spikes)
    bin_steps = max(1, int(round(bin_ms / dt)))
    realized_bin_ms = bin_steps * dt
    n_bins = spikes.shape[0] // bin_steps
    max_lag_bins = max(1, int(round(max_lag_ms / realized_bin_ms)))
    lags = np.arange(max_lag_bins + 1) * realized_bin_ms
    if n_bins <= max_lag_bins + 1:
        return lags, np.full(max_lag_bins + 1, np.nan)
    r = (
        spikes[: n_bins * bin_steps]
        .reshape(n_bins, bin_steps, -1)
        .sum(axis=(1, 2))
        .astype(float)
    )
    # Linear (non-circular) autocorrelation Σ_t r(t)·r(t+ℓ) via zero-padded
    # FFT — O(n log n), so long single-train traces stay fast.
    nfft = 1 << int(np.ceil(np.log2(2 * n_bins)))
    f = np.fft.rfft(r, nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[: max_lag_bins + 1].astype(float)
    overlap = n_bins - np.arange(max_lag_bins + 1)  # terms summed at each lag
    floor = r.mean() ** 2
    if floor <= 0:
        return lags, np.full(max_lag_bins + 1, np.nan)
    ac = ac / overlap / floor
    ac[0] = np.nan  # self-pairs dominate zero-lag
    return lags, ac


def _smooth_ac(ac):
    """3-point-smoothed copy of the autocorrelogram (lag-0 NaN filled), or None
    if it cannot be made finite. Smoothing controls the finite-count shot noise
    a single Poisson spike train carries, so lobe/trough extraction is stable.
    """
    a = np.asarray(ac, dtype=float).copy()
    if a.size > 1 and not np.isfinite(a[0]):
        a[0] = a[1]  # lag-0 is NaN by construction
    if not np.all(np.isfinite(a)):
        return None
    return np.convolve(a, [0.25, 0.5, 0.25], mode="same")


def rhythmicity_scalars(ac_lags, ac, iei_lags, iei_counts, bin_ms=1.0, bio_lag_ms=None):
    """Candidate rhythmicity scalars from an autocorrelogram + IEI histogram.

    Split out from rhythmicity_metrics so the same extraction can run on a
    single train's curves or on curves trial-averaged across seeds (with the averaging procedure chosen explicitly by the caller). Reports:

      - iei_anchored: ac at the IEI primary-mode lag (central-lobe height).
      - lobe_to_trough: central-lobe peak / first-trough depth, both read from
        a lightly smoothed autocorrelogram (the trough is its first local
        minimum), with the trough floored at 0.05 of the unity baseline.
      - biophysical: ac at a supplied network lag bio_lag_ms, or None.
    """
    ac = np.asarray(ac, dtype=float)
    ac_lags = np.asarray(ac_lags, dtype=float)

    def ac_at(lag_ms):
        if lag_ms is None or not np.isfinite(lag_ms) or ac.size < 2:
            return None
        lag_step_ms = ac_lags[1] - ac_lags[0]
        i = int(np.clip(round((lag_ms - ac_lags[0]) / lag_step_ms), 1, ac.size - 1))
        return float(ac[i]) if np.isfinite(ac[i]) else None

    iei_mode_lag = (
        float(iei_lags[int(np.argmax(iei_counts))]) if np.sum(iei_counts) > 0 else None
    )

    sm = _smooth_ac(ac)
    lobe_to_trough = contrast = trough_lag = lobe_lag = None
    if sm is not None:
        trough_i = None
        for i in range(2, sm.size - 1):
            if sm[i] <= sm[i - 1] and sm[i] < sm[i + 1]:
                trough_i = i
                break
        if trough_i is not None and trough_i > 1:
            trough_lag = float(ac_lags[trough_i])
            lobe_i = 1 + int(np.argmax(sm[1:trough_i]))
            lobe_lag = float(ac_lags[lobe_i])
            lobe_v, trough_v = float(sm[lobe_i]), float(sm[trough_i])
            # Unbounded ratio (kept for reference) and the bounded Mexican-hat
            # contrast (lobe−trough)/(lobe+trough) ∈ [0, 1): 0 = flat (lobe=trough),
            # → 1 as the trough goes silent. No trough floor needed.
            lobe_to_trough = lobe_v / max(trough_v, 0.05)
            denom = lobe_v + trough_v
            contrast = (lobe_v - trough_v) / denom if denom > 0 else None

    return {
        "iei_mode_lag": iei_mode_lag,
        "lobe_lag": lobe_lag,
        "trough_lag": trough_lag,
        "iei_anchored": ac_at(iei_mode_lag),
        "lobe_to_trough": lobe_to_trough,
        "contrast": contrast,
        "biophysical": ac_at(bio_lag_ms),
    }


def rhythmicity_metrics(spikes, dt, max_lag_ms=100.0, bin_ms=1.0, bio_lag_ms=None):
    """Spike-time-autocorrelation rhythmicity scalars for a [T, N] raster.

    Builds the IEI histogram and the normalised autocorrelogram, extracts the
    candidate scalars (see rhythmicity_scalars), and returns them together with
    the curves and located lags for plotting. The autocorrelogram has an independence baseline near 1.0; contrast
    instead approaches zero for flat activity. Scalars can be None when
    insufficient data or missing extrema prevent measurement.
    """
    event_times = population_event_times(spikes, dt)
    iei_lags, iei_counts = iei_histogram(event_times, max_lag_ms, bin_ms)
    ac_lags, ac = spike_autocorrelogram(spikes, dt, max_lag_ms, bin_ms)
    out = rhythmicity_scalars(ac_lags, ac, iei_lags, iei_counts, bin_ms, bio_lag_ms)
    out.update(
        iei_lags=iei_lags,
        iei_counts=iei_counts,
        ac_lags=ac_lags,
        ac=ac,
        n_events=int(event_times.size),
    )
    return out
