"""Reusable analysis functions for SNN spike data.

Population rate/CV metrics, rhythmicity (autocorrelation + IEI), and
reporting helpers.
"""

from __future__ import annotations

import numpy as np


def compute_metrics(spk_e, spk_i, dt, model_name="ping", n_e=1024, n_i=256):
    """Compute population metrics from spike rasters. Returns a plain dict."""
    t_sec = len(spk_e) * dt / 1000.0
    rate_e = float(spk_e.sum() / (n_e * t_sec))
    rate_i = float(spk_i.sum() / (n_i * t_sec)) if spk_i is not None else 0.0

    # Population spike count CV in 2ms bins
    bin_steps = max(1, int(2.0 / dt))
    n_bins = len(spk_e) // bin_steps
    if n_bins > 1:
        pop_counts = np.array(
            [spk_e[i * bin_steps : (i + 1) * bin_steps].sum() for i in range(n_bins)]
        )
        pop_cv = float(pop_counts.std() / max(pop_counts.mean(), 1e-9))
    else:
        pop_cv = 0.0

    per_neuron_counts = spk_e.sum(axis=0)
    active_frac = float((per_neuron_counts > 0).sum()) / n_e

    # E-population rhythmicity (nb054). Keep the FULL rhythmicity output, not
    # just contrast: lobe_lag is the autocorrelogram peak → gamma period → f0
    # (the collection's headline quantity), and re-deriving any of these needs
    # the spike raster, which trained cells do not persist. Emitting them here
    # means every epoch record and the init/end snapshots carry them for free.
    # contrast: lobe–trough (pingness), 0 = flat/asynchronous → 1 as sharp
    # volleys separate against near-silence.
    try:
        rhy = rhythmicity_metrics(spk_e, dt)
    except Exception:
        rhy = {}
    contrast = rhy.get("contrast")
    lobe_lag = rhy.get("lobe_lag")

    def _f(x):
        return float(x) if x is not None else None

    return {
        "rate_e": rate_e,
        "rate_i": rate_i,
        "cv": pop_cv,
        "act": active_frac,
        "contrast": float(contrast) if contrast is not None else 0.0,
        # f0_hz DISABLED (2026-07-06): the `1000/lobe_lag` autocorrelogram
        # heuristic is unreliable — across the exp022 gold-star run it returned a
        # constant ≈1000 Hz (lobe_lag≈1 ms) for 84/87 cells, i.e. the first
        # autocorr lobe is a sub-cycle feature, not the gamma period. Emitting a
        # fake frequency is worse than none. The raw rhythmicity scalars below
        # are kept (nothing lost); the true f_γ is measured by the dedicated
        # spectral analysis over pop_traces, not here.
        "f0_hz": None,
        "lobe_lag_ms": _f(lobe_lag),
        "trough_lag_ms": _f(rhy.get("trough_lag")),
        "iei_mode_lag_ms": _f(rhy.get("iei_mode_lag")),
        "lobe_to_trough": _f(rhy.get("lobe_to_trough")),
    }


# =========================================================================
# Rhythmicity: spike-time autocorrelation + IEI histogram (nb054)
# =========================================================================


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
    single train's curves or on curves trial-averaged across seeds (the latter
    gives an unbiased lobe/trough on noisy single-train data). Reports:

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
    the curves and located lags for plotting. All scalars are calibrated so
    flat/Poisson firing sits near the baseline value 1.0.
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


def conductance_loop_score(g_e, g_i):
    """Bounded geometric score for a consistently rotating E/I conductance loop.

    The trajectory is centred and closed, then scored as the isoperimetric
    quotient ``4πA/L²`` multiplied by signed-rotation coherence. The result is
    dimensionless, invariant to uniform conductance scaling, and lies in [0, 1].
    A circular, consistently traversed loop approaches one; an irregular path
    with cancelling rotation approaches zero.
    """
    x = np.asarray(g_e, dtype=float).reshape(-1)
    y = np.asarray(g_i, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 3:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    x = np.r_[x, x[0]]
    y = np.r_[y, y[0]]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    signed_twice_area = float(cross.sum())
    absolute_sweep = float(np.abs(cross).sum())
    perimeter = float(np.hypot(np.diff(x), np.diff(y)).sum())
    if absolute_sweep <= 0 or perimeter <= 0:
        return 0.0
    area = 0.5 * abs(signed_twice_area)
    compactness = min(1.0, 4.0 * np.pi * area / (perimeter * perimeter))
    direction_coherence = abs(signed_twice_area) / absolute_sweep
    return float(np.clip(compactness * direction_coherence, 0.0, 1.0))


def rolling_conductance_loop_score(
    g_e,
    g_i,
    dt,
    *,
    window_ms=40.0,
    stride_ms=5.0,
    normalize_percentiles=(10.0, 95.0),
):
    """Rolling conductance-loop score plus an explicitly run-relative [0, 1] view."""
    x = np.asarray(g_e, dtype=float).reshape(-1)
    y = np.asarray(g_i, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("g_e and g_i must have equal length")
    window_steps = max(3, int(round(float(window_ms) / float(dt))))
    stride_steps = max(1, int(round(float(stride_ms) / float(dt))))
    ends = np.arange(window_steps, x.size + 1, stride_steps)
    raw = np.array(
        [
            conductance_loop_score(
                x[end - window_steps : end], y[end - window_steps : end]
            )
            for end in ends
        ]
    )
    times_ms = ends.astype(float) * float(dt)
    lo, hi = np.percentile(raw, normalize_percentiles) if raw.size else (0.0, 0.0)
    normalized = np.clip((raw - lo) / max(float(hi - lo), 1e-12), 0.0, 1.0)
    return {
        "times_ms": times_ms,
        "raw": raw,
        "normalized": normalized,
        "normalization_low": float(lo),
        "normalization_high": float(hi),
    }
