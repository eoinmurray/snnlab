"""Explicit FFT/Welch spectra and band measurements."""

from __future__ import annotations

import numpy as np
from scipy.signal import periodogram, welch

from ._activity import _integer
from ._spikes import _positive_ms


def power_spectrum(
    trace,
    dt,
    *,
    method="welch",
    center=False,
    detrend="constant",
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    scaling="density",
    aggregation="none",
):
    """PSD of a time trace or (trials, time) array without joining trial boundaries."""
    x = np.asarray(trace, dtype=float)
    dt = _positive_ms(dt, "dt")
    if x.ndim not in {1, 2} or not np.isfinite(x).all():
        raise ValueError("trace must be finite (time,) or (trials, time)")
    if x.ndim == 2 and x.shape[0] == 0:
        raise ValueError("trace requires at least one trial")
    if method not in {"fft", "welch"} or aggregation not in {"none", "mean", "median"}:
        raise ValueError("invalid method or aggregation")
    if (
        not isinstance(center, bool)
        or detrend not in {False, "constant", "linear"}
        or scaling not in {"density", "spectrum"}
    ):
        raise ValueError("invalid center, detrend or scaling")
    length = x.shape[-1]
    segment = (
        length
        if method == "fft"
        else min(256, length)
        if nperseg is None
        else _integer(nperseg, "nperseg", 1)
    )
    if segment > length:
        raise ValueError("nperseg cannot exceed recording length")
    if method == "fft" and (nperseg is not None or noverlap is not None):
        raise ValueError("nperseg/noverlap apply only to Welch spectra")
    overlap = segment // 2 if noverlap is None else _integer(noverlap, "noverlap")
    if method == "welch" and segment and overlap >= segment:
        raise ValueError("noverlap must be smaller than nperseg")
    fft_length = segment if nfft is None else _integer(nfft, "nfft", 1)
    if fft_length < segment:
        raise ValueError("nfft must be at least the segment length")
    metadata = {
        "dt_ms": dt,
        "sampling_hz": 1000 / dt,
        "method": method,
        "center": center,
        "detrend": detrend,
        "window": window
        if isinstance(window, str)
        else [
            value.item() if isinstance(value, np.generic) else value for value in window
        ],
        "nperseg": segment,
        "noverlap": overlap if method == "welch" else 0,
        "nfft": fft_length,
        "scaling": scaling,
        "aggregation": aggregation,
        "trial_count": 1 if x.ndim == 1 else x.shape[0],
        "trial_samples": length,
    }
    if length < 2:
        shape = (0,) if x.ndim == 1 or aggregation != "none" else (x.shape[0], 0)
        return {
            "frequencies_hz": np.empty(0),
            "power": np.empty(shape),
            "status": "short_recording",
            "metadata": metadata,
        }
    if center:
        x = x - x.mean(axis=-1, keepdims=True)
    constant = np.all(np.ptp(x, axis=-1) == 0)
    common = dict(
        fs=1000 / dt,
        window=window,
        detrend=detrend,
        nfft=fft_length,
        scaling=scaling,
        axis=-1,
    )
    if method == "welch":
        frequencies, power = welch(x, nperseg=segment, noverlap=overlap, **common)
    else:
        frequencies, power = periodogram(x, **common)
    if x.ndim == 2 and aggregation != "none":
        power = (
            power.mean(axis=0) if aggregation == "mean" else np.median(power, axis=0)
        )
    return {
        "frequencies_hz": frequencies,
        "power": power,
        "status": "constant_signal" if constant else "ok",
        "metadata": metadata,
    }


def _spectrum(frequencies_hz, power):
    f, p = np.asarray(frequencies_hz, dtype=float), np.asarray(power, dtype=float)
    if (
        f.ndim != 1
        or p.shape != f.shape
        or not np.isfinite(f).all()
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(f < 0)
        or np.any(np.diff(f) <= 0)
    ):
        raise ValueError(
            "spectrum requires matching finite 1D arrays, increasing nonnegative frequencies and nonnegative power"
        )
    return f, p


def _band(band_hz):
    band = np.asarray(band_hz, dtype=float)
    if (
        band.shape != (2,)
        or not np.isfinite(band).all()
        or band[0] < 0
        or band[1] <= band[0]
    ):
        raise ValueError("band_hz must be finite increasing nonnegative bounds")
    return band


def spectral_peak(
    frequencies_hz,
    power,
    band_hz,
    *,
    interpolation="none",
    interpolation_boundary="band",
    clamp_band=True,
):
    """Locate an in-band maximum; optional parabolic interpolation in linear power."""
    f, p = _spectrum(frequencies_hz, power)
    band = _band(band_hz)
    if interpolation_boundary not in {"band", "spectrum"} or not isinstance(
        clamp_band, bool
    ):
        raise ValueError("invalid interpolation boundary or clamp policy")
    if interpolation not in {"none", "parabolic"}:
        raise ValueError("interpolation must be none or parabolic")
    selected = np.flatnonzero((f >= band[0]) & (f <= band[1]))
    metadata = {
        "band_hz": band.tolist(),
        "interpolation": interpolation,
        "interpolation_applied": False,
        "power_domain": "linear",
        "interpolation_boundary": interpolation_boundary,
        "clamp_band": clamp_band,
    }
    result = {
        "frequency_hz": None,
        "peak_power": None,
        "bin_frequency_hz": None,
        "status": "empty_band",
        "metadata": metadata,
    }
    if not len(selected):
        return result
    if p[selected].max() <= 0:
        result["status"] = "no_power"
        return result
    if len(selected) > 1 and np.all(p[selected] == p[selected[0]]):
        result["status"] = "flat_band"
        return result
    k = selected[np.argmax(p[selected])]
    peak_f, peak_p = float(f[k]), float(p[k])
    interior = (
        (k > selected[0] and k < selected[-1])
        if interpolation_boundary == "band"
        else (0 < k < len(f) - 1)
    )
    if interpolation == "parabolic" and interior:
        distances = np.diff(f[k - 1 : k + 2])
        if not np.isclose(distances[0], distances[1], rtol=1e-7):
            raise ValueError(
                "parabolic interpolation requires equally spaced neighbouring frequencies"
            )
        left, middle, right = p[k - 1 : k + 2]
        denominator = left - 2 * middle + right
        if denominator < 0:
            delta = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
            peak_f = float(f[k] + delta * distances[0])
            if clamp_band:
                peak_f = float(np.clip(peak_f, *band))
            peak_p = float(middle - 0.25 * (left - right) * delta)
            metadata["interpolation_applied"] = True
    result.update(
        frequency_hz=peak_f,
        peak_power=peak_p,
        bin_frequency_hz=float(f[k]),
        status="ok",
    )
    return result


def band_power(frequencies_hz, power, band_hz, *, integration="trapezoid"):
    """Integrate a spectral density over the covered band, including interpolated edges."""
    f, p = _spectrum(frequencies_hz, power)
    band = _band(band_hz)
    if integration not in {"trapezoid", "bin_sum"}:
        raise ValueError("integration must be trapezoid or bin_sum")
    metadata = {
        "band_hz": band.tolist(),
        "integration": integration,
        "edge_interpolation": "linear",
        "covered_band_hz": None,
    }
    if len(f) < 2 or max(band[0], f[0]) >= min(band[1], f[-1]):
        return {"power": None, "status": "no_band_coverage", "metadata": metadata}
    low, high = max(band[0], f[0]), min(band[1], f[-1])
    frequencies = np.concatenate(([low], f[(f > low) & (f < high)], [high]))
    if integration == "trapezoid":
        integrated = float(np.trapezoid(np.interp(frequencies, f, p), frequencies))
    else:
        spacing = np.diff(f)
        if not np.allclose(spacing, spacing[0], rtol=1e-7):
            raise ValueError("bin_sum requires uniformly spaced frequencies")
        selected = (f >= band[0]) & (f <= band[1])
        if not selected.any():
            return {"power": None, "status": "no_band_coverage", "metadata": metadata}
        integrated = float(p[selected].sum() * spacing[0])
        metadata["edge_interpolation"] = "none"

    metadata["covered_band_hz"] = [float(low), float(high)]
    return {
        "power": integrated,
        "status": "ok" if low == band[0] and high == band[1] else "partial_coverage",
        "metadata": metadata,
    }


def band_power_fraction(
    frequencies_hz, power, band_hz, *, reference_band_hz, integration="trapezoid"
):
    """Ratio of integrated PSD in a target band and an explicit containing reference band."""
    band, reference = _band(band_hz), _band(reference_band_hz)
    if band[0] < reference[0] or band[1] > reference[1]:
        raise ValueError("reference band must contain target band")
    numerator = band_power(frequencies_hz, power, band, integration=integration)
    denominator = band_power(frequencies_hz, power, reference, integration=integration)
    fraction = None
    status = "undefined_reference"
    if (
        numerator["power"] is not None
        and denominator["power"] is not None
        and denominator["power"] > 0
    ):
        fraction = numerator["power"] / denominator["power"]
        status = (
            "ok"
            if numerator["status"] == denominator["status"] == "ok"
            else "partial_coverage"
        )
    return {
        "fraction": fraction,
        "band_power": numerator["power"],
        "reference_power": denominator["power"],
        "status": status,
        "metadata": {
            "band": numerator["metadata"],
            "reference": denominator["metadata"],
        },
    }
