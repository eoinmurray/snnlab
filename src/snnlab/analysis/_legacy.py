"""Historical aggregate for specialised reporting; preserves permissive inputs.

The public analysis API exposes separate measurements with explicit validation.
"""

import numpy as np

from ._rhythmicity import rhythmicity_metrics


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

    # Retain the historical report keys and missing-measurement fallback.
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
        # The central-lobe lag is not a reliable oscillation-period estimate.
        "f0_hz": None,
        "lobe_lag_ms": _f(lobe_lag),
        "trough_lag_ms": _f(rhy.get("trough_lag")),
        "iei_mode_lag_ms": _f(rhy.get("iei_mode_lag")),
        "lobe_to_trough": _f(rhy.get("lobe_to_trough")),
    }
