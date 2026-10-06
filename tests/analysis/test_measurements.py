"""Reference signals with known rates, spectra, correlations and cycle counts."""

import json

import numpy as np
import pytest

from snnlab import analysis as a


def test_windowed_rates_binning_alignment_and_partial_duration():
    spikes = np.ones((7, 2), dtype=bool)
    rates = a.per_neuron_firing_rates(spikes, 0.5, window=(2, 7))
    np.testing.assert_equal(rates["rates_hz"], [2000, 2000])
    bins = a.binned_spike_counts(
        spikes, 0.5, bin_ms=1, window=(2, 7), trailing="include", alignment="center"
    )
    np.testing.assert_equal(bins["counts"], [[2, 2], [2, 2], [1, 1]])
    np.testing.assert_equal(bins["times_ms"], [1.5, 2.5, 3.25])
    np.testing.assert_equal(bins["duration_ms"], [1, 1, 0.5])
    trace = a.population_rate_trace(
        spikes, 0.5, bin_ms=1, window=(2, 7), trailing="include"
    )
    np.testing.assert_equal(trace["rate_hz"], [2000, 2000, 2000])
    assert (
        a.binned_spike_counts(spikes, 0.5, bin_ms=1)["metadata"]["discarded_steps"] == 1
    )
    with pytest.raises(ValueError):
        a.binned_spike_counts(spikes, 0.5, bin_ms=1, trailing="error")
    with pytest.raises(ValueError):
        a.binned_spike_counts(spikes, 0.5, bin_ms=0.75, rounding="exact")
    with pytest.raises(ValueError):
        a.population_rate_trace(
            spikes, 0.5, bin_ms=1, trailing="include", smoothing_ms=1
        )
    assert (
        len(
            a.population_rate_trace(spikes[:1], 0.5, bin_ms=0.5, smoothing_ms=100)[
                "rate_hz"
            ]
        )
        == 1
    )


def test_new_cv_binning_preserves_previous_float32_calculation():
    rng = np.random.default_rng(4)
    for dtype in (np.float32, np.float64, bool, np.int8):
        spikes = (rng.random((999, 13)) < 0.12).astype(dtype)
        counts = np.array([spikes[i * 3 : (i + 1) * 3].sum() for i in range(333)])
        expected = float(counts.std() / max(counts.mean(), 1e-9))
        assert a.population_spike_count_cv(spikes, 0.6) == expected


def test_trial_boundaries_prevent_artificial_intervals():
    spikes = np.zeros((12, 2), dtype=bool)
    spikes[[1, 3, 5, 6, 8, 10], 0] = True
    spikes[[1, 9], 1] = True
    intervals = a.inter_spike_intervals(spikes, 1, trial_boundaries=[0, 6, 12])
    np.testing.assert_equal(intervals["intervals_ms"][0], [2, 2, 2, 2])
    np.testing.assert_equal(intervals["trial_indices"][0], [0, 0, 1, 1])
    assert len(intervals["intervals_ms"][1]) == 0
    result = a.isi_cv(spikes, 1, trial_boundaries=[0, 6, 12], aggregation="mean")
    assert result["cv"][0] == 0
    assert np.isnan(result["cv"][1])
    assert result["aggregate"] == 0
    cropped = a.inter_spike_intervals(
        spikes, 1, window=(4, 10), trial_boundaries=[0, 6, 12]
    )
    np.testing.assert_equal(cropped["intervals_ms"][0], [2])
    assert a.isi_cv(spikes, 1, min_spikes=7)["status"] == "insufficient_spikes"


def test_correlations_known_positive_negative_and_constant_cells():
    spikes = np.array([[0, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0]])
    result = a.spike_count_correlations(spikes, 1, bin_ms=1, aggregation="mean")
    np.testing.assert_allclose(
        result["correlations"][:3, :3], [[1, 1, -1], [1, 1, -1], [-1, -1, 1]]
    )
    assert np.isnan(result["correlations"][3]).all()
    assert result["pair_count"] == 3
    assert result["aggregate"] == pytest.approx(-1 / 3)
    omitted = a.spike_count_correlations(
        spikes, 1, bin_ms=1, constant="omit", cells=[3, 2, 0]
    )
    np.testing.assert_equal(omitted["cells"], [2, 0])
    np.testing.assert_equal(omitted["excluded_cells"], [3])
    with pytest.raises(ValueError):
        a.spike_count_correlations(spikes, 1, bin_ms=1, constant="error")
    assert a.spike_count_correlations(spikes[:1], 1)["status"] == "no_valid_pairs"


def test_known_sinusoid_fft_peak_and_integrated_variance():
    time = np.arange(1000) / 1000
    trace = 2 * np.sin(2 * np.pi * 40 * time)
    result = a.power_spectrum(trace, 1, method="fft", window="boxcar", detrend=False)
    peak = a.spectral_peak(result["frequencies_hz"], result["power"], (20, 80))
    assert peak["frequency_hz"] == 40
    integrated = a.band_power(result["frequencies_hz"], result["power"], (0, 500))
    assert integrated["power"] == pytest.approx(2.0)
    fraction = a.band_power_fraction(
        result["frequencies_hz"], result["power"], (35, 45), reference_band_hz=(0, 500)
    )
    assert fraction["fraction"] == pytest.approx(1.0)
    assert (
        a.power_spectrum(trace, 1, nperseg=200, noverlap=100)["metadata"]["noverlap"]
        == 100
    )


def test_spectral_trial_aggregation_never_joins_presentations():
    time = np.arange(1000) / 1000
    x = np.sin(2 * np.pi * 40 * time)
    stacked = a.power_spectrum(np.stack([x, 2 * x]), 1, method="fft", window="boxcar")
    assert stacked["power"].shape == (2, 501)
    aggregate = a.power_spectrum(
        np.stack([x, 2 * x]), 1, method="fft", window="boxcar", aggregation="mean"
    )
    np.testing.assert_allclose(aggregate["power"], stacked["power"].mean(axis=0))
    np.testing.assert_allclose(stacked["power"][1], 4 * stacked["power"][0])


def test_parabolic_peak_and_band_edge_integration():
    frequencies = np.arange(6, dtype=float)
    power = 10 - (frequencies - 2.25) ** 2
    peak = a.spectral_peak(frequencies, power, (0, 5), interpolation="parabolic")
    assert peak["frequency_hz"] == pytest.approx(2.25)
    assert peak["peak_power"] == pytest.approx(10)
    assert peak["metadata"]["interpolation_applied"]
    assert a.band_power([0, 1, 2], [2, 2, 2], (0.25, 1.75))["power"] == 3
    assert a.band_power([0, 1, 2], [2, 2, 2], (0, 3))["status"] == "partial_coverage"
    assert a.spectral_peak([0, 1], [0, 0], (0, 1))["frequency_hz"] is None
    assert a.spectral_peak([0, 1], [2, 2], (0, 1))["status"] == "flat_band"
    assert (
        a.band_power_fraction([0, 1], [0, 0], (0, 1), reference_band_hz=(0, 1))[
            "fraction"
        ]
        is None
    )
    assert a.power_spectrum([0], 1)["status"] == "short_recording"
    assert a.power_spectrum(np.zeros(100), 1)["status"] == "constant_signal"


def test_periodic_bursts_cycle_boundaries_and_known_occupancy():
    spikes = np.zeros((100, 2), dtype=bool)
    spikes[[10, 30, 50, 70, 90], :] = True
    bursts = a.detect_population_bursts(spikes, 1, min_separation_ms=10)
    np.testing.assert_equal(bursts["peak_steps"], [10, 30, 50, 70, 90])
    boundaries = a.cycle_boundaries(bursts["peak_steps"], 100)
    np.testing.assert_equal(boundaries["edges"], [0, 20, 40, 60, 80, 100])
    counts = a.cycle_spike_counts(spikes, boundaries["edges"])
    np.testing.assert_equal(counts["counts"], np.ones((5, 2)))
    occupancy = a.cycle_occupancy_distribution(counts["counts"], max_count=2)
    np.testing.assert_equal(occupancy["fractions"], [0, 1, 0])
    assert occupancy["opportunities"] == 10
    full = a.cycle_boundaries(bursts["peak_steps"], 100, policy="peak_to_peak")
    assert full["cycle_count"] == 4
    shifted = a.cycle_boundaries(bursts["peak_steps"] + 200, 100, start_step=200)
    np.testing.assert_equal(
        a.cycle_spike_counts(spikes, shifted["edges"], start_step=200)["counts"],
        counts["counts"],
    )
    np.testing.assert_equal(
        a.cycle_occupancy_distribution(np.array([[0, 1, 2, 3]]), max_count=2)[
            "opportunity_counts"
        ],
        [1, 1, 2],
    )


def test_no_cycles_is_not_zero_occupancy():
    boundaries = a.cycle_boundaries([], 100)
    assert boundaries["status"] == "no_cycles"
    counts = a.cycle_spike_counts(np.zeros((100, 2)), boundaries["edges"])
    result = a.cycle_occupancy_distribution(counts["counts"])
    assert result["opportunities"] == 0
    assert np.isnan(result["fractions"]).all()
    assert a.detect_population_bursts(np.zeros((100, 2)), 1)["status"] == "no_peaks"


@pytest.mark.parametrize(
    "call",
    [
        lambda: a.per_neuron_firing_rates(np.zeros((3, 2)), 0),
        lambda: a.inter_spike_intervals(np.zeros((3, 2)), 1, trial_boundaries=[0, 2]),
        lambda: a.isi_cv(np.zeros((3, 2)), 1, min_spikes=1),
        lambda: a.spike_count_correlations(np.zeros((3, 2)), 1, cells=[0, 0]),
        lambda: a.power_spectrum(np.ones(10), 1, nperseg=20),
        lambda: a.power_spectrum(np.ones(10), 1, noverlap=256),
        lambda: a.power_spectrum([0, np.nan], 1),
        lambda: a.band_power([0, 2, 1], [1, 1, 1], (0, 1)),
        lambda: a.band_power_fraction([0, 1], [1, 1], (0, 2), reference_band_hz=(0, 1)),
        lambda: a.detect_population_bursts(np.zeros((3, 2)), 1, smoothing_ms=-1),
        lambda: a.cycle_boundaries([2, 1], 10),
        lambda: a.cycle_spike_counts(np.zeros((10, 2)), [0, 20]),
        lambda: a.cycle_occupancy_distribution(np.array([[0.5]])),
    ],
)
def test_invalid_estimator_requests(call):
    with pytest.raises(ValueError):
        call()


def test_all_resolved_metadata_is_json_serializable():
    spikes = np.zeros((100, 3), dtype=bool)
    calls = [
        a.per_neuron_firing_rates(spikes, 1),
        a.binned_spike_counts(spikes, 1),
        a.population_rate_trace(spikes, 1),
        a.inter_spike_intervals(spikes, 1),
        a.isi_cv(spikes, 1),
        a.spike_count_correlations(spikes, 1),
        a.power_spectrum(spikes[:, 0], 1),
        a.detect_population_bursts(spikes, 1),
        a.cycle_boundaries([], 100),
        a.cycle_spike_counts(spikes, []),
        a.cycle_occupancy_distribution(np.empty((0, 3), dtype=int)),
        a.spectral_peak([0, 1], [0, 0], (0, 1)),
        a.band_power([0, 1], [0, 0], (0, 1)),
        a.band_power_fraction([0, 1], [0, 0], (0, 1), reference_band_hz=(0, 1)),
    ]
    for result in calls:
        json.dumps(result["metadata"], allow_nan=False)


def test_explicit_spectral_policies_preserve_bin_sum_and_edge_interpolation():
    f = np.arange(6, dtype=float)
    p = 10 - (f - 2.25) ** 2
    regular = a.spectral_peak(f, p, (2, 4), interpolation="parabolic")
    legacy = a.spectral_peak(
        f,
        p,
        (2, 4),
        interpolation="parabolic",
        interpolation_boundary="spectrum",
        clamp_band=False,
    )
    assert regular["frequency_hz"] == 2
    assert legacy["frequency_hz"] == pytest.approx(2.25)
    fraction = a.band_power_fraction(
        f, p, (2, 4), reference_band_hz=(0, 5), integration="bin_sum"
    )
    assert fraction["fraction"] == pytest.approx(p[2:5].sum() / p.sum())
    assert fraction["metadata"]["band"]["edge_interpolation"] == "none"


def test_numpy_estimator_parameters_still_have_json_metadata():
    result = a.power_spectrum(np.arange(20), 1, window=("tukey", np.float32(0.25)))
    json.dumps(result["metadata"], allow_nan=False)
    bursts = a.detect_population_bursts(
        np.zeros((10, 2)), 1, height=np.float32(0), prominence=np.float32(0)
    )
    json.dumps(bursts["metadata"], allow_nan=False)


def test_empty_window_rate_is_undefined_and_silence_is_defined():
    spikes = np.zeros((5, 3), dtype=bool)
    empty = a.per_neuron_firing_rates(spikes, 1, window=(2, 2))
    assert empty["status"] == "empty_window"
    assert np.isnan(empty["rates_hz"]).all()
    np.testing.assert_equal(a.per_neuron_firing_rates(spikes, 1)["rates_hz"], [0, 0, 0])
    assert a.inter_spike_intervals(np.zeros((0, 3)), 1)["status"] == "no_intervals"
