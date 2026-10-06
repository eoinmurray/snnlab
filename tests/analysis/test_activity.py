"""Known-count measurements, representation errors and migration parity."""

import subprocess
import sys

import numpy as np
import pytest

from snnlab import analysis
from snnlab.sim import metrics as compatibility


def test_known_counts_and_population_denominators():
    spikes = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0]])
    assert analysis.firing_rate(spikes, dt=1) == 250.0
    assert analysis.firing_rate(spikes, dt=1, n_cells=6) == 125.0
    assert analysis.active_fraction(spikes) == pytest.approx(2 / 3)
    assert analysis.active_fraction(spikes, n_cells=6) == pytest.approx(1 / 3)
    # Two bins have population counts 2 and 1: population SD=0.5, mean=1.5.
    assert analysis.population_spike_count_cv(spikes, dt=1) == pytest.approx(1 / 3)


def test_cv_floor_rounding_and_discarded_tail():
    spikes = np.array([[1], [0], [1], [1], [1]])
    assert analysis.population_spike_count_cv(spikes, dt=0.75) == pytest.approx(1 / 3)
    assert analysis.population_spike_count_cv(spikes[:2], dt=0.75) == 0
    assert analysis.population_spike_count_cv(np.zeros((6, 2)), dt=1) == 0


def test_empty_time_and_silent_presentations():
    empty = np.zeros((0, 3))
    with pytest.raises(ValueError, match="timestep"):
        analysis.firing_rate(empty, dt=1)
    assert analysis.active_fraction(empty) == 0
    assert analysis.population_spike_count_cv(empty, dt=1) == 0
    silent = np.zeros((10, 3), dtype=bool)
    assert analysis.firing_rate(silent, dt=1) == 0
    assert analysis.active_fraction(silent) == 0


@pytest.mark.parametrize(
    "spikes", [np.ones(3), np.ones((3, 1, 2)), np.ones((3, 0)), [[0, 2]], [[np.nan, 0]]]
)
@pytest.mark.parametrize(
    "function",
    [
        analysis.firing_rate,
        analysis.active_fraction,
        analysis.population_spike_count_cv,
    ],
)
def test_invalid_raster_is_rejected(spikes, function):
    kwargs = {} if function is analysis.active_fraction else {"dt": 1}
    with pytest.raises(ValueError):
        function(spikes, **kwargs)


@pytest.mark.parametrize("dt", [0, -1, np.nan, np.inf])
@pytest.mark.parametrize(
    "function", [analysis.firing_rate, analysis.population_spike_count_cv]
)
def test_invalid_timestep_is_rejected(dt, function):
    with pytest.raises(ValueError, match="dt"):
        function(np.zeros((4, 2)), dt)


@pytest.mark.parametrize("n_cells", [0, -1, True, 2.5])
def test_invalid_population_denominator_is_rejected(n_cells):
    for function in (analysis.firing_rate, analysis.active_fraction):
        kwargs = {"dt": 1} if function is analysis.firing_rate else {}
        with pytest.raises(ValueError, match="n_cells"):
            function(np.zeros((4, 2)), n_cells=n_cells, **kwargs)


def test_flat_public_surface_and_compatibility_exports():
    assert len(analysis.__all__) == 24
    assert not hasattr(analysis, "metrics")
    for name in compatibility.__all__:
        if name != "compute_metrics":
            assert getattr(compatibility, name) is getattr(analysis, name)
    spikes = np.array([[1, 0], [0, 0], [1, 0], [1, 0]])
    report = compatibility.compute_metrics(spikes, None, dt=1, n_e=2)
    assert report["rate_e"] == analysis.firing_rate(spikes, 1)
    assert report["cv"] == analysis.population_spike_count_cv(spikes, 1)
    assert report["act"] == analysis.active_fraction(spikes)
    assert report["rate_i"] == 0
    assert report["f0_hz"] is None


def test_analysis_does_not_import_simulator_or_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import snnlab.analysis; assert 'torch' not in sys.modules; assert 'snnlab.sim' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
