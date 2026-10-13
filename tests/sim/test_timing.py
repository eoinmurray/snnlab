"""Shared time conversions preserve whole-step simulation and exact refractories."""

import pytest

from snnlab.sim.timing import (
    duration_metadata,
    duration_steps,
    refractory_metadata,
    refractory_steps,
)


@pytest.mark.parametrize(
    "dt,steps", [(0.05, 4000), (0.1, 2000), (0.2, 1000), (0.3, 666), (0.6, 333)]
)
def test_selected_grid_whole_trial_steps(dt, steps):
    assert duration_steps(200, dt) == steps
    assert duration_metadata(200, dt) == {
        "nominal_duration_ms": 200.0,
        "duration_steps": steps,
        "realized_duration_ms": steps * dt,
    }


@pytest.mark.parametrize(
    "dt,e_steps,i_steps",
    [(0.05, 24, 12), (0.1, 12, 6), (0.2, 6, 3), (0.3, 4, 2), (0.6, 2, 1)],
)
def test_exact_refraction_conversion_avoids_binary_truncation(dt, e_steps, i_steps):
    assert duration_steps(1.2, dt) == e_steps
    meta = refractory_metadata(1.2, 0.6, dt)
    assert meta["refractory_e_steps"] == e_steps
    assert meta["refractory_i_steps"] == i_steps
    assert meta["realized_refractory_e_ms"] == pytest.approx(1.2)
    assert meta["realized_refractory_i_ms"] == pytest.approx(0.6)


def test_nonintegral_refractory_rejects_quantization():
    with pytest.raises(ValueError, match="not a positive whole number"):
        refractory_steps(0.6, 0.25)
    with pytest.raises(ValueError):
        refractory_steps(0.6, 1.0)
    with pytest.raises(TypeError):
        refractory_steps(0.6, 0.25, policy="nearest")
    with pytest.raises(TypeError):
        refractory_metadata(3.0, 1.5, 1.0, policy="nearest")


def test_zero_refractory_disables_hold():
    assert refractory_steps(0, 0.25) == 0
    assert refractory_metadata(0, 0, 0.25)["realized_refractory_e_ms"] == 0


@pytest.mark.parametrize(
    "duration,dt", [(1, 0), (1, -1), (-1, 1), (float("nan"), 1), (1, float("inf"))]
)
def test_invalid_times_rejected(duration, dt):
    with pytest.raises(ValueError):
        duration_steps(duration, dt)
