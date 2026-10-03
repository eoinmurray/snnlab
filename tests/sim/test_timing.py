"""Shared time conversions preserve whole-step simulation and exact refractories."""

from types import SimpleNamespace

import pytest

from snnlab.sim.config import Config, build_config, set_sim_dt
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


def test_nonintegral_refractory_requires_declared_quantization():
    with pytest.raises(ValueError, match="not a positive whole number"):
        refractory_steps(0.6, 0.25)
    assert refractory_steps(0.6, 0.25, policy="nearest") == 2
    assert refractory_steps(0.6, 1.0, policy="nearest") == 1
    meta = refractory_metadata(3.0, 1.5, 1.0, policy="nearest")
    assert meta["refractory_policy"] == "nearest"
    assert meta["refractory_i_ms"] == 1.5
    assert meta["realized_refractory_i_ms"] == 2.0


@pytest.mark.parametrize(
    "duration,dt", [(1, 0), (1, -1), (-1, 1), (float("nan"), 1), (1, float("inf"))]
)
def test_invalid_times_rejected(duration, dt):
    with pytest.raises(ValueError):
        duration_steps(duration, dt)


def test_set_sim_dt_rejects_empty_trial():
    with pytest.raises(ValueError, match="at least one timestep"):
        set_sim_dt(0.6, 0.2)


def test_simulation_config_retains_generic_and_explicit_values(monkeypatch):
    from snnlab.sim import config

    monkeypatch.setattr(config, "cfg", config.cfg)
    assert Config().refractory_e_ms == 3.0
    assert Config().refractory_i_ms == 1.5
    cfg = build_config(
        SimpleNamespace(
            refractory_e_ms=1.2, refractory_i_ms=0.6, refractory_policy="exact"
        )
    )
    assert (cfg.refractory_e_ms, cfg.refractory_i_ms, cfg.refractory_policy) == (
        1.2,
        0.6,
        "exact",
    )
