"""Shared fixtures for the CLI test suite.

The models.py module carries simulation state as module-level globals (dt, T_ms,
T_steps, the N_* sizes, and the physiological constants tau_gaba, tau_ampa,
max_rate_hz, …). Every entry point mutates these before a forward pass, so a test
that trains, infers, or overrides a time-constant leaves the globals wherever it
last put them. Left unchecked this leaks across tests: e.g. an infer test that
sets a custom tau_gaba changes the GABA decay for every later forward pass, which
made the accuracy and Dale's-law tests fail only in a full-suite run.

We snapshot the module defaults ONCE at import (before any test mutates them) and
restore the whole set around every test, so each test both starts and ends at the
canonical models.py baseline regardless of run order.
"""

from __future__ import annotations

import copy

import pytest

from snnlab.sim import models as M

# Every models.py global any CLI entry point reassigns at runtime. Snapshotted at
# import so the restore target is the true module default, never a leaked value.
_MUTABLE_GLOBALS = (
    "dt", "T_ms", "T_steps",
    "N_IN", "N_HID", "N_INH", "N_OUT", "HIDDEN_SIZES",
    "BATCH_SIZE", "EXACT_K_INITIALIZATION", "SURROGATE_SLOPE", "V_GRAD_DAMPEN",
    "tau_ampa", "tau_gaba",
    "max_rate_hz",
)
_BASELINE = {name: copy.deepcopy(getattr(M, name)) for name in _MUTABLE_GLOBALS}


def _restore_globals():
    for name, value in _BASELINE.items():
        setattr(M, name, copy.deepcopy(value))


@pytest.fixture(autouse=True)
def _reset_model_globals():
    """Pin models.py globals to their module defaults before and after each test."""
    _restore_globals()
    yield
    _restore_globals()
