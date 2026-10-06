"""Isolate tests that exercise the numerical primitives' default constants."""

import pytest

from snnlab.sim import models as M

_DEFAULTS = {
    name: getattr(M, name)
    for name in ("dt", "SURROGATE_SLOPE", "V_GRAD_DAMPEN", "COBA_INTEGRATOR")
}


@pytest.fixture(autouse=True)
def _restore_kernel_defaults():
    for name, value in _DEFAULTS.items():
        setattr(M, name, value)
    yield
    for name, value in _DEFAULTS.items():
        setattr(M, name, value)
