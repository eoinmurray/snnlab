"""Shared contracts and authoring must not depend on simulator implementations."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "snnlab._shared.neurons",
        "snnlab._shared.connectivity",
        "snnlab._shared.timing",
        "snnlab.extensions",
        "snnlab.lang",
    ],
)
def test_contracts_and_authoring_do_not_import_simulator_or_torch(module):
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, sys\n"
            f"importlib.import_module({module!r})\n"
            "assert 'torch' not in sys.modules\n"
            "assert not any(name == 'snnlab.sim' or name.startswith('snnlab.sim.') "
            "for name in sys.modules)\n",
        ],
        check=True,
    )
