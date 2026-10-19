"""The public package API resolves implementations without the retired module."""

import importlib
import re
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import pytest

from snnlab import sim


def test_documented_exports_are_the_public_package_api():
    overview = (
        Path(__file__).resolve().parents[2]
        / "docs/content/docs/api/overview.mdx"
    ).read_text()
    section = overview.split("### Sim\n")[1].split("### Viz\n")[0]
    names = re.findall(r"^\| `sim\.(\w+)`", section, re.MULTILINE)
    assert len(names) == len(set(names))
    assert set(names) == set(sim.__all__)
    for name in names:
        assert getattr(sim, name) is not None
        assert name in dir(sim)
    with pytest.raises(AttributeError, match="no attribute 'missing_export'"):
        sim.missing_export


@pytest.mark.parametrize(
    "module, name",
    [
        ("requests.api", "build"),
        ("requests.api", "simulate"),
        ("requests.api", "infer"),
        ("requests.api", "train"),
        ("graph.executor", "GraphExecutor"),
        ("runtime.recording", "RecordingSpec"),
        ("runtime.interventions", "SpikeReplay"),
        ("diagnostics.tracing", "TraceSpec"),
    ],
)
def test_exports_resolve_to_their_implementation_objects(module, name):
    implementation = importlib.import_module(f"snnlab.sim.{module}")
    assert getattr(sim, name) is getattr(implementation, name)


def test_former_execution_module_is_retired():
    with pytest.raises(ModuleNotFoundError, match=r"snnlab\.sim\.execution"):
        importlib.import_module("snnlab.sim.execution")


def test_package_import_is_lazy_in_a_fresh_process():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from snnlab import sim; import sys; "
            "assert 'torch' not in sys.modules; "
            "assert 'snnlab.sim.execution' not in sys.modules; "
            "assert 'ExecutionSpec' not in dir(sim); "
            "from snnlab.sim import simulate; "
            "assert sim.simulate is simulate; "
            "exports = {name: getattr(sim, name) for name in sim.__all__}; "
            "assert 'snnlab.sim.execution' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "name",
    [
        "ExecutionResult",
        "GraphPlan",
        "GraphRuntimeState",
        "TrainingCheckpoint",
        "simulate",
        "train",
    ],
)
def test_public_annotations_resolve_after_module_moves(name):
    assert get_type_hints(getattr(sim, name))


@pytest.mark.parametrize(
    "package",
    ["requests", "inputs", "graph", "runtime", "training", "artifacts", "diagnostics"],
)
def test_internal_packages_do_not_reexport_the_public_api(package):
    implementation = importlib.import_module(f"snnlab.sim.{package}")
    assert set(sim.__all__).isdisjoint(vars(implementation))


def test_numerical_imports_do_not_load_execution_orchestration():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import snnlab.sim.graph.executor; import sys; "
            "assert 'snnlab.sim.requests.handlers' not in sys.modules; "
            "assert 'snnlab.sim.training.trainer' not in sys.modules",
        ],
        check=True,
    )
