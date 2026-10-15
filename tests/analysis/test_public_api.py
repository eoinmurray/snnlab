"""Analysis measurements remain available through one lazy public API."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

from snnlab import analysis

ROOT = Path(__file__).resolve().parents[2]


def test_documented_exports_are_the_complete_public_api():
    overview = (ROOT / "docs/content/docs/api/overview.mdx").read_text()
    section = overview.split("### Analysis\n")[1].split("## Source organisation")[0]
    names = re.findall(r"^\| `analysis\.(\w+)`", section, re.MULTILINE)
    assert len(names) == len(set(names))
    assert set(names) == set(analysis.__all__)
    for name in names:
        assert callable(getattr(analysis, name))
        assert name in dir(analysis)
    namespace = {}
    exec("from snnlab.analysis import *", namespace)
    assert {name for name in namespace if name != "__builtins__"} == set(names)
    assert all(namespace[name] is getattr(analysis, name) for name in names)
    with pytest.raises(AttributeError, match="missing_export"):
        analysis.missing_export


def test_reference_groups_preserve_all_function_anchors():
    reference = (ROOT / "docs/content/docs/api/analysis.mdx").read_text()
    names = re.findall(r"^### (\w+)\n", reference, re.MULTILINE)
    assert len(names) == len(set(names))
    assert set(names) == set(analysis.__all__)
    for group in (
        "Activity", "Intervals", "Correlations", "Spectra", "Rhythmicity",
        "Cycles", "Conductance",
    ):
        assert f"## {group}\n" in reference
    assert 'id="new-measurement-results"' in reference


def test_package_alias_and_basic_measurements_do_not_load_scipy():
    subprocess.run(
        [sys.executable, "-c", """
import sys
import snnlab as snn
assert 'snnlab.analysis' not in sys.modules
assert 'analysis' in dir(snn)
a = snn.analysis
assert 'numpy' not in sys.modules
assert 'scipy' not in sys.modules
import numpy as np
spikes = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
assert a.firing_rate(spikes, 1) == 500
assert a.active_fraction(spikes) == 1
assert a.population_spike_count_cv(spikes, 1) == 0
assert a.binned_spike_counts(spikes, 1)['counts'].shape == (2, 2)
assert a.per_neuron_firing_rates(spikes, 1)['status'] == 'ok'
assert a.population_rate_trace(spikes, 1)['status'] == 'ok'
assert a.inter_spike_intervals(spikes, 1)['status'] == 'ok'
assert a.isi_cv(spikes, 1, min_spikes=2)['status'] == 'ok'
assert a.spike_count_correlations(spikes, 1)['status'] == 'no_valid_pairs'
assert a.population_event_times(spikes, 1).shape == (4,)
assert a.rhythmicity_metrics(spikes, 1) is not None
assert a.conductance_loop_score([1, 2, 3], [3, 2, 1]) is not None
assert 'scipy' not in sys.modules
assert not any(n.startswith('snnlab.analysis.spectra') or
               n.startswith('snnlab.analysis.cycles') for n in sys.modules)
a.population_rate_trace(spikes, 1, smoothing_ms=1)
assert 'scipy.signal' in sys.modules
assert 'matplotlib' not in sys.modules
assert 'torch' not in sys.modules
assert not any(n == 'snnlab.sim' or n.startswith('snnlab.sim.')
               for n in sys.modules)
"""],
        check=True,
    )


def test_all_exports_remain_independent_of_simulation_and_plotting():
    subprocess.run(
        [sys.executable, "-c", """
import sys
import snnlab as snn
for name in snn.analysis.__all__:
    assert callable(getattr(snn.analysis, name))
assert 'torch' not in sys.modules
assert 'matplotlib' not in sys.modules
assert not any(n == 'snnlab.sim' or n.startswith('snnlab.sim.')
               for n in sys.modules)
for name in ('_activity', '_spikes', '_spectra', '_rhythmicity',
             '_cycles', '_conductance'):
    assert 'snnlab.analysis.' + name not in sys.modules
"""],
        check=True,
    )
