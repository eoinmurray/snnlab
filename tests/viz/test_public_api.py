"""Viz has one public export location, independent of its implementation layout."""

import subprocess
import sys
from typing import get_type_hints

import pytest

from snnlab import viz


def test_all_public_exports_resolve():
    names = viz.__all__
    assert len(names) == len(set(names))
    for name in names:
        assert getattr(viz, name) is not None
        assert name in dir(viz)
    with pytest.raises(AttributeError, match="missing_export"):
        viz.missing_export


def test_package_alias_and_backend_independent_exports_are_lazy():
    subprocess.run(
        [sys.executable, "-c", """
import sys
import snnlab as snn
assert 'snnlab.viz' not in sys.modules
assert 'viz' in dir(snn)
v = snn.viz
assert 'numpy' not in sys.modules
assert 'matplotlib' not in sys.modules
for name in ('Diagram', 'DiagramEdge', 'DiagramGroup', 'DiagramNode',
             'DiagramTheme', 'diagram_to_dot', 'render_diagram'):
    getattr(v, name)
assert 'numpy' not in sys.modules
for name in ('Recording', 'RecordingError', 'FigureRect', 'FigureRegion',
             'grid_layout', 'Theme', 'exponential_trace',
             'projection_activity', 'representative_frame'):
    getattr(v, name)
assert v.FigureRect(0, 0, 1, 1).mpl == (0, 0, 1, 1)
assert v.grid_layout(3, columns=2).shape == (3, 2)
assert 'matplotlib' not in sys.modules
assert 'torch' not in sys.modules
assert not any(n == 'snnlab.sim' or n.startswith('snnlab.sim.')
               for n in sys.modules)
for name in ('FigureGrid', 'Scene', 'Panel', 'FrameTimeline', 'save_animation'):
    getattr(v, name)
assert 'matplotlib' in sys.modules
assert 'torch' not in sys.modules
for package in (v.diagrams, v.plotting):
    assert not any(name in vars(package) for name in v.__all__)
for name in ('animation', 'contracts', 'figure_grid', 'layouts', 'scene', 'styles'):
    assert 'snnlab.viz.' + name not in sys.modules
"""],
        check=True,
    )


@pytest.mark.parametrize(
    "name",
    ["Recording", "Diagram", "DiagramNode", "DiagramEdge", "DiagramGroup",
     "DiagramTheme", "FigureRect", "FigureRegion", "FigureGrid", "Scene",
     "FrameTimeline", "diagram_to_dot", "render_diagram", "grid_layout",
     "save_animation"],
)
def test_public_type_annotations_resolve(name):
    assert get_type_hints(getattr(viz, name))
