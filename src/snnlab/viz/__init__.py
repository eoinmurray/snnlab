"""Composable visualisation tools, imported on demand."""

from importlib import import_module

from ._version import __version__

_EXPORTS = {'FrameTimeline': 'animation', 'save_animation': 'animation', 'Recording': 'contracts', 'RecordingError': 'contracts', 'Diagram': 'diagrams', 'DiagramEdge': 'diagrams', 'DiagramGroup': 'diagrams', 'DiagramNode': 'diagrams', 'DiagramTheme': 'diagrams', 'diagram_to_dot': 'diagrams', 'render_diagram': 'diagrams', 'FigureGrid': 'figure_grid', 'FigureRect': 'figure_grid', 'FigureRegion': 'figure_grid', 'grid_layout': 'layouts', 'load_snnsim_recording': 'loaders', 'Panel': 'scene', 'Scene': 'scene', 'Theme': 'styles', 'exponential_trace': 'transforms', 'projection_activity': 'transforms', 'representative_frame': 'transforms'}
__all__ = [*_EXPORTS, "__version__"]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
