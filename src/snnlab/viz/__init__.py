"""Composable visualisation tools, imported on demand."""

from importlib import import_module

from ._version import __version__

_EXPORTS = {
    "plot_raster": "plotting.primitives",
    "plot_learning_curves": "plotting.primitives",
    "FrameTimeline": "plotting.animation",
    "save_animation": "plotting.animation",
    "Recording": "recording",
    "RecordingError": "recording",
    "Diagram": "diagrams.model",
    "DiagramEdge": "diagrams.model",
    "DiagramGroup": "diagrams.model",
    "DiagramNode": "diagrams.model",
    "DiagramTheme": "diagrams.model",
    "WeightLabel": "diagrams.model",
    "diagram_to_dot": "diagrams.dot",
    "render_diagram": "diagrams.render",
    "FigureGrid": "plotting.grid",
    "FigureRect": "plotting.geometry",
    "FigureRegion": "plotting.geometry",
    "grid_layout": "plotting.geometry",
    "Panel": "plotting.scene",
    "Scene": "plotting.scene",
    "Theme": "plotting.theme",
    "exponential_trace": "transforms",
    "projection_activity": "transforms",
    "representative_frame": "transforms",
}
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
