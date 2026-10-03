"""Composable still-image and animation tools for SNN recordings."""

from ._version import __version__
from .animation import FrameTimeline, save_animation
from .contracts import Recording, RecordingError
from .diagrams import (
    Diagram,
    DiagramEdge,
    DiagramGroup,
    DiagramNode,
    DiagramTheme,
    diagram_to_dot,
    render_diagram,
)
from .figure_grid import FigureGrid, FigureRect, FigureRegion
from .layouts import grid_layout
from .loaders import load_snnsim_recording
from .scene import Panel, Scene
from .styles import Theme
from .transforms import exponential_trace, projection_activity, representative_frame

__all__ = [
    "Diagram",
    "DiagramEdge",
    "DiagramGroup",
    "DiagramNode",
    "DiagramTheme",
    "FrameTimeline",
    "FigureGrid",
    "FigureRect",
    "FigureRegion",
    "Panel",
    "Recording",
    "RecordingError",
    "Scene",
    "Theme",
    "diagram_to_dot",
    "render_diagram",
    "exponential_trace",
    "grid_layout",
    "load_snnsim_recording",
    "projection_activity",
    "representative_frame",
    "save_animation",
    "__version__",
]
