"""Compose, manipulate and analyse qualitative population models.

Approximation definitions and their extension API live in ``closures``.
"""

from . import closures
from .analysis import (
    Equilibrium,
    Linearisation,
    PiecewiseLinear,
    StabilityCrossing,
    Trajectory,
    continuation,
    evaluate,
    find_equilibrium,
    find_stability_crossings,
    integrate,
    linearise,
)
from .controls import PiecewiseConstant
from .history import ConstantHistory, History
from .manipulation import reduce_states, substitute
from .model import Model
from .reduction import MeanIncomingWeights, RealisedWeights, StationaryRate, reduce
from .symbolic import equations, jacobian, latex, summary

__all__ = [
    "closures",
    "Model",
    "MeanIncomingWeights",
    "RealisedWeights",
    "StationaryRate",
    "reduce",
    "reduce_states",
    "substitute",
    "equations",
    "jacobian",
    "latex",
    "summary",
    "evaluate",
    "find_equilibrium",
    "find_stability_crossings",
    "linearise",
    "integrate",
    "continuation",
    "PiecewiseLinear",
    "Equilibrium",
    "Linearisation",
    "StabilityCrossing",
    "Trajectory",
    "PiecewiseConstant",
    "ConstantHistory",
    "History",
]
