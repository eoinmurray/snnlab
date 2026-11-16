"""Runtime plasticity rules.

Each rule is a pure update step on tensors. The simulator calls it once per
time step for every plastic projection.
"""

from snnlab.sim.plasticity import stdp_simple

__all__ = ["stdp_simple"]
