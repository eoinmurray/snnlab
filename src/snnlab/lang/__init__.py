"""Public graph-authoring API, with implementations imported on demand.

Import supported constructors, operations, readouts and recipe helpers directly
from ``snnlab.lang``. Implementation module paths are private.
"""

from importlib import import_module

from ._version import __version__

_EXPORTS = {
    "duration_steps": "._shared.timing",
    "duration_metadata": "._shared.timing",
    "refractory_metadata": "._shared.timing",
    "ConstantGain": "modulation",
    "StepGain": "modulation",
    "RampGain": "modulation",
    "SineGain": "modulation",
    "PiecewiseGain": "modulation",
    "CustomGain": "modulation",
    "ADEX": "components",
    "ALIF": "components",
    "AMPA": "components",
    "GABA": "components",
    "COBA_LIF": "components",
    "CUBA_LIF": "components",
    "COBA_ALIF": "components",
    "CUBA_ALIF": "components",
    "COBA_ADEX": "components",
    "CUBA_ADEX": "components",
    "ExponentialCurrent": "components",
    "CustomNeuron": "components",
    "CustomSynapse": "components",
    "CustomInitializer": "parameters",
    "CustomConstraint": "parameters",
    "LIF": "components",
    "Constant": "parameters",
    "LeakyIntegrator": "components",
    "LowerClampedNormal": "parameters",
    "Modulatory": "components",
    "Network": "network",
    "NonNegative": "parameters",
    "Normal": "parameters",
    "ParameterRef": "network",
    "Population": "network",
    "Projection": "network",
    "Quantity": "units",
    "Signal": "network",
    "SignedNormal": "parameters",
    "Uniform": "parameters",
    "Zeros": "parameters",
    "TrainSpec": "training",
    "Bundle": "compilation.bundles",
    "Diagnostic": "compilation.diagnostics",
    "compile": "compilation.compile",
    "diagram": "diagrams",
    "load_bundle": "compilation.bundles",
    "migrate_refractory": "compilation.graph_validation",
    "validate_graph": "compilation.graph_validation",
    "Hz": "units",
    "mV": "units",
    "ms": "units",
    "nS": "units",
    "uS": "units",
    "nA": "units",
    "linear": "operations",
    "reduce": "operations",
    "divide": "operations",
    "Readout": "readouts",
    "MeanVoltage": "readouts",
    "FinalVoltage": "readouts",
    "SpikeCount": "readouts",
    "SpikeRate": "readouts",
    "CumulativePotential": "readouts",
    "Objective": "training",
    "CrossEntropy": "training",
    "ParameterGroup": "training",
    "Regularizer": "training",
    "UpperRatePenalty": "training",
    "SpikeBudgetPenalty": "training",
    "Optimizer": "training",
    "AdamW": "training",
    "FastSigmoid": "training",
    "CustomObjective": "training",
    "CustomRegularizer": "training",
    "CustomOptimizer": "training",
    "CustomSurrogate": "training",
    "StopGradient": "training",
    "ValidationResult": "compilation.diagnostics",
    "validate_training": "compilation.training_validation",
    "SignalLike": "network",
    "custom_operation": "operations",
}
_ALIASES = {"custom_operation": "custom"}
__all__ = [*_EXPORTS, "__version__"]


def __getattr__(name):
    # Retained, hash-pinned recipes still use the former operations namespace.
    if name == "ops":
        value = import_module(".operations", __name__)
    else:
        module = _EXPORTS.get(name)
        if module is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(import_module(f".{module}", __name__), _ALIASES.get(name, name))
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
