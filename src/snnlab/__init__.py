"""Author, simulate and visualise spiking neural networks.

Subpackages are imported on demand: from snnlab import lang, sim, viz, analysis, extensions.
"""

from importlib import import_module

__version__ = "0.17.0"

_SUBPACKAGES = ("lang", "sim", "viz", "analysis", "extensions")
__all__ = [*_SUBPACKAGES, "__version__"]


def __getattr__(name):
    if name not in _SUBPACKAGES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(f".{name}", __name__)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
