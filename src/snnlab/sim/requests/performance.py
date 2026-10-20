"""Shared public performance policy and execution-path reporting."""

from __future__ import annotations

import copy
import warnings
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Literal


@dataclass(frozen=True, kw_only=True)
class Performance:
    """Execution optimizations shared by build, simulate and train.

    Defaults preserve the automatic execution policy. Compilation and capture
    add startup cost; use ``compile_mode=None`` for short eager runs.
    ``fused_optimizer`` applies only to training and ``dataset_prefetch`` only
    to dataset providers. Device, workload and observability stay on the call.
    """

    compile_mode: Literal["default"] | None = "default"
    compile_block_steps: int | Literal["auto"] = "auto"
    cuda_graphs: Literal["auto", "off"] = "auto"
    precision: Literal["auto", "highest"] = "auto"
    fused_optimizer: Literal["auto", "off"] = "auto"
    dataset_prefetch: int = 1
    projection_mode: Literal["fast", "strict"] = "fast"

    def __post_init__(self):
        if self.compile_mode not in (None, "default"):
            raise ValueError("compile_mode must be None or default")
        if self.compile_block_steps != "auto" and (
            type(self.compile_block_steps) is not int
            or not 1 <= self.compile_block_steps <= 64
        ):
            raise ValueError(
                "compile_block_steps must be auto or an integer from 1 to 64"
            )
        for name in ("cuda_graphs", "fused_optimizer"):
            if getattr(self, name) not in ("auto", "off"):
                raise ValueError(f"{name} must be auto or off")
        if self.precision not in ("auto", "highest"):
            raise ValueError("precision must be auto or highest")
        if (
            type(self.dataset_prefetch) is not int
            or not 0 <= self.dataset_prefetch <= 8
        ):
            raise ValueError("dataset_prefetch must be an integer from 0 to 8")
        if self.projection_mode not in ("fast", "strict"):
            raise ValueError("projection_mode must be fast or strict")


class _Unset(Enum):
    VALUE = "unspecified"

    def __repr__(self):
        return "<unspecified>"


_UNSET = _Unset.VALUE


def _resolve_performance(performance, **legacy):
    if performance is not None and not isinstance(performance, Performance):
        raise TypeError("performance must be a Performance or None")
    supplied = {name: value for name, value in legacy.items() if value is not _UNSET}
    if performance is not None and supplied:
        raise TypeError(
            "performance cannot be combined with legacy performance keywords: "
            + ", ".join(sorted(supplied))
        )
    if supplied:
        policy = Performance(**supplied)
        warnings.warn(
            "Direct performance keywords are deprecated; pass "
            "performance=Performance(...) instead: " + ", ".join(sorted(supplied)),
            DeprecationWarning,
            stacklevel=3,
        )
        return policy
    return performance if performance is not None else Performance()


def _performance_from_spec(spec):
    return Performance(**{f.name: getattr(spec, f.name) for f in fields(Performance)})


def _performance_report(spec, model, metrics, *, device, executed, dataset=None):
    """Report policy separately from observed paths; never imply build ran work."""
    from snnlab.sim.graph.precision import amp_enabled
    from snnlab.sim.inputs.bindings import DatasetSnapshotBinding

    if dataset is None:
        dataset = any(
            isinstance(binding, DatasetSnapshotBinding)
            for data in (spec, spec.validation, spec.test)
            if data is not None
            for binding in data.input_bindings
        )

    parameter = next(model.parameters(), None)
    autocast = amp_enabled(spec.precision, parameter)
    precision = (
        {
            "autocast": "bfloat16" if autocast else "off",
            "matmul": ("high" if spec.precision == "auto" else "highest")
            if str(device).startswith("cuda")
            else "caller",
        }
        if executed
        else {"autocast": "not_run", "matmul": "not_run"}
    )
    paths = copy.deepcopy(model._execution_paths) if executed else []
    replay = copy.deepcopy(metrics.get("cuda_graphs")) if executed else None
    return {
        "schema": "snnlab.performance/v1",
        "requested": asdict(_performance_from_spec(spec)),
        "resolved": {
            "device": str(device),
            "compile_mode": model.compile_mode,
            "compile_block_steps": model.compile_block_steps,
            "projection_mode": model.projection_mode,
            "precision": precision,
            "cuda_graphs": replay
            or {"policy": spec.cuda_graphs, "selected": "not_run"},
            "fused_optimizer": metrics.get("fused_optimizer")
            if executed and spec.kind == "train"
            else None,
            "dataset_prefetch": spec.dataset_prefetch if executed and dataset else None,
            "execution_paths": paths,
        },
        "fallbacks": {
            "compiled_blocks": sorted(
                {
                    reason
                    for path in paths
                    for reason in path.get("fallback_reasons", ())
                }
            ),
            "cuda_graphs": copy.deepcopy((replay or {}).get("fallback_reasons", {})),
        },
    }
