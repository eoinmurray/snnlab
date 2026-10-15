"""Versioned, fail-closed numerical conformance reports for SNNLANG migration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import torch

CONFORMANCE_REPORT_SCHEMA = "tools/snnsim.conformance-report/v1"
ComparisonMode = Literal["exact", "numeric"]


@dataclass(frozen=True)
class ComparisonPolicy:
    mode: ComparisonMode = "exact"
    atol: float = 0.0
    rtol: float = 0.0
    equal_nan: bool = False

    def __post_init__(self):
        if self.mode not in {"exact", "numeric"}:
            raise ValueError(f"unsupported conformance comparison mode {self.mode}")
        if self.atol < 0 or self.rtol < 0:
            raise ValueError("conformance tolerances must be non-negative")
        if self.mode == "exact" and (self.atol or self.rtol):
            raise ValueError("exact conformance policy cannot declare tolerances")


@dataclass(frozen=True)
class TensorComparison:
    layer: str
    field: str
    passed: bool
    reason: str | None
    reference_shape: list[int] | None
    candidate_shape: list[int] | None
    reference_dtype: str | None
    candidate_dtype: str | None
    policy: ComparisonPolicy
    max_abs_error: float | None = None
    max_rel_error: float | None = None


@dataclass(frozen=True)
class ConformanceReport:
    case_id: str
    comparisons: tuple[TensorComparison, ...]
    schema: str = CONFORMANCE_REPORT_SCHEMA

    @property
    def passed(self) -> bool:
        return all(row.passed for row in self.comparisons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "case_id": self.case_id,
            "passed": self.passed,
            "summary": {
                "comparisons": len(self.comparisons),
                "passed": sum(row.passed for row in self.comparisons),
                "failed": sum(not row.passed for row in self.comparisons),
            },
            "comparisons": [asdict(row) for row in self.comparisons],
        }

    def require_passed(self) -> None:
        failures = [
            f"{row.layer}.{row.field}: {row.reason}"
            for row in self.comparisons
            if not row.passed
        ]
        if failures:
            raise AssertionError("conformance failed: " + "; ".join(failures))


def _dtype(value: torch.Tensor) -> str:
    return str(value.dtype).removeprefix("torch.")


def canonical_json_tensor(value: Any) -> torch.Tensor:
    """Encode JSON-compatible structure for exact named conformance comparison."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return torch.tensor(list(encoded), dtype=torch.uint8)


def remap_named_tensors(
    values: Mapping[str, torch.Tensor], mapping: Mapping[str, str]
) -> dict[str, torch.Tensor]:
    """Apply an explicit complete name map without positional or partial fallback."""
    if set(values) != set(mapping):
        raise ValueError(
            f"conformance name map must be complete; missing={sorted(set(values) - set(mapping))}, extra={sorted(set(mapping) - set(values))}"
        )
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("conformance name map contains duplicate destination names")
    return {mapping[name]: values[name] for name in sorted(values)}


def _error_bounds(
    reference: torch.Tensor, candidate: torch.Tensor
) -> tuple[float, float]:
    if reference.numel() == 0:
        return 0.0, 0.0
    # MPS does not support float64, so leave the accelerator before widening
    # the values for stable error measurement.
    reference = reference.detach().to(device="cpu").to(dtype=torch.float64)
    candidate = candidate.detach().to(device="cpu").to(dtype=torch.float64)
    absolute = (reference - candidate).abs()
    relative = absolute / reference.abs().clamp_min(torch.finfo(torch.float64).tiny)
    return float(absolute.max()), float(relative.max())


def compare_conformance_layers(
    case_id: str,
    reference: Mapping[str, Mapping[str, torch.Tensor]],
    candidate: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    policies: Mapping[str, Mapping[str, ComparisonPolicy]] | None = None,
) -> ConformanceReport:
    """Compare named tensor layers with explicit coverage and tolerance policies."""
    policies = policies or {}
    rows: list[TensorComparison] = []
    for layer in sorted(set(reference) | set(candidate)):
        reference_fields = reference.get(layer, {})
        candidate_fields = candidate.get(layer, {})
        for field in sorted(set(reference_fields) | set(candidate_fields)):
            policy = policies.get(layer, {}).get(field, ComparisonPolicy())
            if field not in reference_fields or field not in candidate_fields:
                rows.append(
                    TensorComparison(
                        layer=layer,
                        field=field,
                        passed=False,
                        reason="missing from reference"
                        if field not in reference_fields
                        else "missing from candidate",
                        reference_shape=list(reference_fields[field].shape)
                        if field in reference_fields
                        else None,
                        candidate_shape=list(candidate_fields[field].shape)
                        if field in candidate_fields
                        else None,
                        reference_dtype=_dtype(reference_fields[field])
                        if field in reference_fields
                        else None,
                        candidate_dtype=_dtype(candidate_fields[field])
                        if field in candidate_fields
                        else None,
                        policy=policy,
                    )
                )
                continue
            expected = reference_fields[field]
            actual = candidate_fields[field]
            if expected.shape != actual.shape:
                reason = "shape mismatch"
                passed = False
                max_abs = max_rel = None
            elif expected.dtype != actual.dtype:
                reason = "dtype mismatch"
                passed = False
                max_abs = max_rel = None
            else:
                max_abs, max_rel = _error_bounds(expected, actual)
                expected_value = expected.detach().to(device="cpu")
                actual_value = actual.detach().to(device="cpu")
                if policy.mode == "exact":
                    passed = torch.equal(expected_value, actual_value)
                else:
                    passed = torch.allclose(
                        expected_value,
                        actual_value,
                        atol=policy.atol,
                        rtol=policy.rtol,
                        equal_nan=policy.equal_nan,
                    )
                reason = None if passed else "value mismatch"
            rows.append(
                TensorComparison(
                    layer=layer,
                    field=field,
                    passed=passed,
                    reason=reason,
                    reference_shape=list(expected.shape),
                    candidate_shape=list(actual.shape),
                    reference_dtype=_dtype(expected),
                    candidate_dtype=_dtype(actual),
                    policy=policy,
                    max_abs_error=max_abs,
                    max_rel_error=max_rel,
                )
            )
    declared = {
        (layer, field) for layer, fields in policies.items() for field in fields
    }
    compared = {(row.layer, row.field) for row in rows}
    unused = sorted(declared - compared)
    if unused:
        raise ValueError(f"conformance policies reference absent fields: {unused}")
    return ConformanceReport(case_id=case_id, comparisons=tuple(rows))


def write_conformance_report(path: str | Path, report: ConformanceReport) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return destination
