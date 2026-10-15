"""Validation diagnostics and result records."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Diagnostic:
    severity: str
    code: str
    message: str
    subject: str | None = None

    def line(self) -> str:
        where = f" [{self.subject}]" if self.subject else ""
        return f"{self.severity.upper()} {self.code}{where}: {self.message}"


@dataclass
class ValidationResult:
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "warning"]

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("\n".join(d.line() for d in self.errors))
