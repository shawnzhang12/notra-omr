"""Structured validation issue models for deterministic diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from notra.core.provenance import Provenance


class Severity(str, Enum):
    """Severity level for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One structured validation issue."""

    severity: Severity
    code: str
    message: str
    node_id: str | None = None
    related_node_ids: tuple[str, ...] = ()
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("code must be non-empty")
        if not self.message.strip():
            raise ValueError("message must be non-empty")
        for related_id in self.related_node_ids:
            if not related_id.strip():
                raise ValueError("related_node_ids must contain non-empty values")

    def to_dict(self) -> dict[str, object]:
        """Serialize issue to a JSON-friendly dictionary."""
        payload: dict[str, object] = {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
        }
        if self.node_id is not None:
            payload["node_id"] = self.node_id
        if self.related_node_ids:
            payload["related_node_ids"] = list(self.related_node_ids)
        if self.provenance is not None:
            payload["provenance"] = self.provenance.to_dict()
        return payload


@dataclass(slots=True)
class ValidationReport:
    """Collection wrapper for validation issues with helper queries."""

    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, issue: ValidationIssue) -> None:
        """Append one issue to the report."""
        self.issues.append(issue)

    def extend(self, issues: Iterable[ValidationIssue]) -> None:
        """Append a sequence of issues to the report."""
        self.issues.extend(issues)

    @property
    def has_errors(self) -> bool:
        """Return True when at least one error-level issue is present."""
        return any(issue.severity is Severity.ERROR for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Return True when at least one warning-level issue is present."""
        return any(issue.severity is Severity.WARNING for issue in self.issues)

    def by_severity(self, severity: Severity) -> list[ValidationIssue]:
        """Return issues matching a specific severity."""
        return [issue for issue in self.issues if issue.severity is severity]

    def to_dict(self) -> dict[str, object]:
        """Serialize report to a JSON-friendly dictionary."""
        return {
            "summary": {
                "total": len(self.issues),
                "errors": len(self.by_severity(Severity.ERROR)),
                "warnings": len(self.by_severity(Severity.WARNING)),
                "info": len(self.by_severity(Severity.INFO)),
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }
