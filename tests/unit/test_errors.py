"""Unit tests for structured validation issues."""

from __future__ import annotations

import pytest
from notra.core.errors import Severity, ValidationIssue, ValidationReport
from notra.core.geometry import BBox
from notra.core.provenance import Provenance


def test_validation_issue_serialization() -> None:
    issue = ValidationIssue(
        severity=Severity.ERROR,
        code="MEASURE_DURATION_OVERFLOW",
        message="Voice duration exceeds the measure budget by 1/8.",
        node_id="measure-007",
        related_node_ids=("event-031", "event-032"),
        provenance=Provenance(
            source="fixture:m008_ties",
            producer="validator",
            page=1,
            bbox=BBox(320, 180, 410, 260),
            confidence=0.99,
        ),
    )

    payload = issue.to_dict()
    assert payload["severity"] == "error"
    assert payload["code"] == "MEASURE_DURATION_OVERFLOW"
    assert payload["node_id"] == "measure-007"
    assert payload["related_node_ids"] == ["event-031", "event-032"]
    assert payload["provenance"] == {
        "source": "fixture:m008_ties",
        "producer": "validator",
        "page": 1,
        "bbox": {"x0": 320, "y0": 180, "x1": 410, "y1": 260},
        "confidence": 0.99,
    }


def test_validation_report_queries() -> None:
    report = ValidationReport()
    report.add(ValidationIssue(Severity.INFO, "INFO_CASE", "Informational note."))
    report.add(ValidationIssue(Severity.WARNING, "WARN_CASE", "Potential issue."))
    report.add(ValidationIssue(Severity.ERROR, "ERR_CASE", "Hard validation error."))

    assert report.has_warnings
    assert report.has_errors
    assert len(report.by_severity(Severity.ERROR)) == 1
    assert report.to_dict()["summary"] == {"total": 3, "errors": 1, "warnings": 1, "info": 1}


def test_validation_issue_rejects_empty_fields() -> None:
    with pytest.raises(ValueError):
        ValidationIssue(Severity.ERROR, "", "message")

    with pytest.raises(ValueError):
        ValidationIssue(Severity.ERROR, "CODE", "")

    with pytest.raises(ValueError):
        ValidationIssue(Severity.ERROR, "CODE", "message", related_node_ids=("",))
