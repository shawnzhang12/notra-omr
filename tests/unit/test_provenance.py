"""Unit tests for provenance primitives."""

from __future__ import annotations

import pytest
from notra.core.geometry import BBox
from notra.core.provenance import Provenance


def test_provenance_to_dict() -> None:
    provenance = Provenance(
        source="fixture:m002_four_quarters",
        producer="manual_fixture",
        page=1,
        bbox=BBox(100, 200, 120, 220),
        confidence=1.0,
        notes="golden fixture",
    )

    assert provenance.to_dict() == {
        "source": "fixture:m002_four_quarters",
        "producer": "manual_fixture",
        "page": 1,
        "bbox": {"x0": 100, "y0": 200, "x1": 120, "y1": 220},
        "confidence": 1.0,
        "notes": "golden fixture",
    }


def test_provenance_validation() -> None:
    with pytest.raises(ValueError):
        Provenance(source="", producer="fixture")

    with pytest.raises(ValueError):
        Provenance(source="fixture", producer="", confidence=0.5)

    with pytest.raises(ValueError):
        Provenance(source="fixture", producer="tool", page=0)

    with pytest.raises(ValueError):
        Provenance(source="fixture", producer="tool", confidence=1.1)
