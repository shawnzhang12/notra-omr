"""Unit tests for measure direction features (including wedges)."""

from __future__ import annotations

import pytest
from notra.ir.measure import Direction


def test_direction_supports_wedge_crescendo() -> None:
    direction = Direction(
        id="dir-001",
        kind="wedge",
        value="crescendo",
        placement="below",
        number=2,
    )
    payload = direction.to_dict()

    assert payload["kind"] == "wedge"
    assert payload["value"] == "crescendo"
    assert payload["number"] == 2


def test_direction_rejects_invalid_wedge_value() -> None:
    with pytest.raises(ValueError):
        Direction(id="dir-001", kind="wedge", value="grow")
