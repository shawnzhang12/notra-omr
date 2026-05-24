"""Unit tests for geometry primitives."""

from __future__ import annotations

import pytest
from notra.core.geometry import BBox, PageSize, Point


def test_bbox_properties() -> None:
    box = BBox(10.0, 20.0, 30.0, 60.0)

    assert box.width == 20.0
    assert box.height == 40.0
    assert box.area == 800.0
    assert box.center == Point(20.0, 40.0)
    assert box.to_xywh() == (10.0, 20.0, 20.0, 40.0)


def test_bbox_intersection_and_iou() -> None:
    left = BBox(0.0, 0.0, 10.0, 10.0)
    right = BBox(5.0, 5.0, 15.0, 15.0)

    overlap = left.intersection(right)
    assert overlap == BBox(5.0, 5.0, 10.0, 10.0)
    assert left.intersects(right)
    assert left.iou(right) == pytest.approx(25.0 / 175.0)


def test_bbox_translate_expand_and_contains() -> None:
    box = BBox(10.0, 10.0, 20.0, 20.0)

    assert box.contains(Point(10.0, 10.0))
    assert box.contains(Point(15.0, 15.0))
    assert not box.contains(Point(21.0, 15.0))
    assert box.translate(2.0, -2.0) == BBox(12.0, 8.0, 22.0, 18.0)
    assert box.expand(1.5) == BBox(8.5, 8.5, 21.5, 21.5)


def test_bbox_validation() -> None:
    with pytest.raises(ValueError):
        BBox(1.0, 1.0, 1.0, 2.0)

    with pytest.raises(ValueError):
        BBox.from_xywh(0.0, 0.0, 0.0, 2.0)

    with pytest.raises(ValueError):
        BBox(0.0, 0.0, 2.0, 2.0).expand(-0.1)


def test_page_size_validation_and_serialization() -> None:
    page = PageSize(width=2480, height=3508, unit="px")
    assert page.to_dict() == {"width": 2480, "height": 3508, "unit": "px"}

    with pytest.raises(ValueError):
        PageSize(width=0, height=10)
