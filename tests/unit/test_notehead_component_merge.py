"""Tests for notehead component merging."""

from __future__ import annotations

from notra.layout.staff import StaffBand
from notra.layout.symbol import merge_bisected_components


def test_merge_bisected_components_uses_y_gap_not_x_gap() -> None:
    band = StaffBand(line_ys=(100, 121, 142, 162, 183), interline_px=20.75)
    upper = (180.0, 496.0, 133.5, 483, 128, 507, 137)
    lower = (215.0, 491.0, 145.5, 481, 141, 506, 158)

    merged = merge_bisected_components([upper, lower], [band])

    assert len(merged) == 1
    area, cx, cy, x0, y0, x1, y1 = merged[0]
    assert area == 395.0
    assert 490.0 < cx < 497.0
    assert 139.0 < cy < 143.0
    assert (x0, y0, x1, y1) == (481, 128, 507, 158)
