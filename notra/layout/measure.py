"""Measure geometry: boundaries, beat positions, and measure-level layout.

Measures are defined by barline positions on the page. This module
provides helpers for working with measure x-coordinates and grouping
events by measure.
"""

from __future__ import annotations

from notra.layout.annotations import MeasureBoundary


def build_measure_boundaries(
    barline_xs: list[float],
    *,
    image_width: float,
    min_measure_width: float = 10.0,
    start_x: float = 0.0,
) -> list[MeasureBoundary]:
    """Build measure boundaries from barline x-positions.

    Creates MeasureBoundary objects for each span between consecutive
    barlines (plus the left margin and right margin).

    Args:
        barline_xs: Sorted list of barline x-positions.
        image_width: Width of the page image in pixels.
        min_measure_width: Skip spans narrower than this.
        start_x: Left edge x-coordinate (usually 0).

    Returns:
        List of MeasureBoundary objects, numbered from 1.
    """
    all_xs = [start_x] + sorted(barline_xs) + [image_width]
    boundaries: list[MeasureBoundary] = []

    for i in range(len(all_xs) - 1):
        x0, x1 = all_xs[i], all_xs[i + 1]
        if (x1 - x0) < min_measure_width:
            continue
        boundaries.append(
            MeasureBoundary(
                measure_number=i + 1,
                x_start=x0,
                x_end=x1,
                staff_index=0,
                barline_style="regular",
            )
        )

    return boundaries


def events_in_measure(
    events: list,  # list of events with .cx attribute
    boundary: MeasureBoundary,
    *,
    margin: float = 5.0,
) -> list:
    """Filter events that fall within a measure's x-range.

    Args:
        events: List of objects with a `.cx` attribute.
        boundary: The measure boundary.
        margin: Extra margin to add on each side (px).

    Returns:
        Events whose cx falls within [x_start - margin, x_end + margin).
    """
    x0 = boundary.x_start - margin
    x1 = boundary.x_end + margin
    return [e for e in events if x0 <= e.cx < x1]
