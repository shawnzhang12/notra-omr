"""System-level layout: grouping staves that share barlines.

A system is a group of staves played simultaneously, connected by
a common barline that runs through all staves.
"""

from __future__ import annotations

from notra.layout.page import SystemLayout


def group_systems(
    staff_bands: list,
    barline_xs: list[float],
    *,
    system_gap_min: float = 30.0,
) -> list[SystemLayout]:
    """Group staff bands into systems.

    Staves separated by a vertical gap larger than system_gap_min
    are placed in separate systems.

    Args:
        staff_bands: Detected staff bands, sorted top-to-bottom.
        barline_xs: Detected barline x-positions.
        system_gap_min: Minimum gap between systems in pixels.

    Returns:
        List of SystemLayout objects.
    """
    from notra.layout.page import detect_systems
    return detect_systems(staff_bands, barline_xs, system_gap_min=system_gap_min)
