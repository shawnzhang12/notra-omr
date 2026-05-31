"""Page-level layout: system detection, page-level coordinate reasoning.

Groups staff bands into systems (staves played simultaneously) and
resolves page-level coordinate questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SystemLayout:
    """One system: a group of staves played simultaneously, joined by a barline."""

    system_index: int
    staff_indices: tuple[int, ...]  # indices into the page's staff list
    y_top: int
    y_bottom: int
    x_start: float  # leftmost barline x
    x_end: float  # rightmost barline x


@dataclass
class PageLayout:
    """Page-level layout description."""

    image_width: int
    image_height: int
    systems: list[SystemLayout] = field(default_factory=list)
    staff_count: int = 0

    @property
    def system_count(self) -> int:
        return len(self.systems)


def detect_systems(
    staff_bands: list,
    barline_xs: list[float],
    *,
    system_gap_min: float = 30.0,
) -> list[SystemLayout]:
    """Group staff bands into systems based on barline alignment.

    Staves that share barlines at the same x-positions belong to the
    same system. Staves separated by a large vertical gap are in
    different systems.

    Args:
        staff_bands: Detected staff bands (from staff module).
        barline_xs: Detected barline x-positions.
        system_gap_min: Minimum vertical gap between systems (px).

    Returns:
        List of SystemLayout objects.
    """
    if not staff_bands:
        return []

    bands = list(staff_bands)
    # Sort by y position (top to bottom)
    bands_sorted = sorted(bands, key=lambda b: b.y_top)

    systems: list[SystemLayout] = []
    current_group: list[int] = [0]
    last_y_bottom = bands_sorted[0].y_bottom

    for i in range(1, len(bands_sorted)):
        gap = bands_sorted[i].y_top - last_y_bottom
        if gap > system_gap_min:
            # New system
            systems.append(_make_system(
                len(systems),
                current_group,
                bands_sorted,
                barline_xs,
            ))
            current_group = [i]
        else:
            current_group.append(i)
        last_y_bottom = bands_sorted[i].y_bottom

    # Last system
    if current_group:
        systems.append(_make_system(
            len(systems),
            current_group,
            bands_sorted,
            barline_xs,
        ))

    return systems


def _make_system(
    sys_idx: int,
    staff_indices: list[int],
    bands: list,
    barline_xs: list[float],
) -> SystemLayout:
    y_top = bands[staff_indices[0]].y_top
    y_bottom = bands[staff_indices[-1]].y_bottom

    x_start = 0.0
    x_end = float("inf")
    if barline_xs:
        x_start = barline_xs[0] if barline_xs else 0.0
        x_end = barline_xs[-1] if barline_xs else float("inf")

    return SystemLayout(
        system_index=sys_idx,
        staff_indices=tuple(staff_indices),
        y_top=y_top,
        y_bottom=y_bottom,
        x_start=x_start,
        x_end=x_end,
    )
