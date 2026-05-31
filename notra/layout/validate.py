"""Layout validation: sanity checks on detection results.

Validates that detected layout geometry is internally consistent
and catches common failure modes early.
"""

from __future__ import annotations

from typing import Sequence

from notra.layout.staff import StaffBand


def validate_staff_detection(
    staff_bands: Sequence[StaffBand],
    *,
    min_bands: int = 1,
    max_interline_variance: float = 3.0,
) -> list[str]:
    """Validate staff band detection results.

    Args:
        staff_bands: Detected StaffBand objects.
        min_bands: Minimum expected number of staff bands.
        max_interline_variance: Max allowed interline spacing variance (px).

    Returns:
        List of warning/error messages (empty if valid).
    """
    issues: list[str] = []

    if len(staff_bands) < min_bands:
        issues.append(f"Only {len(staff_bands)} staff bands detected (expected ≥{min_bands})")
        return issues

    # Check interline consistency
    interlines = [b.interline_px for b in staff_bands]
    if interlines:
        mean_il = sum(interlines) / len(interlines)
        for i, il in enumerate(interlines):
            if abs(il - mean_il) > max_interline_variance:
                issues.append(
                    f"Staff band {i} interline ({il:.1f}px) deviates from "
                    f"mean ({mean_il:.1f}px) by >{max_interline_variance}px"
                )

    return issues


def validate_barline_detection(
    barline_xs: Sequence[float],
    image_width: float,
    *,
    min_measure_count: int = 2,
    min_barline_count: int = 1,
) -> list[str]:
    """Validate barline detection results.

    Args:
        barline_xs: Detected barline x-positions.
        image_width: Image width in pixels.
        min_measure_count: Minimum expected number of measures.
        min_barline_count: Minimum expected number of barlines.

    Returns:
        List of issues.
    """
    issues: list[str] = []

    if len(barline_xs) < min_barline_count:
        issues.append(f"Only {len(barline_xs)} barlines detected (expected ≥{min_barline_count})")
        return issues

    # Check that barlines are within image bounds
    for i, x in enumerate(barline_xs):
        if x < 0 or x > image_width:
            issues.append(f"Barline {i} at x={x:.1f} is outside image [0, {image_width}]")

    # Check for suspiciously close barlines
    sorted_xs = sorted(barline_xs)
    for i in range(len(sorted_xs) - 1):
        gap = sorted_xs[i + 1] - sorted_xs[i]
        if gap < 10:
            issues.append(
                f"Barlines at x={sorted_xs[i]:.1f} and {sorted_xs[i + 1]:.1f} "
                f"are too close ({gap:.1f}px)"
            )

    return issues


def validate_notehead_detection(
    notehead_count: int,
    staff_band_count: int,
    *,
    min_notes_per_staff: int = 2,
) -> list[str]:
    """Validate notehead detection results.

    Args:
        notehead_count: Number of detected noteheads.
        staff_band_count: Number of detected staff bands.
        min_notes_per_staff: Minimum notes per staff expected.

    Returns:
        List of issues.
    """
    issues: list[str] = []

    if staff_band_count > 0 and notehead_count < staff_band_count * min_notes_per_staff:
        issues.append(
            f"Only {notehead_count} noteheads detected across {staff_band_count} "
            f"staves (expected ≥{staff_band_count * min_notes_per_staff})"
        )

    return issues
