"""Measure geometry: boundaries, beat positions, and measure-level layout.

Measures are defined by barline positions on the page. This module
provides helpers for working with measure x-coordinates and grouping
events by measure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from notra.layout.annotations import MeasureBoundary
from notra.layout.staff import StaffBand


@dataclass(frozen=True, slots=True)
class MeasureBarlineCandidate:
    """One staff-local barline candidate used for measure boundaries."""

    x: float
    width_px: int
    side_ink_ratio: float
    relative_x: float


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


def detect_measure_barlines(
    ink: np.ndarray,
    gray: np.ndarray,
    staff_bands: list[StaffBand],
) -> dict[int, list[float]]:
    """Detect measure-boundary barlines per staff band.

    The detector is tuned for clean rendered pages.  It deliberately avoids
    accepting every vertical staff-spanning line: stems often span the staff,
    so candidates must also have barline-like width, little side ink after
    staff-line erasure, and plausible spacing.
    """
    if ink.size == 0 or gray.size == 0 or not staff_bands:
        return {}

    result: dict[int, list[float]] = {}
    for staff_index, band in enumerate(staff_bands):
        result[staff_index] = _detect_staff_measure_barlines(ink, gray, band)
    return result


def estimate_staff_x_extent(gray: np.ndarray, band: StaffBand) -> tuple[float, float]:
    """Estimate the horizontal staff extent from the five long staff lines."""
    spans = _staff_line_spans(gray, band)
    if len(spans) < 3:
        return 0.0, float(max(0, gray.shape[1] - 1))
    left = float(np.median([span[0] for span in spans]))
    right = float(np.median([span[1] for span in spans]))
    return left, right


def _detect_staff_measure_barlines(
    ink: np.ndarray,
    gray: np.ndarray,
    band: StaffBand,
) -> list[float]:
    line_ys = [int(y) for y in band.line_ys]
    interline = max(4.0, float(band.interline_px))
    spans = _staff_line_spans(gray, band)
    if len(spans) < 3:
        return []

    left = int(round(float(np.median([span[0] for span in spans]))))
    right = int(round(float(np.median([span[1] for span in spans]))))
    staff_width = max(1.0, float(right - left))
    y0 = max(0, int(round(min(line_ys) - interline * 0.15)))
    y1 = min(ink.shape[0], int(round(max(line_ys) + interline * 0.15)))
    if y1 <= y0 or right <= left:
        return []

    cols: list[int] = []
    for x in range(left, right + 1):
        col = ink[:, x] > 0
        inside = col[y0:y1]
        if int(inside.sum()) < interline * 2.5:
            continue
        if _longest_run(inside) < interline * 2.9:
            continue

        gap_hits = 0
        for y_a, y_b in zip(line_ys, line_ys[1:]):
            gap_y0 = int(round(y_a + 2))
            gap_y1 = int(round(y_b - 2))
            if gap_y1 > gap_y0 and bool(col[gap_y0 : gap_y1 + 1].any()):
                gap_hits += 1
        if gap_hits >= 4:
            cols.append(x)

    clusters = _cluster_columns(cols)
    raw_candidates = [
        _score_barline_cluster(ink, band, cluster, left, staff_width, y0, y1)
        for cluster in clusters
        if len(cluster) > 0
    ]
    raw_candidates = [candidate for candidate in raw_candidates if candidate is not None]

    merged = _merge_close_barline_candidates(raw_candidates, interline)
    accepted: list[tuple[float, bool]] = []
    for group in merged:
        if len(group) >= 2:
            accepted.append((max(group, key=lambda item: item.x).x, False))
            continue

        candidate = group[0]
        if candidate.width_px >= 4 and candidate.side_ink_ratio <= 0.25:
            accepted.append((candidate.x, False))
        elif (
            candidate.width_px >= 3
            and candidate.side_ink_ratio <= 0.05
            and candidate.relative_x >= 0.28
        ):
            accepted.append((candidate.x, True))

    return _prune_measure_barlines(accepted, float(left), interline)


def _score_barline_cluster(
    ink: np.ndarray,
    band: StaffBand,
    cluster: list[int],
    staff_left: int,
    staff_width: float,
    y0: int,
    y1: int,
) -> MeasureBarlineCandidate | None:
    width = max(cluster) - min(cluster) + 1
    if width < 3:
        return None

    interline = max(4.0, float(band.interline_px))
    x = float(sum(cluster)) / float(len(cluster))
    clean = ink.astype(bool).copy()
    for ly in band.line_ys:
        y = int(round(ly))
        clean[max(0, y - 1) : min(clean.shape[0], y + 2), :] = False

    x0 = max(0, int(round(x - interline)))
    x1 = min(ink.shape[1], int(round(x + interline)) + 1)
    vx0 = max(0, min(cluster) - 1)
    vx1 = min(ink.shape[1], max(cluster) + 2)
    if x1 <= x0 or vx1 <= vx0:
        return None

    patch = clean[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    vertical = np.zeros_like(patch)
    vertical[:, vx0 - x0 : vx1 - x0] = patch[:, vx0 - x0 : vx1 - x0]

    vertical_ink = int(vertical.sum())
    side_ink = int(patch.sum()) - vertical_ink
    side_ratio = float(side_ink) / float(max(1, vertical_ink))
    relative_x = (x - float(staff_left)) / staff_width

    return MeasureBarlineCandidate(
        x=x,
        width_px=width,
        side_ink_ratio=side_ratio,
        relative_x=relative_x,
    )


def _staff_line_spans(gray: np.ndarray, band: StaffBand) -> list[tuple[int, int, int]]:
    mask = gray < 240
    spans: list[tuple[int, int, int]] = []
    for ly in band.line_ys:
        y = int(round(ly))
        row = mask[max(0, y - 2) : min(mask.shape[0], y + 3), :].sum(axis=0) > 0
        xs = np.where(row)[0]
        if len(xs) == 0:
            continue
        spans.append(max(_merge_1d_runs(xs, max_gap=6), key=lambda item: item[2]))
    return spans


def _merge_1d_runs(xs: np.ndarray, *, max_gap: int) -> list[tuple[int, int, int]]:
    if len(xs) == 0:
        return []
    runs: list[tuple[int, int, int]] = []
    start = int(xs[0])
    prev = int(xs[0])
    for value in xs[1:]:
        x = int(value)
        if x - prev <= max_gap:
            prev = x
            continue
        runs.append((start, prev, prev - start + 1))
        start = prev = x
    runs.append((start, prev, prev - start + 1))
    return runs


def _cluster_columns(cols: list[int]) -> list[list[int]]:
    if not cols:
        return []
    clusters: list[list[int]] = []
    cluster = [cols[0]]
    for x in cols[1:]:
        if x - cluster[-1] <= 3:
            cluster.append(x)
            continue
        clusters.append(cluster)
        cluster = [x]
    clusters.append(cluster)
    return clusters


def _merge_close_barline_candidates(
    candidates: list[MeasureBarlineCandidate],
    interline: float,
) -> list[list[MeasureBarlineCandidate]]:
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda item: item.x)
    groups: list[list[MeasureBarlineCandidate]] = [[candidates[0]]]
    for candidate in candidates[1:]:
        if candidate.x - groups[-1][-1].x <= interline * 1.2:
            groups[-1].append(candidate)
            continue
        groups.append([candidate])
    return groups


def _prune_measure_barlines(
    candidates: list[tuple[float, bool]],
    staff_left: float,
    interline: float,
) -> list[float]:
    """Remove system-start repeats and weak candidates that break spacing."""
    items = sorted(candidates, key=lambda item: item[0])
    items = [item for item in items if item[0] - staff_left > interline * 7.0]

    changed = True
    while changed and len(items) >= 4:
        changed = False
        xs = [item[0] for item in items]
        gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        median_gap = float(np.median(gaps)) if gaps else 0.0
        if median_gap <= 0:
            break

        small_gap_indices = [
            i
            for i, gap in enumerate(gaps)
            if gap < median_gap * 0.75 and gap < interline * 18.0
        ]
        weak_indices = {i for i, item in enumerate(items) if item[1]}
        removable: set[int] = set()
        for gap_index in small_gap_indices:
            if gap_index in weak_indices:
                removable.add(gap_index)
            if gap_index + 1 in weak_indices:
                removable.add(gap_index + 1)
        if not removable:
            break

        baseline = _spacing_dispersion(items)
        best: tuple[float, int] | None = None
        for idx in removable:
            trial = items[:idx] + items[idx + 1 :]
            score = _spacing_dispersion(trial)
            if best is None or score < best[0]:
                best = (score, idx)

        if best is not None and best[0] <= baseline:
            items.pop(best[1])
            changed = True

    return [item[0] for item in items]


def _spacing_dispersion(items: list[tuple[float, bool]]) -> float:
    xs = [item[0] for item in items]
    if len(xs) < 3:
        return 0.0
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    median_gap = float(np.median(gaps))
    return float(sum(abs(gap - median_gap) for gap in gaps)) / max(1.0, median_gap)


def _longest_run(values: np.ndarray) -> int:
    best = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
            continue
        current = 0
    return best
