"""Deterministic time-signature detection for rendered staff openings.

This is deliberately small and image-processing based.  The current cello
fixtures only need three outcomes:

* common time or omitted explicit time => 4/4
* stacked 2 over 4 => 2/4
* stacked 3 over 4 => 3/4

The detector works in the first staff opening, erases staff lines, groups
staff-relative glyph components, and classifies the upper numeral.  It is not a
general OCR engine; it is an overfit deterministic spine that can later be
replaced or augmented by a learned structural classifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from notra.layout.staff import StaffBand


@dataclass(frozen=True, slots=True)
class TimeSignatureCandidate:
    """One time-signature hypothesis."""

    beats: int
    beat_type: int
    visual_class: str
    confidence: float
    bbox: tuple[int, int, int, int] | None = None

    @property
    def signature(self) -> str:
        return f"{self.beats}/{self.beat_type}"


@dataclass(frozen=True, slots=True)
class _Component:
    area: int
    x0: int
    y0: int
    x1: int
    y1: int
    cx: float
    cy: float

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True, slots=True)
class _Group:
    bbox: tuple[int, int, int, int]
    cx: float
    area: int
    upper_area: int
    lower_area: int
    score: float


def detect_time_signature(
    ink: np.ndarray,
    staff_bands: list[StaffBand],
    *,
    default_beats: int = 4,
    default_beat_type: int = 4,
) -> TimeSignatureCandidate:
    """Detect a simple time signature from the first staff opening.

    Returns a default 4/4 hypothesis when no explicit numeric signature is
    confidently found.  This covers common-time glyphs and pieces that omit the
    time signature after the rendered opening.
    """
    if ink.ndim != 2:
        raise ValueError("ink must be a 2-D binary image")
    if not staff_bands:
        return TimeSignatureCandidate(default_beats, default_beat_type, "default", 0.0)

    band = staff_bands[0]
    groups = _find_time_like_groups(ink, band)
    if not groups:
        return TimeSignatureCandidate(
            default_beats,
            default_beat_type,
            "common_or_default",
            0.25,
        )

    best = max(groups, key=lambda item: item.score)
    digit = _classify_upper_digit(ink, band, best.bbox)
    if digit == "2":
        return TimeSignatureCandidate(2, 4, "2/4", min(0.99, best.score), best.bbox)
    if digit == "3":
        return TimeSignatureCandidate(3, 4, "3/4", min(0.99, best.score), best.bbox)

    return TimeSignatureCandidate(
        default_beats,
        default_beat_type,
        "4/4_or_common",
        min(0.95, best.score),
        best.bbox,
    )


def _find_time_like_groups(ink: np.ndarray, band: StaffBand) -> list[_Group]:
    interline = max(4.0, float(band.interline_px))
    line_min = min(band.line_ys)
    line_max = max(band.line_ys)
    center_y = (line_min + line_max) / 2.0
    staff_left = _estimate_staff_left(ink, band)

    y0 = max(0, int(round(line_min - interline * 1.8)))
    y1 = min(ink.shape[0], int(round(line_max + interline * 1.8)))
    x0 = max(0, int(round(staff_left - interline)))
    # Beginner-score time signatures are in the opening after clef/key and
    # before the first notes.  This cap prevents fingering numbers over later
    # notes from becoming false 2/4 or 3/4 detections.
    x1 = min(ink.shape[1], int(round(staff_left + interline * 11.5)))
    if y1 <= y0 or x1 <= x0:
        return []

    roi = ink[y0:y1, x0:x1].astype(bool).copy()
    _erase_staff_lines(roi, band, y_offset=y0)
    components = _filter_components(
        _connected_components(roi),
        x_offset=x0,
        y_offset=y0,
        band=band,
    )

    groups: list[_Group] = []
    for anchor in sorted(components, key=lambda item: item.cx):
        members = [
            item
            for item in components
            if item.x1 >= anchor.cx - interline * 0.9
            and item.x0 <= anchor.cx + interline * 0.9
        ]
        if not members:
            continue

        gx0 = min(item.x0 for item in members)
        gx1 = max(item.x1 for item in members)
        gy0 = min(item.y0 for item in members)
        gy1 = max(item.y1 for item in members)
        width = gx1 - gx0
        height = gy1 - gy0
        if width > interline * 3.0:
            continue

        upper_area = sum(item.area for item in members if item.cy < center_y)
        lower_area = sum(item.area for item in members if item.cy >= center_y)
        if upper_area < interline * interline * 0.08:
            continue
        if lower_area < interline * interline * 0.08:
            continue

        area = upper_area + lower_area
        cx = sum(item.cx * item.area for item in members) / float(area)
        if cx < staff_left + interline * 4.0:
            continue
        if cx > staff_left + interline * 11.5:
            continue

        score = float(area)
        if height < interline * 2.0:
            score *= 0.4
        if width < interline * 0.9:
            score *= 0.5
        score = score / max(1.0, interline * interline)

        bbox = (int(round(gx0)), int(round(gy0)), int(round(gx1)), int(round(gy1)))
        if any(existing.bbox == bbox for existing in groups):
            continue
        groups.append(
            _Group(
                bbox=bbox,
                cx=cx,
                area=area,
                upper_area=upper_area,
                lower_area=lower_area,
                score=score,
            )
        )

    return groups


def _classify_upper_digit(
    ink: np.ndarray,
    band: StaffBand,
    bbox: tuple[int, int, int, int],
) -> str | None:
    crop = _upper_digit_crop(ink, band, bbox)
    if crop is None:
        return None

    h, w = crop.shape
    if h < 8 or w < 8:
        return None

    third_w = max(1, w // 3)
    third_h = max(1, h // 3)
    left = float(crop[:, :third_w].mean())
    mid = float(crop[:, third_w : min(w, 2 * third_w)].mean())
    right = float(crop[:, min(w, 2 * third_w) :].mean())
    bottom = float(crop[min(h, 2 * third_h) :, :].mean())
    q7 = float(crop[min(h, 2 * third_h) :, third_w : min(w, 2 * third_w)].mean())

    # A 4 has a strong central vertical stroke and a relatively empty right
    # third after staff-line removal.
    if mid > left + 0.12 and mid > right + 0.18 and right < 0.25 and right < left * 0.85:
        return "4"

    # A 2 has a visible lower horizontal/diagonal mass in the upper numeral.
    # This separates it from the rendered 3, whose lower third is weak in the
    # Verovio/Bravura crops except when the 3 has strong right-side mass.
    if bottom > 0.27 and q7 > 0.38:
        return "2"

    if bottom < 0.25 and q7 > 0.38:
        return "3"

    # A 3 is right-heavy relative to the left side, after rejecting 2 and 4.
    if right >= left + 0.025:
        return "3"

    return None


def _upper_digit_crop(
    ink: np.ndarray,
    band: StaffBand,
    bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    interline = max(4.0, float(band.interline_px))
    line_min = min(band.line_ys)
    line_max = max(band.line_ys)
    center_y = (line_min + line_max) / 2.0
    x0, _y0, x1, _y1 = bbox

    pad_x = max(2, int(round(interline * 0.15)))
    top = max(0, int(round(line_min - interline * 0.4)))
    bottom = min(ink.shape[0], int(round(center_y + interline * 0.3)))
    left = max(0, int(round(x0 - pad_x)))
    right = min(ink.shape[1], int(round(x1 + pad_x)))
    if bottom <= top or right <= left:
        return None

    crop = ink[top:bottom, left:right].astype(bool).copy()
    _erase_staff_lines(crop, band, y_offset=top)
    pts = np.argwhere(crop)
    if pts.size == 0:
        return None
    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0) + 1
    return crop[y_min:y_max, x_min:x_max]


def _filter_components(
    components: list[_Component],
    *,
    x_offset: int,
    y_offset: int,
    band: StaffBand,
) -> list[_Component]:
    interline = max(4.0, float(band.interline_px))
    line_min = min(band.line_ys)
    line_max = max(band.line_ys)
    filtered: list[_Component] = []

    for comp in components:
        width = comp.width
        height = comp.height
        if comp.area < interline * interline * 0.08:
            continue
        if width < interline * 0.20 or height < interline * 0.30:
            continue
        if width > interline * 3.2 or height > interline * 2.4:
            continue

        absolute = _Component(
            area=comp.area,
            x0=comp.x0 + x_offset,
            y0=comp.y0 + y_offset,
            x1=comp.x1 + x_offset,
            y1=comp.y1 + y_offset,
            cx=comp.cx + x_offset,
            cy=comp.cy + y_offset,
        )
        if absolute.y1 < line_min - interline * 0.5:
            continue
        if absolute.y0 > line_max + interline * 0.7:
            continue
        filtered.append(absolute)

    return filtered


def _connected_components(binary: np.ndarray) -> list[_Component]:
    h, w = binary.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[_Component] = []

    for y in range(h):
        for x in np.where(binary[y] & ~seen[y])[0]:
            if seen[y, x] or not binary[y, x]:
                continue
            stack = [(y, int(x))]
            seen[y, x] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                cy, cx = stack.pop()
                xs.append(cx)
                ys.append(cy)
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if not (0 <= ny < h and 0 <= nx < w):
                            continue
                        if seen[ny, nx] or not binary[ny, nx]:
                            continue
                        seen[ny, nx] = True
                        stack.append((ny, nx))

            xs_arr = np.asarray(xs)
            ys_arr = np.asarray(ys)
            components.append(
                _Component(
                    area=len(xs),
                    x0=int(xs_arr.min()),
                    y0=int(ys_arr.min()),
                    x1=int(xs_arr.max() + 1),
                    y1=int(ys_arr.max() + 1),
                    cx=float(xs_arr.mean()),
                    cy=float(ys_arr.mean()),
                )
            )

    return components


def _erase_staff_lines(roi: np.ndarray, band: StaffBand, *, y_offset: int) -> None:
    half_width = max(2, int(round(band.interline_px * 0.16)))
    for line_y in band.line_ys:
        y = int(round(line_y - y_offset))
        top = max(0, y - half_width)
        bottom = min(roi.shape[0], y + half_width + 1)
        if bottom > top:
            roi[top:bottom, :] = False


def _estimate_staff_left(ink: np.ndarray, band: StaffBand) -> int:
    starts: list[int] = []
    for y in band.line_ys:
        row_index = int(round(y))
        if not (0 <= row_index < ink.shape[0]):
            continue
        cols = np.where(ink[row_index] > 0)[0]
        if cols.size:
            starts.append(int(cols[0]))
    if starts:
        return min(starts)
    return 0
