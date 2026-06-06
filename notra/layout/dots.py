"""Augmentation-dot detection.

The detector is deliberately staff-relative and conservative. A dot is a small
compact ink component to the right of a notehead/rest, in a legal staff-space
position. It is evidence for duration hypotheses, not a hard duration label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from notra.layout.staff import StaffBand
from notra.layout.symbol import NoteheadCandidate


@dataclass(frozen=True, slots=True)
class AugmentationDotCandidate:
    """One detected augmentation-dot component."""

    event_index: int
    cx: float
    cy: float
    bbox: tuple[int, int, int, int]
    area: int
    staff_band_index: int
    confidence: float


@dataclass(frozen=True, slots=True)
class _Component:
    area: int
    cx: float
    cy: float
    bbox: tuple[int, int, int, int]


def detect_augmentation_dots(
    ink: np.ndarray,
    staff_bands: list[StaffBand],
    noteheads: list[NoteheadCandidate] | tuple[NoteheadCandidate, ...],
    *,
    rests: list[NoteheadCandidate] | tuple[NoteheadCandidate, ...] = (),
    rest_event_index_offset: int | None = None,
    max_dots: int = 1,
) -> tuple[dict[int, int], list[AugmentationDotCandidate]]:
    """Detect augmentation dots for note/rest events.

    Returns:
        A map from pipeline event index to dot count, plus the accepted dot
        components for debug overlays.
    """
    if ink.size == 0 or not staff_bands or max_dots <= 0:
        return {}, []

    h, w = ink.shape
    binary = ink.astype(bool)
    offset = len(noteheads) if rest_event_index_offset is None else rest_event_index_offset
    symbols: list[tuple[int, NoteheadCandidate]] = [
        (idx, item) for idx, item in enumerate(noteheads)
    ]
    symbols.extend((offset + idx, item) for idx, item in enumerate(rests))

    dot_map: dict[int, int] = {}
    dot_candidates: list[AugmentationDotCandidate] = []

    for event_index, symbol in symbols:
        if symbol.staff_band_index < 0 or symbol.staff_band_index >= len(staff_bands):
            continue
        band = staff_bands[symbol.staff_band_index]
        interline = max(4.0, float(band.interline_px))
        x0, y0, x1, y1 = _search_window(symbol, interline, w, h)
        if x1 <= x0 or y1 <= y0:
            continue

        patch = binary[y0:y1, x0:x1].copy()
        _erase_staff_lines(patch, band, patch_y0=y0)
        accepted: list[AugmentationDotCandidate] = []
        for component in _connected_components(patch, x0=x0, y0=y0):
            confidence = _score_dot_component(component, symbol, band, interline)
            if confidence <= 0.0:
                continue
            accepted.append(
                AugmentationDotCandidate(
                    event_index=event_index,
                    cx=component.cx,
                    cy=component.cy,
                    bbox=component.bbox,
                    area=component.area,
                    staff_band_index=symbol.staff_band_index,
                    confidence=confidence,
                )
            )

        if not accepted:
            continue
        accepted.sort(key=lambda item: (item.cx, -item.confidence))
        kept = accepted[:max_dots]
        dot_map[event_index] = len(kept)
        dot_candidates.extend(kept)

    return dot_map, dot_candidates


def _search_window(
    symbol: NoteheadCandidate,
    interline: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    sx0, sy0, sx1, sy1 = symbol.bbox
    x0 = int(round(max(sx1 + interline * 0.10, symbol.cx + interline * 0.45)))
    x1 = int(round(min(image_width, sx1 + interline * 2.35)))
    y_center = float(symbol.cy)
    if symbol.source == "rest_candidate":
        y_center = (sy0 + sy1) / 2.0
    y0 = int(round(max(0, y_center - interline * 1.05)))
    y1 = int(round(min(image_height, y_center + interline * 1.05)))
    return x0, y0, x1, y1


def _erase_staff_lines(patch: np.ndarray, band: StaffBand, *, patch_y0: int) -> None:
    for line_y in band.line_ys:
        local_y = int(round(line_y)) - patch_y0
        if local_y < 0 or local_y >= patch.shape[0]:
            continue
        y0 = max(0, local_y - 1)
        y1 = min(patch.shape[0], local_y + 2)
        patch[y0:y1, :] = False


def _connected_components(patch: np.ndarray, *, x0: int, y0: int) -> list[_Component]:
    h, w = patch.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[_Component] = []
    ys, xs = np.nonzero(patch)

    for start_y, start_x in zip(ys.tolist(), xs.tolist(), strict=False):
        if visited[start_y, start_x] or not patch[start_y, start_x]:
            continue

        stack = [(start_x, start_y)]
        visited[start_y, start_x] = True
        points: list[tuple[int, int]] = []
        while stack:
            px, py = stack.pop()
            points.append((px, py))
            for ny in range(max(0, py - 1), min(h, py + 2)):
                for nx in range(max(0, px - 1), min(w, px + 2)):
                    if visited[ny, nx] or not patch[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((nx, ny))

        area = len(points)
        comp_xs = [point[0] for point in points]
        comp_ys = [point[1] for point in points]
        min_x = min(comp_xs)
        max_x = max(comp_xs)
        min_y = min(comp_ys)
        max_y = max(comp_ys)
        components.append(
            _Component(
                area=area,
                cx=x0 + sum(comp_xs) / float(area),
                cy=y0 + sum(comp_ys) / float(area),
                bbox=(x0 + min_x, y0 + min_y, x0 + max_x + 1, y0 + max_y + 1),
            )
        )

    return components


def _score_dot_component(
    component: _Component,
    symbol: NoteheadCandidate,
    band: StaffBand,
    interline: float,
) -> float:
    x0, y0, x1, y1 = component.bbox
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        return 0.0

    area = float(component.area)
    density = area / float(width * height)
    aspect = width / float(height)
    if density < 0.35:
        return 0.0
    if not (0.45 <= aspect <= 2.20):
        return 0.0
    if not (interline * 0.10 <= width <= interline * 0.70):
        return 0.0
    if not (interline * 0.10 <= height <= interline * 0.70):
        return 0.0
    if not (interline * interline * 0.015 <= area <= interline * interline * 0.35):
        return 0.0

    sx0, _sy0, sx1, _sy1 = symbol.bbox
    if component.cx <= sx1:
        return 0.0
    if component.cx - sx0 > interline * 3.0:
        return 0.0

    step = band.staff_step_from_y(component.cy)
    target_steps = _legal_dot_steps(symbol.staff_step)
    step_error = min(abs(step - target) for target in target_steps)
    if step_error > 1.10:
        return 0.0

    size_score = _score_range(
        width / interline,
        low=0.10,
        ideal_low=0.18,
        ideal_high=0.45,
        high=0.70,
    )
    height_score = _score_range(
        height / interline,
        low=0.10,
        ideal_low=0.18,
        ideal_high=0.45,
        high=0.70,
    )
    aspect_score = _score_range(aspect, low=0.45, ideal_low=0.75, ideal_high=1.45, high=2.20)
    position_score = max(0.0, 1.0 - step_error / 1.10)
    density_score = _score_range(density, low=0.35, ideal_low=0.55, ideal_high=1.00, high=1.00)
    return float(
        0.20 * size_score
        + 0.20 * height_score
        + 0.20 * aspect_score
        + 0.25 * position_score
        + 0.15 * density_score
    )


def _legal_dot_steps(staff_step: float) -> tuple[float, ...]:
    rounded = round(staff_step)
    if abs(staff_step - rounded) <= 0.25 and rounded % 2 == 0:
        return (float(rounded + 1), float(rounded - 1), staff_step)
    return (staff_step,)


def _score_range(
    value: float,
    *,
    low: float,
    ideal_low: float,
    ideal_high: float,
    high: float,
) -> float:
    if value < low or value > high:
        return 0.0
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < ideal_low:
        return (value - low) / max(ideal_low - low, 1e-6)
    return (high - value) / max(high - ideal_high, 1e-6)
