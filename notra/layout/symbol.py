"""Symbol-level detection primitives for OMR.

Detects noteheads, stems, clef regions, and barlines from binarized images.
All detection is geometry-first — using the staff band as the coordinate frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from notra.layout.staff import StaffBand

# ---------------------------------------------------------------------------
# Notehead detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NoteheadCandidate:
    """One detected notehead with its pixel geometry."""

    cx: float
    cy: float
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1
    area: float
    is_filled: bool
    staff_step: float
    staff_band_index: int = 0
    source: str = "manual"
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class StemCandidate:
    """One detected stem associated with a notehead."""

    notehead_cx: float
    notehead_cy: float
    direction: str  # "up" or "down"
    top_y: int
    bottom_y: int
    center_x: float
    length_px: int


def _score_range(
    value: float,
    *,
    low: float,
    ideal_low: float,
    ideal_high: float,
    high: float,
) -> float:
    """Score a scalar against a trapezoidal acceptance window."""
    if value < low or value > high:
        return 0.0
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < ideal_low:
        width = max(ideal_low - low, 1e-6)
        return max(0.0, (value - low) / width)
    width = max(high - ideal_high, 1e-6)
    return max(0.0, (high - value) / width)


def _notehead_confidence(
    *,
    area: float,
    bbox: tuple[int, int, int, int],
    staff_step: float,
    interline: float,
    source: str,
) -> float:
    """Return a calibrated heuristic confidence for pseudo-label triage.

    This is not a model probability. It is a deterministic quality score used
    to separate high-confidence pseudo-labels from crops that need review.
    """
    x0, y0, x1, y1 = bbox
    width_ratio = max(0.0, (x1 - x0 + 1.0) / interline)
    height_ratio = max(0.0, (y1 - y0 + 1.0) / interline)
    area_ratio = max(0.0, area / max(interline * interline, 1.0))
    aspect = width_ratio / max(height_ratio, 1e-6)
    staff_step_error = abs(staff_step - round(staff_step))

    width_score = _score_range(width_ratio, low=0.45, ideal_low=0.75, ideal_high=1.35, high=1.90)
    height_score = _score_range(
        height_ratio,
        low=0.35,
        ideal_low=0.60,
        ideal_high=1.25,
        high=1.65,
    )
    area_score = _score_range(area_ratio, low=0.20, ideal_low=0.38, ideal_high=0.95, high=1.55)
    aspect_score = _score_range(aspect, low=0.55, ideal_low=0.75, ideal_high=1.55, high=2.25)
    staff_score = max(0.0, 1.0 - staff_step_error / 0.60)
    shape_score = (
        width_score * 0.25
        + height_score * 0.25
        + area_score * 0.25
        + aspect_score * 0.15
        + staff_score * 0.10
    )

    base_source = source.split(":", maxsplit=1)[0]
    source_prior = {
        "connected_component": 0.90,
        "line_position": 0.76,
        "grayscale_darkness": 0.68,
        "manual": 1.00,
    }.get(base_source, 0.55)
    if "split" in source:
        source_prior = min(source_prior, 0.72)

    confidence = shape_score * 0.82 + source_prior * 0.18
    return round(float(min(1.0, max(0.0, confidence))), 3)


def _detect_noteheads_grayscale(
    gray: np.ndarray,
    staff_bands: list,
    interline: float,
    min_area: float,
    max_area: float,
    min_w: float,
    max_w: float,
    min_h: float,
    max_h: float,
) -> list[NoteheadCandidate]:
    """Detect noteheads from grayscale by finding local darkness minima.

    Operates on ORIGINAL grayscale (not binarized). Staff-line damage
    is avoided by not erasing lines.
    """
    candidates: list[NoteheadCandidate] = []
    if gray.size == 0 or not staff_bands:
        return candidates

    seen: set[tuple[int, int]] = set()
    h_img, w_img = gray.shape

    for band_idx, band in enumerate(staff_bands):
        # Staff region with padding
        y0 = max(0, int(band.y_bottom - interline * 1.5))
        y1 = min(h_img, int(band.y_top + interline * 1.5))
        if y1 <= y0:
            continue

        staff_strip = gray[y0:y1, :].astype(np.float32)

        # Local darkness: noteheads are dark blobs. Compute the darkest
        # pixels in each column, then find local minima.
        # Use a threshold: pixels darker than median by 1.5× MAD
        flat = staff_strip.reshape(-1)
        median_val = float(np.median(flat))
        mad = float(np.median(np.abs(flat - median_val))) * 1.4826  # scale to std
        threshold = median_val - 1.5 * max(mad, 5.0)
        threshold = max(10, threshold)

        # Mark dark pixels
        dark_mask = staff_strip < threshold

        # Find connected dark components (pure numpy flood-fill)
        labeled = np.zeros_like(dark_mask, dtype=np.int32)
        label_id = 0
        for y in range(dark_mask.shape[0]):
            for x in range(dark_mask.shape[1]):
                if dark_mask[y, x] and labeled[y, x] == 0:
                    label_id += 1
                    stack = [(y, x)]
                    while stack:
                        cy, cx = stack.pop()
                        if not (0 <= cy < dark_mask.shape[0] and 0 <= cx < dark_mask.shape[1]):
                            continue
                        if not dark_mask[cy, cx] or labeled[cy, cx] != 0:
                            continue
                        labeled[cy, cx] = label_id
                        stack.extend([(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)])
        num_features = label_id

        for lbl in range(1, num_features + 1):
            ys, xs = np.where(labeled == lbl)
            if len(ys) == 0:
                continue

            cy_img = float(np.mean(ys)) + y0
            cx_img = float(np.mean(xs))
            area = len(ys)
            bw = float(xs.max() - xs.min() + 1)
            bh = float(ys.max() - ys.min() + 1)

            # Size filter
            if area < min_area or area > max_area:
                continue
            if bw < min_w or bw > max_w:
                continue
            if bh < min_h or bh > max_h:
                continue

            # Staff step
            step = band.staff_step_from_y(cy_img)
            if step < -8.0 or step > 15.0:
                continue  # ledger line sanity

            pos_key = (band_idx, int(round(cx_img)))
            if pos_key in seen:
                continue
            seen.add(pos_key)

            # Fill classification: check original grayscale darkness
            # in the component center
            cx_i = int(round(cx_img))
            cy_i = int(round(cy_img))
            if 0 <= cy_i < h_img and 0 <= cx_i < w_img:
                center_dark = float(gray[cy_i, cx_i])
                is_filled = center_dark < threshold + 20
            else:
                is_filled = True

            bbox = (
                int(xs.min()),
                int(ys.min() + y0),
                int(xs.max()),
                int(ys.max() + y0),
            )
            candidates.append(
                NoteheadCandidate(
                    cx=cx_img,
                    cy=cy_img,
                    bbox=bbox,
                    area=float(area),
                    is_filled=is_filled,
                    staff_step=step,
                    staff_band_index=band_idx,
                    source="grayscale_darkness",
                    confidence=_notehead_confidence(
                        area=float(area),
                        bbox=bbox,
                        staff_step=step,
                        interline=interline,
                        source="grayscale_darkness",
                    ),
                )
            )

    return candidates


def detect_noteheads(
    ink: np.ndarray,
    staff_bands: list,
    *,
    remove_staff_lines: bool = True,
    line_erase_half_width: int | None = None,
    gray: np.ndarray | None = None,
    use_grayscale_fallback: bool = False,
    use_line_position_pass: bool = False,
) -> list[NoteheadCandidate]:
    """Detect notehead candidates from a binarized image using CCL.

    Staff lines are erased before component analysis to prevent them from
    merging with noteheads.

    Args:
        ink: Binarized image (uint8, 0=background, 1=ink).
        staff_bands: List of StaffBand detected in the image.
        remove_staff_lines: If True, erase staff line pixels before detection.
        line_erase_half_width: Half-width of staff line erasure band (px).
            Defaults to ~25% of interline spacing.

    Returns:
        List of NoteheadCandidate objects sorted by x-coordinate.
    """
    if ink.size == 0 or not staff_bands:
        return []

    interline = float(np.median([b.interline_px for b in staff_bands]))
    interline = max(4.0, interline)

    # Size thresholds relative to interline spacing (staff-normalized).
    # Tighter to reject phantom fragments from staff-line erasure.
    min_area = interline * interline * 0.25
    max_area = interline * interline * 2.00
    min_w = interline * 0.50
    max_w = interline * 2.00
    min_h = interline * 0.40
    max_h = interline * 1.50

    # --- Pass 1: erased image CCL (catches space-position noteheads) ---
    cleaned = ink.copy()
    if remove_staff_lines:
        if line_erase_half_width is None:
            line_erase_half_width = 1
        _erase_staff_lines(cleaned, staff_bands, half_width=line_erase_half_width)
    components = _find_connected_components(cleaned)
    if len(components) > 1:
        components = merge_bisected_components(components, staff_bands)
    candidates: list[NoteheadCandidate] = []
    seen_positions: set[tuple[int, int]] = set()

    _accept_components(
        components,
        ink,
        staff_bands,
        interline,
        min_area,
        max_area,
        min_w,
        max_w,
        min_h,
        max_h,
        candidates,
        seen_positions,
        gray=gray,
    )

    candidates = _split_chord_noteheads(candidates, interline, staff_bands)

    if use_line_position_pass:
        _detect_line_position_noteheads(
            ink,
            staff_bands,
            interline,
            candidates,
            seen_positions,
            min_area,
            max_area,
            min_w,
            max_w,
            min_h,
            max_h,
        )

    # --- Pass 2: grayscale darkness components (anti-aliased fallback) ---
    if use_grayscale_fallback and gray is not None and gray.size > 0:
        gray_candidates = _detect_noteheads_grayscale(
            gray,
            staff_bands,
            interline,
            min_area,
            max_area,
            min_w,
            max_w,
            min_h,
            max_h,
        )
        for nh in gray_candidates:
            pos_key = (nh.staff_band_index, int(round(nh.cx)))
            if pos_key in seen_positions:
                continue
            seen_positions.add(pos_key)
            candidates.append(nh)

    # --- Size-consistency filter ---
    # Noteheads are all approximately one staff-space in size.
    # Phantom fragments are much smaller; merged blobs much larger.
    # Keep only candidates within ±60% of the median area.
    if len(candidates) >= 5:
        areas = np.array([c.area for c in candidates], dtype=np.float64)
        med_area = float(np.median(areas))
        if med_area > 0:
            lo = med_area * 0.35
            hi = med_area * 1.65
            candidates = [c for c in candidates if lo <= c.area <= hi]

    candidates.sort(key=lambda n: n.cx)
    return candidates


def _accept_components(
    comps: list,
    img: np.ndarray,
    staff_bands: list,
    interline: float,
    min_area: float,
    max_area: float,
    min_w: float,
    max_w: float,
    min_h: float,
    max_h: float,
    candidates: list[NoteheadCandidate],
    seen: set[tuple[int, int]],
    gray: np.ndarray | None = None,
    source: str = "connected_component",
) -> None:
    """Filter components by size/position and add to candidates list."""
    for comp in comps:
        area, cx, cy, x0, y0, x1, y1 = comp
        bw = x1 - x0 + 1.0
        bh = y1 - y0 + 1.0

        if area < min_area or area > max_area:
            continue
        if bw < min_w or bw > max_w:
            continue
        if bh < min_h or bh > max_h:
            continue

        band_idx, band = _nearest_band_with_index(cy, staff_bands)
        if band is None:
            continue

        step = band.staff_step_from_y(cy)

        # Filter candidates far from their staff band
        # Allow up to 5 ledger lines above/below staff
        if step < -8.0 or step > 15.0:
            continue

        # Deduplicate by (band_idx, rounded cx)
        pos_key = (band_idx, int(round(cx)))
        if pos_key in seen:
            continue
        seen.add(pos_key)

        is_filled = _classify_notehead_filled(
            img, int(x0), int(y0), int(x1), int(y1), interline, gray=gray
        )

        bbox = (int(x0), int(y0), int(x1), int(y1))
        candidates.append(
            NoteheadCandidate(
                cx=cx,
                cy=cy,
                bbox=bbox,
                area=area,
                is_filled=is_filled,
                staff_step=step,
                staff_band_index=band_idx,
                source=source,
                confidence=_notehead_confidence(
                    area=area,
                    bbox=bbox,
                    staff_step=step,
                    interline=interline,
                    source=source,
                ),
            )
        )


def _classify_notehead_filled(
    ink: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    interline: float,
    gray: np.ndarray | None = None,
) -> bool:
    """Classify a notehead as filled (quarter/shorter) or open (half/whole).

    Uses grayscale center-darkness when available — more robust than binary
    ink ratio, especially with Sauvola binarization.
    """
    # Grayscale method: compare center darkness vs edge darkness.
    # A filled notehead has a dark center surrounded by lighter edges.
    # An open notehead has a light center (paper) with dark outline.
    if gray is not None and gray.size > 0:
        h_g, w_g = gray.shape

        # Dynamic center-patch size: 1/3 of the smaller bbox dimension
        bw = x1 - x0 + 1
        bh = y1 - y0 + 1
        half_s = max(1, int(min(bw, bh) * 0.2))
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2

        # Center patch
        yc0 = max(0, cy - half_s)
        yc1 = min(h_g, cy + half_s + 1)
        xc0 = max(0, cx - half_s)
        xc1 = min(w_g, cx + half_s + 1)

        # Edge ring: pixels near bbox border
        margin = max(1, half_s // 2)
        ey0 = max(0, y0 + margin)
        ey1 = min(h_g, y1 - margin + 1)
        ex0 = max(0, x0 + margin)
        ex1 = min(w_g, x1 - margin + 1)

        if yc1 > yc0 and xc1 > xc0 and ey1 > ey0 and ex1 > ex0:
            center_vals = gray[yc0:yc1, xc0:xc1].ravel()

            # Edge pixels: top, bottom, left, right strips near border
            top_strip = gray[max(0, y0) : min(h_g, y0 + margin + 1), max(0, x0) : min(w_g, x1 + 1)]
            bot_strip = gray[max(0, y1 - margin) : min(h_g, y1 + 1), max(0, x0) : min(w_g, x1 + 1)]
            left_strip = gray[ey0:ey1, max(0, x0) : min(w_g, x0 + margin + 1)]
            right_strip = gray[ey0:ey1, max(0, x1 - margin) : min(w_g, x1 + 1)]
            edge_parts = []
            for s in [top_strip, bot_strip, left_strip, right_strip]:
                if s.size > 0:
                    edge_parts.append(s.ravel())
            if edge_parts:
                edge_vals = np.concatenate(edge_parts)
                center_median = float(np.median(center_vals))
                edge_median = float(np.median(edge_vals))
                # Filled: center is much darker than edge
                # Open: center is similar to or lighter than edge (paper showing through)
                return center_median < edge_median - 20

    # Fallback: binary ink ratio
    h, w = ink.shape
    y0_c = max(0, y0)
    y1_c = min(h, y1 + 1)
    x0_c = max(0, x0)
    x1_c = min(w, x1 + 1)
    region = ink[y0_c:y1_c, x0_c:x1_c]

    if region.size == 0:
        return True

    ink_ratio = int(region.sum()) / region.size
    return ink_ratio >= 0.45


# ---------------------------------------------------------------------------
# Stem detection
# ---------------------------------------------------------------------------


def detect_stems(
    ink: np.ndarray,
    noteheads: list[NoteheadCandidate],
    staff_bands: list,
    *,
    gray: np.ndarray | None = None,
) -> list[StemCandidate]:
    """Detect stems near detected noteheads.

    For each notehead, searches vertically above/below for a vertical
    ink run that indicates a stem. When a grayscale image is provided,
    uses a lower darkness threshold to detect anti-aliased stem edges
    that may be lost in Otsu binarization.

    Args:
        ink: Binarized image.
        noteheads: Detected noteheads (must have bbox info).
        staff_bands: Detected staff bands.
        gray: Optional grayscale image for robust stem detection.

    Returns:
        List of StemCandidate, one per notehead that has a stem.
    """
    if ink.size == 0:
        return []

    interline = float(np.median([b.interline_px for b in staff_bands]))
    interline = max(4.0, interline)
    stem_min_length = int(interline * 2.0)  # stems ≥ ~2 interline spaces

    # Use a fixed threshold for grayscale stem detection — pixels darker
    # than 80 are ink (stems are typically 0-40 in rendered PNGs).
    use_gray = gray is not None and gray.size > 0
    if use_gray:
        assert gray is not None
        gray_img = gray
        dark_threshold = 80
    else:
        gray_img = None
        dark_threshold = 128  # unused

    stems: list[StemCandidate] = []
    h, w = ink.shape

    for nh in noteheads:
        x0, y0, x1, y1 = nh.bbox

        # Determine stem direction: default "up" for notes below center,
        # "down" for notes above center. But we need the nearest band.
        band = _nearest_band(nh.cy, staff_bands)
        if band is None:
            continue

        # If notehead is above center line, stem goes down; below → up
        if nh.staff_step > 4.0:
            direction = "down"
            # Search below notehead
            search_start = y1
            search_end = min(h, y1 + int(interline * 5))
        else:
            direction = "up"
            # Search above notehead
            search_start = max(0, y0 - int(interline * 5))
            search_end = y0

        # Search a window of columns for a vertical ink run (stem).
        # Stems in rendered music are 3-5 px wide; a single-column search
        # is too fragile. For up-stems, scan the right side of the notehead
        # bbox. For down-stems, scan the left side. Use the longest run
        # found across all candidate columns.
        half_window = max(2, int(interline * 0.35))  # ~5-10 px depending on DPI
        if direction == "up":
            col_start = max(0, x1 - half_window)
            col_end = min(w - 1, x1 + half_window)
        else:
            col_start = max(0, x0 - half_window)
            col_end = min(w - 1, x0 + half_window)

        best_stem_start: int | None = None
        best_stem_end: int | None = None
        best_stem_x: int | None = None
        best_length = 0

        for sx in range(col_start, col_end + 1):
            run_start: int | None = None
            run_end: int | None = None
            dark_count = 0
            total_count = 0
            for y in range(search_start, search_end):
                if 0 <= y < h and 0 <= sx < w:
                    if use_gray:
                        assert gray_img is not None
                        val = 1 if int(gray_img[y, sx]) < dark_threshold else 0
                    else:
                        val = int(ink[y, sx])
                else:
                    val = 0
                total_count += 1
                if val > 0:
                    dark_count += 1
                    if run_start is None:
                        run_start = y
                    run_end = y
                else:
                    if run_start is not None and run_end is not None:
                        length = run_end - run_start + 1
                        if length >= stem_min_length and length > best_length:
                            best_length = length
                            best_stem_start = run_start
                            best_stem_end = run_end
                            best_stem_x = sx
                    run_start = None
                    run_end = None
            if run_start is not None and run_end is not None:
                length = run_end - run_start + 1
                if length >= stem_min_length and length > best_length:
                    best_length = length
                    best_stem_start = run_start
                    best_stem_end = run_end
                    best_stem_x = sx

            # Fallback: if no continuous run found but the column has high
            # dark-pixel density (>50%), treat the entire search range as a
            # stem (anti-aliased stems are often fragmented).
            if best_length < stem_min_length and use_gray:
                if total_count > 0 and dark_count / total_count > 0.50:
                    best_length = total_count
                    best_stem_start = search_start
                    best_stem_end = search_end - 1
                    best_stem_x = sx

        if best_stem_start is not None and best_stem_end is not None and best_stem_x is not None:
            stems.append(
                StemCandidate(
                    notehead_cx=nh.cx,
                    notehead_cy=nh.cy,
                    direction=direction,
                    top_y=best_stem_start if direction == "up" else best_stem_end,
                    bottom_y=best_stem_end if direction == "up" else best_stem_start,
                    center_x=float(best_stem_x),
                    length_px=best_length,
                )
            )

    return stems


# ---------------------------------------------------------------------------
# Barline detection
# ---------------------------------------------------------------------------


def detect_barlines(
    ink: np.ndarray,
    staff_bands: list,
    *,
    min_length_ratio: float = 5.0,
) -> list[float]:
    """Detect barline x-positions by finding vertical runs spanning staff height.

    Args:
        ink: Binarized image.
        staff_bands: Detected staff bands.
        min_length_ratio: Minimum barline length as ratio of interline spacing.

    Returns:
        Sorted list of barline x-coordinates.
    """
    if ink.size == 0 or not staff_bands:
        return []

    interline = float(np.median([b.interline_px for b in staff_bands]))
    min_run = int(max(24, interline * min_length_ratio))

    h, w = ink.shape
    candidate_cols: list[int] = []

    for x in range(w):
        col = ink[:, x]
        run = _longest_ink_run(col)
        if run >= min_run:
            candidate_cols.append(x)

    if not candidate_cols:
        return []

    # Merge nearby columns (within 4px)
    merged: list[float] = []
    cluster: list[int] = [candidate_cols[0]]
    for x in candidate_cols[1:]:
        if x - cluster[-1] <= 4:
            cluster.append(x)
            continue
        merged.append(float(sum(cluster)) / float(len(cluster)))
        cluster = [x]
    merged.append(float(sum(cluster)) / float(len(cluster)))

    return sorted(merged)


# ---------------------------------------------------------------------------
# Clef detection
# ---------------------------------------------------------------------------


def detect_clef_region(
    ink: np.ndarray,
    band: "StaffBand",
    *,
    barline_xs: list[float] | None = None,
    search_width_px: int = 420,
) -> tuple[str, int]:
    """Detect the clef type for a staff band by analyzing its left region.

    For rendered scores at known scale, uses ink density heuristics around
    the expected clef line positions.

    Args:
        ink: Binarized image.
        band: The staff band to analyze.
        barline_xs: Known barline x-positions (clef is left of first barline).
        search_width_px: Maximum width to search for clef.

    Returns:
        (clef_sign, clef_line) tuple, e.g. ("G", 2) for treble.
    """
    h, w = ink.shape
    # Search region: from left edge to first barline or search_width_px
    right_limit = search_width_px
    if barline_xs:
        min_valid_x = int(max(40.0, band.interline_px * 3.0))
        valid_barlines = [int(x) for x in barline_xs if int(x) >= min_valid_x]
        if valid_barlines:
            right_limit = min(right_limit, min(valid_barlines))
    right_limit = min(w, right_limit)
    if right_limit <= 0:
        return ("G", 2)  # default

    # Extract clef region — start from first column with ink (after margin)
    top = max(0, band.y_bottom - int(band.interline_px * 2))
    bottom = min(h, band.y_top + int(band.interline_px * 2))
    staff_slice = ink[top:bottom, 0:right_limit]

    if staff_slice.size == 0:
        return ("G", 2)

    # Find first column with ink (clef starts after left margin)
    col_sums = staff_slice.sum(axis=0)
    ink_cols = np.where(col_sums > 0)[0]
    if len(ink_cols) == 0:
        return ("G", 2)
    clef_x0 = ink_cols[0]

    # Extract ~3 interline widths from first ink column
    clef_w = int(band.interline_px * 4)
    clef_x1 = min(staff_slice.shape[1], clef_x0 + clef_w)
    clef_region = staff_slice[:, clef_x0:clef_x1]

    if clef_region.size == 0:
        return ("G", 2)

    vproj = clef_region.sum(axis=1).astype(np.float64)

    region_h = len(vproj)
    norm_h = max(1.0, region_h / band.interline_px)
    score_g = _score_treble(vproj, band, top) / norm_h
    score_f = _score_bass(vproj, band, top) / norm_h
    score_c = _score_alto(vproj, band, top) / norm_h

    # Pragmatic: G2 is the default for most staves.
    # F4 for bass instruments; C3 only when absolutely dominant (viola parts).
    if score_f > score_g * 2.2:
        return ("F", 4)

    if score_c > score_g * 6.0 and score_c > score_f * 6.0:
        return ("C", 3)

    return ("G", 2)


def _score_treble(vproj: np.ndarray, band: "StaffBand", top: int) -> float:
    """Score the treble clef hypothesis.

    Treble clef: the G curl wraps around line 2 (from bottom), and the
    vertical stroke extends well above and below. Characteristic ink
    concentration around the G line and the upper curl region.
    """
    # band.line_ys is top-to-bottom:
    # line5(top)=idx0, line4=idx1, line3=idx2, line2=idx3, line1(bottom)=idx4
    line2_y = int(band.line_ys[3] - top)  # line 2 from bottom
    line5_y = int(band.line_ys[0] - top)  # top line
    half_step = int(max(1, band.interline_px / 2.0))

    # Ink around line 2 (G line)
    y2_low = max(0, line2_y - half_step)
    y2_high = min(len(vproj), line2_y + half_step + 1)
    score_g_line = float(vproj[y2_low:y2_high].sum()) if y2_high > y2_low else 0.0

    # Ink in upper region (curl above staff)
    y_upper_low = max(0, line5_y - half_step)
    y_upper_high = min(len(vproj), line5_y + half_step * 5)
    score_upper = (
        float(vproj[y_upper_low:y_upper_high].sum()) if y_upper_high > y_upper_low else 0.0
    )

    # Treble has substantial ink above staff
    return score_g_line * 2.0 + score_upper


def _score_bass(vproj: np.ndarray, band: "StaffBand", top: int) -> float:
    """Score the bass clef hypothesis.

    Bass clef: two dots on either side of line 4, and the curved shape
    on line 4. Characteristic dot pairs.
    """
    # line4 from bottom = idx1 when line_ys is top-to-bottom
    line4_y = int(band.line_ys[1] - top)
    half_step = int(max(1, band.interline_px / 2.0))

    # Dots are in spaces 3 and 4 (above and below line 4)
    space3_y = int(line4_y - half_step)
    space4_y = int(line4_y + half_step)

    def _sum_around(y: int) -> float:
        lo = max(0, y - half_step // 2)
        hi = min(len(vproj), y + half_step // 2 + 1)
        return float(vproj[lo:hi].sum()) if hi > lo else 0.0

    score_dot_above = _sum_around(space3_y)
    score_dot_below = _sum_around(space4_y)

    # The two dots should both have ink
    dot_score = min(score_dot_above, score_dot_below) * 3.0

    return dot_score


def _score_alto(vproj: np.ndarray, band: "StaffBand", top: int) -> float:
    """Score the alto clef hypothesis.

    Alto clef: two vertical strokes meeting at line 3 (middle line).
    Unlike treble, alto clef has minimal ink ABOVE the staff.
    """
    line3_y = int(band.line_ys[2] - top)
    line5_y = int(band.line_ys[0] - top)
    half_step = int(max(1, band.interline_px / 2.0))

    y_low = max(0, line3_y - half_step * 3)
    y_high = min(len(vproj), line3_y + half_step * 3 + 1)
    score = float(vproj[y_low:y_high].sum()) if y_high > y_low else 0.0

    # Penalty: ink well above the staff indicates treble clef, not alto.
    y_above = max(0, line5_y - half_step * 2)
    y_above_end = min(len(vproj), line5_y + half_step * 4)
    above_score = float(vproj[y_above:y_above_end].sum()) if y_above_end > y_above else 0.0
    score -= above_score * 0.5

    return max(0.0, score)


# ---------------------------------------------------------------------------
# Key signature detection
# ---------------------------------------------------------------------------


def detect_key_signature(
    ink: np.ndarray,
    band: "StaffBand",
    clef_sign: str,
    *,
    barline_xs: list[float] | None = None,
    search_width_px: int = 150,
) -> int:
    """Detect the key signature fifths value from the staff's left margin.

    Counts sharp/flat symbols between the clef and the first barline.
    Returns fifths value: positive = sharps, negative = flats.

    Args:
        ink: Binarized image.
        band: Staff band.
        clef_sign: Detected clef ("G", "F", "C").
        barline_xs: Barline x-positions (key sig is left of first barline).
        search_width_px: Max search width for key sig.

    Returns:
        Fifths value (e.g., +2 for D major, -3 for E-flat major).
    """
    h, w = ink.shape
    right_limit = search_width_px
    if barline_xs:
        right_limit = min(right_limit, int(barline_xs[0]) - 5)
    right_limit = min(w, max(20, right_limit))

    # Key sig is after the clef. Use a generous clef width estimate.
    clef_width = int(band.interline_px * 5.0)
    x_start = clef_width
    if x_start >= right_limit:
        return 0

    # Extract key signature region
    y0 = max(0, band.y_bottom - int(band.interline_px))
    y1 = min(h, band.y_top + int(band.interline_px))
    region = ink[y0:y1, x_start:right_limit]
    if region.size == 0:
        return 0

    # Count isolated ink components in the region
    # Key sig symbols are small, vertically oriented marks
    min_symbol_area = band.interline_px * band.interline_px * 0.15
    max_symbol_area = band.interline_px * band.interline_px * 2.0

    components = _find_connected_components(region)
    symbol_count = 0
    for comp in components:
        area = comp[0]
        bw = comp[5] - comp[3] + 1.0
        bh = comp[6] - comp[4] + 1.0
        if area < min_symbol_area or area > max_symbol_area:
            continue
        # Key sig symbols are taller than wide
        if bh > bw * 1.2:
            symbol_count += 1

    # Determine sharps vs flats based on the clef and position
    # For treble clef: sharps appear on F/C/G/D/A/E/B lines
    # For bass clef: sharps appear on different lines
    # We just return the count; the sign is determined by context
    # (sharps are ♯, flats are ♭ — we assume sharps for positive count,
    # but this is a simplification)

    # Heuristic: if we detect components, assume they're accidentals
    # Limit to reasonable range (-7 to +7)
    fifths = min(7, max(-7, symbol_count))
    return fifths


# ---------------------------------------------------------------------------
# Accidentals detection
# ---------------------------------------------------------------------------


def detect_accidentals(
    ink: np.ndarray,
    noteheads: list[NoteheadCandidate],
    staff_bands: list,
    *,
    search_width: float | None = None,
) -> dict[int, int]:  # notehead_index -> alter (-2..2)
    """Detect accidentals (sharps, flats, naturals) left of noteheads.

    Args:
        ink: Binarized image.
        noteheads: Detected noteheads.
        staff_bands: Staff bands.
        search_width: How far left of notehead to search (default: 1.5 × interline).

    Returns:
        Dict mapping notehead index to alter value (-1=flat, 0=natural, 1=sharp, 2=double-sharp).
    """
    if not noteheads or not staff_bands:
        return {}

    interline = float(np.median([b.interline_px for b in staff_bands]))
    if search_width is None:
        search_width = interline * 1.5

    h, w = ink.shape
    results: dict[int, int] = {}

    for idx, nh in enumerate(noteheads):
        x0 = nh.bbox[0]
        search_x0 = max(0, int(x0 - search_width))
        search_x1 = max(0, int(x0 - interline * 0.2))

        if search_x1 <= search_x0:
            continue

        # Look for characteristic ink patterns in the search region
        band = _nearest_band(nh.cy, staff_bands)
        if band is None:
            continue

        search_y0 = max(0, int(nh.cy - interline * 2))
        search_y1 = min(h, int(nh.cy + interline * 2))

        region = ink[search_y0:search_y1, search_x0:search_x1]
        if region.size == 0:
            continue

        alter = _classify_accidental(region, interline)
        if alter != 0:
            results[idx] = alter

    return results


def _classify_accidental(region: np.ndarray, interline: float) -> int:
    """Classify an accidental region as sharp, flat, natural, or none.

    Uses simple heuristics: sharp has two vertical bars + two horizontals,
    flat has a curved shape, natural has two vertical bars + two horizontals
    in a box shape.
    """
    total_ink = float(region.sum())
    total_pixels = float(region.size)
    ink_ratio = total_ink / max(total_pixels, 1.0)

    # If there's no significant ink, no accidental
    if ink_ratio < 0.05:
        return 0

    # For clean rendered scores, check column projection
    col_sums = region.sum(axis=0)
    nonzero_cols = int((col_sums > 0).sum())

    # Sharps and naturals have more vertical structure (wider)
    # Flats are more compact horizontally
    if nonzero_cols < int(interline * 0.8):
        # Compact → likely flat
        return -1
    elif nonzero_cols < int(interline * 1.3):
        # Medium → natural
        return 0  # natural = no alter needed (but we should mark it)
    else:
        # Wider → sharp
        return 1


# ---------------------------------------------------------------------------
# Duration classification
# ---------------------------------------------------------------------------


def classify_duration(
    notehead: NoteheadCandidate,
    has_stem: bool,
    stem: StemCandidate | None = None,
    *,
    flag_count: int = 0,
    dot_count: int = 0,
) -> tuple[int, int]:
    """Classify note duration from notehead type and stem properties.

    Returns (numerator, denominator) for the note's duration fraction.

    - Filled + stem = quarter (1/4) or shorter with flags
    - Open + stem = half (1/2)
    - Open + no stem = whole (1/1)
    - Filled + no stem = whole (breve style, rare) → treat as quarter
    """
    if notehead.is_filled:
        if not has_stem:
            # Filled without stem: breve or whole in some notation styles
            # Default to quarter
            base_num, base_den = 1, 4
        elif flag_count == 1:
            base_num, base_den = 1, 8
        elif flag_count == 2:
            base_num, base_den = 1, 16
        elif flag_count == 3:
            base_num, base_den = 1, 32
        elif flag_count == 4:
            base_num, base_den = 1, 64
        else:
            base_num, base_den = 1, 4  # quarter
    else:
        # Open notehead
        if not has_stem:
            base_num, base_den = 1, 1  # whole
        else:
            base_num, base_den = 1, 2  # half

    # Apply dots: each dot adds half the previous value
    if dot_count > 0:
        base = base_num / base_den
        multiplier = sum(1.0 / (2 ** (i + 1)) for i in range(dot_count))
        value = base * (1.0 + multiplier)
        # Convert back to fraction
        from fractions import Fraction

        f = Fraction(value).limit_denominator(128)
        return f.numerator, f.denominator

    return base_num, base_den


# ---------------------------------------------------------------------------
# Rest detection
# ---------------------------------------------------------------------------


def detect_rests(
    ink: np.ndarray,
    staff_bands: list,
    *,
    interline: float | None = None,
    noteheads: list[NoteheadCandidate] | None = None,
) -> list[NoteheadCandidate]:
    """Detect rest symbols in staff bands using constrained shape heuristics.

    Whole rests hang from line 4 (step ~6). Half rests sit on line 3 (step ~4).
    Quarter/eighth rests are near the staff center with tall aspect and
    moderate ink density.

    Components near known noteheads or with stem-like aspect are rejected.

    Args:
        ink: Binarized image.
        staff_bands: Detected staff bands.
        interline: Interline spacing (computed from bands if None).
        noteheads: Known notehead positions for proximity filtering.

    Returns:
        List of NoteheadCandidate representing rest positions (≤ ~2 per staff).
    """
    if ink.size == 0 or not staff_bands:
        return []

    if interline is None:
        interline = float(np.median([b.interline_px for b in staff_bands]))
    interline = max(4.0, interline)

    cleaned = ink.copy()
    _erase_staff_lines(cleaned, staff_bands, half_width=1)

    # Build exclusion mask from noteheads
    exclude_regions: list[tuple[float, float, float]] = []  # (cx, cy, radius)
    if noteheads:
        for nh in noteheads:
            exclude_regions.append((nh.cx, nh.cy, interline * 2.0))

    def _near_notehead(cx: float, cy: float) -> bool:
        for ex, ey, er in exclude_regions:
            if abs(cx - ex) < er and abs(cy - ey) < er:
                return True
        return False

    # Tight size thresholds
    min_area = interline * interline * 0.40
    max_area = interline * interline * 4.00
    min_w = interline * 0.60
    max_w = interline * 2.50
    min_h = interline * 0.40
    max_h = interline * 2.50

    components = _find_connected_components(cleaned)
    rests: list[NoteheadCandidate] = []
    rest_count_per_staff: dict[int, int] = {}

    for comp in components:
        area, cx, cy, x0, y0, x1, y1 = comp
        bw = x1 - x0 + 1.0
        bh = y1 - y0 + 1.0
        if area < min_area or area > max_area:
            continue
        if bw < min_w or bw > max_w or bh < min_h or bh > max_h:
            continue

        # Reject stem-like (very tall, very thin)
        if bh > bw * 3.0 and bw < interline * 0.8:
            continue

        aspect = bw / max(bh, 1.0)
        band_idx, band = _nearest_band_with_index(cy, staff_bands)
        if band is None:
            continue
        if cy < band.y_bottom - interline * 0.5 or cy > band.y_top + interline * 0.5:
            continue

        step = band.staff_step_from_y(cy)

        # Reject if near a notehead
        if _near_notehead(cx, cy):
            continue

        # Limit rest count per staff (real scores have few rests)
        if rest_count_per_staff.get(band_idx, 0) >= 3:
            continue

        is_rest = False

        # Whole rest: wide, hanging from line 4, step ~6
        if aspect > 1.3 and 5.0 <= step <= 7.0:
            region_ink = float(
                cleaned[
                    max(0, y0) : min(ink.shape[0], y1 + 1), max(0, x0) : min(ink.shape[1], x1 + 1)
                ].sum()
            )
            region_px = (y1 - y0 + 1) * (x1 - x0 + 1)
            if region_px > 0 and 0.30 < region_ink / region_px < 0.85:
                is_rest = True

        # Half rest: wide, sitting on line 3, step ~4
        elif aspect > 1.3 and 3.0 <= step <= 5.0:
            region_ink = float(
                cleaned[
                    max(0, y0) : min(ink.shape[0], y1 + 1), max(0, x0) : min(ink.shape[1], x1 + 1)
                ].sum()
            )
            region_px = (y1 - y0 + 1) * (x1 - x0 + 1)
            if region_px > 0 and 0.30 < region_ink / region_px < 0.85:
                is_rest = True

        # Quarter/eighth rest: tall, near staff center, moderate ink
        elif aspect < 0.9 and 2.0 <= step <= 6.0:
            region_ink = float(
                cleaned[
                    max(0, y0) : min(ink.shape[0], y1 + 1), max(0, x0) : min(ink.shape[1], x1 + 1)
                ].sum()
            )
            region_px = (y1 - y0 + 1) * (x1 - x0 + 1)
            if region_px > 0 and 0.20 < region_ink / region_px < 0.70:
                is_rest = True

        if not is_rest:
            continue

        rest_count_per_staff[band_idx] = rest_count_per_staff.get(band_idx, 0) + 1
        rests.append(
            NoteheadCandidate(
                cx=cx,
                cy=cy,
                bbox=(int(x0), int(y0), int(x1), int(y1)),
                area=area,
                is_filled=False,
                staff_step=step,
                staff_band_index=band_idx,
                source="rest_candidate",
                confidence=0.70,
            )
        )

    rests.sort(key=lambda r: r.cx)
    return rests


# ---------------------------------------------------------------------------
# Flag detection (for duration classification)
# ---------------------------------------------------------------------------


def detect_flags(
    ink: np.ndarray,
    stems: list[StemCandidate],
    *,
    interline: float | None = None,
    staff_bands: list | None = None,
) -> dict[int, int]:
    """Count flags on stems for duration classification.

    Flags are short horizontal marks at the end of a stem.
    Up-stems: flags on right side near top.
    Down-stems: flags on left side near bottom.

    Args:
        ink: Binarized image.
        stems: Detected stems (indexed by position in list).
        interline: Interline spacing.
        staff_bands: Used to compute interline if not given.

    Returns:
        Dict mapping stem index → flag count (0-4).
    """
    if interline is None and staff_bands:
        interline = float(np.median([b.interline_px for b in staff_bands]))
    if interline is None:
        interline = 8.0

    flag_map: dict[int, int] = {}
    h, w = ink.shape

    for s_idx, stem in enumerate(stems):
        if stem.direction == "up":
            scan_y = stem.top_y
            scan_x0 = int(stem.center_x)
            scan_x1 = min(w, int(stem.center_x + interline * 1.5))
        else:
            scan_y = stem.bottom_y
            scan_x0 = max(0, int(stem.center_x - interline * 1.5))
            scan_x1 = int(stem.center_x)

        if scan_x1 <= scan_x0:
            continue

        flags_found = 0
        search_r = int(interline * 0.5)
        for dy in range(-search_r, search_r + 1):
            y = scan_y + dy
            if y < 0 or y >= h:
                continue
            row = ink[y, scan_x0:scan_x1]
            min_fw = int(interline * 0.3)
            run = 0
            for val in row.flat:
                if int(val) > 0:
                    run += 1
                else:
                    if run >= min_fw:
                        flags_found += 1
                    run = 0
            if run >= min_fw:
                flags_found += 1

        flag_count = max(0, min(4, flags_found // 2))
        if flag_count > 0:
            flag_map[s_idx] = flag_count

    return flag_map


# ---------------------------------------------------------------------------
# Voice assignment from stem direction
# ---------------------------------------------------------------------------


def assign_voice_from_stems(
    noteheads: list[NoteheadCandidate],
    stem_map: dict[int, "StemCandidate"],
) -> dict[int, int]:
    """Assign voice (1 or 2) based on stem direction.

    Stems up → voice 1. Stems down → voice 2.
    Stemless noteheads inherit from nearest stemmed notehead in same staff.

    Returns dict mapping notehead index → voice number.
    """
    voice_map: dict[int, int] = {}
    for idx, nh in enumerate(noteheads):
        stem = stem_map.get(idx)
        voice_map[idx] = (
            1 if (stem is not None and stem.direction == "up") else (2 if stem is not None else 1)
        )

    # Stemless noteheads: inherit from nearest stemmed in same staff
    for idx, nh in enumerate(noteheads):
        if idx in stem_map:
            continue
        best_dist = float("inf")
        best_voice = 1
        for jdx, other in enumerate(noteheads):
            if jdx == idx or jdx not in stem_map:
                continue
            if nh.staff_band_index != other.staff_band_index:
                continue
            dist = abs(nh.cx - other.cx)
            if dist < best_dist:
                best_dist = dist
                best_voice = voice_map.get(jdx, 1)
        # Reassign only if nearby stemmed note found
        if best_dist < 50:  # ~6 interline spaces at scale 48
            voice_map[idx] = best_voice

    return voice_map


# ---------------------------------------------------------------------------
# Notehead recall: merge components bisected by staff lines
# ---------------------------------------------------------------------------


def merge_bisected_components(
    components: list[tuple[float, float, float, int, int, int, int]],
    staff_bands: list,
) -> list[tuple[float, float, float, int, int, int, int]]:
    """Merge CCL components split by staff line erasure.

    When a notehead sits on a staff line, erasing the line can split the
    component into two pieces. This merges vertically aligned pairs with
    a small gap between them.

    Args:
        components: Raw CCL output from _find_connected_components.
        staff_bands: Staff bands for interline reference.

    Returns:
        Merged component list.
    """
    if len(components) < 2 or not staff_bands:
        return components

    interline = float(np.median([b.interline_px for b in staff_bands]))
    max_gap = max(1.0, interline * 0.2)
    max_x_off = interline * 0.5

    comps = sorted(components, key=lambda c: (c[2], c[1]))
    merged: list[tuple[float, float, float, int, int, int, int]] = []
    used: set[int] = set()

    for i in range(len(comps)):
        if i in used:
            continue
        a = comps[i]
        best_j = -1
        best_gap = float("inf")
        for j in range(i + 1, len(comps)):
            if j in used:
                continue
            b = comps[j]
            if abs(a[1] - b[1]) > max_x_off:
                continue
            if a[6] < b[4]:
                gap = b[4] - a[6]
            elif b[6] < a[4]:
                gap = a[4] - b[6]
            else:
                continue
            if 0 < gap < max_gap and gap < best_gap:
                best_gap = gap
                best_j = j

        if best_j >= 0:
            b = comps[best_j]
            ma = a[0] + b[0]
            merged.append(
                (
                    ma,
                    (a[1] * a[0] + b[1] * b[0]) / ma,
                    (a[2] * a[0] + b[2] * b[0]) / ma,
                    min(a[3], b[3]),
                    min(a[4], b[4]),
                    max(a[5], b[5]),
                    max(a[6], b[6]),
                )
            )
            used.add(i)
            used.add(best_j)
        else:
            merged.append(a)
            used.add(i)

    return merged


# ---------------------------------------------------------------------------
# Line-position notehead detection (projection-based)
# ---------------------------------------------------------------------------


def _detect_line_position_noteheads(
    ink: np.ndarray,
    staff_bands: list,
    interline: float,
    candidates: list[NoteheadCandidate],
    seen: set[tuple[int, int]],
    min_area: float,
    max_area: float,
    min_w: float,
    max_w: float,
    min_h: float,
    max_h: float,
) -> None:
    """Detect noteheads sitting directly on staff lines via projection.

    For each staff line, scan horizontally within a narrow vertical band
    (±0.5 interline). Where the vertical ink density peaks locally, a
    notehead sits on the line. We extract the local bounding box and
    add a NoteheadCandidate if it passes size/position filters.
    """
    h, w = ink.shape
    half_band = int(max(1, interline * 0.6))

    for band_idx, band in enumerate(staff_bands):
        # Scan the 5 staff lines plus 3 ledger lines above and below
        # Ledger lines are spaced at interline intervals from the staff edges
        scan_ys = list(band.line_ys)
        for n in range(1, 6):  # 1-5 ledger lines above
            scan_ys.append(int(band.y_bottom - band.interline_px * n))
        for n in range(1, 6):  # 1-5 ledger lines below
            scan_ys.append(int(band.y_top + band.interline_px * n))
        for line_y in scan_ys:
            ly = int(line_y)
            if ly < half_band or ly >= h - half_band:
                continue

            # Extract the narrow band around this staff line
            y0_band = ly - half_band
            y1_band = ly + half_band + 1
            band_slice = ink[y0_band:y1_band, :]

            # Horizontal projection: ink density per column in this band
            col_density = band_slice.sum(axis=0).astype(np.float32)

            # Smooth to find peaks
            kernel = np.ones(max(1, int(interline * 0.3)), dtype=np.float32)
            kernel = kernel / kernel.sum()
            smooth = np.convolve(col_density, kernel, mode="same")

            # Baseline: median density (staff line alone)
            baseline = float(np.median(smooth[smooth > 0])) if (smooth > 0).any() else 0.0
            threshold = baseline + max(0.3, interline * 0.15)

            # Find columns above threshold
            above = smooth >= threshold
            if not above.any():
                continue

            # Cluster nearby columns into notehead candidates
            in_cluster = False
            cluster_start = 0
            for x in range(w):
                if above[x] and not in_cluster:
                    in_cluster = True
                    cluster_start = x
                elif not above[x] and in_cluster:
                    in_cluster = False
                    cluster_end = x - 1
                    cluster_width = cluster_end - cluster_start + 1

                    if cluster_width < min_w or cluster_width > max_w * 1.5:
                        continue

                    # Compute center and bounding box
                    cx = float(cluster_start + cluster_end) / 2.0
                    cy = float(ly)

                    # Estimate vertical extent by checking ink above/below
                    local_y0 = max(0, ly - int(interline * 0.7))
                    local_y1 = min(h, ly + int(interline * 0.7) + 1)
                    local = ink[
                        local_y0:local_y1,
                        max(0, int(cx - interline)) : min(w, int(cx + interline) + 1),
                    ]
                    if local.size == 0:
                        continue

                    # Find actual vertical bounds
                    rows_with_ink = np.where(local.sum(axis=1) > 0)[0]
                    if len(rows_with_ink) < 2:
                        continue
                    actual_y0 = local_y0 + int(rows_with_ink[0])
                    actual_y1 = local_y0 + int(rows_with_ink[-1])
                    bh = actual_y1 - actual_y0 + 1.0

                    if bh < min_h or bh > max_h * 1.5:
                        continue

                    area = float(local.sum())
                    if area < min_area * 0.5 or area > max_area * 2.0:
                        continue

                    step = band.staff_step_from_y(cy)
                    if step < -8.0 or step > 15.0:
                        continue

                    pos_key = (band_idx, int(round(cx)))
                    if pos_key in seen:
                        continue
                    seen.add(pos_key)

                    is_filled = _classify_notehead_filled(
                        ink,
                        max(0, int(cx - interline)),
                        actual_y0,
                        min(w, int(cx + interline) + 1),
                        actual_y1,
                        interline,
                    )

                    bbox = (
                        max(0, int(cx - interline)),
                        actual_y0,
                        min(w, int(cx + interline) + 1),
                        actual_y1,
                    )
                    candidates.append(
                        NoteheadCandidate(
                            cx=cx,
                            cy=cy,
                            bbox=bbox,
                            area=area,
                            is_filled=is_filled,
                            staff_step=step,
                            staff_band_index=band_idx,
                            source="line_position",
                            confidence=_notehead_confidence(
                                area=area,
                                bbox=bbox,
                                staff_step=step,
                                interline=interline,
                                source="line_position",
                            ),
                        )
                    )


# ---------------------------------------------------------------------------
# Chord splitting
# ---------------------------------------------------------------------------


def _split_chord_noteheads(
    candidates: list[NoteheadCandidate],
    interline: float,
    staff_bands: list,
) -> list[NoteheadCandidate]:
    """Split components that contain two vertically stacked chord noteheads.

    When noteheads on adjacent staff positions (e.g., E4+G4) are close,
    CCL merges them into one tall component. We detect this by height
    and split vertically into two noteheads.
    """
    if not candidates:
        return candidates

    result: list[NoteheadCandidate] = []
    for nh in candidates:
        x0, y0, x1, y1 = nh.bbox
        bh = y1 - y0 + 1.0

        # Only split if extremely tall (clearly 2+ noteheads stacked)
        # Conservative: height > 3× interline
        if bh < interline * 3.0:
            result.append(nh)
            continue

        band = staff_bands[nh.staff_band_index] if nh.staff_band_index < len(staff_bands) else None
        if band is None:
            result.append(nh)
            continue

        # Only split if the two halves would map to different staff steps
        # (different pitches). Otherwise it's just a tall single notehead.
        mid_y = (y0 + y1) / 2.0
        upper_step = int(round(band.staff_step_from_y((y0 + mid_y) / 2.0)))
        lower_step = int(round(band.staff_step_from_y((mid_y + y1) / 2.0)))
        if upper_step == lower_step:
            result.append(nh)
            continue

        # Split vertically at midpoint
        upper_bbox = (x0, y0, x1, int(mid_y))
        upper_step_value = band.staff_step_from_y((y0 + mid_y) / 2.0)
        upper_source = f"{nh.source}:split"
        upper = NoteheadCandidate(
            cx=nh.cx,
            cy=(y0 + mid_y) / 2.0,
            bbox=upper_bbox,
            area=nh.area / 2.0,
            is_filled=nh.is_filled,
            staff_step=upper_step_value,
            staff_band_index=nh.staff_band_index,
            source=upper_source,
            confidence=min(
                nh.confidence,
                _notehead_confidence(
                    area=nh.area / 2.0,
                    bbox=upper_bbox,
                    staff_step=upper_step_value,
                    interline=interline,
                    source=upper_source,
                ),
            ),
        )
        lower_bbox = (x0, int(mid_y) + 1, x1, y1)
        lower_step_value = band.staff_step_from_y((mid_y + y1) / 2.0)
        lower_source = f"{nh.source}:split"
        lower = NoteheadCandidate(
            cx=nh.cx,
            cy=(mid_y + y1) / 2.0,
            bbox=lower_bbox,
            area=nh.area / 2.0,
            is_filled=nh.is_filled,
            staff_step=lower_step_value,
            staff_band_index=nh.staff_band_index,
            source=lower_source,
            confidence=min(
                nh.confidence,
                _notehead_confidence(
                    area=nh.area / 2.0,
                    bbox=lower_bbox,
                    staff_step=lower_step_value,
                    interline=interline,
                    source=lower_source,
                ),
            ),
        )
        result.append(upper)
        result.append(lower)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _erase_staff_lines(ink: np.ndarray, bands: list, half_width: int) -> None:
    """Zero out pixels near staff lines in-place."""
    for band in bands:
        for line_y in band.line_ys:
            top = max(0, int(line_y) - half_width)
            bottom = min(ink.shape[0], int(line_y) + half_width + 1)
            ink[top:bottom, :] = 0


def _find_connected_components(
    ink: np.ndarray,
) -> list[tuple[float, float, float, int, int, int, int]]:
    """Find connected components via flood-fill.

    Returns list of (area, cx, cy, x0, y0, x1, y1) tuples.
    """
    h, w = ink.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    components: list[tuple[float, float, float, int, int, int, int]] = []

    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for y in range(h):
        for x in range(w):
            if ink[y, x] == 0 or visited[y, x] != 0:
                continue
            stack = [(y, x)]
            visited[y, x] = 1
            min_x, max_x = x, x
            min_y, max_y = y, y
            pixel_count = 0
            sum_x, sum_y = 0.0, 0.0

            while stack:
                cy, cx = stack.pop()
                pixel_count += 1
                sum_x += float(cx)
                sum_y += float(cy)
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if ink[ny, nx] == 0 or visited[ny, nx] != 0:
                        continue
                    visited[ny, nx] = 1
                    stack.append((ny, nx))

            if pixel_count > 0:
                components.append(
                    (
                        float(pixel_count),
                        sum_x / float(pixel_count),
                        sum_y / float(pixel_count),
                        min_x,
                        min_y,
                        max_x,
                        max_y,
                    )
                )

    return components


def _nearest_band(y: float, bands: list) -> "StaffBand | None":
    """Find the nearest staff band to a y-coordinate."""
    _, band = _nearest_band_with_index(y, bands)
    return band


def _nearest_band_with_index(y: float, bands: list) -> tuple[int, "StaffBand | None"]:
    """Find the index and nearest staff band to a y-coordinate."""
    if not bands:
        return -1, None
    best_idx = 0
    best_dist = float("inf")
    for idx, band in enumerate(bands):
        dist = abs(band.y_center - y)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx, bands[best_idx]


def _longest_ink_run(column: np.ndarray) -> int:
    """Find the longest run of ink pixels in a column."""
    max_run = 0
    run = 0
    for value in column.flat:
        if int(value) > 0:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run
