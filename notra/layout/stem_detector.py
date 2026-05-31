"""
Global stem detection: find all vertical dark runs in the page,
filter by geometry, merge fragments, reject barlines, then
attach to noteheads by geometric proximity.

Replaces the old per-notehead single-column search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StemCandidate:
    x0: int
    y0: int
    x1: int
    y1: int
    x_center: float
    height: int
    width: int
    score: float = 0.0
    staff_id: int = -1
    direction: str = ""  # "up" or "down"


def detect_stems_global(
    binary: np.ndarray,
    staff_bands: list,
    noteheads: list,
) -> tuple[list[StemCandidate], dict[int, StemCandidate]]:
    """Detect stems globally and associate with noteheads.

    Args:
        binary: Binarized image (0=bg, 1=ink, Sauvola output).
        staff_bands: Detected staff bands.
        noteheads: Detected notehead candidates.

    Returns:
        (all_stems, stem_map) where stem_map is {notehead_index: StemCandidate}
        for associated stems.
    """
    if binary.size == 0 or not staff_bands:
        return [], {}

    h, w = binary.shape
    interline = float(np.median([b.interline_px for b in staff_bands]))
    interline = max(4.0, interline)

    # Thresholds in staff-space units
    min_stem_height = int(interline * 1.5)
    max_stem_height = int(interline * 5.5)
    max_stem_width = int(interline * 0.45)
    barline_min_height = int(interline * 5.75)
    max_col_x_gap = max(1, int(interline * 0.15))
    max_fragment_y_gap = int(interline * 0.50)

    # --- Step 1: Column-run detection ---
    # For each x-column, find all vertical dark runs
    raw_runs: list[tuple[int, int, int]] = []  # (x, y0, y1)

    for x in range(w):
        col = binary[:, x]
        in_run = False
        run_start = 0
        for y in range(h):
            if col[y] > 0:
                if not in_run:
                    run_start = y
                    in_run = True
            else:
                if in_run:
                    run_len = y - run_start
                    if run_len >= min_stem_height:
                        raw_runs.append((x, run_start, y - 1))
                    in_run = False
        if in_run:
            run_len = h - run_start
            if run_len >= min_stem_height:
                raw_runs.append((x, run_start, h - 1))

    if not raw_runs:
        return [], {}

    # --- Step 2: Merge adjacent columns into stem bboxes ---
    # Cluster by x-proximity and y-overlap
    from operator import itemgetter
    raw_runs.sort(key=itemgetter(0))  # sort by x

    merged: list[list[tuple[int, int, int]]] = []
    used = set()

    for i, (xi, y0i, y1i) in enumerate(raw_runs):
        if i in used:
            continue
        cluster = [(xi, y0i, y1i)]
        used.add(i)
        # Look ahead for adjacent columns with overlapping y-ranges
        for j in range(i + 1, len(raw_runs)):
            if j in used:
                continue
            xj, y0j, y1j = raw_runs[j]
            if xj - xi > max_col_x_gap:
                break  # sorted by x, too far
            # Check y-overlap
            overlap = min(y1i, y1j) - max(y0i, y0j)
            if overlap >= min_stem_height * 0.5:
                cluster.append((xj, y0j, y1j))
                used.add(j)

        if len(cluster) >= 1:
            merged.append(cluster)

    # Convert clusters to StemCandidates
    candidates: list[StemCandidate] = []
    for cluster in merged:
        xs = [r[0] for r in cluster]
        ys0 = [r[1] for r in cluster]
        ys1 = [r[2] for r in cluster]
        x0 = min(xs)
        x1 = max(xs)
        y0 = min(ys0)
        y1 = max(ys1)
        stem_h = y1 - y0 + 1
        stem_w = x1 - x0 + 1

        if stem_h < min_stem_height or stem_h > max_stem_height:
            continue
        if stem_w > max_stem_width:
            continue

        candidates.append(
            StemCandidate(
                x0=x0, y0=y0, x1=x1, y1=y1,
                x_center=float(np.mean(xs)),
                height=stem_h,
                width=stem_w,
            )
        )

    if not candidates:
        return [], {}

    # --- Step 3: Merge collinear fragments ---
    # Stems broken by staff lines should be merged
    candidates.sort(key=lambda c: c.x_center)
    merged_candidates: list[StemCandidate] = []
    used_c = set()

    for i, a in enumerate(candidates):
        if i in used_c:
            continue
        best = a
        for j, b in enumerate(candidates):
            if j <= i or j in used_c:
                continue
            if abs(a.x_center - b.x_center) <= max(1.0, interline * 0.20):
                y_gap = b.y0 - a.y1
                if 0 <= y_gap <= max_fragment_y_gap:
                    # Merge: extend a's y-range to include b
                    new_y0 = min(a.y0, b.y0)
                    new_y1 = max(a.y1, b.y1)
                    best = StemCandidate(
                        x0=min(a.x0, b.x0), y0=new_y0,
                        x1=max(a.x1, b.x1), y1=new_y1,
                        x_center=(a.x_center + b.x_center) / 2,
                        height=new_y1 - new_y0 + 1,
                        width=max(a.width, b.width),
                    )
                    used_c.add(j)
        merged_candidates.append(best)

    # --- Step 4: Reject barlines ---
    filtered: list[StemCandidate] = []
    for c in merged_candidates:
        if c.height >= barline_min_height:
            continue  # too tall = barline
        if c.width > interline * 0.60:
            continue  # too wide = not a stem
        filtered.append(c)

    # --- Step 5: Assign staff IDs ---
    for c in filtered:
        cy = (c.y0 + c.y1) / 2
        best_band = -1
        best_dist = float("inf")
        for bi, band in enumerate(staff_bands):
            dist = abs(cy - (band.y_top + band.y_bottom) / 2)
            if dist < best_dist:
                best_dist = dist
                best_band = bi
        c.staff_id = best_band

    # --- Step 6: Associate stems to noteheads ---
    stem_map: dict[int, StemCandidate] = {}

    for nh_idx, nh in enumerate(noteheads):
        nh_x0, nh_y0, nh_x1, nh_y1 = nh.bbox
        nh_cy = nh.cy
        nh_band = getattr(nh, "staff_band_index", 0)

        best_stem = None
        best_score = -float("inf")

        for stem in filtered:
            if stem.staff_id != nh_band:
                continue

            sx = stem.x_center
            sy0 = stem.y0
            sy1 = stem.y1

            # Up-stem: attaches to RIGHT side, extends UPWARD
            right_dist = abs(sx - nh_x1)
            up_attach = np.exp(-right_dist / max(1.0, interline * 0.25))
            up_touch = 1.0 if sy1 >= nh_y0 - interline * 0.5 else 0.0
            up_dir = 1.0 if sy0 < nh_cy - interline * 0.3 else -1.0
            up_score = up_attach * 2.0 + up_touch + up_dir

            # Down-stem: attaches to LEFT side, extends DOWNWARD
            left_dist = abs(sx - nh_x0)
            down_attach = np.exp(-left_dist / max(1.0, interline * 0.25))
            down_touch = 1.0 if sy0 <= nh_y1 + interline * 0.5 else 0.0
            down_dir = 1.0 if sy1 > nh_cy + interline * 0.3 else -1.0
            down_score = down_attach * 2.0 + down_touch + down_dir

            if up_score >= down_score and up_score > best_score:
                best_score = up_score
                best_stem = stem
                stem.direction = "up"
            elif down_score > best_score:
                best_score = down_score
                best_stem = stem
                stem.direction = "down"

        if best_stem is not None and best_score > -1.0:
            stem_map[nh_idx] = best_stem

    return filtered, stem_map
