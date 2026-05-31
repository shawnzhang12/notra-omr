"""
Beam detection: find near-horizontal dark rectangles connecting stem endpoints.

Uses the Sauvola binary image. Beams are thick horizontal runs above/below
the staff, connecting two or more stems. Detected geometrically via:

1. Horizontal run detection per row
2. Merge adjacent rows into beam bboxes
3. Associate with nearby stem endpoints
4. Determine beam level (1=eighth, 2=sixteenth, 3=32nd)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BeamCandidate:
    x0: int
    y0: int
    x1: int
    y1: int
    thickness: float  # in interline units
    level: int  # 1=eighth beam, 2=sixteenth, 3=32nd
    staff_id: int
    connected_stems: list[int]  # stem indices
    score: float


def detect_beams(
    binary: np.ndarray,
    staff_bands: list,
    stem_candidates: list,  # from stem_detector.StemCandidate
    interline: float,
) -> list[BeamCandidate]:
    """Detect beam candidates from binary image.

    Args:
        binary: Sauvola-binarized image.
        staff_bands: Staff bands.
        stem_candidates: Global stem candidates.
        interline: Staff interline spacing in pixels.

    Returns:
        List of BeamCandidate with connected stem information.
    """
    if binary.size == 0 or not staff_bands:
        return []

    h, w = binary.shape

    # Beam thickness: 0.25-0.45 interline spaces
    min_thickness = max(2, int(interline * 0.20))
    max_thickness = int(interline * 0.55)
    min_run_length = int(interline * 1.5)  # beams span at least 1.5 staff spaces

    # --- Step 1: Find horizontal dark runs in each row ---
    raw_runs: list[tuple[int, int, int]] = []  # (y, x0, x1)

    for y in range(h):
        row = binary[y, :]
        in_run = False
        run_start = 0
        for x in range(w):
            if row[x] > 0:
                if not in_run:
                    run_start = x
                    in_run = True
            else:
                if in_run:
                    run_len = x - run_start
                    if run_len >= min_run_length:
                        raw_runs.append((y, run_start, x - 1))
                    in_run = False
        if in_run:
            if w - run_start >= min_run_length:
                raw_runs.append((y, run_start, w - 1))

    if not raw_runs:
        return []

    # --- Step 1.5: Reject runs overlapping staff lines ---
    staff_line_ys: set[int] = set()
    for band in staff_bands:
        for ly in band.line_ys:
            staff_line_ys.add(int(ly))

    filtered_runs: list[tuple[int, int, int]] = []
    for y, x0, x1 in raw_runs:
        is_staff = any(abs(y - sl) <= 2 for sl in staff_line_ys)
        if not is_staff:
            filtered_runs.append((y, x0, x1))

    if not filtered_runs:
        return []
    raw_runs = filtered_runs

    # --- Step 2: Merge adjacent rows into beam bboxes ---
    # Sort by y, then cluster vertically adjacent rows with overlapping x-ranges
    from operator import itemgetter
    raw_runs.sort(key=itemgetter(0))

    clusters: list[list[tuple[int, int, int]]] = []
    used = set()

    for i, (yi, x0i, x1i) in enumerate(raw_runs):
        if i in used:
            continue
        cluster = [(yi, x0i, x1i)]
        used.add(i)
        for j in range(i + 1, len(raw_runs)):
            if j in used:
                continue
            yj, x0j, x1j = raw_runs[j]
            if yj - yi > max_thickness + 2:
                break
            # Check x-overlap: runs must share horizontal span
            overlap = min(x1i, x1j) - max(x0i, x0j)
            if overlap >= min_run_length * 0.5:
                cluster.append((yj, x0j, x1j))
                used.add(j)
        clusters.append(cluster)

    # Convert to beam candidates
    beams: list[BeamCandidate] = []
    for cluster in clusters:
        ys = [r[0] for r in cluster]
        xs0 = [r[1] for r in cluster]
        xs1 = [r[2] for r in cluster]
        y0 = min(ys)
        y1 = max(ys)
        x0 = int(np.median(xs0))
        x1 = int(np.median(xs1))
        thickness = y1 - y0 + 1

        if thickness < min_thickness or thickness > max_thickness:
            continue

        # Filter: beams should be roughly horizontal (slope < 5°)
        # Check x-span consistency across rows
        if len(cluster) >= 3:
            x_starts = np.array(xs0)
            start_var = float(np.std(x_starts)) / max(float(np.mean(x_starts)), 1.0)
            if start_var > 0.1:
                continue

        # Beam level: always 1 (eighth) for now.
        # Proper multi-beam detection (two parallel beams separated
        # by a gap) needs separate implementation.
        level = 1

        beams.append(BeamCandidate(
            x0=int(x0), y0=y0, x1=int(x1), y1=y1,
            thickness=thickness / interline,
            level=level,
            staff_id=-1,
            connected_stems=[],
            score=0.5,
        ))

    # Assign staff IDs
    for b in beams:
        by = (b.y0 + b.y1) / 2
        best = -1
        best_dist = float("inf")
        for bi, band in enumerate(staff_bands):
            d = abs(by - (band.y_top + band.y_bottom) / 2)
            if d < best_dist:
                best_dist = d
                best = bi
        b.staff_id = best

    # --- Step 4: Associate beams with nearby stem endpoints ---
    for b in beams:
        for si, stem in enumerate(stem_candidates):
            if stem.staff_id != b.staff_id:
                continue
            sx = stem.x_center
            # Beam must overlap the stem horizontally
            if b.x0 - interline * 0.5 <= sx <= b.x1 + interline * 0.5:
                # Beam must be near stem top (up-stem) or bottom (down-stem)
                stem_top = stem.y0 if stem.direction == "up" else stem.y1
                stem_bot = stem.y1 if stem.direction == "up" else stem.y0
                beam_dist = min(abs(stem_top - b.y1), abs(stem_bot - b.y0),
                               abs(stem_top - b.y0), abs(stem_bot - b.y1))
                if beam_dist <= interline * 0.5:
                    b.connected_stems.append(si)

    # Filter: beams must connect at least 2 stems
    beams = [b for b in beams if len(b.connected_stems) >= 2]

    return beams
