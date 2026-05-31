"""Staff-line detection, band grouping, and staff-to-pitch coordinate mapping.

The staff is the fundamental coordinate frame for OMR. Everything — notes,
clefs, accidentals, dynamics — is positioned relative to the five staff lines.
This module detects those lines and maps page y-coordinates to diatonic staff
steps, which are then mapped to concrete pitches via a clef reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Diatonic reference tables
# ---------------------------------------------------------------------------

DIATONIC_STEPS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G")
_DIATONIC_INDEX: dict[str, int] = {s: i for i, s in enumerate(DIATONIC_STEPS)}

# For each clef (sign, line), the reference note and its octave.
# "line" is 1-indexed (bottom line = 1).
_CLEF_REFERENCE: dict[tuple[str, int], tuple[str, int]] = {
    ("G", 2): ("G", 4),   # Treble: G clef on line 2 → G4
    ("F", 4): ("F", 3),   # Bass:   F clef on line 4 → F3
    ("C", 3): ("C", 4),   # Alto:   C clef on line 3 → C4
    ("C", 4): ("C", 4),   # Tenor:  C clef on line 4 → C4
    ("G", 1): ("G", 4),   # French violin: G on line 1
    ("F", 3): ("F", 3),   # Baritone: F on line 3
    ("C", 1): ("C", 4),   # Soprano: C on line 1
    ("C", 2): ("C", 4),   # Mezzo-soprano: C on line 2
    ("C", 5): ("C", 4),   # Baritone: C on line 5
}


# ---------------------------------------------------------------------------
# StaffBand
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StaffBand:
    """One five-line staff band detected from an image.

    Attributes:
        line_ys: Y-coordinates (pixels) of the five staff lines, bottom-to-top
                 (line 1 through line 5).
        interline_px: Mean interline spacing in pixels.
    """

    line_ys: tuple[int, int, int, int, int]
    interline_px: float

    # Derived properties ---------------------------------------------------
    @property
    def y_top(self) -> int:
        """Y-coordinate of the top staff line (line 5)."""
        return self.line_ys[-1]

    @property
    def y_bottom(self) -> int:
        """Y-coordinate of the bottom staff line (line 1)."""
        return self.line_ys[0]

    @property
    def y_center(self) -> float:
        """Y-coordinate of the middle line (line 3)."""
        return float(self.line_ys[2])

    @property
    def staff_height(self) -> int:
        """Vertical span of the five-line staff in pixels."""
        return self.y_top - self.y_bottom

    def staff_step_from_y(self, y: float) -> float:
        """Convert a page y-coordinate to a continuous staff step.

        Step 0.0 = bottom line (line 1). Step 8.0 = top line (line 5).
        Fractional values represent positions between lines/spaces.
        Negative values are below the staff; values > 8.0 are above.

        The mapping is linear: two half-steps per interline space.
        """
        half_step_px = self.interline_px / 2.0
        return (self.y_center - y) / half_step_px + 4.0


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_staff_lines(
    gray: np.ndarray,
    *,
    binarize: bool = True,
    smooth_sigma: float = 2.0,
    threshold_sigma: float = 1.2,
) -> list[int]:
    """Detect staff-line y-positions from a grayscale image.

    Uses horizontal projection profile with smoothing kernel,
    thresholding, and peak clustering.

    Args:
        gray: 2-D uint8 grayscale image.
        binarize: If True, apply Otsu threshold before projection.
        smooth_sigma: Sigma for the Gaussian smoothing kernel width.
        threshold_sigma: Number of std deviations above mean for peak threshold.

    Returns:
        Sorted list of y-coordinates of detected staff lines.
    """
    if gray.size == 0:
        return []

    if binarize:
        ink = _otsu_binarize(gray)
    else:
        ink = gray.astype(np.float32)

    # Horizontal projection: count ink pixels per row
    row_density = ink.sum(axis=1).astype(np.float32)
    if row_density.sum() <= 0:
        return []

    # Smooth with Gaussian-like kernel
    kernel_size = max(3, int(round(smooth_sigma * 3)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = _gaussian_kernel(kernel_size, smooth_sigma)
    smooth = np.convolve(row_density, kernel, mode="same")

    # Threshold
    mean_val = float(np.mean(smooth))
    std_val = float(np.std(smooth))
    threshold = mean_val + threshold_sigma * std_val

    # Find candidate rows above threshold
    candidate_rows = np.where(smooth >= threshold)[0].tolist()
    if not candidate_rows:
        return []

    # Cluster nearby rows (within 3px) and take centroid
    merged: list[int] = []
    cluster: list[int] = [candidate_rows[0]]
    for row in candidate_rows[1:]:
        if row - cluster[-1] <= 3:
            cluster.append(row)
            continue
        merged.append(int(round(float(sum(cluster)) / float(len(cluster)))))
        cluster = [row]
    merged.append(int(round(float(sum(cluster)) / float(len(cluster)))))

    # --- Filter text-like lines ---
    # Staff lines span most of the page width (even if broken by notation).
    # Text/titles produce short runs only. Filter lines whose total ink
    # span (first ink to last ink column) doesn't cover ≥30% of image width.
    if binarize and merged:
        ink = _otsu_binarize(gray)
        img_w = ink.shape[1]
        min_span = int(img_w * 0.50)
        filtered: list[int] = []
        for y in merged:
            row = ink[y, :]
            ink_cols = np.where(row > 0)[0]
            if len(ink_cols) > 0:
                span = int(ink_cols[-1] - ink_cols[0])
                if span >= min_span:
                    filtered.append(y)
        merged = filtered

    return merged


def group_staff_bands(
    staff_lines: Sequence[int],
    *,
    min_interline: float = 4.0,
    max_interline: float = 50.0,
    tolerance_ratio: float = 0.35,
) -> tuple[list[StaffBand], float]:
    """Group detected staff lines into five-line staff bands.

    Uses the median interline spacing to identify groups of five consecutive
    lines with regular spacing.

    Args:
        staff_lines: Sorted y-positions from detect_staff_lines.
        min_interline, max_interline: Valid interline spacing bounds (px).
        tolerance_ratio: Max allowed deviation from median spacing, as a
            fraction of the spacing.

    Returns:
        (bands, global_interline) tuple. bands may be empty if no valid
        groupings are found.
    """
    lines = list(staff_lines)
    if len(lines) < 5:
        return [], 12.0

    # --- Filter isolated lines ---
    # Real staff lines appear in dense groups (5 lines within ~4× interline).
    # Isolated lines (text, artifacts) have no neighbors within 2× interline.
    # Remove them before band formation to prevent phantom bands.
    if len(lines) > 5:
        diffs_all = np.diff(np.asarray(lines, dtype=np.float32))
        if diffs_all.size > 0:
            valid_diffs_all = diffs_all[(diffs_all >= 3) & (diffs_all <= 80)]
            med_diff = float(np.median(valid_diffs_all)) if valid_diffs_all.size else 12.0
        else:
            med_diff = 12.0
        med_diff = max(4.0, med_diff)
        keep: list[int] = []
        for i, y in enumerate(lines):
            has_near_above = i > 0 and (y - lines[i - 1]) <= med_diff * 2.0
            has_near_below = i + 1 < len(lines) and (lines[i + 1] - y) <= med_diff * 2.0
            if has_near_above or has_near_below:
                keep.append(y)
        if len(keep) >= 5:
            lines = keep

    # Estimate global interline spacing from line-to-line differences.
    # Use 25th percentile — interline spacing is the SMALLEST consistent
    # gap; larger gaps are between-staff or between-system spacing.
    diffs = np.diff(np.asarray(lines, dtype=np.float32))
    valid_diffs = diffs[(diffs >= min_interline) & (diffs <= max_interline)]
    if valid_diffs.size == 0:
        interline = 12.0
    else:
        interline = float(np.percentile(valid_diffs, 25))
    interline = max(min_interline, interline)

    # --- Interpolate missing staff lines ---
    # Faint lines may fall below the detection threshold, creating gaps
    # of 2× or 3× interline. Fill these with synthetic lines so
    # group_staff_bands can form complete 5-line groups.
    lines = _interpolate_missing_lines(lines, interline)

    # Walk through lines looking for groups of 5 with consistent spacing
    bands: list[StaffBand] = []
    idx = 0
    while idx + 4 < len(lines):
        block = lines[idx : idx + 5]
        block_diffs = np.diff(np.asarray(block, dtype=np.float32))
        tolerance = max(2.0, interline * tolerance_ratio)
        if np.all(np.abs(block_diffs - interline) <= tolerance):
            bands.append(
                StaffBand(
                    line_ys=(block[0], block[1], block[2], block[3], block[4]),
                    interline_px=float(np.mean(block_diffs)),
                )
            )
            idx += 5
            continue
        idx += 1

    if not bands:
        return [], interline

    interline_est = float(np.median([band.interline_px for band in bands]))
    return bands, interline_est


# ---------------------------------------------------------------------------
# Pitch mapping
# ---------------------------------------------------------------------------


def staff_step_to_pitch(
    step: float,
    clef_sign: str,
    clef_line: int,
) -> tuple[str, int, int]:
    """Map a staff-step value to a (step_name, octave, alter) tuple.

    Args:
        step: Continuous staff step (0.0 = bottom line, 8.0 = top line).
        clef_sign: Clef letter ("G", "F", "C").
        clef_line: Line the clef sits on (1-5).

    Returns:
        (diatonic_step, octave, alter) where diatonic_step is "A"-"G",
        octave is the MIDI octave number, and alter is 0 (no accidental
        applied — this is the *diatonic* pitch at that position).
    """
    # Round to nearest integer step (noteheads sit on lines or spaces)
    discrete_step = int(round(step))

    # Determine the pitch at staff step 0 for this clef
    # Treble (G on line 2): line 2 = G4, so bottom line (step 0) = E4
    # Bass   (F on line 4): line 4 = F3, so bottom line (step 0) = G2
    # Alto   (C on line 3): line 3 = C4, so bottom line (step 0) = F3
    _STEP0_PITCH: dict[tuple[str, int], tuple[str, int]] = {
        ("G", 2): ("E", 4),   # Treble
        ("F", 4): ("G", 2),   # Bass
        ("C", 3): ("F", 3),   # Alto
        ("C", 4): ("E", 3),   # Tenor
        ("G", 1): ("D", 4),   # French violin
        ("F", 3): ("D", 2),   # Baritone
        ("C", 1): ("E", 3),   # Soprano
        ("C", 2): ("D", 3),   # Mezzo-soprano
        ("C", 5): ("A", 2),   # Baritone C clef
    }

    key = (clef_sign, clef_line)
    base_note, base_octave = _STEP0_PITCH.get(key, ("E", 4))
    base_idx = _DIATONIC_INDEX[base_note]

    # Walk diatonic steps from base, with proper octave tracking.
    # Octave increments when crossing from B to C (index 1→2).
    delta = discrete_step  # delta from step 0

    if delta >= 0:
        note_idx = base_idx
        octave = base_octave
        for _ in range(delta):
            # Check if next note crosses octave boundary (B → C)
            if note_idx == 1:  # B
                note_idx = 2  # C
                octave += 1
            else:
                note_idx += 1
                if note_idx >= 7:
                    note_idx = 0
                    # Octave does NOT increment at G→A
    else:
        note_idx = base_idx
        octave = base_octave
        for _ in range(-delta):
            # Check if previous note crosses octave boundary (C → B)
            if note_idx == 2:  # C
                note_idx = 1  # B
                octave -= 1
            else:
                note_idx -= 1
                if note_idx < 0:
                    note_idx = 6
                    # Octave does NOT decrement at A→G

    # Clamp octave to valid range
    octave = max(0, min(9, octave))

    return DIATONIC_STEPS[note_idx], octave, 0


def staff_step_range_for_band(band: StaffBand) -> tuple[float, float]:
    """Return the (min_step, max_step) for a staff band.

    Includes one ledger line above and below.
    """
    return -2.0, 10.0


# ---------------------------------------------------------------------------
# Missing-line interpolation
# ---------------------------------------------------------------------------


def _interpolate_missing_lines(lines: list[int], interline: float) -> list[int]:
    """Fill gaps of 2×–3× interline with synthetic staff lines.

    When a staff line is too faint to be detected, the gap between
    adjacent detected lines will be ~2× interline. This function inserts
    a synthetic line at the midpoint of such gaps, recovering the
    missing line so group_staff_bands can form complete 5-line groups.
    """
    if len(lines) < 3 or interline <= 0:
        return lines

    # Identify staff-group lines (have neighbors within 3× interline)
    is_staff: list[bool] = []
    for i in range(len(lines)):
        has_near = (i > 0 and lines[i] - lines[i-1] <= interline * 2.0) or \
                   (i+1 < len(lines) and lines[i+1] - lines[i] <= interline * 2.0)
        is_staff.append(has_near)

    result: list[int] = []
    for i in range(len(lines)):
        result.append(lines[i])
        if i + 1 >= len(lines):
            break
        # Only interpolate if both lines are in a staff group
        if not is_staff[i] or not is_staff[i+1]:
            continue
        gap = lines[i + 1] - lines[i]

        # Gap of ~2× interline → one missing line
        if interline * 1.5 < gap <= interline * 2.5:
            mid = int(round(lines[i] + gap / 2.0))
            result.append(mid)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _otsu_binarize(gray: np.ndarray) -> np.ndarray:
    """Binarize a grayscale image using Otsu's method. Returns uint8 0/1."""
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total = float(gray.size)
    if total <= 0:
        return np.zeros_like(gray, dtype=np.uint8)

    probs = hist / total
    omega = np.cumsum(probs)
    mu = np.cumsum(probs * np.arange(256, dtype=np.float64))
    mu_total = float(mu[-1])

    sigma_b: np.ndarray = np.zeros(256, dtype=np.float64)
    for t in range(256):
        w0 = omega[t]
        w1 = 1.0 - w0
        if w0 <= 1e-12 or w1 <= 1e-12:
            continue
        mu0 = mu[t] / w0
        mu1 = (mu_total - mu[t]) / w1
        sigma_b[t] = w0 * w1 * (mu0 - mu1) * (mu0 - mu1)

    threshold = int(np.argmax(sigma_b))

    # Auto-detect ink polarity: ink is the minority class
    # For dark-on-light (scans): ink is below threshold → gray < threshold
    # For light-on-dark (alpha channel): ink is above threshold → gray > threshold
    below_count = float((gray <= threshold).sum())
    above_count = float(gray.size) - below_count
    if below_count <= above_count:
        # Dark ink on light background (typical)
        return (gray < threshold).astype(np.uint8)
    else:
        # Light ink on dark background (alpha channel, inverted)
        return (gray > threshold).astype(np.uint8)


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a 1-D Gaussian kernel."""
    half = size // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


# ---------------------------------------------------------------------------
# Adaptive binarization (Sauvola) — preserves anti-aliased edges
# ---------------------------------------------------------------------------


def sauvola_binarize(
    gray: np.ndarray,
    *,
    window: int = 41,
    k: float = 0.2,
    R: float = 128.0,
) -> np.ndarray:
    """Binarize using Sauvola's adaptive thresholding.

    Uses integral images for O(1) per-pixel mean/variance computation.
    Much better than global Otsu for anti-aliased renders because local
    statistics preserve gray edges as ink.

    Args:
        gray: 2-D uint8 grayscale image.
        window: Local window size in pixels (odd, default 41).
        k: Sensitivity (0.2 = more ink, 0.5 = less ink).
        R: Dynamic range of std (default 128 for 8-bit).

    Returns:
        Binary uint8 image (0=background, 1=ink).
    """
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    img = gray.astype(np.float64)
    h, w = img.shape

    if window % 2 == 0:
        window += 1
    half = window // 2

    padded = np.pad(img, half, mode="edge")

    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = padded
    integral = integral.cumsum(axis=0).cumsum(axis=1)

    integral_sq = np.zeros_like(integral)
    integral_sq[1:, 1:] = padded * padded
    integral_sq = integral_sq.cumsum(axis=0).cumsum(axis=1)

    n = window * window
    mean = (integral[window:, window:] - integral[window:, :w] -
            integral[:h, window:] + integral[:h, :w]) / n

    sq_mean = (integral_sq[window:, window:] - integral_sq[window:, :w] -
               integral_sq[:h, window:] + integral_sq[:h, :w]) / n

    variance = sq_mean - mean * mean
    std = np.sqrt(np.maximum(variance, 0.0))

    threshold = mean * (1.0 + k * (std / R - 1.0))

    binary = (img <= threshold).astype(np.uint8)
    return binary
