"""Weak segmentation mask generation from deterministic OMR geometry.

These masks are pseudo-labels, not ground truth.  They are designed to bootstrap
the first tiny U-Net by converting stable staff-relative detections into
semantic class maps.  Human correction or model-assisted relabeling can replace
individual classes later without changing the training data contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from notra.layout.beam_detector import BeamCandidate
from notra.layout.staff import StaffBand
from notra.layout.stem_detector import StemCandidate as GlobalStemCandidate
from notra.layout.symbol import NoteheadCandidate
from notra.pipeline.config import PipelineConfig
from notra.pipeline.stages import (
    detect_accidentals_stage,
    detect_clefs_stage,
    detect_layout_stage,
    detect_noteheads_stage,
    detect_rests_stage,
    detect_stems_stage,
    load_image_stage,
)
from notra.vision.schema import SegmentationClass
from notra.vision.segmentation import SegmentationInstanceExtractor

MASK_PALETTE: dict[SegmentationClass, tuple[int, int, int]] = {
    SegmentationClass.BACKGROUND: (255, 255, 255),
    SegmentationClass.STAFF_LINE: (32, 32, 32),
    SegmentationClass.NOTEHEAD_FILLED: (230, 57, 70),
    SegmentationClass.NOTEHEAD_OPEN: (244, 162, 97),
    SegmentationClass.STEM: (42, 157, 143),
    SegmentationClass.BEAM: (38, 70, 83),
    SegmentationClass.BARLINE: (69, 123, 157),
    SegmentationClass.LEDGER_LINE: (29, 53, 87),
    SegmentationClass.CLEF: (131, 56, 236),
    SegmentationClass.REST: (255, 183, 3),
    SegmentationClass.ACCIDENTAL: (0, 150, 199),
    SegmentationClass.ARTICULATION: (255, 0, 110),
    SegmentationClass.TIE_SLUR: (90, 24, 154),
    SegmentationClass.DYNAMIC_MARK: (106, 76, 147),
    SegmentationClass.TEXT: (120, 120, 120),
    SegmentationClass.FINGERING: (251, 133, 0),
}


@dataclass(frozen=True, slots=True)
class PseudoMaskConfig:
    """Controls deterministic pseudo-mask generation."""

    staff_line_half_width: int = 1
    barline_half_width: int = 2
    stem_half_width: int = 1
    clef_region_interlines: float = 4.5
    notehead_padding_interlines: float = 0.10
    min_notehead_staff_step: float = -4.0
    max_notehead_staff_step: float = 12.0
    min_notehead_confidence: float = 0.0
    label_notehead_ellipse: bool = True
    include_clef_region: bool = True
    include_accidentals: bool = False
    overwrite_priority: tuple[SegmentationClass, ...] = field(
        default_factory=lambda: (
            SegmentationClass.STAFF_LINE,
            SegmentationClass.BARLINE,
            SegmentationClass.STEM,
            SegmentationClass.BEAM,
            SegmentationClass.CLEF,
            SegmentationClass.REST,
            SegmentationClass.ACCIDENTAL,
            SegmentationClass.NOTEHEAD_OPEN,
            SegmentationClass.NOTEHEAD_FILLED,
        )
    )


@dataclass(frozen=True, slots=True)
class PseudoMaskResult:
    """One generated pseudo-label mask with diagnostics."""

    image_path: Path
    mask: np.ndarray
    class_pixel_counts: dict[str, int]
    symbol_counts: dict[str, int]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_summary(self) -> dict[str, object]:
        return {
            "image_path": str(self.image_path),
            "height": int(self.mask.shape[0]),
            "width": int(self.mask.shape[1]),
            "class_pixel_counts": self.class_pixel_counts,
            "symbol_counts": self.symbol_counts,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def generate_pseudo_segmentation_mask(
    image_path: str | Path,
    *,
    config: PseudoMaskConfig | None = None,
    pipeline_config: PipelineConfig | None = None,
) -> PseudoMaskResult:
    """Run deterministic stages and convert their hypotheses into a class mask."""
    image_path = Path(image_path)
    config = config or PseudoMaskConfig()
    pipeline_config = pipeline_config or PipelineConfig.for_image(image_path)

    ctx: dict[str, Any] = {
        "image_path": str(image_path),
        "errors": [],
        "warnings": [],
        "metrics": {},
    }
    ctx.update(pipeline_config.to_context())

    stage_fns = [
        load_image_stage,
        detect_layout_stage,
        detect_clefs_stage,
        detect_noteheads_stage,
        detect_rests_stage,
        detect_stems_stage,
    ]
    if config.include_accidentals:
        stage_fns.append(detect_accidentals_stage)

    for stage_fn in stage_fns:
        try:
            stage_fn(ctx)
        except Exception as exc:  # pragma: no cover - defensive artifact path
            ctx.setdefault("errors", []).append(f"{stage_fn.__name__}: {exc}")

    gray = ctx.get("gray")
    ink = ctx.get("ink")
    if not isinstance(gray, np.ndarray):
        raise RuntimeError(f"{image_path}: image stage did not produce grayscale array")
    if not isinstance(ink, np.ndarray):
        ink = _binarize_dark_ink(gray)

    mask = build_pseudo_segmentation_mask(
        ink=ink,
        staff_bands=ctx.get("staff_bands", []),
        noteheads=ctx.get("notehead_candidates", []),
        rests=ctx.get("rest_candidates", []),
        stems=list(ctx.get("stem_map", {}).values()),
        beams=ctx.get("beam_candidates", []),
        barline_xs=ctx.get("barline_xs", []),
        system_members=ctx.get("system_members", []),
        accidental_map=ctx.get("accidental_map", {}),
        config=config,
    )

    instances = SegmentationInstanceExtractor().extract(
        mask,
        staff_bands=ctx.get("staff_bands", []),
    )
    symbol_counts: dict[str, int] = {}
    for instance in instances:
        symbol_counts[instance.class_name] = symbol_counts.get(instance.class_name, 0) + 1

    return PseudoMaskResult(
        image_path=image_path,
        mask=mask,
        class_pixel_counts=class_pixel_counts(mask),
        symbol_counts=symbol_counts,
        errors=tuple(str(item) for item in ctx.get("errors", [])),
        warnings=tuple(str(item) for item in ctx.get("warnings", [])),
    )


def build_pseudo_segmentation_mask(
    *,
    ink: np.ndarray,
    staff_bands: list[StaffBand],
    noteheads: list[NoteheadCandidate] | None = None,
    rests: list[NoteheadCandidate] | None = None,
    stems: list[Any] | None = None,
    beams: list[BeamCandidate] | None = None,
    barline_xs: list[float] | None = None,
    system_members: list[list[int]] | None = None,
    accidental_map: dict[int, str] | None = None,
    config: PseudoMaskConfig | None = None,
) -> np.ndarray:
    """Build a semantic class map from already-computed deterministic hypotheses."""
    if ink.ndim != 2:
        raise ValueError("ink must be a 2-D binary image")

    config = config or PseudoMaskConfig()
    noteheads = noteheads or []
    rests = rests or []
    stems = stems or []
    beams = beams or []
    barline_xs = barline_xs or []
    system_members = system_members or []
    accidental_map = accidental_map or {}

    mask = np.zeros(ink.shape, dtype=np.uint8)
    interline = _median_interline(staff_bands)

    _paint_staff_lines(mask, staff_bands, config.staff_line_half_width)
    _paint_barlines(mask, ink, staff_bands, barline_xs, system_members, config.barline_half_width)
    _paint_stems(mask, stems, config.stem_half_width)
    _paint_beams(mask, beams)

    if config.include_clef_region:
        _paint_clef_regions(mask, ink, staff_bands, barline_xs, config.clef_region_interlines)

    _paint_notehead_like(mask, rests, SegmentationClass.REST, interline, ink, config)
    if config.include_accidentals:
        _paint_accidentals(mask, noteheads, accidental_map, interline, ink)

    open_heads = [item for item in noteheads if not item.is_filled]
    filled_heads = [item for item in noteheads if item.is_filled]
    _paint_notehead_like(mask, open_heads, SegmentationClass.NOTEHEAD_OPEN, interline, ink, config)
    _paint_notehead_like(
        mask,
        filled_heads,
        SegmentationClass.NOTEHEAD_FILLED,
        interline,
        ink,
        config,
    )

    return mask


def save_pseudo_mask_artifacts(
    result: PseudoMaskResult,
    output_dir: str | Path,
    *,
    stem: str | None = None,
) -> dict[str, str]:
    """Save label-index mask, color overlay, and per-page summary JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or result.image_path.stem

    mask_path = output_dir / f"{stem}.mask.png"
    overlay_path = output_dir / f"{stem}.overlay.png"
    summary_path = output_dir / f"{stem}.summary.json"

    Image.fromarray(result.mask, mode="L").save(mask_path)
    colorize_mask(result.mask).save(overlay_path)
    summary_path.write_text(
        json.dumps(result.to_summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "summary_path": str(summary_path),
    }


def colorize_mask(mask: np.ndarray) -> Image.Image:
    """Convert a label-index mask to an RGB visualization."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D")
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in MASK_PALETTE.items():
        rgb[mask == int(class_id)] = color
    return Image.fromarray(rgb, mode="RGB")


def class_pixel_counts(mask: np.ndarray) -> dict[str, int]:
    """Return non-background class pixel counts keyed by class name."""
    counts: dict[str, int] = {}
    for cls in SegmentationClass:
        count = int((mask == int(cls)).sum())
        if count > 0 and cls is not SegmentationClass.BACKGROUND:
            counts[cls.symbol_name] = count
    return counts


def _paint_staff_lines(mask: np.ndarray, staff_bands: list[StaffBand], half_width: int) -> None:
    for band in staff_bands:
        for y in band.line_ys:
            _paint_rect(
                mask,
                0,
                int(round(y)) - half_width,
                mask.shape[1],
                int(round(y)) + half_width + 1,
                SegmentationClass.STAFF_LINE,
            )


def _paint_barlines(
    mask: np.ndarray,
    ink: np.ndarray,
    staff_bands: list[StaffBand],
    barline_xs: list[float],
    system_members: list[list[int]],
    half_width: int,
) -> None:
    if not staff_bands:
        return
    systems = system_members or [list(range(len(staff_bands)))]
    for x in barline_xs:
        xi = int(round(x))
        for members in systems:
            for staff_idx in members:
                if staff_idx < 0 or staff_idx >= len(staff_bands):
                    continue
                band = staff_bands[staff_idx]
                pad = max(2, int(round(band.interline_px * 0.35)))
                y0 = max(0, band.y_bottom - pad)
                y1 = min(mask.shape[0], band.y_top + pad + 1)
                x0 = max(0, xi - half_width)
                x1 = min(mask.shape[1], xi + half_width + 1)
                if y1 <= y0 or x1 <= x0:
                    continue
                column = (ink[y0:y1, x0:x1] > 0).max(axis=1).astype(np.uint8)
                min_run = max(4, int(round((band.y_top - band.y_bottom) * 0.85)))
                if _longest_one_run(column) < min_run:
                    continue
                _paint_rect(
                    mask,
                    xi - half_width,
                    y0,
                    xi + half_width + 1,
                    y1,
                    SegmentationClass.BARLINE,
                )


def _paint_stems(mask: np.ndarray, stems: list[Any], half_width: int) -> None:
    for stem in stems:
        if isinstance(stem, GlobalStemCandidate):
            x0 = stem.x0
            x1 = stem.x1 + 1
            y0 = stem.y0
            y1 = stem.y1 + 1
        else:
            cx = int(round(float(getattr(stem, "center_x", 0.0))))
            x0 = cx - half_width
            x1 = cx + half_width + 1
            y0 = int(min(getattr(stem, "top_y", 0), getattr(stem, "bottom_y", 0)))
            y1 = int(max(getattr(stem, "top_y", 0), getattr(stem, "bottom_y", 0))) + 1
        _paint_rect(mask, x0, y0, x1, y1, SegmentationClass.STEM)


def _paint_beams(mask: np.ndarray, beams: list[BeamCandidate]) -> None:
    for beam in beams:
        _paint_rect(mask, beam.x0, beam.y0, beam.x1 + 1, beam.y1 + 1, SegmentationClass.BEAM)


def _paint_clef_regions(
    mask: np.ndarray,
    ink: np.ndarray,
    staff_bands: list[StaffBand],
    barline_xs: list[float],
    width_interlines: float,
) -> None:
    for band in staff_bands:
        x_limit = int(round(width_interlines * band.interline_px))
        if barline_xs:
            positive_barlines = [x for x in barline_xs if x > band.interline_px]
            if positive_barlines:
                x_limit = min(x_limit, int(round(min(positive_barlines))))
        y0 = max(0, int(round(band.y_bottom - band.interline_px * 2.0)))
        y1 = min(mask.shape[0], int(round(band.y_top + band.interline_px * 2.0)))
        x1 = min(mask.shape[1], max(1, x_limit))
        region = ink[y0:y1, 0:x1] > 0
        if region.size:
            target = mask[y0:y1, 0:x1]
            target[region] = int(SegmentationClass.CLEF)


def _paint_notehead_like(
    mask: np.ndarray,
    items: list[NoteheadCandidate],
    cls: SegmentationClass,
    interline: float,
    ink: np.ndarray,
    config: PseudoMaskConfig,
) -> None:
    padding = max(1, int(round(interline * config.notehead_padding_interlines)))
    for item in items:
        if cls in {SegmentationClass.NOTEHEAD_FILLED, SegmentationClass.NOTEHEAD_OPEN}:
            if item.staff_step < config.min_notehead_staff_step:
                continue
            if item.staff_step > config.max_notehead_staff_step:
                continue
            if item.confidence < config.min_notehead_confidence:
                continue
        x0, y0, x1, y1 = item.bbox
        x0 -= padding
        y0 -= padding
        x1 += padding + 1
        y1 += padding + 1
        if config.label_notehead_ellipse and cls in {
            SegmentationClass.NOTEHEAD_FILLED,
            SegmentationClass.NOTEHEAD_OPEN,
        }:
            _paint_ellipse(mask, x0, y0, x1, y1, cls)
        else:
            _paint_foreground_bbox(mask, ink, x0, y0, x1, y1, cls)


def _paint_accidentals(
    mask: np.ndarray,
    noteheads: list[NoteheadCandidate],
    accidental_map: dict[int, str],
    interline: float,
    ink: np.ndarray,
) -> None:
    pad_x = int(round(interline * 1.6))
    pad_y = int(round(interline * 1.0))
    for notehead_index in accidental_map:
        if notehead_index < 0 or notehead_index >= len(noteheads):
            continue
        nh = noteheads[notehead_index]
        x0, y0, _x1, y1 = nh.bbox
        _paint_foreground_bbox(
            mask,
            ink,
            int(x0 - pad_x),
            int(y0 - pad_y),
            int(x0),
            int(y1 + pad_y),
            SegmentationClass.ACCIDENTAL,
        )


def _paint_rect(
    mask: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    cls: SegmentationClass,
) -> None:
    h, w = mask.shape
    left = max(0, min(w, x0))
    right = max(0, min(w, x1))
    top = max(0, min(h, y0))
    bottom = max(0, min(h, y1))
    if right <= left or bottom <= top:
        return
    mask[top:bottom, left:right] = int(cls)


def _paint_foreground_bbox(
    mask: np.ndarray,
    ink: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    cls: SegmentationClass,
) -> None:
    h, w = mask.shape
    left = max(0, min(w, x0))
    right = max(0, min(w, x1))
    top = max(0, min(h, y0))
    bottom = max(0, min(h, y1))
    if right <= left or bottom <= top:
        return
    foreground = ink[top:bottom, left:right] > 0
    target = mask[top:bottom, left:right]
    target[foreground] = int(cls)


def _paint_ellipse(
    mask: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    cls: SegmentationClass,
) -> None:
    h, w = mask.shape
    left = max(0, min(w, x0))
    right = max(0, min(w, x1))
    top = max(0, min(h, y0))
    bottom = max(0, min(h, y1))
    if right <= left or bottom <= top:
        return

    yy, xx = np.ogrid[top:bottom, left:right]
    cx = (left + right - 1) / 2.0
    cy = (top + bottom - 1) / 2.0
    rx = max((right - left) / 2.0, 1.0)
    ry = max((bottom - top) / 2.0, 1.0)
    ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    target = mask[top:bottom, left:right]
    target[ellipse] = int(cls)


def _median_interline(staff_bands: list[StaffBand]) -> float:
    if not staff_bands:
        return 12.0
    return max(4.0, float(np.median([band.interline_px for band in staff_bands])))


def _binarize_dark_ink(gray: np.ndarray) -> np.ndarray:
    threshold = int(np.percentile(gray, 25))
    return (gray <= threshold).astype(np.uint8)


def _longest_one_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values.tolist():
        if int(value) > 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
