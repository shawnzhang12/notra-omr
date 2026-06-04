"""Notehead pseudo-label artifact generation.

The goal is to bootstrap training data without pretending deterministic OMR
output is ground truth.  Each candidate keeps source, confidence, staff-relative
position, and a triage label so high-confidence positives can train a small
segmentation model while uncertain crops remain available for review.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from notra.layout.symbol import NoteheadCandidate
from notra.pipeline import stages
from notra.pipeline.config import PipelineConfig


@dataclass(frozen=True, slots=True)
class NoteheadPseudoLabelConfig:
    """Controls notehead pseudo-label triage and artifacts."""

    positive_threshold: float = 0.82
    uncertain_threshold: float = 0.55
    crop_padding_interlines: float = 0.75
    write_overlays: bool = True
    write_crops: bool = True


@dataclass(frozen=True, slots=True)
class NoteheadPseudoLabel:
    """One notehead candidate exported for pseudo-label training or review."""

    index: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    area: float
    is_filled: bool
    staff_step: float
    staff_band_index: int
    source: str
    confidence: float
    label: str

    @classmethod
    def from_candidate(
        cls,
        index: int,
        candidate: NoteheadCandidate,
        config: NoteheadPseudoLabelConfig,
    ) -> "NoteheadPseudoLabel":
        if candidate.confidence >= config.positive_threshold:
            label = "positive"
        elif candidate.confidence >= config.uncertain_threshold:
            label = "uncertain"
        else:
            label = "reject"

        return cls(
            index=index,
            bbox=candidate.bbox,
            center=(candidate.cx, candidate.cy),
            area=candidate.area,
            is_filled=candidate.is_filled,
            staff_step=candidate.staff_step,
            staff_band_index=candidate.staff_band_index,
            source=candidate.source,
            confidence=candidate.confidence,
            label=label,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "bbox": list(self.bbox),
            "center": {"x": self.center[0], "y": self.center[1]},
            "area": self.area,
            "is_filled": self.is_filled,
            "staff_step": self.staff_step,
            "staff_band_index": self.staff_band_index,
            "source": self.source,
            "confidence": self.confidence,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class MusicXMLNoteheadCounts:
    """Ground-truth-ish symbol counts derivable from fixture MusicXML."""

    pitched_noteheads: int
    rests: int

    @property
    def total_events(self) -> int:
        return self.pitched_noteheads + self.rests

    def to_dict(self) -> dict[str, int]:
        return {
            "pitched_noteheads": self.pitched_noteheads,
            "rests": self.rests,
            "total_events": self.total_events,
        }


@dataclass(frozen=True, slots=True)
class NoteheadPseudoPage:
    """Pseudo-label result for one rendered page."""

    image_path: Path
    labels: tuple[NoteheadPseudoLabel, ...]
    interline_px: float
    staff_count: int
    system_count: int
    musicxml_counts: MusicXMLNoteheadCounts | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def label_counts(self) -> dict[str, int]:
        counts = {"positive": 0, "uncertain": 0, "reject": 0}
        for label in self.labels:
            counts[label.label] = counts.get(label.label, 0) + 1
        return counts

    def to_summary(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "image_path": str(self.image_path),
            "detected_notehead_candidates": len(self.labels),
            "label_counts": self.label_counts,
            "interline_px": self.interline_px,
            "staff_count": self.staff_count,
            "system_count": self.system_count,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }
        if self.musicxml_counts is not None:
            gt_heads = self.musicxml_counts.pitched_noteheads
            payload["musicxml_counts"] = self.musicxml_counts.to_dict()
            payload["positive_count_error"] = self.label_counts["positive"] - gt_heads
            payload["candidate_count_error"] = len(self.labels) - gt_heads
        return payload


def generate_notehead_pseudo_page(
    image_path: str | Path,
    *,
    musicxml_path: str | Path | None = None,
    config: NoteheadPseudoLabelConfig | None = None,
    pipeline_config: PipelineConfig | None = None,
) -> NoteheadPseudoPage:
    """Run deterministic layout/notehead stages and classify candidates."""
    image_path = Path(image_path)
    config = config or NoteheadPseudoLabelConfig()
    pipeline_config = pipeline_config or PipelineConfig.for_image(image_path)

    ctx: dict[str, Any] = {
        "image_path": str(image_path),
        "errors": [],
        "warnings": [],
        "metrics": {},
    }
    ctx.update(pipeline_config.to_context())

    for stage_fn in (
        stages.load_image_stage,
        stages.detect_layout_stage,
        stages.detect_clefs_stage,
        stages.detect_noteheads_stage,
    ):
        try:
            stage_fn(ctx)
        except Exception as exc:  # pragma: no cover - artifact path
            ctx.setdefault("errors", []).append(f"{stage_fn.__name__}: {exc}")

    candidates = ctx.get("notehead_candidates", [])
    labels = tuple(
        NoteheadPseudoLabel.from_candidate(idx, candidate, config)
        for idx, candidate in enumerate(candidates)
    )
    counts = _read_musicxml_notehead_counts(musicxml_path) if musicxml_path else None

    return NoteheadPseudoPage(
        image_path=image_path,
        labels=labels,
        interline_px=float(ctx.get("interline_px", 0.0) or 0.0),
        staff_count=len(ctx.get("staff_bands", [])),
        system_count=len(ctx.get("system_members", [])),
        musicxml_counts=counts,
        errors=tuple(str(item) for item in ctx.get("errors", [])),
        warnings=tuple(str(item) for item in ctx.get("warnings", [])),
    )


def save_notehead_pseudo_artifacts(
    page: NoteheadPseudoPage,
    output_dir: str | Path,
    *,
    config: NoteheadPseudoLabelConfig | None = None,
    stem: str | None = None,
) -> dict[str, str]:
    """Write summary, candidate manifest, overlay, and optional crops."""
    config = config or NoteheadPseudoLabelConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or page.image_path.stem

    summary_path = output_dir / f"{stem}.summary.json"
    candidates_path = output_dir / f"{stem}.candidates.jsonl"
    summary_path.write_text(
        json.dumps(page.to_summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    candidates_path.write_text(
        "".join(json.dumps(label.to_dict(), sort_keys=True) + "\n" for label in page.labels),
        encoding="utf-8",
    )

    paths = {
        "summary_path": str(summary_path),
        "candidates_path": str(candidates_path),
    }
    if config.write_overlays:
        overlay_path = output_dir / f"{stem}.noteheads.overlay.png"
        _save_overlay(page, overlay_path)
        paths["overlay_path"] = str(overlay_path)
    if config.write_crops:
        crops_dir = output_dir / f"{stem}.crops"
        _save_crops(page, crops_dir, config)
        paths["crops_dir"] = str(crops_dir)
    return paths


def _read_musicxml_notehead_counts(path: str | Path | None) -> MusicXMLNoteheadCounts | None:
    if path is None:
        return None
    xml_path = Path(path)
    if not xml_path.exists():
        return None

    root = ET.parse(xml_path).getroot()
    pitched = 0
    rests = 0
    for note in root.findall(".//note"):
        if note.find("pitch") is not None:
            pitched += 1
        elif note.find("rest") is not None:
            rests += 1
    return MusicXMLNoteheadCounts(pitched_noteheads=pitched, rests=rests)


def _save_overlay(page: NoteheadPseudoPage, output_path: Path) -> None:
    image = Image.open(page.image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {
        "positive": (28, 168, 76),
        "uncertain": (235, 168, 24),
        "reject": (215, 48, 39),
    }
    for label in page.labels:
        x0, y0, x1, y1 = label.bbox
        color = colors.get(label.label, (64, 64, 64))
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _save_crops(
    page: NoteheadPseudoPage,
    crops_dir: Path,
    config: NoteheadPseudoLabelConfig,
) -> None:
    image = Image.open(page.image_path).convert("RGB")
    width, height = image.size
    pad = max(2, int(round(page.interline_px * config.crop_padding_interlines)))
    crops_dir.mkdir(parents=True, exist_ok=True)

    for label in page.labels:
        x0, y0, x1, y1 = label.bbox
        left = max(0, x0 - pad)
        top = max(0, y0 - pad)
        right = min(width, x1 + pad + 1)
        bottom = min(height, y1 + pad + 1)
        crop = image.crop((left, top, right, bottom))
        crop_name = (
            f"{page.image_path.stem}_{label.index:04d}_{label.label}_{label.confidence:.2f}.png"
        )
        crop.save(crops_dir / crop_name)
