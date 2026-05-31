"""Shared vision-layer dataclasses for segmentation and symbol instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from notra.core.geometry import BBox


class SegmentationClass(IntEnum):
    """Semantic mask labels for the first practical OMR segmentation head."""

    BACKGROUND = 0
    STAFF_LINE = 1
    NOTEHEAD_FILLED = 2
    NOTEHEAD_OPEN = 3
    STEM = 4
    BEAM = 5
    BARLINE = 6
    LEDGER_LINE = 7
    CLEF = 8
    REST = 9
    ACCIDENTAL = 10
    ARTICULATION = 11
    TIE_SLUR = 12
    DYNAMIC_MARK = 13
    TEXT = 14
    FINGERING = 15

    @property
    def symbol_name(self) -> str:
        return self.name.lower()


@dataclass(frozen=True, slots=True)
class SymbolRelation:
    """Candidate relation between two symbol instances."""

    target_symbol_id: str
    relation: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "target_symbol_id": self.target_symbol_id,
            "relation": self.relation,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class SymbolInstance:
    """One extracted visual primitive in staff-relative coordinates."""

    symbol_id: str
    class_name: str
    bbox: BBox
    center_x: float
    center_y: float
    confidence: float = 1.0
    staff_index: int | None = None
    staff_step: float | None = None
    measure_id: str | None = None
    pitch_label: str | None = None
    duration_label: str | None = None
    mask_area: int = 0
    relations: tuple[SymbolRelation, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "symbol_id": self.symbol_id,
            "class": self.class_name,
            "bbox": self.bbox.to_dict(),
            "center": {"x": self.center_x, "y": self.center_y},
            "confidence": self.confidence,
            "mask_area": self.mask_area,
        }
        if self.staff_index is not None:
            payload["staff_index"] = self.staff_index
        if self.staff_step is not None:
            payload["staff_step"] = self.staff_step
        if self.measure_id is not None:
            payload["measure_id"] = self.measure_id
        if self.pitch_label is not None:
            payload["pitch_label"] = self.pitch_label
        if self.duration_label is not None:
            payload["duration_label"] = self.duration_label
        if self.relations:
            payload["relations"] = [relation.to_dict() for relation in self.relations]
        return payload


@dataclass(frozen=True, slots=True)
class SegmentationModelConfig:
    """Architecture-independent config for a future segmentation model."""

    architecture: str = "tiny_unet"
    encoder: str = "lightweight"
    input_channels: int = 1
    class_count: int = len(SegmentationClass)
    crop_height: int = 512
    crop_width: int = 512
