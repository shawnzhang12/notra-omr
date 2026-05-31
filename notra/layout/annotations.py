"""Pipeline annotation dataclasses shared across OMR stages and artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from notra.core.geometry import BBox
from notra.ir.score import Score
from notra.layout.staff import StaffBand


@dataclass(frozen=True, slots=True)
class StaffAnnotation:
    """Detected staff-level context used by later symbolic stages."""

    staff_index: int
    band: StaffBand
    clef_sign: str = "G"
    clef_line: int = 2
    key_fifths: int = 0

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "staff_index": self.staff_index,
            "line_ys": list(self.band.line_ys),
            "interline_px": self.band.interline_px,
            "clef_sign": self.clef_sign,
            "clef_line": self.clef_line,
            "key_fifths": self.key_fifths,
        }


@dataclass(frozen=True, slots=True)
class NoteEventAnnotation:
    """One recognized symbol-level event projected into staff semantics."""

    event_index: int
    staff_index: int
    staff_step: float
    diatonic_step: str
    octave: int
    alter: int
    duration_num: int
    duration_den: int
    is_rest: bool = False
    is_chord: bool = False
    voice: int = 1
    cx: float = 0.0
    cy: float = 0.0
    bbox: BBox | None = None

    def __post_init__(self) -> None:
        if self.duration_num <= 0:
            raise ValueError("duration_num must be > 0")
        if self.duration_den <= 0:
            raise ValueError("duration_den must be > 0")
        if self.voice <= 0:
            raise ValueError("voice must be >= 1")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dictionary."""
        payload: dict[str, object] = {
            "event_index": self.event_index,
            "staff_index": self.staff_index,
            "staff_step": self.staff_step,
            "diatonic_step": self.diatonic_step,
            "octave": self.octave,
            "alter": self.alter,
            "duration_num": self.duration_num,
            "duration_den": self.duration_den,
            "is_rest": self.is_rest,
            "is_chord": self.is_chord,
            "voice": self.voice,
            "cx": self.cx,
            "cy": self.cy,
        }
        if self.bbox is not None:
            payload["bbox"] = self.bbox.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class MeasureBoundary:
    """Horizontal measure span for one staff/system context."""

    measure_number: int
    x_start: float
    x_end: float
    staff_index: int
    system_index: int = 0
    barline_style: str = "regular"

    def __post_init__(self) -> None:
        if self.measure_number <= 0:
            raise ValueError("measure_number must be >= 1")
        if self.x_end <= self.x_start:
            raise ValueError("x_end must be greater than x_start")
        if self.staff_index < 0:
            raise ValueError("staff_index must be >= 0")
        if self.system_index < 0:
            raise ValueError("system_index must be >= 0")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "measure_number": self.measure_number,
            "x_start": self.x_start,
            "x_end": self.x_end,
            "staff_index": self.staff_index,
            "system_index": self.system_index,
            "barline_style": self.barline_style,
        }


@dataclass(slots=True)
class PageAnnotations:
    """Page-level debug artifact for layout and note-level hypotheses."""

    image_width: int
    image_height: int
    staff_annotations: list[StaffAnnotation] = field(default_factory=list)
    note_events: list[NoteEventAnnotation] = field(default_factory=list)
    measure_boundaries: list[MeasureBoundary] = field(default_factory=list)
    barline_xs: list[float] = field(default_factory=list)
    interline_px: float = 12.0

    @property
    def staff_count(self) -> int:
        """Detected staff-band count."""
        return len(self.staff_annotations)

    @property
    def staff_band_count(self) -> int:
        """Alias for compatibility with evaluation scripts."""
        return self.staff_count

    @property
    def note_count(self) -> int:
        """Detected note/rest event count after semantic assignment."""
        return len(self.note_events)

    @property
    def measure_count(self) -> int:
        """Count of detected measure spans."""
        return len(self.measure_boundaries)

    def to_summary(self) -> dict[str, Any]:
        """Return compact scalar stats for logs/artifacts."""
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "staff_count": self.staff_count,
            "note_count": self.note_count,
            "measure_count": self.measure_count,
            "barline_count": len(self.barline_xs),
            "interline_px": self.interline_px,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly nested dictionary."""
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "staff_annotations": [item.to_dict() for item in self.staff_annotations],
            "note_events": [item.to_dict() for item in self.note_events],
            "measure_boundaries": [item.to_dict() for item in self.measure_boundaries],
            "barline_xs": list(self.barline_xs),
            "interline_px": self.interline_px,
        }


@dataclass(slots=True)
class PipelineResult:
    """Structured return object for a full pipeline run."""

    page_annotations: PageAnnotations
    score: Score | None
    musicxml: str
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True when no stage errors were recorded."""
        return not self.errors
