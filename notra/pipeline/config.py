"""Typed configuration for Notra recognition profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ClefDetectionConfig:
    """Controls deterministic clef handling."""

    force_bass_clef: bool | None = None


@dataclass(frozen=True, slots=True)
class NoteheadDetectionConfig:
    """Controls notehead candidate extraction and conservative rescue passes."""

    use_grayscale_fallback: bool = False
    use_line_position_pass: bool = False
    low_density_grayscale_rescue: bool = True
    low_density_threshold_per_staff: float = 22.0
    grayscale_rescue_growth_cap: float = 1.5


@dataclass(frozen=True, slots=True)
class LayoutDetectionConfig:
    """Controls deterministic page/staff layout behavior."""

    upscale_factor: int = 0


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Recognition profile translated into the legacy stage context.

    The stage functions still use a mutable context dict. This config is the
    typed boundary so profile choices stay explicit and testable.
    """

    profile_name: str = "default"
    layout: LayoutDetectionConfig = field(default_factory=LayoutDetectionConfig)
    clef: ClefDetectionConfig = field(default_factory=ClefDetectionConfig)
    noteheads: NoteheadDetectionConfig = field(default_factory=NoteheadDetectionConfig)

    @classmethod
    def default(cls) -> "PipelineConfig":
        return cls()

    @classmethod
    def cello(cls) -> "PipelineConfig":
        return cls(
            profile_name="cello",
            clef=ClefDetectionConfig(force_bass_clef=True),
            noteheads=NoteheadDetectionConfig(
                use_grayscale_fallback=False,
                use_line_position_pass=False,
                low_density_grayscale_rescue=True,
                low_density_threshold_per_staff=22.0,
                grayscale_rescue_growth_cap=1.5,
            ),
        )

    @classmethod
    def for_image(cls, image_path: str | Path) -> "PipelineConfig":
        path_text = str(image_path).lower()
        if "/cello/" in path_text or path_text.endswith("cello"):
            return cls.cello()
        return cls.default()

    def to_context(self) -> dict[str, Any]:
        ctx: dict[str, Any] = {
            "profile_name": self.profile_name,
            "upscale_factor": self.layout.upscale_factor,
            "use_grayscale_notehead_fallback": self.noteheads.use_grayscale_fallback,
            "use_line_position_noteheads": self.noteheads.use_line_position_pass,
            "low_density_grayscale_rescue": self.noteheads.low_density_grayscale_rescue,
            "cello_low_density_threshold": self.noteheads.low_density_threshold_per_staff,
            "cello_gray_growth_cap": self.noteheads.grayscale_rescue_growth_cap,
        }
        if self.clef.force_bass_clef is not None:
            ctx["force_bass_clef"] = self.clef.force_bass_clef
        return ctx
