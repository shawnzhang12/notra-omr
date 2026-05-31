"""Class-based recognition entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from notra.layout.annotations import PipelineResult
from notra.pipeline.config import PipelineConfig
from notra.pipeline.stages import STAGE_ORDER, run_full_pipeline


class OMRRecognizer:
    """Small orchestrator around the staged Notra pipeline."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        stages: tuple[tuple[str, Any], ...] | None = None,
    ) -> None:
        self.config = config
        self.stages = stages or STAGE_ORDER

    @classmethod
    def for_image(cls, image_path: str | Path) -> "OMRRecognizer":
        return cls(PipelineConfig.for_image(image_path))

    @classmethod
    def cello(cls) -> "OMRRecognizer":
        return cls(PipelineConfig.cello())

    def recognize(
        self,
        image_path: str | Path,
        *,
        structural: dict[str, Any] | None = None,
    ) -> PipelineResult:
        config = self.config or PipelineConfig.for_image(image_path)
        return run_full_pipeline(
            str(image_path),
            stages=self.stages,
            structural=structural,
            config=config,
        )
