"""Artifact management for the OMR pipeline.

Provides ArtifactManager for saving/loading intermediate pipeline
results to enable debugging and reproducible evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from notra.layout.annotations import PipelineResult


@dataclass
class ArtifactManager:
    """Manages pipeline artifact storage and retrieval."""

    root_dir: Path

    def save_result(self, result: PipelineResult, *, name: str) -> Path:
        """Save a pipeline result to disk.

        Args:
            result: The pipeline result to save.
            name: A name for this artifact set (e.g., image stem).

        Returns:
            Path to the artifact directory.
        """
        artifact_dir = self.root_dir / name
        artifact_dir.mkdir(parents=True, exist_ok=True)

        if result.musicxml:
            (artifact_dir / "output.musicxml").write_text(
                result.musicxml, encoding="utf-8"
            )

        if result.score is not None:
            from notra.ir.serialize import score_to_json
            (artifact_dir / "output.ir.json").write_text(
                score_to_json(result.score), encoding="utf-8"
            )

        summary = {
            "success": result.success,
            "errors": result.errors,
            "warnings": result.warnings,
            "metrics": result.metrics,
            "annotations": result.page_annotations.to_summary(),
        }
        (artifact_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        return artifact_dir

    def load_summary(self, name: str) -> dict[str, Any]:
        """Load a pipeline summary from disk."""
        path = self.root_dir / name / "summary.json"
        return json.loads(path.read_text(encoding="utf-8"))
