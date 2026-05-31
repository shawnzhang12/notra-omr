"""Pipeline orchestrator for notra OMR.

Provides the `run_pipeline` entry point that drives the full
recognition pipeline and returns structured results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from notra.pipeline.config import PipelineConfig
from notra.pipeline.recognizer import OMRRecognizer


def recognize_image(
    image_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    save_musicxml: bool = True,
    save_ir_json: bool = False,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    """Run the full OMR pipeline on a single image.

    This is the primary public entry point for notra OMR recognition.

    Args:
        image_path: Path to the input image.
        output_dir: Optional directory to save outputs. If None, no files
            are written beyond what save_musicxml/save_ir_json specify.
        save_musicxml: If True and output_dir is set, write the MusicXML result.
        save_ir_json: If True and output_dir is set, write the IR JSON.

    Returns:
        Dict with keys: 'success', 'musicxml', 'score', 'annotations',
        'errors', 'warnings', 'metrics'.
    """
    image_path = Path(image_path)
    recognizer = OMRRecognizer(config or PipelineConfig.for_image(image_path))
    result = recognizer.recognize(image_path)

    output: dict[str, Any] = {
        "success": result.success,
        "musicxml": result.musicxml,
        "score": result.score,
        "annotations": result.page_annotations,
        "errors": result.errors,
        "warnings": result.warnings,
        "metrics": result.metrics,
    }

    if output_dir is not None and result.success:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if save_musicxml and result.musicxml:
            musicxml_path = out / f"{image_path.stem}.musicxml"
            musicxml_path.write_text(result.musicxml, encoding="utf-8")

        if save_ir_json and result.score is not None:
            from notra.ir.serialize import score_to_json
            ir_path = out / f"{image_path.stem}.ir.json"
            ir_path.write_text(score_to_json(result.score), encoding="utf-8")

    return output


def recognize_batch(
    image_paths: list[str | Path],
    *,
    output_dir: str | Path | None = None,
    config: PipelineConfig | None = None,
) -> list[dict[str, Any]]:
    """Run the pipeline on multiple images, returning results in order.

    Args:
        image_paths: List of image paths.
        output_dir: Optional base output directory.

    Returns:
        List of result dicts, one per image.
    """
    results: list[dict[str, Any]] = []
    for img_path in image_paths:
        result = recognize_image(img_path, output_dir=output_dir, config=config)
        results.append(result)
    return results
