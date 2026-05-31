"""Generate weak semantic segmentation labels for OMR pages.

Example:
  uv run python scripts/generate_segmentation_pseudolabels.py \
    --images-root tests/fixtures/images/cello \
    --output-dir artifacts/training/pseudolabels/cello
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from notra.pipeline.config import PipelineConfig
from notra.vision.pseudolabels import (
    generate_pseudo_segmentation_mask,
    save_pseudo_mask_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OMR segmentation pseudo-labels.")
    parser.add_argument(
        "--images-root",
        default="tests/fixtures/images/cello",
        help="Root directory containing page images.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/training/pseudolabels/cello",
        help="Output directory for masks, overlays, summaries, and manifest.",
    )
    parser.add_argument(
        "--glob",
        default="**/*.png",
        help="Image glob relative to images-root.",
    )
    parser.add_argument(
        "--profile",
        default="auto",
        choices=["auto", "default", "cello"],
        help="Pipeline profile for pseudo-label generation.",
    )
    return parser.parse_args()


def _pipeline_config(profile: str, image_path: Path) -> PipelineConfig:
    if profile == "default":
        return PipelineConfig.default()
    if profile == "cello":
        return PipelineConfig.cello()
    return PipelineConfig.for_image(image_path)


def main() -> int:
    args = parse_args()
    images_root = Path(args.images_root)
    output_dir = Path(args.output_dir)

    image_paths = sorted(path for path in images_root.glob(args.glob) if path.is_file())
    if not image_paths:
        raise RuntimeError(f"no images matched {args.glob!r} under {images_root}")

    records: list[dict[str, Any]] = []
    for image_path in image_paths:
        rel = image_path.relative_to(images_root)
        page_stem = "__".join(rel.with_suffix("").parts)
        page_dir = output_dir / rel.parent

        result = generate_pseudo_segmentation_mask(
            image_path,
            pipeline_config=_pipeline_config(str(args.profile), image_path),
        )
        paths = save_pseudo_mask_artifacts(result, page_dir, stem=rel.stem)

        record: dict[str, Any] = {
            "image_path": str(image_path),
            "relative_image_path": str(rel),
            "page_id": page_stem,
            "mask_path": paths["mask_path"],
            "overlay_path": paths["overlay_path"],
            "summary_path": paths["summary_path"],
            "class_pixel_counts": result.class_pixel_counts,
            "symbol_counts": result.symbol_counts,
            "errors": list(result.errors),
            "warnings": list(result.warnings),
        }
        records.append(record)
        status = "ERR" if result.errors else "OK"
        print(
            f"{status} {rel}: "
            f"classes={len(result.class_pixel_counts)} symbols={sum(result.symbol_counts.values())}"
        )

    manifest = {
        "schema_version": "0.1",
        "images_root": str(images_root),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[done] wrote {len(records)} pseudo-label records to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
