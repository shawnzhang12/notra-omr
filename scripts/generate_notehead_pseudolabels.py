"""Generate notehead pseudo-label manifests, overlays, and review crops.

Example:
  uv run python scripts/generate_notehead_pseudolabels.py \
    --images-root tests/fixtures/images/cello \
    --golden-root tests/fixtures/golden/cello \
    --output-dir artifacts/training/noteheads/cello \
    --profile cello
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from notra.pipeline.config import PipelineConfig
from notra.vision.notehead_pseudolabels import (
    NoteheadPseudoLabelConfig,
    generate_notehead_pseudo_page,
    save_notehead_pseudo_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate notehead pseudo-label artifacts.")
    parser.add_argument(
        "--images-root",
        default="tests/fixtures/images/cello",
        help="Root directory containing rendered page images.",
    )
    parser.add_argument(
        "--golden-root",
        default="tests/fixtures/golden/cello",
        help="Optional MusicXML fixture root used for count-level diagnostics.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/training/noteheads/cello",
        help="Output directory for manifests, overlays, crops, and summary files.",
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
        help="Pipeline profile for notehead pseudo-label generation.",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.82,
        help="Confidence score required to mark a candidate as positive.",
    )
    parser.add_argument(
        "--uncertain-threshold",
        type=float,
        default=0.55,
        help="Confidence score required to keep a candidate as uncertain.",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Do not write color-coded page overlays.",
    )
    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="Do not write candidate crop images.",
    )
    return parser.parse_args()


def _pipeline_config(profile: str, image_path: Path) -> PipelineConfig:
    if profile == "default":
        return PipelineConfig.default()
    if profile == "cello":
        return PipelineConfig.cello()
    return PipelineConfig.for_image(image_path)


def _musicxml_path(golden_root: Path, image_rel: Path) -> Path | None:
    if image_rel.parent == Path("."):
        candidate = golden_root / f"{image_rel.stem}.musicxml"
    else:
        candidate = golden_root / f"{image_rel.parts[0]}.musicxml"
    return candidate if candidate.exists() else None


def main() -> int:
    args = parse_args()
    images_root = Path(args.images_root)
    golden_root = Path(args.golden_root)
    output_dir = Path(args.output_dir)
    config = NoteheadPseudoLabelConfig(
        positive_threshold=float(args.positive_threshold),
        uncertain_threshold=float(args.uncertain_threshold),
        write_overlays=not bool(args.no_overlays),
        write_crops=not bool(args.no_crops),
    )

    image_paths = sorted(path for path in images_root.glob(args.glob) if path.is_file())
    if not image_paths:
        raise RuntimeError(f"no images matched {args.glob!r} under {images_root}")

    records: list[dict[str, Any]] = []
    count_errors_positive: list[int] = []
    count_errors_candidates: list[int] = []

    for image_path in image_paths:
        rel = image_path.relative_to(images_root)
        page_stem = "__".join(rel.with_suffix("").parts)
        page_dir = output_dir / rel.parent
        musicxml_path = _musicxml_path(golden_root, rel)
        page = generate_notehead_pseudo_page(
            image_path,
            musicxml_path=musicxml_path,
            config=config,
            pipeline_config=_pipeline_config(str(args.profile), image_path),
        )
        paths = save_notehead_pseudo_artifacts(page, page_dir, config=config, stem=rel.stem)

        summary = page.to_summary()
        gt_heads = None
        if page.musicxml_counts is not None:
            gt_heads = page.musicxml_counts.pitched_noteheads
            count_errors_positive.append(abs(page.label_counts["positive"] - gt_heads))
            count_errors_candidates.append(abs(len(page.labels) - gt_heads))

        record = {
            "image_path": str(image_path),
            "relative_image_path": str(rel),
            "page_id": page_stem,
            "musicxml_path": str(musicxml_path) if musicxml_path else None,
            "artifact_paths": paths,
            "summary": summary,
        }
        records.append(record)

        status = "ERR" if page.errors else "OK"
        print(
            f"{status} {rel}: "
            f"gt_heads={gt_heads if gt_heads is not None else '?'} "
            f"candidates={len(page.labels)} positive={page.label_counts['positive']} "
            f"uncertain={page.label_counts['uncertain']} reject={page.label_counts['reject']}"
        )

    manifest: dict[str, Any] = {
        "schema_version": "0.1",
        "images_root": str(images_root),
        "golden_root": str(golden_root),
        "output_dir": str(output_dir),
        "positive_threshold": config.positive_threshold,
        "uncertain_threshold": config.uncertain_threshold,
        "record_count": len(records),
        "records": records,
    }
    if count_errors_positive:
        manifest["mean_abs_positive_count_error"] = mean(count_errors_positive)
        manifest["mean_abs_candidate_count_error"] = mean(count_errors_candidates)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[done] wrote {len(records)} notehead records to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
