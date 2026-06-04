# Notehead Iteration Loop

This is the practical path for improving cello noteheads without training on
garbage labels.

## Current Baseline

The cello layout spine is deterministic and stable enough to be the coordinate
frame for notehead work:

1. page image
2. staff bands
3. systems
4. staff-local measure barlines
5. clef and time-signature context
6. notehead candidates

The notehead detector is still noisy. Count-only metrics currently show high
recall and poor precision, so pseudo-labeling every candidate as ground truth
would train the model to reproduce false positives.

## Method

1. Detect candidates with deterministic passes.
2. Attach provenance to every candidate:
   `connected_component`, `grayscale_darkness`, `line_position`, or split source.
3. Attach a deterministic confidence score from staff-relative geometry:
   bbox size, area, aspect ratio, staff-step alignment, and source prior.
4. Export three bins:
   `positive` for high-confidence pseudo-labels,
   `uncertain` for review/model-assisted relabeling,
   `reject` for likely false positives.
5. Train segmentation only on strict positives first.
6. Use model predictions to propose more candidates, not to overwrite the
   deterministic symbolic spine.
7. Validate against MusicXML counts, overlays, crops, and eventually rendered
   symbolic diffs.

## Commands

Generate notehead review artifacts:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/generate_notehead_pseudolabels.py \
  --images-root tests/fixtures/images/cello \
  --golden-root tests/fixtures/golden/cello \
  --output-dir artifacts/training/noteheads/cello \
  --profile cello
```

Generate stricter semantic segmentation masks for tiny U-Net bootstrapping:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/generate_segmentation_pseudolabels.py \
  --images-root tests/fixtures/images/cello \
  --output-dir artifacts/training/pseudolabels/cello \
  --profile cello \
  --min-notehead-confidence 0.82
```

## Confidence

High confidence:

- Staff/system/measure geometry is the right frame for cello noteheads.
- Pseudo-labels need candidate confidence and review bins before training.
- Strict positives are safer than high-recall noisy masks for a first U-Net.

Moderate confidence:

- The current heuristic confidence score is useful for triage.
- Per-source thresholds will improve after inspecting overlays and crops.

Low confidence:

- A segmentation model trained on current labels will improve end-to-end OMR
  without additional cleanup. That needs evidence from held-out fixtures.
