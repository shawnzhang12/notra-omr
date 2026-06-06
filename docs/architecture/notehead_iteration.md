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

## Dynamic Inference Policy

The deployable loop should not use one global threshold forever.

1. First pass:
   use conservative detection and a learned threshold to get high-confidence
   noteheads.
2. Build local symbolic hypotheses:
   staff position, pitch, stem attachment, duration candidates, measure
   assignment, and measure-duration validity.
3. Detect violations:
   impossible measure duration, missing stem relation, illegal spacing, too few
   events for a measure, or too many events for a measure.
4. Second pass:
   run relaxed grayscale/line-position proposals only in suspicious regions or
   as a full-page candidate pool during early development.
5. Dynamic selection:
   keep the best candidate set that satisfies the symbolic constraint. In the
   current training script, MusicXML notehead count is used as a stand-in
   target. Later this target must come from duration/measure decoding, not from
   ground truth.
6. Repeat until constraints stabilize or the page is marked for human review.

Thresholding can only select from candidates that exist. If candidate recall is
below the target, no threshold policy can reach 100%; the detector must run a
broader proposal pass.

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

Train/evaluate the threshold policy on an 80/20 cello split:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train_notehead_threshold_policy.py \
  --images-root tests/fixtures/images/cello \
  --golden-root tests/fixtures/golden/cello \
  --output-dir artifacts/training/noteheads/policy \
  --profile cello \
  --include-relaxed-rescue
```

Current result with hash split and relaxed rescue:

- learned threshold: `0.890`
- fixed threshold train: `1/14` exact, MAE `10.429`
- fixed threshold validation: `0/4` exact, MAE `6.000`
- oracle target-count upper bound train: `14/14` exact, MAE `0.000`
- oracle target-count upper bound validation: `4/4` exact, MAE `0.000`
- leak-free measure solver train: `1/14` exact pages, MAE `14.571`,
  valid measures `203/203`
- leak-free measure solver validation: `0/4` exact pages, MAE `13.250`,
  valid measures `87/87`

That proves the relaxed candidate pool has enough recall for count-level
oracle selection on these fixtures. The leak-free solver is the real inference
metric. Measure-duration validity is now solved for this split, but full OMR is
not: many measures can be duration-valid while still selecting the wrong number
of visual noteheads.

Show measure-level overlays for the validation split:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/debug_leak_free_measures.py \
  --policy artifacts/training/noteheads/policy/policy.json \
  --split validation \
  --include-relaxed-rescue \
  --output-dir artifacts/debug/noteheads/leak_free_measures
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
