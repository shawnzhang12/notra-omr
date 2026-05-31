# Notra OMR Recovery Notes

This file preserves the useful ideas from the dirty tree without keeping the
whole experimental sprawl in the runtime package.

## Highest-Value Ideas To Keep

1. Deterministic layout spine first.
   Page normalization, staff lines, systems, barlines, measure spans, and
   staff-relative coordinates should be stable before model output is trusted.

2. Dense primitive segmentation before general object detection.
   Staff lines, noteheads, stems, beams, ledger lines, ties/slurs, and barlines
   are dense overlapping primitives. A small U-Net-style semantic segmentation
   model is a better first model than YOLO for these classes.

3. Detection should over-generate.
   The visual layer should emit multiple hypotheses per ambiguous blob:
   filled notehead, open notehead, rest fragment, dot, false positive, and
   possible duration candidates. The decoder should choose.

4. Rhythm is constrained decoding, not local classification.
   A measure-level dynamic program should select note/rest durations that fill
   the active time signature. False positives must be legal skip transitions.

5. Stems and beams are global primitives.
   Detect stems page-wide as vertical runs, then attach them to noteheads.
   Detect beams as horizontal dark rectangles connecting stem endpoints. Do not
   search only inside each notehead bbox.

6. Cello validation should stay small and reproducible.
   The current practical validation loop is the 18 cello MusicXML files plus
   corresponding page screenshots. That is the first supervised overfit target.

7. Heavy retrieval experiments are not core OMR.
   Hash-KNN, timm embeddings, screenshot-to-fixture retrieval, and large
   classifier sweeps are useful negative evidence: they prove retrieval-style
   pipelines do not solve symbolic OMR. Keep the lesson, not the artifact pile.

## Dirty Tree Decisions

Kept in core:
- Typed pipeline profiles and recognizer orchestration.
- Staff geometry and staff-relative coordinate mapping.
- Global stem and beam detector concepts.
- Measure-constrained rhythm solver concepts.
- Vision schema, mask instance extraction, and optional tiny U-Net factory.
- Cello MusicXML and screenshot fixtures for local validation.

Preserved as ideas only:
- Large-model/timm/ConvNeXt retrieval baselines.
- Structural classifier training sweeps.
- Advanced fixture corpus generation.
- Viewer fixture navigation changes.
- One-off debug scripts for individual pieces.
- Executive milestone artifact dumps.

Rejected for core right now:
- Diffusion models for recognition.
- VLMs as primary recognizers.
- Monolithic screenshot-to-MusicXML classifiers.
- Generic object detection as the first notehead/stem strategy.

## Model Stack Decision

Short term:
- Use deterministic layout plus `tiny_unet` semantic segmentation masks.
- Use connected components over masks to create `SymbolInstance` objects.
- Use grammar/measure validation to reject bad interpretations.

Medium term:
- Add a YOLO/RT-DETR-style detector only for sparse symbols such as clefs,
  rests, dynamics, articulations, text blocks, and rehearsal marks.
- Add a relation scorer for notehead-stem-beam-dot-accidental links.
- Add active-learning export of failed crops and correction labels.

Long term:
- Evaluate SegFormer-B0 after mask labels, metrics, and crop generation are
  stable. It should be an optional model backend, not a core dependency.

## Accuracy Standard

The target is not "looks plausible." The target is validated symbolic recovery:

1. staff/system/measure geometry correct,
2. notehead/stem/beam/rest/dot primitives detected with high recall,
3. symbol relations assembled into a graph,
4. pitch and duration decoded under clef/key/time constraints,
5. measures validate exactly,
6. MusicXML exports render close to ground truth,
7. every failure leaves artifacts that localize the cause.

Confidence: high for the architecture, moderate for the exact model choice until
segmentation masks and train/test metrics exist.
