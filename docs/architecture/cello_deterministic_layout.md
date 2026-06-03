# Cello Deterministic Layout Reference

This document captures the parts of the cello pipeline that currently validate
well against the 18 rendered cello fixtures.  It is a reference point for
expanding the system without blurring proven deterministic stages with weaker
symbol-recognition stages.

Current validated strengths:

- Bass clef detection: 18/18 cello fixtures.
- Time signature detection: 18/18 cello fixtures.
- Staff/system/measure-count layout: 18/18 cello fixtures, 290/290 measures.

Not yet in the "works great" set:

- Notehead precision/recall.
- Stem, beam, accidental, rest, duration, voice, and full MusicXML semantic
  reconstruction.

## Deterministic Cello Layout Flow

```mermaid
flowchart TD
    A["Cello page image<br/>tests/fixtures/images/cello/*/page-001.png"] --> B["load_image_stage"]
    B --> C["Gray image<br/>ctx['gray']"]
    C --> D["Sauvola binarization<br/>ctx['ink']"]

    C --> E["Projection staff-line detector<br/>detect_staff_lines + group_staff_bands"]
    C --> F["Rendered staff-line detector<br/>detect_staff_bands_from_horizontal_runs"]
    E --> G{"profile_name == cello?"}
    F --> G
    G -->|"use better rendered bands"| H["StaffBand list<br/>5 line ys + interline"]

    H --> I["_detect_system_members"]
    I --> J["System membership<br/>one staff per cello system"]

    H --> K["estimate_staff_x_extent"]
    D --> L["detect_measure_barlines"]
    C --> L
    K --> L
    L --> M["Staff-local barline candidates"]

    M --> N["Candidate filters<br/>vertical run + 4 staff-space hits"]
    N --> O["Reject stem-like lines<br/>width + side ink after staff-line erase"]
    O --> P["Merge repeat/double bars<br/>dedupe close x positions"]
    P --> Q["Suppress start-repeat bars<br/>near staff-left opening"]
    Q --> R["Weak-candidate spacing prune"]
    R --> S["barline_by_system"]

    H --> T["detect_clef_region<br/>or force cello F4"]
    H --> U["detect_time_signature<br/>opening glyph classifier"]
    S --> V["assemble_measures_stage"]
    K --> V
    V --> W["MeasureBoundary spans<br/>staff-left to detected right barlines"]

    S --> X["Layout evaluator<br/>scripts/eval_cello_layout_metrics.py"]
    X --> Y["18/18 exact<br/>290 predicted / 290 golden measures"]
```

## Barline Candidate Logic

```mermaid
flowchart TD
    A["One StaffBand"] --> B["Find staff x extent<br/>median long-run span over 5 staff lines"]
    B --> C["Scan x columns inside staff extent"]
    C --> D{"Column has enough vertical ink?"}
    D -->|"no"| C
    D -->|"yes"| E{"Longest vertical run crosses staff?"}
    E -->|"no"| C
    E -->|"yes"| F{"Hits all 4 interline spaces?"}
    F -->|"no"| C
    F -->|"yes"| G["Cluster nearby candidate columns"]

    G --> H["MeasureBarlineCandidate<br/>x, width_px, side_ink_ratio, relative_x"]
    H --> I{"Repeat/double group?"}
    I -->|"yes"| J["Accept rightmost group x"]
    I -->|"no"| K{"Regular-width barline?<br/>width >= 4 and side_ink_ratio <= 0.25"}
    K -->|"yes"| L["Accept strong candidate"]
    K -->|"no"| M{"Narrow clean candidate?<br/>width >= 3, side_ink_ratio <= 0.05,<br/>relative_x >= 0.28"}
    M -->|"yes"| N["Accept weak candidate"]
    M -->|"no"| O["Reject as stem/noise"]

    J --> P["Candidate list"]
    L --> P
    N --> P
    P --> Q["Drop system-opening repeat bars<br/>x - staff_left <= 7 interlines"]
    Q --> R["Remove only weak candidates<br/>that create implausibly short spacing"]
    R --> S["Final measure-ending barlines"]
```

## Validated Metrics

```mermaid
flowchart LR
    A["18 cello fixtures"] --> B["Golden MusicXML"]
    A --> C["Rendered PNG pages"]
    B --> D["Ground-truth measure count"]
    C --> E["Deterministic layout"]
    E --> F["Predicted per-system barline count"]
    D --> G["eval_cello_layout_metrics.py"]
    F --> G
    G --> H["Exact: 18/18"]
    G --> I["MAE: 0.000"]
    G --> J["Total: 290/290 measures"]
```

## Expansion Rules

When adding segmentation or learned detectors, keep this deterministic spine as
the baseline and comparison target.

1. New models may propose candidates, but candidates must still become explicit
   staff-relative hypotheses.
2. Staff, system, and measure geometry should remain inspectable before symbol
   semantics run.
3. Do not let noteheads, stems, beams, rests, or duration decoding overwrite
   the validated layout artifacts without a metric-backed improvement.
4. Every expanded stage should have an evaluator equivalent to
   `scripts/eval_cello_layout_metrics.py`.

Reference commands:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/eval_cello_layout_metrics.py
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/pipeline/test_cello_measure_barlines.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/eval_cello_stage_metrics.py
```
