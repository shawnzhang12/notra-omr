# Milestone 1: Deterministic Notation Compiler Spine

## Objective
Build a deterministic end-to-end pipeline for curated fixtures:

`fixture/image -> IR -> validation -> MusicXML -> render -> diffable artifacts`

Milestone 1 is about correctness, determinism, and debuggability, not model training.

## Working Agreement (Validation-First)
Most steps require maintainer sign-off before continuing.

- Gate A: approve foundation/tooling and command surface.
- Gate B: approve IR v0 schema and serialization shape.
- Gate C: approve validation error model and strictness.
- Gate D: approve MusicXML subset and export conventions.
- Gate E: approve fixture corpus format and first case quality.

No gate is treated as complete until validated by you.

## In Scope
- Package/tooling foundation.
- Canonical IR and JSON serialization.
- Deterministic validation.
- MusicXML export for a constrained subset.
- Rendering path for visual inspection.
- Artifact-first pipeline output.
- Golden tests on curated fixtures.

## Out of Scope
- Neural model training/inference.
- Large-scale dataset ingestion pipelines.
- Benchmark claims.
- Full MusicXML or full MEI coverage.

## Step Plan

### 1. Project Foundation and Tooling
Deliverables:
- `pyproject.toml` with package metadata and tool config.
- `notra` CLI entrypoint with `--help` and subcommand stubs.
- `ruff`, `pytest`, `mypy`, `pre-commit` configured.
- `Makefile` developer commands.

Acceptance:
- `uv run notra --help` works.
- `uv run ruff check .` runs.
- `uv run pytest` runs (even with placeholder tests).

Gate A: maintainer approval required.

### 2. Core Primitives
Deliverables:
- `core/geometry.py`, `core/ids.py`, `core/provenance.py`, `core/errors.py`.
- Baseline types: `BBox`, `Point`, `Provenance`, `ValidationIssue`.

Acceptance:
- Unit tests for type behavior and serialization.

### 3. IR v0
Deliverables:
- Canonical semantic score model (score/part/measure/voice/events).
- Strict duration + pitch representations.

Acceptance:
- Fixtures can be represented without lossy hacks.

Gate B: maintainer approval required.

### 4. IR Serialization
Deliverables:
- Stable JSON encoding/decoding.
- CLI: `notra validate <ir.json>` (schema + semantic validation pipeline hook).

Acceptance:
- Roundtrip tests: `IR -> JSON -> IR`.

### 5. Semantic Validation
Deliverables:
- Duration, measure-balance, pitch, tie, and voice checks.
- Structured error output with stable node ids.

Acceptance:
- Invalid fixtures produce deterministic error reports.

Gate C: maintainer approval required.

### 6. MusicXML Export (Subset)
Deliverables:
- `IR -> MusicXML` for single part/staff core notation.
- CLI: `notra convert <ir.json> --to musicxml --out <file>`.

Acceptance:
- Exported files parse and render consistently.

Gate D: maintainer approval required.

### 7. Rendering + Visual Diff
Deliverables:
- `notra render <musicxml> --out <svg>`.
- Visual diff utility for regression inspection.

Acceptance:
- Golden render comparisons for fixture subset.

### 8. Fixture Corpus v1
Deliverables:
- First curated fixture set with expected IR + MusicXML + SVG.
- Fixture metadata and naming conventions.

Acceptance:
- Pipeline tests consume fixtures deterministically.

Gate E: maintainer approval required.

### 9. Pipeline Artifacts
Deliverables:
- `pipeline run` writes artifact directory with stage outputs.
- Debug metadata/report for each run.

Acceptance:
- Single command run is inspectable without re-executing stages.

### 10. Viewer MVP
Deliverables:
- Load artifact dir, show source/render/IR/errors.

Acceptance:
- Manual debugging loop is practical for fixture failures.

## Definition of Done (Milestone 1)
- 20 curated fixtures pass deterministic pipeline checks.
- IR format is documented and versioned.
- Validation errors are structured, stable, and test-covered.
- MusicXML subset export is reliable and regression-tested.
- Artifacts + rendering provide practical debugging workflow.

## Execution Status
- [x] Step 1 complete (Gate A approved)
- [x] Step 2 complete
- [x] Step 3 complete (pending Gate B)
- [x] Step 4 complete
- [ ] Step 5 complete (pending Gate C)
- [ ] Step 6 complete (pending Gate D)
- [ ] Step 7 complete
- [ ] Step 8 complete (pending Gate E)
- [ ] Step 9 complete
- [ ] Step 10 complete
