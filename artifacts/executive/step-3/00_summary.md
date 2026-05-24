# Step 3 Summary: IR v0 + Real Note Outputs

## Status

`IMPLEMENTED - READY FOR GATE B VALIDATION`

## What You Asked For (Now Present)

- Real note-level IR with durations and pitches.
- Real MusicXML output generated from IR.
- Real validation report output for both valid and invalid measures.

## Executive Relevance to OMR End Goal

This is the first deterministic symbolic spine that future recognizers must target:

- recognizer output -> Notra IR
- Notra IR -> semantic validation
- Notra IR -> MusicXML export

Without this spine, model accuracy is not actionable.

## Fast Validation Path

1. Open `examples/01_valid_four_quarters.ir.json`.
2. Open `examples/03_valid_four_quarters.musicxml`.
3. Open `examples/02_valid_four_quarters.validation.json` (should show zero errors).
4. Open `examples/05_invalid_underfill.validation.json` (should show deterministic underflow error).
5. Open `10_pipeline_value_visual.svg`.

## Core Files Added

- `notra/ir/*.py` for score semantics and serialization.
- `notra/ir/validate.py` for measure-duration checks.
- `notra/exporters/musicxml.py` for deterministic export.
- `notra/cli/validate.py` and `notra/cli/convert.py` now execute real work.
