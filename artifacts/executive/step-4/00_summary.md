# Step 4 Summary: Expanded Validation Semantics

## Status

`IMPLEMENTED - READY FOR MANUAL DEEP VALIDATION`

## New Semantic Rules Added

- Duplicate event ID detection across score.
- Duplicate voice ID detection within measure.
- Non-monotonic measure number warnings.
- Voice container ID vs event voice mismatch warnings.
- Tie integrity checks:
  - stop without start
  - continue without start
  - stop pitch mismatch within same voice
  - invalid tie encoding (continue + start/stop)
  - unclosed ties at part end

## High-Value Relevance to OMR

These checks directly constrain model outputs to musically coherent structures and prevent silent corruption in exported MusicXML.

## Fast Validation

1. Open `examples/03_valid_tie_chain.validation.json` (should be zero issues).
2. Open `examples/04_invalid_semantics.validation.json` (should include multiple deterministic codes).
3. Run commands in `10_commands.md`.
