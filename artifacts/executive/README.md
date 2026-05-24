# Executive Validation Pack

This folder is optimized for fast maintainer validation.

## One-Minute Review Path

0. Open `ONE_MINUTE.md`.
1. Open `gate-a/00_summary.md` for foundation/tooling status.
2. Open `step-3/00_summary.md` for score semantics and real notation outputs.
3. Open `step-4/00_summary.md` for expanded semantic validation rules.
4. Check concrete examples in `step-3/examples/` and `step-4/examples/`.

## Artifact Organization

- `gate-a/`: tooling and CLI readiness evidence.
- `step-2/`: core primitive implementation evidence.
- `step-3/`: first real notation outputs (`IR -> validate -> MusicXML`).
- `step-4/`: deeper semantic integrity checks and adversarial invalid cases.

## Expected Use

Each subfolder is designed to answer:

- What changed?
- Why it matters?
- How to validate quickly?
- What to approve before moving to next gate?
