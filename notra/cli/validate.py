"""`notra validate` command."""

from __future__ import annotations

import argparse
import json
import sys

from notra.ir.serialize import score_from_json
from notra.ir.validate import validate_score


def register(subparsers) -> None:
    """Register the validate subcommand."""
    parser = subparsers.add_parser("validate", help="Validate a Notra IR JSON file.")
    parser.add_argument("input", help="Path to Notra IR JSON file.")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Execute the validate subcommand."""
    try:
        with open(args.input, "r", encoding="utf-8") as handle:
            score = score_from_json(handle.read())
        report = validate_score(score)
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(f"validation failed to run: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.to_dict(), indent=2))
    return 1 if report.has_errors else 0
