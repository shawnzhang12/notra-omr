"""`notra convert` command."""

from __future__ import annotations

import argparse
import sys

from notra.exporters.json_ir import export_score_to_json
from notra.exporters.musicxml import export_score_to_musicxml
from notra.ir.serialize import score_from_json


def register(subparsers) -> None:
    """Register the convert subcommand."""
    parser = subparsers.add_parser("convert", help="Convert Notra IR to target formats.")
    parser.add_argument("input", help="Path to Notra IR JSON file.")
    parser.add_argument("--to", required=True, choices=["musicxml", "json", "mei"])
    parser.add_argument("--out", required=True, help="Output file path.")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Execute the convert subcommand."""
    try:
        with open(args.input, "r", encoding="utf-8") as handle:
            score = score_from_json(handle.read())

        if args.to == "musicxml":
            output = export_score_to_musicxml(score)
        elif args.to == "json":
            output = export_score_to_json(score)
        else:
            print("MEI export is not implemented yet.", file=sys.stderr)
            return 2

        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(output)
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(f"conversion failed: {exc}", file=sys.stderr)
        return 2

    print(f"wrote {args.to} output to {args.out}")
    return 0
