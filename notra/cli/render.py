"""`notra render` command."""

from __future__ import annotations

import argparse


def register(subparsers) -> None:
    """Register the render subcommand."""
    parser = subparsers.add_parser("render", help="Render notation files to image outputs.")
    parser.add_argument("input", help="Path to MusicXML or MEI input file.")
    parser.add_argument("--out", required=True, help="Rendered output path.")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Execute the render subcommand."""
    print(f"`notra render` is not implemented yet. Input: {args.input}, out: {args.out}")
    return 2
