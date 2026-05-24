"""`notra recognize` command."""

from __future__ import annotations

import argparse


def register(subparsers) -> None:
    """Register the recognize subcommand."""
    parser = subparsers.add_parser("recognize", help="Run OMR recognition pipeline.")
    parser.add_argument("input", help="Path to input image.")
    parser.add_argument("--out", required=True, help="Path to output IR JSON.")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Execute the recognize subcommand."""
    print(f"`notra recognize` is not implemented yet. Input: {args.input}, out: {args.out}")
    return 2
