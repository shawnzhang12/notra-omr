"""`notra diff` command."""

from __future__ import annotations

import argparse


def register(subparsers) -> None:
    """Register the diff subcommand."""
    parser = subparsers.add_parser("diff", help="Compare two IR or export artifacts.")
    parser.add_argument("left", help="Path to first artifact.")
    parser.add_argument("right", help="Path to second artifact.")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Execute the diff subcommand."""
    print(f"`notra diff` is not implemented yet. Left: {args.left}, right: {args.right}")
    return 2
