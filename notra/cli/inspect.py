"""`notra inspect` command."""

from __future__ import annotations

import argparse


def register(subparsers) -> None:
    """Register the inspect subcommand."""
    parser = subparsers.add_parser("inspect", help="Inspect a pipeline artifact directory.")
    parser.add_argument("artifact_dir", help="Path to run artifact directory.")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Execute the inspect subcommand."""
    print(f"`notra inspect` is not implemented yet. Artifact dir: {args.artifact_dir}")
    return 2
