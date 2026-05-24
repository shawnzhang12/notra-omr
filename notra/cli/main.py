"""Main CLI entrypoint for Notra."""

from __future__ import annotations

import argparse
from typing import Callable

from notra.cli import convert, diff, inspect, recognize, render, validate

Handler = Callable[[argparse.Namespace], int]


def build_parser() -> argparse.ArgumentParser:
    """Build and return the root command parser."""
    parser = argparse.ArgumentParser(
        prog="notra",
        description="Notra OMR command-line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    for module in (validate, convert, render, inspect, diff, recognize):
        module.register(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Handler | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
