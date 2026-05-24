"""Smoke tests for repository scaffolding."""

from notra import __version__


def test_package_exposes_version() -> None:
    """The package should expose a version string."""
    assert isinstance(__version__, str)
    assert __version__
