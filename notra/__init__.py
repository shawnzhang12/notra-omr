"""Top-level package for Notra OMR."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("notra-omr")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
