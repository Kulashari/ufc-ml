"""Leakage-safe UFC fight probability training and inference."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ufc-predictor")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
