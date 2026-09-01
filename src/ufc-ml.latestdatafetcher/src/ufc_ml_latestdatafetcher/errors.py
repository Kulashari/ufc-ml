"""Fetcher-specific exceptions surfaced cleanly by the CLI."""

from __future__ import annotations


class LatestDataFetcherError(Exception):
    """Base class for expected fetcher failures."""


class FetcherConfigurationError(LatestDataFetcherError):
    """The fetcher configuration is absent or invalid."""


class SourceFetchError(LatestDataFetcherError):
    """A source page could not be retrieved or was a browser challenge."""


class SourceParseError(LatestDataFetcherError):
    """A retrieved page did not satisfy the expected UFCStats schema."""


class DatasetValidationError(LatestDataFetcherError):
    """Candidate local data is incomplete or internally inconsistent."""


__all__ = [
    "DatasetValidationError",
    "FetcherConfigurationError",
    "LatestDataFetcherError",
    "SourceFetchError",
    "SourceParseError",
]
