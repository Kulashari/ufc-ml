"""Domain exceptions raised by the UFC predictor package."""

from __future__ import annotations


class UFCPredictorError(Exception):
    """Base class for errors that callers may safely present to users."""


class ConfigurationError(UFCPredictorError):
    """Raised when project configuration is missing or invalid."""


class DataLoadError(UFCPredictorError):
    """Raised when a configured data asset cannot be loaded."""


class DataValidationError(UFCPredictorError):
    """Raised when a dataset violates a required invariant."""


class SchemaValidationError(DataValidationError):
    """Raised when columns or data types do not match the expected schema."""


class LeakageValidationError(DataValidationError):
    """Raised when a post-fight or otherwise forbidden column is detected."""


class SplitValidationError(DataValidationError):
    """Raised when chronological dataset splits are invalid."""


class SnapshotValidationError(DataValidationError):
    """Raised when fighter snapshots violate their point-in-time cutoff."""


class FeatureRegistryError(DataValidationError):
    """Raised when the feature dictionary and dataset disagree."""


class FingerprintMismatchError(DataValidationError):
    """Raised when a data asset does not match its expected SHA256 digest."""


class FighterNotFoundError(UFCPredictorError):
    """Raised when a fighter cannot be found in the snapshot store."""


class AmbiguousFighterError(UFCPredictorError):
    """Raised when a fighter name resolves to more than one fighter ID."""
