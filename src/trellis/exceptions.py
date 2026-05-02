from __future__ import annotations


class TrellisError(Exception):
    """Base class for all Trellis-specific errors."""


class DatasetNotFoundError(TrellisError, FileNotFoundError):
    """Raised when a dataset's underlying data is not present at its location."""


class DatasetLoadError(TrellisError):
    """Raised when loading a dataset fails."""


class DatasetSaveError(TrellisError):
    """Raised when saving a dataset fails."""
