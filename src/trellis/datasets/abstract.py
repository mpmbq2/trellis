from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# ------------------------------------------------------------------
# AbstractDataset
# ------------------------------------------------------------------


class AbstractDataset(ABC):
    """Abstract base class for dataset I/O in data pipelines.

    Subclasses implement :meth:`load`, :meth:`save`, and :meth:`exists`.
    Instantiate concrete classes directly (e.g. ``CSVDataset(...)``).
    """

    def __init__(self, *, location: str | None = None) -> None:
        """Initialize the dataset.

        Args:
            location: Optional address for the data (path, URI, etc.).
                Subclasses may require additional constructor arguments.
        """
        self.location = location

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(location={self.location!r})"

    @abstractmethod
    def load(self) -> Any:
        """Load data from the dataset location."""
        ...

    @abstractmethod
    def save(self, data: Any) -> None:
        """Save *data* to the dataset location."""
        ...

    @abstractmethod
    def exists(self) -> bool:
        """Return whether the dataset is present at its location."""
        ...
