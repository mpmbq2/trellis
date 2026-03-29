from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractDatasource(ABC):
    """Abstract base class for read-only data sources in data pipelines.

    Unlike :class:`~trellis.datasets.abstract.AbstractDataset`, datasources
    are strictly read-only: they expose no ``save`` method.  Use them to
    represent external or upstream data that the pipeline consumes but never
    writes back to (e.g. a vendor API, a read-only database view, or a
    shared input file).

    Subclasses implement :meth:`load` and :meth:`exists`.
    Instantiate concrete classes directly (e.g. ``CSVDatasource(...)``).
    """

    def __init__(self, *, location: str | None = None) -> None:
        """Initialize the datasource.

        Args:
            location: Optional address for the data (path, URI, etc.).
                Subclasses may require additional constructor arguments.
        """
        self.location = location

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(location={self.location!r})"

    @abstractmethod
    def load(self) -> Any:
        """Load data from the datasource location."""
        ...

    @abstractmethod
    def exists(self) -> bool:
        """Return whether the datasource is present at its location."""
        ...
