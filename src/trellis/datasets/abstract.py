from __future__ import annotations

from abc import ABC


class AbstractDataset(ABC):
    """Abstract base class for dataset I/O operations in data pipelines.

    This class provides a universal interface for all datasets, allowing
    pipeline developers to abstract away I/O implementation details.

    Concrete subclasses should be registered via the ``@AbstractDataset.register``
    decorator so they can be instantiated through the factory method
    ``AbstractDataset.create()``.
    """

    pass
