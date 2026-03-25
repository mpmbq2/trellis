from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class AbstractDataset(ABC):
    """Abstract base class for dataset I/O operations in data pipelines.

    This class provides a universal interface for all datasets, allowing
    pipeline developers to abstract away I/O implementation details.

    Concrete subclasses should be registered via the ``@AbstractDataset.register``
    decorator so they can be instantiated through the factory method
    ``AbstractDataset.create()``.
    """

    _registry: ClassVar[dict[str, type[AbstractDataset]]] = {}

    def __init__(self, path: str | None = None, **kwargs: Any) -> None:
        """Initialize the dataset.

        Args:
            path: Optional path to the dataset location.
            **kwargs: Additional configuration parameters stored as metadata.
        """
        self._path = path
        self._metadata: dict[str, Any] = kwargs

    # ------------------------------------------------------------------
    # Registry / factory
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str):
        """Class decorator that registers a dataset subclass under *name*.

        Usage::

            @AbstractDataset.register("parquet")
            class ParquetDataset(AbstractDataset):
                ...

        Raises:
            ValueError: If *name* is already registered.
        """

        def decorator(subclass: type[AbstractDataset]) -> type[AbstractDataset]:
            if name in cls._registry:
                raise ValueError(
                    f"Dataset type '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> AbstractDataset:
        """Instantiate a registered dataset subclass by *name*.

        Args:
            name: The registered name of the dataset type (e.g. ``"parquet"``).
            **kwargs: Arguments forwarded to the subclass constructor.

        Returns:
            An instance of the registered subclass.

        Raises:
            KeyError: If *name* has not been registered.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys())) or "(none)"
            raise KeyError(
                f"Unknown dataset type '{name}'. Available types: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def list_types(cls) -> list[str]:
        """Return the names of all registered dataset types."""
        return sorted(cls._registry.keys())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> str | None:
        """Get the dataset path."""
        return self._path

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> Any:
        """Load data from the dataset.

        Returns:
            The loaded data.
        """
        ...

    @abstractmethod
    def save(self, data: Any) -> None:
        """Save data to the dataset.

        Args:
            data: The data to save.
        """
        ...

    @abstractmethod
    def exists(self) -> bool:
        """Check if the dataset exists.

        Returns:
            True if the dataset exists, False otherwise.
        """
        ...

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def get_metadata(self) -> dict[str, Any]:
        """Get dataset metadata.

        Returns:
            A shallow copy of the metadata dictionary.
        """
        return self._metadata.copy()

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata key-value pair.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self._metadata[key] = value

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"{self.__class__.__name__}(path={self._path!r},"
            f" metadata={self._metadata!r})"
        )
