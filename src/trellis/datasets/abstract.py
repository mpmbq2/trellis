from typing import Any, Optional, Dict
from abc import ABC, abstractmethod


class AbstractDataset(ABC):
    """
    Abstract base class for dataset I/O operations in data pipelines.

    This class provides a universal interface for all datasets, allowing
    pipeline developers to abstract away I/O implementation details.
    """

    def __init__(self, path: Optional[str] = None, **kwargs):
        """
        Initialize the dataset.

        Args:
            path: Optional path to the dataset location
            **kwargs: Additional configuration parameters
        """
        self._path = path
        self._metadata: Dict[str, Any] = kwargs

    @property
    def path(self) -> Optional[str]:
        """Get the dataset path."""
        return self._path

    @abstractmethod
    def load(self) -> Any:
        """
        Load data from the dataset.

        Returns:
            The loaded data

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Load method is not implemented")

    @abstractmethod
    def save(self, data: Any) -> None:
        """
        Save data to the dataset.

        Args:
            data: The data to save

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Save method is not implemented")

    @abstractmethod
    def exists(self) -> bool:
        """
        Check if the dataset exists.

        Returns:
            True if the dataset exists, False otherwise

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Exists method is not implemented")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get dataset metadata.

        Returns:
            Dictionary containing dataset metadata
        """
        return self._metadata.copy()

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set a metadata key-value pair.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate the dataset.

        Returns:
            True if dataset is valid, False otherwise

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Validate method is not implemented")

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return f"{self.__class__.__name__}(path={self._path!r}, metadata={self._metadata!r})"
