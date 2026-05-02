from __future__ import annotations

from typing import Any, Literal

import fsspec  # type: ignore
import pandas as pd  # type: ignore
import polars as pl  # type: ignore

from trellis.datasets.abstract import AbstractDataset
from trellis.exceptions import (
    DatasetLoadError,
    DatasetNotFoundError,
    DatasetSaveError,
)


class CSVDataset(AbstractDataset):
    """CSV dataset implementation supporting polars and pandas backends.

    Supports local and remote storage via fsspec-compatible paths (S3, GCS, etc.).
    """

    _location: str

    def __init__(
        self,
        *,
        location: str,
    ) -> None:
        """Initialize CSV dataset.

        Args:
            location: Path or URI to the CSV file (local or remote via fsspec).
        """
        super().__init__(location=location)
        self._location = location

    def load(
        self,
        *,
        backend: Literal["polars", "pandas"] = "polars",
        lazy: bool = False,
    ) -> Any:
        """Load data from the CSV location.

        Args:
            backend: Which library to use ("polars" or "pandas").
            lazy: If True and backend is "polars", load as LazyFrame.

        Returns:
            polars DataFrame or LazyFrame, or pandas DataFrame.

        Raises:
            DatasetNotFoundError: If the CSV file is not present at the location.
            DatasetLoadError: If reading the CSV fails for any other reason.
        """
        try:
            if backend == "polars":
                if lazy:
                    return pl.scan_csv(self._location)
                return pl.read_csv(self._location)

            return pd.read_csv(self._location)
        except FileNotFoundError as e:
            raise DatasetNotFoundError(f"CSV not found at {self._location!r}") from e
        except Exception as e:
            raise DatasetLoadError(
                f"Failed to load CSV from {self._location!r}: {e}"
            ) from e

    def save(self, data: Any) -> None:
        """Save data to the CSV location.

        Backend is inferred from the data type (polars vs pandas).

        Args:
            data: A polars DataFrame/LazyFrame or pandas DataFrame.

        Raises:
            DatasetSaveError: If writing the CSV fails.
        """
        if not isinstance(data, (pl.DataFrame, pl.LazyFrame, pd.DataFrame)):
            raise TypeError(f"Unsupported data type: {type(data)}")

        try:
            if isinstance(data, pl.DataFrame):
                data.write_csv(self._location)
            elif isinstance(data, pl.LazyFrame):
                data.sink_csv(self._location)
            else:
                data.to_csv(self._location, index=False)
        except Exception as e:
            raise DatasetSaveError(
                f"Failed to save CSV to {self._location!r}: {e}"
            ) from e

    def exists(self) -> bool:
        """Return whether the CSV file exists at its location."""
        fs, path = fsspec.core.url_to_fs(self._location)
        return fs.exists(path)
