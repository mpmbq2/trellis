from __future__ import annotations

from typing import Any, Literal

import fsspec  # type: ignore
import pandas as pd  # type: ignore
import polars as pl  # type: ignore

from trellis.datasets.abstract import AbstractDataset


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
        """
        if backend == "polars":
            if lazy:
                return pl.scan_csv(self._location)
            return pl.read_csv(self._location)

        return pd.read_csv(self._location)

    def save(self, data: Any) -> None:
        """Save data to the CSV location.

        Backend is inferred from the data type (polars vs pandas).

        Args:
            data: A polars DataFrame/LazyFrame or pandas DataFrame.
        """
        if isinstance(data, pl.DataFrame):
            data.write_csv(self._location)
        elif isinstance(data, pl.LazyFrame):
            data.sink_csv(self._location)
        elif isinstance(data, pd.DataFrame):
            data.to_csv(self._location, index=False)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def exists(self) -> bool:
        """Return whether the CSV file exists at its location."""
        fs, path = fsspec.core.url_to_fs(self._location)
        return fs.exists(path)
