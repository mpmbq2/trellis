from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import polars as pl
from polars import LazyFrame
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from trellis.datasets.abstract import AbstractDataset


@AbstractDataset.register("parquet")
class ParquetDataset(AbstractDataset):
    """Parquet dataset I/O with polars and pandas DataFrame support.

    This dataset handles reading and writing Parquet files using PyArrow
    for I/O and supports both polars and pandas DataFrames.

    Args:
        path: Path to the Parquet file.
        compression: Compression codec - "snappy" (default), "gzip", "zstd",
            "lz4", "bz2", or "none".
        row_group_size: Number of rows per row group (default: 1048576).
        use_threads: Whether to use multithreading (default: True).
        **kwargs: Additional metadata and options.
    """

    def __init__(
        self,
        path: str,
        compression: str = "snappy",
        row_group_size: int = 1048576,
        use_threads: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Parquet dataset.

        Args:
            path: Path to the Parquet file.
            compression: Compression codec.
            row_group_size: Number of rows per row group.
            use_threads: Whether to use multithreading.
            **kwargs: Additional configuration options.
        """
        super().__init__(path=path, **kwargs)
        self._compression = compression
        self._row_group_size = row_group_size
        self._use_threads = use_threads

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def compression(self) -> str:
        """Get the compression codec."""
        return self._compression

    @property
    def row_group_size(self) -> int:
        """Get the row group size."""
        return self._row_group_size

    @property
    def use_threads(self) -> bool:
        """Check if multithreading is enabled."""
        return self._use_threads

    # ------------------------------------------------------------------
    # I/O operations
    # ------------------------------------------------------------------

    def load(self, as_type: str = "polars", lazy: bool = False) -> Any:
        """Load data from the Parquet file.

        Args:
            as_type: Output format - "polars" (default) or "pandas".
            lazy: If True and as_type="polars", return a LazyFrame for
                deferred execution (default: False).

        Returns:
            A polars DataFrame/LazyFrame or pandas DataFrame containing the Parquet data.

        Raises:
            ValueError: If as_type is not "polars" or "pandas".
            FileNotFoundError: If the file does not exist.
            RuntimeError: If there's an error reading the file.
        """
        if as_type not in ("polars", "pandas"):
            raise ValueError(f"as_type must be 'polars' or 'pandas', got {as_type!r}")

        if self._path is None:
            raise ValueError("Path must be specified to load a dataset")

        if not self.exists():
            raise FileNotFoundError(f"Parquet file not found: {self._path}")

        try:
            if lazy:
                return pl.scan_parquet(self._path)
            df = pl.read_parquet(self._path)
            if as_type == "pandas":
                return df.to_pandas()
            return df
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Parquet from {self._path!r}: {e}"
            ) from e

    def lazy_load(self) -> LazyFrame:
        """Return a Polars LazyFrame for deferred execution.

        This enables query optimization and memory-efficient processing
        of large Parquet files.

        Returns:
            A Polars LazyFrame representing the Parquet data.

        Raises:
            ValueError: If path is not specified.
            FileNotFoundError: If the file does not exist.
        """
        if self._path is None:
            raise ValueError("Path must be specified to load a dataset")

        if not self.exists():
            raise FileNotFoundError(f"Parquet file not found: {self._path}")

        return pl.scan_parquet(self._path)

    def save(self, data: Any) -> None:
        """Save a DataFrame to the Parquet file.

        Automatically detects whether the data is a polars or pandas DataFrame.

        Args:
            data: A polars or pandas DataFrame to save.

        Raises:
            TypeError: If data is not a polars or pandas DataFrame.
            RuntimeError: If there's an error writing the file.
        """
        if self._path is None:
            raise ValueError("Path must be specified to save a dataset")

        if isinstance(data, pl.DataFrame):
            table = data.to_arrow()
        elif isinstance(data, pd.DataFrame):
            table = pa.Table.from_pandas(data)
        else:
            raise TypeError(
                f"ParquetDataset.save() expects a polars or pandas DataFrame, "
                f"got {type(data).__name__}"
            )

        try:
            pq.write_table(
                table,
                self._path,
                compression=self._compression,
                row_group_size=self._row_group_size,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to save Parquet to {self._path!r}: {e}") from e

    def exists(self) -> bool:
        """Check if the Parquet file exists.

        Returns:
            True if the file exists, False otherwise.
        """
        if self._path is None:
            return False

        try:
            return Path(self._path).exists()
        except Exception:
            return False
