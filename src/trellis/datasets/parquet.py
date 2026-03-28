from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from trellis.datasets.abstract import AbstractDataset

# Check for optional dependencies
if sys.version_info >= (3, 10):
    from importlib.util import find_spec
else:
    from importlib.util import find_spec  # type: ignore[attr-defined]

HAS_POLARS = find_spec("polars") is not None
HAS_PANDAS = find_spec("pandas") is not None

if HAS_POLARS:
    import polars as pl  # type: ignore[import-not-found]
    from polars import LazyFrame
else:
    pl = None  # type: ignore[misc,assignment]
    LazyFrame = None  # type: ignore[misc,assignment]

if HAS_PANDAS:
    import pandas as pd  # type: ignore[import-untyped]
else:
    pd = None  # type: ignore[misc,assignment]


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
            ImportError: If the requested library is not installed.
            ValueError: If as_type is not "polars" or "pandas".
            FileNotFoundError: If the file does not exist.
            RuntimeError: If there's an error reading the file.
        """
        if as_type not in ("polars", "pandas"):
            raise ValueError(f"as_type must be 'polars' or 'pandas', got {as_type!r}")

        if as_type == "polars" and not HAS_POLARS:
            raise ImportError(
                "polars is required to load ParquetDataset as polars. "
                "Install with: pip install trellis[polars] or pip install polars"
            )

        if as_type == "pandas" and not HAS_PANDAS:
            raise ImportError(
                "pandas is required to load ParquetDataset as pandas. "
                "Install with: pip install trellis[pandas] or pip install pandas"
            )

        if self._path is None:
            raise ValueError("Path must be specified to load a dataset")

        if not self.exists():
            raise FileNotFoundError(f"Parquet file not found: {self._path}")

        try:
            if as_type == "polars":
                if lazy:
                    return pl.scan_parquet(  # pyright: ignore[reportOptionalMemberAccess]
                        self._path,
                    )
                else:
                    return pl.read_parquet(  # pyright: ignore[reportOptionalMemberAccess]
                        self._path,
                    )
            else:
                table = pq.read_table(
                    self._path,
                    use_threads=self._use_threads,
                )
                return table.to_pandas()

        except Exception as e:
            raise RuntimeError(
                f"Failed to load Parquet from {self._path!r}: {e}"
            ) from e

    def lazy_load(self) -> LazyFrame:  # type: ignore[override]
        """Return a Polars LazyFrame for deferred execution.

        This enables query optimization and memory-efficient processing
        of large Parquet files.

        Returns:
            A Polars LazyFrame representing the Parquet data.

        Raises:
            ImportError: If polars is not installed.
            ValueError: If path is not specified.
            FileNotFoundError: If the file does not exist.
        """
        if not HAS_POLARS:
            raise ImportError(
                "polars is required to use lazy_load(). "
                "Install with: pip install trellis[polars] or pip install polars"
            )

        if self._path is None:
            raise ValueError("Path must be specified to load a dataset")

        if not self.exists():
            raise FileNotFoundError(f"Parquet file not found: {self._path}")

        return pl.scan_parquet(  # pyright: ignore[reportOptionalMemberAccess]
            self._path,
        )

    def save(self, data: Any) -> None:
        """Save a DataFrame to the Parquet file.

        Automatically detects whether the data is a polars or pandas DataFrame.

        Args:
            data: A polars or pandas DataFrame to save.

        Raises:
            ImportError: If required libraries are not installed.
            TypeError: If data is not a polars or pandas DataFrame.
            RuntimeError: If there's an error writing the file.
        """
        if self._path is None:
            raise ValueError("Path must be specified to save a dataset")

        # Detect and convert to pyarrow table
        if HAS_POLARS and isinstance(data, pl.DataFrame):  # pyright: ignore[reportOptionalMemberAccess]
            table = data.to_arrow()
        elif HAS_PANDAS and isinstance(data, pd.DataFrame):  # pyright: ignore[reportOptionalMemberAccess]
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
