from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
from polars import LazyFrame
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from trellis.datasets.abstract import AbstractDataset


def _sql_string_literal(value: str) -> str:
    """Return *value* as a single-quoted SQL string literal (escaping quotes)."""
    return "'" + value.replace("'", "''") + "'"


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

    def query(self, sql: str, as_type: str = "polars") -> pd.DataFrame | pl.DataFrame:
        """Run a DuckDB SQL query against this Parquet file.

        Pass a SQL string that uses the placeholder ``{dataset}`` in a ``FROM``
        clause. It is replaced with ``read_parquet(...)`` for this dataset's path
        so DuckDB can project or filter without wrapping the file in an extra CTE.

        Example::

            SELECT name, age FROM {dataset} WHERE age > 28

        Args:
            sql: SQL containing ``FROM {dataset}`` (see :func:`str.format`).
            as_type: ``"polars"`` (default) or ``"pandas"`` for the result type.

        Returns:
            Query result as a polars or pandas DataFrame.

        Raises:
            ValueError: If *as_type* is invalid, the path is not set, *sql* lacks
                ``FROM {dataset}``, or *sql* has invalid format placeholders.
            FileNotFoundError: If the file does not exist.
            RuntimeError: If DuckDB fails to execute the query.
        """
        if as_type not in ("polars", "pandas"):
            raise ValueError(f"as_type must be 'polars' or 'pandas', got {as_type!r}")

        if self._path is None:
            raise ValueError("Path must be specified to query a dataset")

        if not self.exists():
            raise FileNotFoundError(f"Parquet file not found: {self._path}")

        # Require FROM {dataset} placeholder for read substitution.
        if not re.search(r"(?i)FROM\s+\{dataset\}", sql):
            raise ValueError(
                'Query must include the placeholder "FROM {dataset}" so the Parquet '
                "file can be substituted (e.g. "
                "'SELECT col1, col2 FROM {dataset} WHERE age > 0'). "
                "Use literal {dataset} exactly; other braces in SQL must be doubled "
                "as {{ and }} when using str.format rules."
            )

        path_abs = str(Path(self._path).resolve())
        path_lit = _sql_string_literal(path_abs)
        read_expr = f"read_parquet({path_lit})"

        try:
            full_sql = sql.format(dataset=read_expr)
        except KeyError as e:
            raise ValueError(
                "Invalid query string for .format(): only the {dataset} placeholder "
                "is supported; escape any other braces as {{ and }}."
            ) from e

        try:
            relation = duckdb.sql(full_sql)
            if as_type == "pandas":
                return relation.df()
            return relation.pl()
        except Exception as e:
            raise RuntimeError(
                f"Failed to query Parquet at {self._path!r} with DuckDB: {e}"
            ) from e

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
