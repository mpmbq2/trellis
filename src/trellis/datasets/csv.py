from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, cast

import duckdb
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
from polars import LazyFrame
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.csv as pa_csv  # type: ignore[import-untyped]

from trellis.datasets.abstract import AbstractDataset


def _sql_string_literal(value: str) -> str:
    """Return *value* as a single-quoted SQL string literal (escaping quotes)."""
    return "'" + value.replace("'", "''") + "'"


@AbstractDataset.register("csv")
class CSVDataset(AbstractDataset):
    """CSV dataset I/O with polars and pandas DataFrame support.

    This dataset handles reading and writing CSV files using PyArrow
    for I/O and supports both polars and pandas DataFrames.

    Args:
        path: Path to the CSV file.
        delimiter: Field delimiter (default: ",").
        has_header: Whether the CSV has a header row (default: True).
        encoding: Character encoding (default: "utf-8").
        **kwargs: Additional metadata and options.
    """

    def __init__(
        self,
        path: str,
        delimiter: str = ",",
        has_header: bool = True,
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> None:
        """Initialize the CSV dataset.

        Args:
            path: Path to the CSV file.
            delimiter: Field delimiter character.
            has_header: Whether the file has a header row.
            encoding: Character encoding for the file.
            **kwargs: Additional configuration options.
        """
        super().__init__(path=path, **kwargs)
        self._delimiter = delimiter
        self._has_header = has_header
        self._encoding = encoding

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def delimiter(self) -> str:
        """Get the field delimiter."""
        return self._delimiter

    @property
    def has_header(self) -> bool:
        """Check if CSV has a header row."""
        return self._has_header

    @property
    def encoding(self) -> str:
        """Get the character encoding."""
        return self._encoding

    # ------------------------------------------------------------------
    # I/O operations
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_encoding(encoding: str) -> str:
        """Normalize encoding string for Polars compatibility."""
        encoding_map: dict[str, str] = {
            "utf-8": "utf8",
            "utf8": "utf8",
            "utf-8-lossy": "utf8-lossy",
        }
        return encoding_map.get(encoding.lower(), encoding)

    def load(self, as_type: str = "polars", lazy: bool = False) -> Any:
        """Load data from the CSV file.

        Args:
            as_type: Output format - "polars" (default) or "pandas".
            lazy: If True and as_type="polars", return a LazyFrame for
                deferred execution (default: False).

        Returns:
            A polars DataFrame/LazyFrame or pandas DataFrame containing the CSV data.

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
            raise FileNotFoundError(f"CSV file not found: {self._path}")

        try:
            normalized_encoding = cast(
                Literal["utf8", "utf8-lossy"],
                self._normalize_encoding(self._encoding),
            )
            if lazy:
                return pl.scan_csv(
                    self._path,
                    separator=self._delimiter,
                    has_header=self._has_header,
                    encoding=normalized_encoding,
                )
            df = pl.read_csv(
                self._path,
                separator=self._delimiter,
                has_header=self._has_header,
                encoding=normalized_encoding,
            )
            if as_type == "pandas":
                return df.to_pandas()
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV from {self._path!r}: {e}") from e

    def lazy_load(self) -> LazyFrame:
        """Return a Polars LazyFrame for deferred execution.

        This enables query optimization and memory-efficient processing
        of large CSV files.

        Returns:
            A Polars LazyFrame representing the CSV data.

        Raises:
            ValueError: If path is not specified.
            FileNotFoundError: If the file does not exist.
        """
        if self._path is None:
            raise ValueError("Path must be specified to load a dataset")

        if not self.exists():
            raise FileNotFoundError(f"CSV file not found: {self._path}")

        normalized_encoding = cast(
            Literal["utf8", "utf8-lossy"], self._normalize_encoding(self._encoding)
        )
        return pl.scan_csv(
            self._path,
            separator=self._delimiter,
            has_header=self._has_header,
            encoding=normalized_encoding,
        )

    def save(self, data: Any) -> None:
        """Save a DataFrame to the CSV file.

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
                f"CSVDataset.save() expects a polars or pandas DataFrame, "
                f"got {type(data).__name__}"
            )

        try:
            # PyArrow's WriteOptions may support delimiter in newer versions
            # We try with delimiter first, fallback to without if needed
            try:
                write_options = pa_csv.WriteOptions(
                    include_header=self._has_header,
                    delimiter=self._delimiter,
                )
            except TypeError:
                # Fallback for older pyarrow versions that don't support delimiter
                write_options = pa_csv.WriteOptions(
                    include_header=self._has_header,
                )

            pa_csv.write_csv(
                table,
                self._path,
                write_options=write_options,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to save CSV to {self._path!r}: {e}") from e

    def query(self, sql: str, as_type: str = "polars") -> pd.DataFrame | pl.DataFrame:
        """Run a DuckDB SQL query against this CSV file.

        Pass a SQL string that uses the placeholder ``{dataset}`` in a ``FROM``
        clause. It is replaced with a ``read_csv(...)`` call using this dataset's
        path, delimiter, header, and encoding, so DuckDB can project or filter
        without first loading the whole file into a separate relation.

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
            raise FileNotFoundError(f"CSV file not found: {self._path}")

        # Require FROM {dataset} placeholder for read substitution.
        if not re.search(r"(?i)FROM\s+\{dataset\}", sql):
            raise ValueError(
                'Query must include the placeholder "FROM {dataset}" so the CSV file '
                "can be substituted (e.g. "
                "'SELECT col1, col2 FROM {dataset} WHERE age > 0'). "
                "Use literal {dataset} exactly; other braces in SQL must be doubled "
                "as {{ and }} when using str.format rules."
            )

        path_abs = str(Path(self._path).resolve())
        path_lit = _sql_string_literal(path_abs)
        delim_lit = _sql_string_literal(self._delimiter)
        # Polars uses "utf8"; DuckDB's CSV reader expects names like "utf-8".
        norm = self._normalize_encoding(self._encoding)
        duck_encoding = "utf-8" if norm in ("utf8", "utf8-lossy") else self._encoding
        encoding_lit = _sql_string_literal(duck_encoding)
        header_sql = "true" if self._has_header else "false"
        read_expr = (
            f"read_csv({path_lit}, delim={delim_lit}, "
            f"header={header_sql}, encoding={encoding_lit})"
        )

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
                f"Failed to query CSV at {self._path!r} with DuckDB: {e}"
            ) from e

    def exists(self) -> bool:
        """Check if the CSV file exists.

        Returns:
            True if the file exists, False otherwise.
        """
        if self._path is None:
            return False

        try:
            return Path(self._path).exists()
        except Exception:
            return False
