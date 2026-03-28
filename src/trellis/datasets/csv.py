from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd  # type: ignore[import-untyped]
import polars as pl
from polars import LazyFrame
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.csv as pa_csv  # type: ignore[import-untyped]

from trellis.datasets.abstract import AbstractDataset


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
