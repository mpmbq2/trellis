from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.csv as pa_csv  # type: ignore[import-untyped]

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
else:
    pl = None  # type: ignore[misc,assignment]

if HAS_PANDAS:
    import pandas as pd  # type: ignore[import-untyped]
else:
    pd = None  # type: ignore[misc,assignment]


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

    def load(self, as_type: str = "polars") -> Any:
        """Load data from the CSV file.

        Args:
            as_type: Output format - "polars" (default) or "pandas".

        Returns:
            A polars or pandas DataFrame containing the CSV data.

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
                "polars is required to load CSVDataset as polars. "
                "Install with: pip install trellis[polars] or pip install polars"
            )

        if as_type == "pandas" and not HAS_PANDAS:
            raise ImportError(
                "pandas is required to load CSVDataset as pandas. "
                "Install with: pip install trellis[pandas] or pip install pandas"
            )

        if self._path is None:
            raise ValueError("Path must be specified to load a dataset")

        if not self.exists():
            raise FileNotFoundError(f"CSV file not found: {self._path}")

        try:
            # Read CSV using pyarrow
            parse_options = pa_csv.ParseOptions(
                delimiter=self._delimiter,
            )

            read_options = pa_csv.ReadOptions(
                use_threads=True,
                encoding=self._encoding,
                autogenerate_column_names=not self._has_header,
            )

            table = pa_csv.read_csv(
                self._path,
                parse_options=parse_options,
                read_options=read_options,
            )

            # Convert to requested type
            if as_type == "polars":
                return pl.DataFrame(table)  # pyright: ignore[reportOptionalMemberAccess]
            else:  # pandas
                return table.to_pandas()

        except Exception as e:
            raise RuntimeError(f"Failed to load CSV from {self._path!r}: {e}") from e

    def save(self, data: Any) -> None:
        """Save a DataFrame to the CSV file.

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
