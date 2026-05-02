from __future__ import annotations

from typing import Any, Literal

import ibis  # type: ignore
from ibis.expr.types import Table  # type: ignore

import pandas as pd  # type: ignore
import polars as pl  # type: ignore

from trellis._url_utils import redact_url_password
from trellis.datasets.abstract import AbstractDataset
from trellis.exceptions import (
    DatasetLoadError,
    DatasetNotFoundError,
    DatasetSaveError,
)


class SQLDataset(AbstractDataset):
    """SQL dataset implementation supporting ibis, polars, and pandas backends.

    Uses ibis-framework for database connectivity.
    Supports any backend that ibis supports: DuckDB, PostgreSQL, SQLite, MySQL, etc.

    Example:
        # SQLite
        ds = SQLDataset(location="sqlite:///path/to/db.db", table_name="my_table")

        # DuckDB (in-memory)
        ds = SQLDataset(location="duckdb://", table_name="my_table")

        # PostgreSQL
        ds = SQLDataset(location="postgres://user:pass@host:5432/db", table_name="my_table")
    """

    _connection: ibis.BaseBackend
    _table_name: str
    _location: str

    def __init__(
        self,
        *,
        location: str,
        table_name: str,
    ) -> None:
        """Initialize SQL dataset.

        Args:
            location: ibis connection URL (e.g., "sqlite:///path.db", "duckdb://", "postgres://...").
            table_name: Name of the table in the database.
        """
        super().__init__(location=location)
        self._location = location
        self._table_name = table_name
        self._connection = ibis.connect(location)

    @property
    def connection(self) -> ibis.BaseBackend:
        """Return the ibis connection for direct use."""
        return self._connection

    def load(
        self,
        *,
        backend: Literal["polars", "pandas", "ibis"] = "polars",
    ) -> Any:
        """Load data from the SQL table.

        Args:
            backend: Which library to use ("polars", "pandas", or "ibis").

        Returns:
            polars DataFrame, pandas DataFrame, or ibis Table expression.

        Raises:
            DatasetNotFoundError: If the table does not exist in the database.
            DatasetLoadError: If reading the table fails for any other reason.
        """
        if backend not in ("polars", "pandas", "ibis"):
            raise ValueError(f"Unsupported backend: {backend}")

        if not self.exists():
            raise DatasetNotFoundError(
                f"Table {self._table_name!r} not found at "
                f"{redact_url_password(self._location)!r}"
            )

        try:
            table_expr = self._connection.table(self._table_name)

            if backend == "ibis":
                return table_expr
            elif backend == "polars":
                return self._connection.to_polars(table_expr)
            else:
                return self._connection.to_pandas(table_expr)
        except Exception as e:
            raise DatasetLoadError(
                f"Failed to load table {self._table_name!r} from "
                f"{redact_url_password(self._location)!r}: {e}"
            ) from e

    def save(
        self,
        data: Any,
        *,
        if_exists: Literal["fail", "replace", "append"] = "replace",
    ) -> None:
        """Save data to the SQL table.

        Backend is inferred from the data type (ibis Table, polars DataFrame/LazyFrame,
        pandas DataFrame, or pyarrow Table).

        Args:
            data: The data to save.
            if_exists: Behavior when table exists:
                - "fail": Raise an error if table exists.
                - "replace": Drop and recreate the table (default).
                - "append": Insert rows into existing table.

        Raises:
            FileExistsError: If ``if_exists="fail"`` and the table already exists.
            DatasetSaveError: If writing the table fails.
        """
        # Handle lazy polars DataFrame by collecting
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        # Convert ibis Table to a materialized form for saving
        if isinstance(data, Table):
            # Check if the table is from the same backend
            table_backend = data._find_backend(use_default=False)
            if table_backend is None or table_backend is not self._connection:
                # Table is from a different backend or unbound, materialize it
                obj = data.to_pandas()
            else:
                # Table is from our backend, can use it directly
                obj = data
        elif isinstance(data, pl.DataFrame):
            obj = data
        elif isinstance(data, pd.DataFrame):
            obj = data
        elif hasattr(data, "__arrow_c_stream__") or hasattr(data, "to_arrow"):
            # PyArrow Table or compatible
            import pyarrow as pa  # type: ignore

            if isinstance(data, pa.Table):
                obj = data
            else:
                obj = data.to_arrow() if hasattr(data, "to_arrow") else data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        if if_exists not in ("fail", "replace", "append"):
            raise ValueError(f"Invalid if_exists value: {if_exists}")

        if if_exists == "fail" and self.exists():
            raise FileExistsError(
                f"Table '{self._table_name}' already exists. "
                "Use if_exists='replace' or if_exists='append' to overwrite or append."
            )

        try:
            if if_exists == "fail":
                self._connection.create_table(
                    self._table_name,
                    obj=obj,
                    overwrite=False,
                )
            elif if_exists == "replace":
                self._connection.create_table(
                    self._table_name,
                    obj=obj,
                    overwrite=True,
                )
            else:  # append
                if not self.exists():
                    self._connection.create_table(
                        self._table_name,
                        obj=obj,
                        overwrite=False,
                    )
                else:
                    self._connection.insert(
                        self._table_name,
                        obj=obj,
                    )
        except Exception as e:
            raise DatasetSaveError(
                f"Failed to save table {self._table_name!r} to "
                f"{redact_url_password(self._location)!r}: {e}"
            ) from e

    def exists(self) -> bool:
        """Return whether the table exists in the database."""
        return self._table_name in self._connection.list_tables()

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of this dataset's configuration.

        The connection URL has any embedded password redacted.
        """
        return {
            "type": self.__class__.__name__,
            "location": redact_url_password(self._location),
            "table_name": self._table_name,
        }
