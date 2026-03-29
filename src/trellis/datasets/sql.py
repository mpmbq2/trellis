from __future__ import annotations

from typing import Any, Literal

import ibis  # type: ignore
from ibis.expr.types import Table  # type: ignore

import pandas as pd  # type: ignore
import polars as pl  # type: ignore

from trellis.datasets.abstract import AbstractDataset


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
        """
        table_expr = self._connection.table(self._table_name)

        if backend == "ibis":
            return table_expr
        elif backend == "polars":
            return self._connection.to_polars(table_expr)
        elif backend == "pandas":
            return self._connection.to_pandas(table_expr)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

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

        if if_exists == "fail":
            if self.exists():
                raise FileExistsError(
                    f"Table '{self._table_name}' already exists. "
                    "Use if_exists='replace' or if_exists='append' to overwrite or append."
                )
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
        elif if_exists == "append":
            if not self.exists():
                # Table doesn't exist, create it
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
        else:
            raise ValueError(f"Invalid if_exists value: {if_exists}")

    def exists(self) -> bool:
        """Return whether the table exists in the database."""
        return self._table_name in self._connection.list_tables()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(location={self._location!r}, table_name={self._table_name!r})"
