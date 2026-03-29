from __future__ import annotations

from typing import Any, Literal

import ibis  # type: ignore

from trellis.datasources.abstract import AbstractDatasource


class SQLDatasource(AbstractDatasource):
    """Read-only SQL datasource supporting ibis, polars, and pandas backends.

    Uses ibis-framework for database connectivity.
    Supports any backend that ibis supports: DuckDB, PostgreSQL, SQLite, MySQL, etc.

    Unlike :class:`~trellis.datasets.sql.SQLDataset`, this class exposes no
    ``save`` method — it is intended for external or shared databases that the
    pipeline reads from but never writes back to.

    Example:
        # SQLite
        ds = SQLDatasource(location="sqlite:///path/to/db.db", table_name="my_table")

        # DuckDB (in-memory)
        ds = SQLDatasource(location="duckdb://", table_name="my_table")

        # PostgreSQL
        ds = SQLDatasource(location="postgres://user:pass@host:5432/db", table_name="my_table")
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
        """Initialize SQL datasource.

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
            raise ValueError(f"Unsupported backend: {backend!r}")

    def exists(self) -> bool:
        """Return whether the table exists in the database."""
        return self._table_name in self._connection.list_tables()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"location={self._location!r}, "
            f"table_name={self._table_name!r})"
        )
