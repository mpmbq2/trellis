from __future__ import annotations

import ibis
import pandas as pd
import polars as pl
import pytest

from trellis.datasources import SQLDatasource


@pytest.fixture
def sample_data_polars() -> pl.DataFrame:
    return pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


@pytest.fixture
def sample_data_pandas() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


@pytest.fixture
def sql_datasource(tmp_path, sample_data_polars) -> SQLDatasource:
    """SQLDatasource backed by a SQLite database pre-seeded with sample data."""
    db_path = str(tmp_path / "test.db")
    location = f"sqlite:///{db_path}"

    # Seed the table via ibis directly — SQLDatasource has no save().
    conn = ibis.connect(location)
    conn.create_table("test_table", obj=sample_data_polars)

    return SQLDatasource(location=location, table_name="test_table")


@pytest.fixture
def empty_sql_datasource(tmp_path) -> SQLDatasource:
    """SQLDatasource pointing at a database with no tables yet."""
    db_path = str(tmp_path / "empty.db")
    return SQLDatasource(location=f"sqlite:///{db_path}", table_name="test_table")


# -----------------------------------------------------------------------------
# Load tests
# -----------------------------------------------------------------------------


def test_load_polars(sql_datasource, sample_data_polars) -> None:
    loaded = sql_datasource.load(backend="polars")
    assert isinstance(loaded, pl.DataFrame)
    assert loaded.equals(sample_data_polars)


def test_load_pandas(sql_datasource) -> None:
    loaded = sql_datasource.load(backend="pandas")
    assert isinstance(loaded, pd.DataFrame)
    assert list(loaded["a"]) == [1, 2, 3]
    assert list(loaded["b"]) == ["x", "y", "z"]


def test_load_ibis(sql_datasource) -> None:
    loaded = sql_datasource.load(backend="ibis")
    assert isinstance(loaded, ibis.expr.types.Table)
    materialized = loaded.execute()
    assert len(materialized) == 3


def test_load_default_backend_is_polars(sql_datasource, sample_data_polars) -> None:
    loaded = sql_datasource.load()
    assert isinstance(loaded, pl.DataFrame)
    assert loaded.equals(sample_data_polars)


def test_load_invalid_backend(sql_datasource) -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        sql_datasource.load(backend="invalid")  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Exists tests
# -----------------------------------------------------------------------------


def test_exists_true_when_table_present(sql_datasource) -> None:
    assert sql_datasource.exists() is True


def test_exists_false_when_table_absent(empty_sql_datasource) -> None:
    assert empty_sql_datasource.exists() is False


# -----------------------------------------------------------------------------
# Connection property
# -----------------------------------------------------------------------------


def test_connection_property(sql_datasource) -> None:
    conn = sql_datasource.connection
    assert conn is not None
    assert "test_table" in conn.list_tables()


# -----------------------------------------------------------------------------
# No save method
# -----------------------------------------------------------------------------


def test_has_no_save_method(sql_datasource) -> None:
    assert not hasattr(sql_datasource, "save")


# -----------------------------------------------------------------------------
# Repr
# -----------------------------------------------------------------------------


def test_repr(sql_datasource) -> None:
    expected = (
        f"SQLDatasource(location={sql_datasource._location!r}, table_name='test_table')"
    )
    assert repr(sql_datasource) == expected
