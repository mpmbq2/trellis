from __future__ import annotations

import ibis
import pandas as pd
import polars as pl
import pytest

from trellis.datasets import SQLDataset


@pytest.fixture
def sql_dataset(tmp_path):
    """Create a SQLDataset backed by SQLite."""
    db_path = str(tmp_path / "test.db")
    return SQLDataset(location=f"sqlite:///{db_path}", table_name="test_table")


@pytest.fixture
def sample_data_polars():
    return pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


@pytest.fixture
def sample_data_pandas():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


# -----------------------------------------------------------------------------
# Load tests
# -----------------------------------------------------------------------------


def test_save_polars_load_polars(sql_dataset, sample_data_polars):
    sql_dataset.save(sample_data_polars)
    loaded = sql_dataset.load(backend="polars")
    assert isinstance(loaded, pl.DataFrame)
    assert loaded.equals(sample_data_polars)


def test_save_polars_load_pandas(sql_dataset, sample_data_polars):
    sql_dataset.save(sample_data_polars)
    loaded = sql_dataset.load(backend="pandas")
    assert isinstance(loaded, pd.DataFrame)
    # Compare values by converting to lists
    assert list(loaded["a"]) == [1, 2, 3]
    assert list(loaded["b"]) == ["x", "y", "z"]


def test_save_pandas_load_pandas(sql_dataset, sample_data_pandas):
    sql_dataset.save(sample_data_pandas)
    loaded = sql_dataset.load(backend="pandas")
    assert isinstance(loaded, pd.DataFrame)
    # Compare values by converting to lists
    assert list(loaded["a"]) == list(sample_data_pandas["a"])
    assert list(loaded["b"]) == list(sample_data_pandas["b"])


def test_save_pandas_load_polars(sql_dataset, sample_data_pandas):
    sql_dataset.save(sample_data_pandas)
    loaded = sql_dataset.load(backend="polars")
    assert isinstance(loaded, pl.DataFrame)


def test_save_polars_load_ibis(sql_dataset, sample_data_polars):
    sql_dataset.save(sample_data_polars)
    loaded = sql_dataset.load(backend="ibis")
    assert isinstance(loaded, ibis.expr.types.Table)
    # Verify we can execute the table
    materialized = loaded.execute()
    assert len(materialized) == 3


def test_save_ibis_table_load_polars(sql_dataset, sample_data_polars, tmp_path):
    """Test saving an ibis Table expression."""
    # Create a separate connection with a table
    db_path = str(tmp_path / "source.db")
    source_conn = ibis.connect(f"sqlite:///{db_path}")
    source_conn.create_table("source_table", obj=sample_data_polars)

    # Get the table expression
    table_expr = source_conn.table("source_table")

    # Save the ibis table to our SQLDataset
    sql_dataset.save(table_expr)

    # Verify it loaded correctly
    loaded = sql_dataset.load(backend="polars")
    assert isinstance(loaded, pl.DataFrame)
    assert loaded.equals(sample_data_polars)


# -----------------------------------------------------------------------------
# Save tests
# -----------------------------------------------------------------------------


def test_save_lazyframe(sql_dataset):
    data = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    sql_dataset.save(data)
    loaded = sql_dataset.load(backend="polars")
    assert isinstance(loaded, pl.DataFrame)
    expected = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert loaded.equals(expected)


# -----------------------------------------------------------------------------
# if_exists tests
# -----------------------------------------------------------------------------


def test_if_exists_fail_when_table_exists(sql_dataset, sample_data_polars):
    sql_dataset.save(sample_data_polars)
    with pytest.raises(FileExistsError, match="already exists"):
        sql_dataset.save(sample_data_polars, if_exists="fail")


def test_if_exists_replace_overwrites(sql_dataset, sample_data_polars):
    sql_dataset.save(sample_data_polars)
    new_data = pl.DataFrame({"a": [4, 5], "b": ["w", "v"]})
    sql_dataset.save(new_data, if_exists="replace")
    loaded = sql_dataset.load(backend="polars")
    assert loaded.shape == (2, 2)
    expected = pl.DataFrame({"a": [4, 5], "b": ["w", "v"]})
    assert loaded.equals(expected)


def test_if_exists_append_adds_rows(sql_dataset, sample_data_polars):
    sql_dataset.save(sample_data_polars)
    more_data = pl.DataFrame({"a": [4, 5], "b": ["w", "v"]})
    sql_dataset.save(more_data, if_exists="append")
    loaded = sql_dataset.load(backend="polars")
    assert loaded.shape == (5, 2)


def test_if_exists_append_creates_table_if_not_exists(sql_dataset, sample_data_polars):
    # Table doesn't exist yet, but if_exists="append" should still work
    sql_dataset.save(sample_data_polars, if_exists="append")
    loaded = sql_dataset.load(backend="polars")
    assert loaded.shape == (3, 2)


# -----------------------------------------------------------------------------
# Existence tests
# -----------------------------------------------------------------------------


def test_exists_false_before_save(sql_dataset):
    assert sql_dataset.exists() is False


def test_exists_true_after_save(sql_dataset, sample_data_polars):
    sql_dataset.save(sample_data_polars)
    assert sql_dataset.exists() is True


# -----------------------------------------------------------------------------
# Connection property
# -----------------------------------------------------------------------------


def test_connection_property(sql_dataset):
    """Test that we can access the ibis connection directly."""
    conn = sql_dataset.connection
    assert conn is not None
    # Verify it's a real connection by creating a table
    conn.create_table("conn_test", schema=ibis.schema({"x": "int64"}))
    assert "conn_test" in conn.list_tables()


# -----------------------------------------------------------------------------
# Repr
# -----------------------------------------------------------------------------


def test_repr(sql_dataset):
    expected = (
        f"SQLDataset(location={sql_dataset._location!r}, table_name='test_table')"
    )
    assert repr(sql_dataset) == expected


# -----------------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------------


def test_save_unsupported_type(sql_dataset):
    with pytest.raises(TypeError, match="Unsupported data type"):
        sql_dataset.save("not a dataframe")


def test_invalid_backend(sql_dataset, sample_data_polars):
    sql_dataset.save(sample_data_polars)
    with pytest.raises(ValueError, match="Unsupported backend"):
        sql_dataset.load(backend="invalid")


def test_invalid_if_exists(sql_dataset, sample_data_polars):
    with pytest.raises(ValueError, match="Invalid if_exists value"):
        sql_dataset.save(sample_data_polars, if_exists="invalid")
