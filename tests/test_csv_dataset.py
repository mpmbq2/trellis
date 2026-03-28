from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from trellis.datasets import CSVDataset


@pytest.fixture
def csv_path(tmp_path):
    return str(tmp_path / "test.csv")


def test_save_polars_load_polars(csv_path):
    ds = CSVDataset(location=csv_path)
    data = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    ds.save(data)
    loaded = ds.load(backend="polars")
    assert isinstance(loaded, pl.DataFrame)
    assert loaded.equals(data)


def test_save_polars_load_pandas(csv_path):
    ds = CSVDataset(location=csv_path)
    data = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    ds.save(data)
    loaded = ds.load(backend="pandas")
    assert isinstance(loaded, pd.DataFrame)


def test_save_pandas_load_pandas(csv_path):
    ds = CSVDataset(location=csv_path)
    data = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    ds.save(data)
    loaded = ds.load(backend="pandas")
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.equals(data)


def test_save_pandas_load_polars(csv_path):
    ds = CSVDataset(location=csv_path)
    data = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    ds.save(data)
    loaded = ds.load(backend="polars")
    assert isinstance(loaded, pl.DataFrame)


def test_lazy_load_polars(csv_path):
    ds = CSVDataset(location=csv_path)
    data = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    ds.save(data)
    loaded = ds.load(backend="polars", lazy=True)
    assert isinstance(loaded, pl.LazyFrame)
    assert loaded.collect().equals(data)


def test_save_lazyframe(csv_path):
    ds = CSVDataset(location=csv_path)
    data = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    ds.save(data)
    loaded = ds.load(backend="polars")
    assert isinstance(loaded, pl.DataFrame)
    assert loaded.equals(pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))


def test_exists_false_before_save(csv_path):
    ds = CSVDataset(location=csv_path)
    assert ds.exists() is False


def test_exists_true_after_save(csv_path):
    ds = CSVDataset(location=csv_path)
    data = pl.DataFrame({"a": [1]})
    ds.save(data)
    assert ds.exists() is True


def test_repr(csv_path):
    ds = CSVDataset(location=csv_path)
    assert repr(ds) == f"CSVDataset(location={csv_path!r})"


def test_save_unsupported_type(csv_path):
    ds = CSVDataset(location=csv_path)
    with pytest.raises(TypeError, match="Unsupported data type"):
        ds.save("not a dataframe")
