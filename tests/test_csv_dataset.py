from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from trellis.datasets import CSVDataset
from trellis.exceptions import DatasetLoadError, DatasetNotFoundError, DatasetSaveError


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


def test_describe(csv_path):
    ds = CSVDataset(location=csv_path)
    assert ds.describe() == {"type": "CSVDataset", "location": csv_path}


def test_save_unsupported_type(csv_path):
    ds = CSVDataset(location=csv_path)
    with pytest.raises(TypeError, match="Unsupported data type"):
        ds.save("not a dataframe")


def test_load_missing_raises_dataset_not_found(tmp_path):
    ds = CSVDataset(location=str(tmp_path / "missing.csv"))
    with pytest.raises(DatasetNotFoundError) as exc_info:
        ds.load()
    assert exc_info.value.__cause__ is not None


def test_dataset_not_found_is_also_filenotfound(tmp_path):
    """Pipeline code using ``except FileNotFoundError`` keeps working."""
    ds = CSVDataset(location=str(tmp_path / "missing.csv"))
    with pytest.raises(FileNotFoundError):
        ds.load()


def test_load_malformed_raises_dataset_load_error(tmp_path):
    bad_path = tmp_path / "bad.csv"
    # Bytes that polars cannot parse as UTF-8 CSV.
    bad_path.write_bytes(b"\xff\xfe\xfa not,a,csv\n\xc3\x28")
    ds = CSVDataset(location=str(bad_path))
    with pytest.raises(DatasetLoadError) as exc_info:
        ds.load()
    assert exc_info.value.__cause__ is not None


def test_save_to_unwritable_path_raises_dataset_save_error(tmp_path):
    # Path inside a non-existent directory — write fails at the OS layer.
    ds = CSVDataset(location=str(tmp_path / "no_such_dir" / "out.csv"))
    with pytest.raises(DatasetSaveError) as exc_info:
        ds.save(pl.DataFrame({"a": [1]}))
    assert exc_info.value.__cause__ is not None
