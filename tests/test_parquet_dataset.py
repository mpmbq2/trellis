from __future__ import annotations

import polars as pl
import pytest

from trellis.datasets import ParquetDataset
from trellis.exceptions import DatasetLoadError, DatasetNotFoundError, DatasetSaveError


def test_load_missing_raises_dataset_not_found(tmp_path):
    ds = ParquetDataset(location=str(tmp_path / "missing.parquet"))
    with pytest.raises(DatasetNotFoundError) as exc_info:
        ds.load()
    assert exc_info.value.__cause__ is not None


def test_dataset_not_found_is_also_filenotfound(tmp_path):
    ds = ParquetDataset(location=str(tmp_path / "missing.parquet"))
    with pytest.raises(FileNotFoundError):
        ds.load()


def test_load_malformed_raises_dataset_load_error(tmp_path):
    bad_path = tmp_path / "bad.parquet"
    bad_path.write_bytes(b"this is not a parquet file")
    ds = ParquetDataset(location=str(bad_path))
    with pytest.raises(DatasetLoadError) as exc_info:
        ds.load()
    assert exc_info.value.__cause__ is not None


def test_save_to_unwritable_path_raises_dataset_save_error(tmp_path):
    ds = ParquetDataset(location=str(tmp_path / "no_such_dir" / "out.parquet"))
    with pytest.raises(DatasetSaveError) as exc_info:
        ds.save(pl.DataFrame({"a": [1]}))
    assert exc_info.value.__cause__ is not None
