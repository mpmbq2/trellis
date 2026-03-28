from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest

from trellis.datasets import AbstractDataset, ParquetDataset


class TestParquetDatasetInit:
    """Test ParquetDataset initialization."""

    def test_basic_init(self) -> None:
        """Test basic initialization with path."""
        ds = ParquetDataset(path="/tmp/test.parquet")

        assert ds.path == "/tmp/test.parquet"
        assert ds.compression == "snappy"
        assert ds.row_group_size == 1048576
        assert ds.use_threads is True

    def test_init_with_custom_options(self) -> None:
        """Test initialization with custom parquet options."""
        ds = ParquetDataset(
            path="/tmp/test.parquet",
            compression="gzip",
            row_group_size=10000,
            use_threads=False,
        )

        assert ds.path == "/tmp/test.parquet"
        assert ds.compression == "gzip"
        assert ds.row_group_size == 10000
        assert ds.use_threads is False

    def test_init_with_metadata(self) -> None:
        """Test initialization with metadata."""
        ds = ParquetDataset(
            path="/tmp/test.parquet",
            custom_key="custom_value",
            another_key=42,
        )

        metadata = ds.get_metadata()
        assert metadata["custom_key"] == "custom_value"
        assert metadata["another_key"] == 42

    def test_repr(self) -> None:
        """Test string representation."""
        ds = ParquetDataset(path="/tmp/test.parquet")
        repr_str = repr(ds)

        assert "ParquetDataset" in repr_str
        assert "/tmp/test.parquet" in repr_str


class TestParquetDatasetPolars:
    """Test ParquetDataset with polars DataFrames."""

    def test_save_and_load_polars(self) -> None:
        """Test saving and loading with polars DataFrames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            df_original = pl.DataFrame(
                {
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [25, 30, 35],
                    "city": ["NYC", "LA", "Chicago"],
                }
            )

            # Save
            ds = ParquetDataset(path=path)
            ds.save(df_original)

            # Load as polars (default)
            df_loaded = ds.load()

            assert isinstance(df_loaded, pl.DataFrame)
            assert df_loaded.shape == df_original.shape
            assert list(df_loaded.columns) == list(df_original.columns)

    def test_save_polars_load_as_pandas(self) -> None:
        """Test saving polars DataFrame and loading as pandas."""
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            df_original = pl.DataFrame(
                {
                    "col1": [1, 2, 3],
                    "col2": ["a", "b", "c"],
                }
            )

            ds = ParquetDataset(path=path)
            ds.save(df_original)

            # Load as pandas
            df_loaded = ds.load(as_type="pandas")

            assert isinstance(df_loaded, pd.DataFrame)
            assert df_loaded.shape == (3, 2)
            assert list(df_loaded.columns) == ["col1", "col2"]

    def test_save_with_custom_compression_polars(self) -> None:
        """Test saving with gzip compression using polars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            df_original = pl.DataFrame(
                {
                    "col1": [1, 2, 3],
                    "col2": ["a", "b", "c"],
                }
            )

            ds = ParquetDataset(path=path, compression="gzip")
            ds.save(df_original)

            # Load and verify
            df_loaded = ds.load()
            assert df_loaded.shape == df_original.shape


class TestParquetDatasetPandas:
    """Test ParquetDataset with pandas DataFrames."""

    def test_save_and_load_pandas(self) -> None:
        """Test saving and loading with pandas DataFrames."""
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            df_original = pd.DataFrame(
                {
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [25, 30, 35],
                    "city": ["NYC", "LA", "Chicago"],
                }
            )

            # Save
            ds = ParquetDataset(path=path)
            ds.save(df_original)

            # Load as pandas
            df_loaded = ds.load(as_type="pandas")

            assert isinstance(df_loaded, pd.DataFrame)
            assert df_loaded.shape == df_original.shape
            assert list(df_loaded.columns) == list(df_original.columns)

    def test_save_pandas_load_as_polars(self) -> None:
        """Test saving pandas DataFrame and loading as polars."""
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            df_original = pd.DataFrame(
                {
                    "col1": [1, 2, 3],
                    "col2": ["a", "b", "c"],
                }
            )

            ds = ParquetDataset(path=path)
            ds.save(df_original)

            # Load as polars (default)
            df_loaded = ds.load()

            assert isinstance(df_loaded, pl.DataFrame)
            assert df_loaded.shape == (3, 2)


class TestParquetDatasetIO:
    """Test ParquetDataset I/O operations."""

    def test_exists_returns_true_for_existing_file(self) -> None:
        """Test exists() returns True for existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            # Create a file
            df = pl.DataFrame({"col": [1, 2, 3]})
            ds = ParquetDataset(path=path)
            ds.save(df)

            assert ds.exists() is True

    def test_exists_returns_false_for_nonexistent_file(self) -> None:
        """Test exists() returns False for non-existent file."""
        ds = ParquetDataset(path="/nonexistent/path/file.parquet")
        assert ds.exists() is False

    def test_exists_returns_false_with_no_path(self) -> None:
        """Test exists() returns False when no path is set."""
        ds = ParquetDataset(path="test.parquet")
        ds._path = None
        assert ds.exists() is False

    def test_load_nonexistent_file_raises_error(self) -> None:
        """Test loading a non-existent file raises FileNotFoundError."""
        ds = ParquetDataset(path="/nonexistent/path/file.parquet")

        with pytest.raises(FileNotFoundError):
            ds.load()

    def test_load_invalid_as_type_raises_error(self) -> None:
        """Test loading with invalid as_type raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            df = pl.DataFrame({"col": [1, 2, 3]})
            ds = ParquetDataset(path=path)
            ds.save(df)

            with pytest.raises(ValueError, match="as_type must be"):
                ds.load(as_type="invalid")

    def test_save_wrong_type_raises_error(self) -> None:
        """Test saving non-DataFrame raises TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")
            ds = ParquetDataset(path=path)

            with pytest.raises(TypeError, match="polars or pandas DataFrame"):
                ds.save({"key": "value"})

    def test_save_without_path_raises_error(self) -> None:
        """Test saving without a path raises ValueError."""
        ds = ParquetDataset(path="test.parquet")
        ds._path = None

        df = pl.DataFrame({"col": [1, 2, 3]})

        with pytest.raises(ValueError, match="Path must be specified"):
            ds.save(df)

    def test_load_without_path_raises_error(self) -> None:
        """Test loading without a path raises ValueError."""
        ds = ParquetDataset(path="test.parquet")
        ds._path = None

        with pytest.raises(ValueError, match="Path must be specified"):
            ds.load()


class TestParquetDatasetFactory:
    """Test ParquetDataset factory pattern."""

    def test_create_via_factory(self) -> None:
        """Test creating ParquetDataset via AbstractDataset.create()."""
        ds = AbstractDataset.create("parquet", path="/tmp/test.parquet")

        assert isinstance(ds, ParquetDataset)
        assert ds.path == "/tmp/test.parquet"

    def test_create_via_factory_with_options(self) -> None:
        """Test factory creation with options."""
        ds = AbstractDataset.create(
            "parquet",
            path="/tmp/test.parquet",
            compression="zstd",
            row_group_size=5000,
        )

        assert isinstance(ds, ParquetDataset)
        assert ds.compression == "zstd"
        assert ds.row_group_size == 5000

    def test_registered_in_list_types(self) -> None:
        """Test that 'parquet' appears in registered types."""
        types = AbstractDataset.list_types()
        assert "parquet" in types


class TestParquetDatasetDataTypes:
    """Test ParquetDataset with various data types."""

    def test_save_and_load_numeric_types_polars(self) -> None:
        """Test handling of numeric data types with polars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            df_original = pl.DataFrame(
                {
                    "int_col": [1, 2, 3],
                    "float_col": [1.5, 2.5, 3.5],
                }
            )

            ds = ParquetDataset(path=path)
            ds.save(df_original)
            df_loaded = ds.load()

            assert df_loaded.shape == (3, 2)
            assert list(df_loaded.columns) == ["int_col", "float_col"]

    def test_save_and_load_null_values_polars(self) -> None:
        """Test handling of null values with polars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            df_original = pl.DataFrame(
                {
                    "col1": [1, None, 3],
                    "col2": ["a", "b", None],
                }
            )

            ds = ParquetDataset(path=path)
            ds.save(df_original)
            df_loaded = ds.load()

            assert df_loaded.shape == (3, 2)

    def test_save_and_load_empty_dataframe_polars(self) -> None:
        """Test saving and loading an empty DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            df_original = pl.DataFrame(
                {
                    "col1": [],
                    "col2": [],
                }
            )

            ds = ParquetDataset(path=path)
            ds.save(df_original)
            df_loaded = ds.load()

            assert df_loaded.shape == (0, 2)


class TestParquetDatasetRoundTrip:
    """Test round-trip compatibility between polars and pandas."""

    def test_polars_to_pandas_roundtrip(self) -> None:
        """Test round-trip from polars to pandas."""
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            # Create polars DataFrame
            df_polars = pl.DataFrame(
                {
                    "a": [1, 2, 3],
                    "b": ["x", "y", "z"],
                }
            )

            # Save with polars
            ds = ParquetDataset(path=path)
            ds.save(df_polars)

            # Load as pandas
            df_pandas = ds.load(as_type="pandas")

            assert isinstance(df_pandas, pd.DataFrame)
            assert df_pandas.shape == (3, 2)

    def test_pandas_to_polars_roundtrip(self) -> None:
        """Test round-trip from pandas to polars."""
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            # Create pandas DataFrame
            df_pandas = pd.DataFrame(
                {
                    "a": [1, 2, 3],
                    "b": ["x", "y", "z"],
                }
            )

            # Save with pandas
            ds = ParquetDataset(path=path)
            ds.save(df_pandas)

            # Load as polars
            df_polars = ds.load()

            assert isinstance(df_polars, pl.DataFrame)
            assert df_polars.shape == (3, 2)


class TestParquetDatasetCompression:
    """Test ParquetDataset with different compression codecs."""

    def test_compression_snappy(self) -> None:
        """Test saving with snappy compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            df = pl.DataFrame({"col": [1, 2, 3]})
            ds = ParquetDataset(path=path, compression="snappy")
            ds.save(df)

            assert ds.exists()
            df_loaded = ds.load()
            assert df_loaded.shape == (3, 1)

    def test_compression_zstd(self) -> None:
        """Test saving with zstd compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            df = pl.DataFrame({"col": [1, 2, 3]})
            ds = ParquetDataset(path=path, compression="zstd")
            ds.save(df)

            assert ds.exists()
            df_loaded = ds.load()
            assert df_loaded.shape == (3, 1)


class TestParquetDatasetLazyLoading:
    """Test ParquetDataset lazy loading functionality."""

    def test_lazy_load_returns_lazyframe(self) -> None:
        """Test that lazy_load() returns a polars LazyFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            df_original = pl.DataFrame(
                {
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [25, 30, 35],
                }
            )

            ds = ParquetDataset(path=path)
            ds.save(df_original)

            lf = ds.lazy_load()

            assert isinstance(lf, pl.LazyFrame)

    def test_lazy_load_collect_produces_correct_data(self) -> None:
        """Test that collecting lazy load produces correct DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            df_original = pl.DataFrame(
                {
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [25, 30, 35],
                }
            )

            ds = ParquetDataset(path=path)
            ds.save(df_original)

            lf = ds.lazy_load()
            df_loaded = lf.collect()

            assert isinstance(df_loaded, pl.DataFrame)
            assert df_loaded.shape == df_original.shape
            assert list(df_loaded.columns) == list(df_original.columns)

    def test_lazy_load_with_filter(self) -> None:
        """Test that lazy load supports filter operations before collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            df_original = pl.DataFrame(
                {
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [25, 30, 35],
                }
            )

            ds = ParquetDataset(path=path)
            ds.save(df_original)

            df_filtered = ds.lazy_load().filter(pl.col("age") > 25).collect()

            assert df_filtered.shape[0] == 2
            assert all(df_filtered["age"] > 25)

    def test_load_lazy_true_returns_lazyframe(self) -> None:
        """Test that load(lazy=True) returns a polars LazyFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            df_original = pl.DataFrame(
                {
                    "col1": [1, 2, 3],
                    "col2": ["a", "b", "c"],
                }
            )

            ds = ParquetDataset(path=path)
            ds.save(df_original)

            result = ds.load(lazy=True)

            assert isinstance(result, pl.LazyFrame)

    def test_load_lazy_true_collect_equals_eager(self) -> None:
        """Test that lazy load collection matches eager load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")

            df_original = pl.DataFrame(
                {
                    "a": [10, 20, 30],
                    "b": ["x", "y", "z"],
                }
            )

            ds = ParquetDataset(path=path)
            ds.save(df_original)

            df_eager = ds.load()
            df_lazy = ds.load(lazy=True).collect()

            assert df_eager.shape == df_lazy.shape
            assert list(df_eager.columns) == list(df_lazy.columns)

    def test_lazy_load_without_path_raises_error(self) -> None:
        """Test lazy_load without path raises ValueError."""
        ds = ParquetDataset(path="test.parquet")
        ds._path = None

        with pytest.raises(ValueError, match="Path must be specified"):
            ds.lazy_load()

    def test_lazy_load_nonexistent_file_raises_error(self) -> None:
        """Test lazy_load on non-existent file raises FileNotFoundError."""
        ds = ParquetDataset(path="/nonexistent/path/file.parquet")

        with pytest.raises(FileNotFoundError):
            ds.lazy_load()
