from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest

from trellis.datasets import AbstractDataset, CSVDataset


class TestCSVDatasetInit:
    """Test CSVDataset initialization."""

    def test_basic_init(self) -> None:
        """Test basic initialization with path."""
        ds = CSVDataset(path="/tmp/test.csv")

        assert ds.path == "/tmp/test.csv"
        assert ds.delimiter == ","
        assert ds.has_header is True
        assert ds.encoding == "utf-8"

    def test_init_with_custom_options(self) -> None:
        """Test initialization with custom CSV options."""
        ds = CSVDataset(
            path="/tmp/test.csv",
            delimiter=";",
            has_header=False,
            encoding="latin1",
        )

        assert ds.path == "/tmp/test.csv"
        assert ds.delimiter == ";"
        assert ds.has_header is False
        assert ds.encoding == "latin1"

    def test_init_with_metadata(self) -> None:
        """Test initialization with metadata."""
        ds = CSVDataset(
            path="/tmp/test.csv",
            custom_key="custom_value",
            another_key=42,
        )

        metadata = ds.get_metadata()
        assert metadata["custom_key"] == "custom_value"
        assert metadata["another_key"] == 42

    def test_repr(self) -> None:
        """Test string representation."""
        ds = CSVDataset(path="/tmp/test.csv")
        repr_str = repr(ds)

        assert "CSVDataset" in repr_str
        assert "/tmp/test.csv" in repr_str


class TestCSVDatasetPolars:
    """Test CSVDataset with polars DataFrames."""

    def test_save_and_load_polars(self) -> None:
        """Test saving and loading with polars DataFrames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            df_original = pl.DataFrame(
                {
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [25, 30, 35],
                    "city": ["NYC", "LA", "Chicago"],
                }
            )

            # Save
            ds = CSVDataset(path=path)
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
            path = str(Path(tmpdir) / "test.csv")

            df_original = pl.DataFrame(
                {
                    "col1": [1, 2, 3],
                    "col2": ["a", "b", "c"],
                }
            )

            ds = CSVDataset(path=path)
            ds.save(df_original)

            # Load as pandas
            df_loaded = ds.load(as_type="pandas")

            assert isinstance(df_loaded, pd.DataFrame)
            assert df_loaded.shape == (3, 2)
            assert list(df_loaded.columns) == ["col1", "col2"]

    def test_save_with_custom_delimiter_polars(self) -> None:
        """Test saving with semicolon delimiter using polars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            df_original = pl.DataFrame(
                {
                    "col1": [1, 2, 3],
                    "col2": ["a", "b", "c"],
                }
            )

            ds = CSVDataset(path=path, delimiter=";")
            ds.save(df_original)

            # Load with same delimiter
            df_loaded = ds.load()
            assert df_loaded.shape == df_original.shape


class TestCSVDatasetPandas:
    """Test CSVDataset with pandas DataFrames."""

    def test_save_and_load_pandas(self) -> None:
        """Test saving and loading with pandas DataFrames."""
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            df_original = pd.DataFrame(
                {
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [25, 30, 35],
                    "city": ["NYC", "LA", "Chicago"],
                }
            )

            # Save
            ds = CSVDataset(path=path)
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
            path = str(Path(tmpdir) / "test.csv")

            df_original = pd.DataFrame(
                {
                    "col1": [1, 2, 3],
                    "col2": ["a", "b", "c"],
                }
            )

            ds = CSVDataset(path=path)
            ds.save(df_original)

            # Load as polars (default)
            df_loaded = ds.load()

            assert isinstance(df_loaded, pl.DataFrame)
            assert df_loaded.shape == (3, 2)

    def test_save_with_custom_delimiter_pandas(self) -> None:
        """Test saving with semicolon delimiter using pandas."""
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            df_original = pd.DataFrame(
                {
                    "col1": [1, 2, 3],
                    "col2": ["a", "b", "c"],
                }
            )

            ds = CSVDataset(path=path, delimiter=";")
            ds.save(df_original)

            # Load with same delimiter
            df_loaded = ds.load(as_type="pandas")
            assert df_loaded.shape == df_original.shape


class TestCSVDatasetIO:
    """Test CSVDataset I/O operations."""

    def test_load_without_header(self) -> None:
        """Test loading a CSV without header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            # Write a CSV without header manually
            with open(path, "w") as f:
                f.write("1,Alice,25\n")
                f.write("2,Bob,30\n")
                f.write("3,Charlie,35\n")

            ds = CSVDataset(path=path, has_header=False)
            df_loaded = ds.load()

            assert df_loaded.shape == (3, 3)
            assert len(df_loaded.columns) == 3
            # Polars auto-generates column names like column_1, column_2, column_3
            assert "column_1" in df_loaded.columns

    def test_exists_returns_true_for_existing_file(self) -> None:
        """Test exists() returns True for existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            # Create a file
            with open(path, "w") as f:
                f.write("col1,col2\n1,2\n")

            ds = CSVDataset(path=path)
            assert ds.exists() is True

    def test_exists_returns_false_for_nonexistent_file(self) -> None:
        """Test exists() returns False for non-existent file."""
        ds = CSVDataset(path="/nonexistent/path/file.csv")
        assert ds.exists() is False

    def test_exists_returns_false_with_no_path(self) -> None:
        """Test exists() returns False when no path is set."""
        ds = CSVDataset(path="test.csv")
        ds._path = None
        assert ds.exists() is False

    def test_load_nonexistent_file_raises_error(self) -> None:
        """Test loading a non-existent file raises FileNotFoundError."""
        ds = CSVDataset(path="/nonexistent/path/file.csv")

        with pytest.raises(FileNotFoundError):
            ds.load()

    def test_load_invalid_as_type_raises_error(self) -> None:
        """Test loading with invalid as_type raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            with open(path, "w") as f:
                f.write("col1,col2\n1,2\n")

            ds = CSVDataset(path=path)

            with pytest.raises(ValueError, match="as_type must be"):
                ds.load(as_type="invalid")

    def test_save_wrong_type_raises_error(self) -> None:
        """Test saving non-DataFrame raises TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")
            ds = CSVDataset(path=path)

            with pytest.raises(TypeError, match="polars or pandas DataFrame"):
                ds.save({"key": "value"})

    def test_save_without_path_raises_error(self) -> None:
        """Test saving without a path raises ValueError."""
        ds = CSVDataset(path="test.csv")
        ds._path = None

        df = pl.DataFrame({"col": [1, 2, 3]})

        with pytest.raises(ValueError, match="Path must be specified"):
            ds.save(df)

    def test_load_without_path_raises_error(self) -> None:
        """Test loading without a path raises ValueError."""
        ds = CSVDataset(path="test.csv")
        ds._path = None

        with pytest.raises(ValueError, match="Path must be specified"):
            ds.load()


class TestCSVDatasetQuery:
    """Test CSVDataset.query() with DuckDB."""

    def test_query_returns_polars(self) -> None:
        """Test query returns polars DataFrame by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            df_original = pl.DataFrame(
                {
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [25, 30, 35],
                }
            )
            ds = CSVDataset(path=path)
            ds.save(df_original)

            out = ds.query("SELECT * FROM {dataset} WHERE age > 28")
            assert isinstance(out, pl.DataFrame)
            assert out.shape == (2, 2)
            assert set(out["name"].to_list()) == {"Bob", "Charlie"}

    def test_query_returns_pandas(self) -> None:
        """Test query returns pandas DataFrame when as_type is pandas."""
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            df_original = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
            ds = CSVDataset(path=path)
            ds.save(df_original)

            out = ds.query("SELECT * FROM {dataset} WHERE col1 = 2", as_type="pandas")
            assert isinstance(out, pd.DataFrame)
            assert out.shape == (1, 2)
            assert out.iloc[0]["col2"] == "b"

    def test_query_respects_csv_delimiter(self) -> None:
        """Test read_csv in query uses dataset delimiter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            df_original = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
            ds = CSVDataset(path=path, delimiter=";")
            ds.save(df_original)

            out = ds.query("SELECT * FROM {dataset} WHERE x = 1")
            assert isinstance(out, pl.DataFrame)
            assert out.shape == (1, 2)

    def test_query_requires_from_dataset_placeholder(self) -> None:
        """Test query without FROM {dataset} raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")
            with open(path, "w") as f:
                f.write("a,b\n1,2\n")
            ds = CSVDataset(path=path)

            with pytest.raises(ValueError, match=r"FROM \{dataset\}"):
                ds.query("SELECT * FROM t")

    def test_query_invalid_as_type_raises(self) -> None:
        """Test query with invalid as_type raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")
            with open(path, "w") as f:
                f.write("a,b\n1,2\n")
            ds = CSVDataset(path=path)

            with pytest.raises(ValueError, match="as_type must be"):
                ds.query("SELECT * FROM {dataset}", as_type="spark")

    def test_query_nonexistent_file_raises(self) -> None:
        """Test query on missing file raises FileNotFoundError."""
        ds = CSVDataset(path="/nonexistent/path/file.csv")

        with pytest.raises(FileNotFoundError):
            ds.query("SELECT * FROM {dataset}")

    def test_query_without_path_raises(self) -> None:
        """Test query without path raises ValueError."""
        ds = CSVDataset(path="x.csv")
        ds._path = None

        with pytest.raises(ValueError, match="Path must be specified"):
            ds.query("SELECT * FROM {dataset}")


class TestCSVDatasetFactory:
    """Test CSVDataset factory pattern."""

    def test_create_via_factory(self) -> None:
        """Test creating CSVDataset via AbstractDataset.create()."""
        ds = AbstractDataset.create("csv", path="/tmp/test.csv")

        assert isinstance(ds, CSVDataset)
        assert ds.path == "/tmp/test.csv"

    def test_create_via_factory_with_options(self) -> None:
        """Test factory creation with options."""
        ds = AbstractDataset.create(
            "csv",
            path="/tmp/test.csv",
            delimiter="|",
            has_header=False,
        )

        assert isinstance(ds, CSVDataset)
        assert ds.delimiter == "|"
        assert ds.has_header is False

    def test_registered_in_list_types(self) -> None:
        """Test that 'csv' appears in registered types."""
        types = AbstractDataset.list_types()
        assert "csv" in types


class TestCSVDatasetDataTypes:
    """Test CSVDataset with various data types."""

    def test_save_and_load_numeric_types_polars(self) -> None:
        """Test handling of numeric data types with polars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            df_original = pl.DataFrame(
                {
                    "int_col": [1, 2, 3],
                    "float_col": [1.5, 2.5, 3.5],
                }
            )

            ds = CSVDataset(path=path)
            ds.save(df_original)
            df_loaded = ds.load()

            assert df_loaded.shape == (3, 2)
            assert list(df_loaded.columns) == ["int_col", "float_col"]

    def test_save_and_load_string_data_polars(self) -> None:
        """Test handling of string data with special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            df_original = pl.DataFrame(
                {
                    "text": [
                        "Hello, World!",
                        "Line\nBreak",
                        'Quote"Test',
                    ],
                }
            )

            ds = CSVDataset(path=path)
            ds.save(df_original)
            df_loaded = ds.load()

            assert df_loaded.shape == (3, 1)
            assert list(df_loaded.columns) == ["text"]

    def test_save_and_load_empty_dataframe_polars(self) -> None:
        """Test saving and loading an empty DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            df_original = pl.DataFrame(
                {
                    "col1": [],
                    "col2": [],
                }
            )

            ds = CSVDataset(path=path)
            ds.save(df_original)
            df_loaded = ds.load()

            assert df_loaded.shape == (0, 2)


class TestCSVDatasetRoundTrip:
    """Test round-trip compatibility between polars and pandas."""

    def test_polars_to_pandas_roundtrip(self) -> None:
        """Test round-trip from polars to pandas."""
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            # Create polars DataFrame
            df_polars = pl.DataFrame(
                {
                    "a": [1, 2, 3],
                    "b": ["x", "y", "z"],
                }
            )

            # Save with polars
            ds = CSVDataset(path=path)
            ds.save(df_polars)

            # Load as pandas
            df_pandas = ds.load(as_type="pandas")

            assert isinstance(df_pandas, pd.DataFrame)
            assert df_pandas.shape == (3, 2)

    def test_pandas_to_polars_roundtrip(self) -> None:
        """Test round-trip from pandas to polars."""
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            # Create pandas DataFrame
            df_pandas = pd.DataFrame(
                {
                    "a": [1, 2, 3],
                    "b": ["x", "y", "z"],
                }
            )

            # Save with pandas
            ds = CSVDataset(path=path)
            ds.save(df_pandas)

            # Load as polars
            df_polars = ds.load()

            assert isinstance(df_polars, pl.DataFrame)
            assert df_polars.shape == (3, 2)


class TestCSVDatasetLazyLoading:
    """Test CSVDataset lazy loading functionality."""

    def test_lazy_load_returns_lazyframe(self) -> None:
        """Test that lazy_load() returns a polars LazyFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            df_original = pl.DataFrame(
                {
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [25, 30, 35],
                }
            )

            ds = CSVDataset(path=path)
            ds.save(df_original)

            lf = ds.lazy_load()

            assert isinstance(lf, pl.LazyFrame)

    def test_lazy_load_collect_produces_correct_data(self) -> None:
        """Test that collecting lazy load produces correct DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            df_original = pl.DataFrame(
                {
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [25, 30, 35],
                }
            )

            ds = CSVDataset(path=path)
            ds.save(df_original)

            lf = ds.lazy_load()
            df_loaded = lf.collect()

            assert isinstance(df_loaded, pl.DataFrame)
            assert df_loaded.shape == df_original.shape
            assert list(df_loaded.columns) == list(df_original.columns)

    def test_lazy_load_with_filter(self) -> None:
        """Test that lazy load supports filter operations before collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            df_original = pl.DataFrame(
                {
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [25, 30, 35],
                }
            )

            ds = CSVDataset(path=path)
            ds.save(df_original)

            df_filtered = ds.lazy_load().filter(pl.col("age") > 25).collect()

            assert df_filtered.shape[0] == 2
            assert all(df_filtered["age"] > 25)

    def test_load_lazy_true_returns_lazyframe(self) -> None:
        """Test that load(lazy=True) returns a polars LazyFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            df_original = pl.DataFrame(
                {
                    "col1": [1, 2, 3],
                    "col2": ["a", "b", "c"],
                }
            )

            ds = CSVDataset(path=path)
            ds.save(df_original)

            result = ds.load(lazy=True)

            assert isinstance(result, pl.LazyFrame)

    def test_load_lazy_true_collect_equals_eager(self) -> None:
        """Test that lazy load collection matches eager load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")

            df_original = pl.DataFrame(
                {
                    "a": [10, 20, 30],
                    "b": ["x", "y", "z"],
                }
            )

            ds = CSVDataset(path=path)
            ds.save(df_original)

            df_eager = ds.load()
            df_lazy = ds.load(lazy=True).collect()

            assert df_eager.shape == df_lazy.shape
            assert list(df_eager.columns) == list(df_lazy.columns)

    def test_lazy_load_without_path_raises_error(self) -> None:
        """Test lazy_load without path raises ValueError."""
        ds = CSVDataset(path="test.csv")
        ds._path = None

        with pytest.raises(ValueError, match="Path must be specified"):
            ds.lazy_load()

    def test_lazy_load_nonexistent_file_raises_error(self) -> None:
        """Test lazy_load on non-existent file raises FileNotFoundError."""
        ds = CSVDataset(path="/nonexistent/path/file.csv")

        with pytest.raises(FileNotFoundError):
            ds.lazy_load()
