from __future__ import annotations

import trellis
from trellis.exceptions import (
    DatasetLoadError,
    DatasetNotFoundError,
    DatasetSaveError,
    TrellisError,
)


def test_trellis_error_is_exception() -> None:
    assert issubclass(TrellisError, Exception)


def test_dataset_not_found_inherits_trellis_and_filenotfound() -> None:
    assert issubclass(DatasetNotFoundError, TrellisError)
    assert issubclass(DatasetNotFoundError, FileNotFoundError)


def test_dataset_load_error_inherits_trellis() -> None:
    assert issubclass(DatasetLoadError, TrellisError)


def test_dataset_save_error_inherits_trellis() -> None:
    assert issubclass(DatasetSaveError, TrellisError)


def test_errors_re_exported_from_top_level() -> None:
    assert trellis.TrellisError is TrellisError
    assert trellis.DatasetNotFoundError is DatasetNotFoundError
    assert trellis.DatasetLoadError is DatasetLoadError
    assert trellis.DatasetSaveError is DatasetSaveError


def test_dataset_not_found_caught_as_filenotfound() -> None:
    """Existing pipeline code using `except FileNotFoundError` must keep working."""
    try:
        raise DatasetNotFoundError("missing")
    except FileNotFoundError as e:
        assert isinstance(e, DatasetNotFoundError)
        assert isinstance(e, TrellisError)
