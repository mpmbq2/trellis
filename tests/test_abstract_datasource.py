from __future__ import annotations

import pytest

from trellis.datasources import AbstractDatasource


class _StubDatasource(AbstractDatasource):
    def load(self) -> str:
        return "ok"

    def exists(self) -> bool:
        return True


def test_cannot_instantiate_abstract_datasource() -> None:
    with pytest.raises(TypeError):
        AbstractDatasource(location="x")  # type: ignore[abstract]


def test_concrete_subclass_load_and_exists() -> None:
    ds = _StubDatasource(location="/tmp/x")
    assert ds.load() == "ok"
    assert ds.exists() is True
    assert ds.location == "/tmp/x"


def test_repr_includes_class_and_location() -> None:
    ds = _StubDatasource(location="/data/a.csv")
    assert repr(ds) == "_StubDatasource(location='/data/a.csv')"
