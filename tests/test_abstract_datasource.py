from __future__ import annotations

import json
from typing import Any

import pytest

from trellis.datasources import AbstractDatasource


class _StubDatasource(AbstractDatasource):
    def load(self) -> str:
        return "ok"

    def exists(self) -> bool:
        return True


class _RichDatasource(AbstractDatasource):
    """Stub that extends describe() with extra fields."""

    def __init__(self, *, location: str, extra: str) -> None:
        super().__init__(location=location)
        self._extra = extra

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "location": self.location,
            "extra": self._extra,
        }

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


def test_describe_default_keys() -> None:
    ds = _StubDatasource(location="/data/a.csv")
    assert ds.describe() == {"type": "_StubDatasource", "location": "/data/a.csv"}


def test_describe_is_json_serializable() -> None:
    ds = _StubDatasource(location="/data/a.csv")
    json.dumps(ds.describe())


def test_describe_subclass_extension() -> None:
    ds = _RichDatasource(location="/data/a.csv", extra="hello")
    assert ds.describe() == {
        "type": "_RichDatasource",
        "location": "/data/a.csv",
        "extra": "hello",
    }


def test_repr_derived_from_describe() -> None:
    ds = _RichDatasource(location="/data/a.csv", extra="hello")
    assert repr(ds) == "_RichDatasource(location='/data/a.csv', extra='hello')"
