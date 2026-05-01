from __future__ import annotations

import json
from typing import Any

import pytest

from trellis.datasets import AbstractDataset


class _StubDataset(AbstractDataset):
    def load(self) -> str:
        return "ok"

    def save(self, data: object) -> None:
        return None

    def exists(self) -> bool:
        return True


class _RichDataset(AbstractDataset):
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

    def save(self, data: object) -> None:
        return None

    def exists(self) -> bool:
        return True


def test_cannot_instantiate_abstract_dataset() -> None:
    with pytest.raises(TypeError):
        AbstractDataset(location="x")  # type: ignore[abstract]


def test_concrete_subclass_load_save_exists() -> None:
    ds = _StubDataset(location="/tmp/x")
    assert ds.load() == "ok"
    assert ds.exists() is True
    ds.save(None)
    assert ds.location == "/tmp/x"


def test_repr_includes_class_and_location() -> None:
    ds = _StubDataset(location="/data/a.csv")
    assert repr(ds) == "_StubDataset(location='/data/a.csv')"


def test_describe_default_keys() -> None:
    ds = _StubDataset(location="/data/a.csv")
    assert ds.describe() == {"type": "_StubDataset", "location": "/data/a.csv"}


def test_describe_is_json_serializable() -> None:
    ds = _StubDataset(location="/data/a.csv")
    json.dumps(ds.describe())


def test_describe_subclass_extension() -> None:
    ds = _RichDataset(location="/data/a.csv", extra="hello")
    assert ds.describe() == {
        "type": "_RichDataset",
        "location": "/data/a.csv",
        "extra": "hello",
    }


def test_repr_derived_from_describe() -> None:
    ds = _RichDataset(location="/data/a.csv", extra="hello")
    assert repr(ds) == "_RichDataset(location='/data/a.csv', extra='hello')"


def test_repr_does_not_mutate_describe() -> None:
    ds = _RichDataset(location="/data/a.csv", extra="hello")
    before = ds.describe()
    repr(ds)
    after = ds.describe()
    assert before == after
    assert "type" in after
