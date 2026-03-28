from __future__ import annotations

import pytest

from trellis.datasets import AbstractDataset


class _StubDataset(AbstractDataset):
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
