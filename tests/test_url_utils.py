from __future__ import annotations

from trellis._url_utils import redact_url_password


def test_redacts_password_in_url_with_user_and_port() -> None:
    assert (
        redact_url_password("postgres://alice:secret@host:5432/db")
        == "postgres://alice:***@host:5432/db"
    )


def test_redacts_password_without_port() -> None:
    assert (
        redact_url_password("postgres://alice:secret@host/db")
        == "postgres://alice:***@host/db"
    )


def test_no_password_returns_unchanged() -> None:
    assert redact_url_password("postgres://alice@host/db") == "postgres://alice@host/db"


def test_no_userinfo_returns_unchanged() -> None:
    assert redact_url_password("sqlite:///path/to.db") == "sqlite:///path/to.db"


def test_local_path_returns_unchanged() -> None:
    assert redact_url_password("/local/path.csv") == "/local/path.csv"


def test_empty_string_returns_unchanged() -> None:
    assert redact_url_password("") == ""
