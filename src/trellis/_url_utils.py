from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def redact_url_password(url: str) -> str:
    """Return *url* with any embedded password replaced by ``***``.

    If *url* does not contain a password component, it is returned unchanged.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.password is None:
        return url
    user = parts.username or ""
    host = parts.hostname or ""
    netloc = f"{user}:***@{host}"
    if parts.port is not None:
        netloc += f":{parts.port}"
    return urlunsplit(parts._replace(netloc=netloc))
