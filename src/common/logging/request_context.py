"""Request context utilities for logging.

Provides a context variable to propagate a request_id across call boundaries,
plus convenience helpers and a context manager for scoped usage.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar
from typing import Optional

_request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Return the current request_id from context, if any."""
    return _request_id_var.get()


def set_request_id(request_id: Optional[str]) -> None:
    """Set the request_id in the current context. Pass None to clear."""
    if request_id is None:
        clear_request_id()
        return
    _request_id_var.set(request_id)


def clear_request_id() -> None:
    """Clear the current request_id from context."""
    _request_id_var.set(None)


@contextlib.contextmanager
def request_context(request_id: str) -> Iterator[None]:
    """Set request_id for the duration of the context manager scope."""
    token = _request_id_var.set(request_id)
    try:
        yield
    finally:
        _request_id_var.reset(token)
