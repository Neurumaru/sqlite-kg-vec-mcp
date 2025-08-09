from __future__ import annotations

import contextvars
import datetime as _dt
import json
import sys
from contextlib import contextmanager
from typing import Optional, TextIO

# Import the standard library logging under an alias to avoid any naming collision
# with this POC module file name.
import logging as _logging


# Context variable for request id propagation
_request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id",
    default=None,
)


def get_request_id() -> Optional[str]:
    return _request_id_var.get()


def set_request_id(request_id: Optional[str]) -> None:
    if request_id is None:
        clear_request_id()
        return
    _request_id_var.set(request_id)


def clear_request_id() -> None:
    # Reset to None; safe even if not set
    _request_id_var.set(None)


@contextmanager
def request_context(request_id: str):
    token = _request_id_var.set(request_id)
    try:
        yield
    finally:
        _request_id_var.reset(token)


class _ContextRequestIdFilter(_logging.Filter):
    def filter(self, record: _logging.LogRecord) -> bool:  # noqa: D401
        """Attach request_id from contextvars to the log record."""
        record.request_id = _request_id_var.get()
        return True


def _iso8601(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()


class JsonFormatter(_logging.Formatter):
    def format(self, record: _logging.LogRecord) -> str:  # noqa: D401
        """Render log record as JSON with optional request_id."""
        payload = {
            "timestamp": _iso8601(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        req_id = getattr(record, "request_id", None)
        if req_id:
            payload["request_id"] = req_id
        return json.dumps(payload, ensure_ascii=False)


class LogfmtFormatter(_logging.Formatter):
    @staticmethod
    def _escape(value: object) -> str:
        if value is None:
            return "null"
        s = str(value)
        if any(ch.isspace() for ch in s) or "=" in s or '"' in s:
            s = s.replace('"', '\\"')
            return f'"{s}"'
        return s

    def format(self, record: _logging.LogRecord) -> str:  # noqa: D401
        """Render log record in logfmt k=v pairs with optional request_id."""
        parts = [
            f"timestamp={self._escape(_iso8601(record.created))}",
            f"level={self._escape(record.levelname)}",
            f"logger={self._escape(record.name)}",
            f"msg={self._escape(record.getMessage())}",
        ]
        req_id = getattr(record, "request_id", None)
        if req_id:
            parts.append(f"request_id={self._escape(req_id)}")
        return " ".join(parts)


def get_logger(
    name: str,
    *,
    fmt: str = "json",
    level: str = "INFO",
    stream: TextIO = sys.stdout,
) -> _logging.Logger:
    """Create a logger instance with JSON or logfmt formatter.

    - Includes a filter that attaches `request_id` from contextvars
    - Does not propagate to root to avoid duplicate logs
    - Replaces existing handlers on the named logger for deterministic output
    """
    logger = _logging.getLogger(name)
    logger.setLevel(getattr(_logging, level.upper(), _logging.INFO))
    logger.propagate = False

    # Remove existing handlers to avoid duplication during tests
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = _logging.StreamHandler(stream)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    elif fmt == "logfmt":
        handler.setFormatter(LogfmtFormatter())
    else:
        raise ValueError("fmt must be 'json' or 'logfmt'")

    handler.addFilter(_ContextRequestIdFilter())
    logger.addHandler(handler)
    return logger



