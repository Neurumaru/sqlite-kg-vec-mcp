from __future__ import annotations

import io
import json
import importlib.util
from pathlib import Path

# Dynamically load the module from file to comply with POC rule (no __init__.py)
_MODULE_PATH = Path(__file__).with_name("logging.py")
_SPEC = importlib.util.spec_from_file_location("logging_poc", _MODULE_PATH)
assert _SPEC and _SPEC.loader
logging_poc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(logging_poc)  # type: ignore[arg-type]

get_logger = logging_poc.get_logger
request_context = logging_poc.request_context
set_request_id = logging_poc.set_request_id
clear_request_id = logging_poc.clear_request_id


def test_json_formatter_without_context():
    stream = io.StringIO()
    logger = get_logger("test.json.noctx", fmt="json", stream=stream)
    logger.info("hello")
    out = stream.getvalue().strip()
    payload = json.loads(out)
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert "request_id" not in payload


def test_json_formatter_with_request_context():
    stream = io.StringIO()
    logger = get_logger("test.json.ctx", fmt="json", stream=stream)
    with request_context("req-123"):
        logger.info("hi")
    out = stream.getvalue().strip()
    payload = json.loads(out)
    assert payload["message"] == "hi"
    assert payload["request_id"] == "req-123"


def test_logfmt_formatter_without_context():
    stream = io.StringIO()
    logger = get_logger("test.logfmt.noctx", fmt="logfmt", stream=stream)
    logger.info("hello world")
    out = stream.getvalue().strip()
    # Basic shape validations
    assert "level=INFO" in out
    assert "logger=test.logfmt.noctx" in out
    assert "msg=\"hello world\"" in out
    assert "request_id=" not in out


def test_logfmt_formatter_with_manual_set_and_clear():
    stream = io.StringIO()
    logger = get_logger("test.logfmt.ctx", fmt="logfmt", stream=stream)
    set_request_id("abc")
    try:
        logger.warning("warn msg")
    finally:
        clear_request_id()
    out = stream.getvalue().strip()
    assert "level=WARNING" in out
    assert "request_id=abc" in out


