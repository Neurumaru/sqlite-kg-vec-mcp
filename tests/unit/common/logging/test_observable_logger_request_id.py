from __future__ import annotations

import io
import json
import logging
import unittest

from src.common.logging.config import LoggingConfig, LogLevel, configure_structured_logging
from src.common.logging.request_context import request_context
from src.common.observability.logger import get_observable_logger


class TestObservableLoggerRequestId(unittest.TestCase):
    def setUp(self):
        # Configure structlog to write JSON to a buffer for assertion
        configure_structured_logging(
            LoggingConfig(level=LogLevel.INFO, format="json", output="console")
        )

    def test_request_id_included(self):
        # Capture logging output
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        root = logging.getLogger()
        # Remove existing handlers to avoid duplication
        for h in list(root.handlers):
            root.removeHandler(h)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)

        logger = get_observable_logger("unit_test_component", "test")

        with request_context("req-xyz"):
            logger.info("unit_event", extra="value")

        output = stream.getvalue().strip()
        self.assertTrue(output)
        payload = json.loads(output)
        self.assertEqual(payload.get("request_id"), "req-xyz")
        self.assertEqual(payload.get("event"), "unit_event")


if __name__ == "__main__":
    unittest.main()
