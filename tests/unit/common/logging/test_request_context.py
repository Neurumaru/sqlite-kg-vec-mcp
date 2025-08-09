from __future__ import annotations

import unittest

from src.common.logging.request_context import (
    clear_request_id,
    get_request_id,
    request_context,
    set_request_id,
)


class TestRequestContext(unittest.TestCase):
    def setUp(self):
        clear_request_id()

    def test_get_set_clear(self):
        self.assertIsNone(get_request_id())
        set_request_id("abc")
        self.assertEqual(get_request_id(), "abc")
        clear_request_id()
        self.assertIsNone(get_request_id())

    def test_context_manager(self):
        self.assertIsNone(get_request_id())
        with request_context("req-1"):
            self.assertEqual(get_request_id(), "req-1")
        self.assertIsNone(get_request_id())


if __name__ == "__main__":
    unittest.main()
