"""Tests for configuration loader module."""

from __future__ import annotations

import unittest

from src.common.config.loader import (
    CoercionError,
    KeySpec,
    MissingRequiredError,
    load_from_env,
    redact,
)


class TestConfigLoader(unittest.TestCase):
    """Test cases for configuration loader."""

    def test_defaults_and_types(self):
        """Test loading with defaults and type coercion."""
        specs = [
            KeySpec("APP_PORT", int, default=8000),
            KeySpec("DEBUG", bool, default=False),
            KeySpec("SERVICE", str, default="svc"),
            KeySpec("RATIO", float, default=0.5),
        ]
        env = {"APP_PORT": "9000", "DEBUG": "yes", "RATIO": "0.75"}
        cfg = load_from_env(specs, env=env)
        self.assertEqual(cfg["APP_PORT"], 9000)
        self.assertIs(cfg["DEBUG"], True)
        self.assertEqual(cfg["SERVICE"], "svc")
        self.assertAlmostEqual(cfg["RATIO"], 0.75)

    def test_required_missing(self):
        """Test missing required keys raise error."""
        specs = [KeySpec("API_KEY", str, required=True), KeySpec("DB_URL", str, required=True)]
        with self.assertRaises(MissingRequiredError) as ctx:
            load_from_env(specs, env={})
        self.assertEqual(set(ctx.exception.missing_keys), {"API_KEY", "DB_URL"})

    def test_bool_variants(self):
        """Test various boolean value formats."""
        specs = [KeySpec("FLAG", bool, required=True)]
        for v in ["1", "true", "t", "yes", "y", "on", "TRUE", "On"]:
            cfg = load_from_env(specs, env={"FLAG": v})
            self.assertIs(cfg["FLAG"], True)
        for v in ["0", "false", "f", "no", "n", "off", "FALSE", "Off"]:
            cfg = load_from_env(specs, env={"FLAG": v})
            self.assertIs(cfg["FLAG"], False)

    def test_redact(self):
        """Test secret value redaction."""
        specs = [
            KeySpec("API_KEY", str, required=True, secret=True),
            KeySpec("PUBLIC_URL", str, default="http://example"),
        ]
        cfg = load_from_env(specs, env={"API_KEY": "supersecret"})
        safe = redact(cfg, specs)
        self.assertEqual(safe["API_KEY"], "***")
        self.assertEqual(safe["PUBLIC_URL"], "http://example")

    def test_invalid_int(self):
        """Test invalid integer coercion."""
        specs = [KeySpec("PORT", int, required=True)]
        with self.assertRaises(CoercionError):
            load_from_env(specs, env={"PORT": "abc"})


if __name__ == "__main__":
    unittest.main()
