from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    module_path = Path(__file__).with_name("config_loader.py")
    spec = importlib.util.spec_from_file_location("config_loader_poc", module_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so dataclasses can resolve forward refs
    sys.modules[spec.name] = mod  # type: ignore[index]
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


def test_defaults_and_types():
    m = _load_module()
    specs = [
        m.EnvSpec("PORT", int, default=8080),
        m.EnvSpec("DEBUG", bool, default=False),
        m.EnvSpec("NAME", str, default="svc"),
    ]
    env = {"PORT": "9090", "DEBUG": "true"}
    cfg = m.load_config(specs, env=env)
    assert cfg["PORT"] == 9090 and isinstance(cfg["PORT"], int)
    assert cfg["DEBUG"] is True and isinstance(cfg["DEBUG"], bool)
    assert cfg["NAME"] == "svc"


def test_required_missing_fail_fast():
    m = _load_module()
    specs = [
        m.EnvSpec("API_KEY", str, required=True),
        m.EnvSpec("DB_URL", str, required=True),
    ]
    try:
        m.load_config(specs, env={})
        assert False, "Expected MissingRequiredEnvError"
    except m.MissingRequiredEnvError as exc:
        assert set(exc.missing_keys) == {"API_KEY", "DB_URL"}


def test_bool_parsing_true_false_variants():
    m = _load_module()
    specs = [m.EnvSpec("FLAG", bool, required=True)]
    true_values = ["1", "true", "t", "yes", "y", "on", "TRUE", "On"]
    false_values = ["0", "false", "f", "no", "n", "off", "FALSE", "Off"]

    for v in true_values:
        cfg = m.load_config(specs, env={"FLAG": v})
        assert cfg["FLAG"] is True

    for v in false_values:
        cfg = m.load_config(specs, env={"FLAG": v})
        assert cfg["FLAG"] is False


def test_secret_redaction():
    m = _load_module()
    specs = [
        m.EnvSpec("API_KEY", str, required=True, secret=True),
        m.EnvSpec("PUBLIC_URL", str, default="http://example"),
    ]
    cfg = m.load_config(specs, env={"API_KEY": "supersecret"})
    redacted = m.redact_for_logging(cfg, specs)
    assert redacted["API_KEY"] == "***"
    assert redacted["PUBLIC_URL"] == "http://example"


