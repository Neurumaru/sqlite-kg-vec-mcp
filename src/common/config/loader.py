"""
Configuration loader with typed environment variable support.

This module provides utilities for loading and validating configuration
from environment variables with type coercion and default values.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

_MISSING = object()


@dataclass(frozen=True)
class KeySpec:
    """Specification for a configuration key from environment variables."""

    env: str
    target_type: type
    default: Any = _MISSING
    required: bool = False
    secret: bool = False


class MissingRequiredError(ValueError):
    """Raised when required configuration keys are missing."""

    def __init__(self, missing_keys: Sequence[str]):
        self.missing_keys: tuple[str, ...] = tuple(missing_keys)
        super().__init__(f"Missing required configuration keys: {', '.join(self.missing_keys)}")


class CoercionError(ValueError):
    """Raised when type coercion fails."""


_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def _coerce(value: Any, target_type: type) -> Any:
    """
    Coerce a value to the specified type.

    Supports bool, int, float, str and custom types.
    Returns None for None values.

    Args:
        value: Value to coerce
        target_type: Target type

    Returns:
        Coerced value

    Raises:
        CoercionError: If coercion fails
    """
    if value is None:
        return None

    # Handle boolean type specially
    if target_type is bool:
        return _coerce_bool(value)

    # Handle string type
    if target_type is str:
        return str(value)

    # Handle numeric types
    if target_type in (int, float):
        return _coerce_numeric(value, target_type)

    # Handle custom types
    return _coerce_custom(value, target_type)


def _coerce_bool(value: Any) -> bool:
    """Coerce value to boolean."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in _TRUE_VALUES:
        return True
    if s in _FALSE_VALUES:
        return False
    raise CoercionError(f"Cannot parse boolean from: {value!r}")


def _coerce_numeric(value: Any, target_type: type) -> Any:
    """Coerce value to int or float."""
    try:
        return target_type(str(value).strip())
    except Exception as exc:  # noqa: BLE001
        raise CoercionError(f"Cannot parse {target_type.__name__} from: {value!r}") from exc


def _coerce_custom(value: Any, target_type: type) -> Any:
    """Coerce value to custom type."""
    try:
        return target_type(value)
    except Exception as exc:  # noqa: BLE001
        raise CoercionError(
            f"Cannot coerce {value!r} to {getattr(target_type, '__name__', target_type)}"
        ) from exc


def load_from_env(
    specs: Iterable[KeySpec],
    *,
    env: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Load typed configuration from environment variables.

    - Supports defaults and required checks
    - Performs type coercion
    - Allows injecting a custom env mapping for testing
    """
    source = os.environ if env is None else env
    result: dict[str, Any] = {}
    missing: list[str] = []

    for spec in specs:
        raw = source.get(spec.env, _MISSING)
        if raw is _MISSING:
            if spec.required and spec.default is _MISSING:
                missing.append(spec.env)
                continue
            value = None if spec.default is _MISSING else spec.default
        else:
            value = raw

        coerced = _coerce(value, spec.target_type) if value is not None else None
        result[spec.env] = coerced

    if missing:
        raise MissingRequiredError(missing)

    return result


def redact(config: Mapping[str, Any], specs: Iterable[KeySpec]) -> dict[str, Any]:
    """
    Redact secret values from configuration dictionary.

    Args:
        config: Configuration dictionary
        specs: Key specifications (to identify secret keys)

    Returns:
        Configuration dictionary with secret values replaced by '***'
    """
    secret_keys = {s.env for s in specs if s.secret}
    safe: dict[str, Any] = {}
    for key, value in config.items():
        if key in secret_keys and value not in (None, ""):
            safe[key] = "***"
        else:
            safe[key] = value
    return safe
