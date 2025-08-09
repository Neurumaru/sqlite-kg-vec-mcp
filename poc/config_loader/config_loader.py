from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Type


_MISSING = object()


@dataclass(frozen=True)
class EnvSpec:
    key: str
    target_type: Type
    default: Any = _MISSING
    required: bool = False
    secret: bool = False


class MissingRequiredEnvError(ValueError):
    def __init__(self, missing_keys: Sequence[str]):
        self.missing_keys: Tuple[str, ...] = tuple(missing_keys)
        super().__init__(
            f"Missing required environment variables: {', '.join(self.missing_keys)}"
        )


class ConfigTypeError(ValueError):
    pass


_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def _coerce(value: Any, target_type: Type) -> Any:
    if value is None:
        return None

    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in _TRUE_VALUES:
            return True
        if s in _FALSE_VALUES:
            return False
        raise ConfigTypeError(f"Cannot parse boolean from: {value!r}")

    if target_type is int:
        try:
            return int(str(value).strip())
        except Exception as exc:  # noqa: BLE001
            raise ConfigTypeError(f"Cannot parse int from: {value!r}") from exc

    if target_type is float:
        try:
            return float(str(value).strip())
        except Exception as exc:  # noqa: BLE001
            raise ConfigTypeError(f"Cannot parse float from: {value!r}") from exc

    if target_type is str:
        return str(value)

    # Fallback: try direct construction
    try:
        return target_type(value)
    except Exception as exc:  # noqa: BLE001
        raise ConfigTypeError(
            f"Cannot coerce {value!r} to {getattr(target_type, '__name__', target_type)}"
        ) from exc


def load_config(
    specs: Iterable[EnvSpec],
    *,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Load typed configuration from environment by the given specs.

    - Uses provided `env` mapping or `os.environ` by default
    - Applies defaults; collects missing required vars and raises `MissingRequiredEnvError`
    - Coerces values to the given target types
    """
    source = os.environ if env is None else env
    result: Dict[str, Any] = {}
    missing: list[str] = []

    for spec in specs:
        raw = source.get(spec.key, _MISSING)  # type: ignore[arg-type]
        if raw is _MISSING:
            if spec.required and spec.default is _MISSING:
                missing.append(spec.key)
                continue
            value = None if spec.default is _MISSING else spec.default
        else:
            value = raw

        # Apply type coercion only if value is not None
        coerced = _coerce(value, spec.target_type) if value is not None else None
        result[spec.key] = coerced

    if missing:
        raise MissingRequiredEnvError(missing)

    return result


def redact_for_logging(config: Mapping[str, Any], specs: Iterable[EnvSpec]) -> Dict[str, Any]:
    """Redact secret values from config for safe logging."""
    secret_keys = {s.key for s in specs if s.secret}
    redacted: Dict[str, Any] = {}
    for key, value in config.items():
        if key in secret_keys and value not in (None, ""):
            redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted



