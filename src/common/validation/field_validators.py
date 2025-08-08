"""
공통 필드 검증 함수들.
"""

import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


def validate_positive_integer(value: int, field_name: str, max_value: Optional[int] = None) -> int:
    """양의 정수 검증."""
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")

    if value <= 0:
        raise ValueError(f"{field_name} must be positive")

    if max_value is not None and value > max_value:
        raise ValueError(f"{field_name} must be less than or equal to {max_value}")

    return value


def validate_positive_number(
    value: float, field_name: str, max_value: Optional[float] = None
) -> float:
    """양의 실수 검증."""
    if not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a number")

    if value <= 0:
        raise ValueError(f"{field_name} must be greater than 0")

    if max_value is not None and value > max_value:
        raise ValueError(f"{field_name} must be less than or equal to {max_value}")

    return float(value)


def validate_port(port: int) -> int:
    """포트 번호 검증."""
    return validate_positive_integer(port, "Port", 65535)


def validate_timeout(timeout: float, max_timeout: float = 3600.0) -> float:
    """타임아웃 검증."""
    return validate_positive_number(timeout, "Timeout", max_timeout)


def validate_dimension(dimension: int, max_dimension: int = 4096) -> int:
    """차원 수 검증."""
    return validate_positive_integer(dimension, "Vector dimension", max_dimension)


def validate_api_key(
    api_key: Optional[str], provider: str, required_prefix: Optional[str] = None
) -> Optional[str]:
    """API 키 검증."""
    if api_key is None:
        return None

    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(f"{provider} API key must be a non-empty string")

    if required_prefix and not api_key.startswith(required_prefix):
        raise ValueError(f"{provider} API key must start with '{required_prefix}'")

    # 최소 길이 검증
    if len(api_key) < 10:
        raise ValueError(f"{provider} API key is too short")

    return api_key


def validate_url(url: str, field_name: str = "URL") -> str:
    """URL 형식 검증."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"{field_name} must be a non-empty string")

    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            raise ValueError(f"Invalid {field_name} format")
    except Exception as e:
        raise ValueError(f"Invalid {field_name} format: {e}") from e

    return url


def validate_file_path(
    file_path: str, field_name: str = "File path", must_exist: bool = False
) -> str:
    """파일 경로 검증."""
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError(f"{field_name} must be a non-empty string")

    path = Path(file_path)

    # 상대 경로에서 상위 디렉토리 참조 검증
    if ".." in path.parts:
        raise ValueError(f"{field_name} cannot contain parent directory references (..)")

    if must_exist and not path.exists():
        raise ValueError(f"{field_name} file does not exist: {file_path}")

    return file_path


def validate_temperature(temperature: float, min_temp: float = 0.0, max_temp: float = 2.0) -> float:
    """온도 값 검증."""
    if not isinstance(temperature, int | float):
        raise ValueError("Temperature must be a number")

    if not min_temp <= temperature <= max_temp:
        raise ValueError(f"Temperature must be between {min_temp} and {max_temp}")

    return float(temperature)


def validate_email(email: str) -> str:
    """이메일 주소 검증."""
    if not isinstance(email, str) or not email.strip():
        raise ValueError("Email address must be a non-empty string")

    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(email_pattern, email):
        raise ValueError("Invalid email address format")

    return email


def validate_host(host: str) -> str:
    """호스트명 검증."""
    if not isinstance(host, str) or not host.strip():
        raise ValueError("Host name must be a non-empty string")

    # IP 주소 또는 도메인명 검증
    host_pattern = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$|^[a-zA-Z0-9.-]+$"
    if not re.match(host_pattern, host):
        raise ValueError("Invalid host name format")

    return host
