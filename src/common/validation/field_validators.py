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
        raise ValueError(f"{field_name}은(는) 정수여야 합니다")

    if value <= 0:
        raise ValueError(f"{field_name}은(는) 양수여야 합니다")

    if max_value is not None and value > max_value:
        raise ValueError(f"{field_name}은(는) {max_value} 이하여야 합니다")

    return value


def validate_positive_number(
    value: float, field_name: str, max_value: Optional[float] = None
) -> float:
    """양의 실수 검증."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name}은(는) 숫자여야 합니다")

    if value <= 0:
        raise ValueError(f"{field_name}은(는) 0보다 커야 합니다")

    if max_value is not None and value > max_value:
        raise ValueError(f"{field_name}은(는) {max_value} 이하여야 합니다")

    return float(value)


def validate_port(port: int) -> int:
    """포트 번호 검증."""
    return validate_positive_integer(port, "포트 번호", 65535)


def validate_timeout(timeout: float, max_timeout: float = 3600.0) -> float:
    """타임아웃 검증."""
    return validate_positive_number(timeout, "타임아웃", max_timeout)


def validate_dimension(dimension: int, max_dimension: int = 4096) -> int:
    """차원 수 검증."""
    return validate_positive_integer(dimension, "차원", max_dimension)


def validate_api_key(
    api_key: Optional[str], provider: str, required_prefix: Optional[str] = None
) -> Optional[str]:
    """API 키 검증."""
    if api_key is None:
        return None

    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(f"{provider} API 키는 비어 있지 않은 문자열이어야 합니다")

    if required_prefix and not api_key.startswith(required_prefix):
        raise ValueError(f"{provider} API 키는 '{required_prefix}'로 시작해야 합니다")

    # 최소 길이 검증
    if len(api_key) < 10:
        raise ValueError(f"{provider} API 키가 너무 짧습니다")

    return api_key


def validate_url(url: str, field_name: str = "URL") -> str:
    """URL 형식 검증."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"{field_name}은(는) 비어 있지 않은 문자열이어야 합니다")

    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            raise ValueError(f"올바르지 않은 {field_name} 형식입니다")
    except Exception as e:
        raise ValueError(f"올바르지 않은 {field_name} 형식입니다: {e}")

    return url


def validate_file_path(
    file_path: str, field_name: str = "파일 경로", must_exist: bool = False
) -> str:
    """파일 경로 검증."""
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError(f"{field_name}는 비어 있지 않은 문자열이어야 합니다")

    path = Path(file_path)

    # 상대 경로에서 상위 디렉토리 참조 검증
    if ".." in path.parts:
        raise ValueError(f"{field_name}에 상위 디렉토리 참조(..)는 허용되지 않습니다")

    if must_exist and not path.exists():
        raise ValueError(f"{field_name}에 지정된 파일이 존재하지 않습니다: {file_path}")

    return file_path


def validate_temperature(temperature: float, min_temp: float = 0.0, max_temp: float = 2.0) -> float:
    """온도 값 검증."""
    if not isinstance(temperature, (int, float)):
        raise ValueError("온도는 숫자여야 합니다")

    if not min_temp <= temperature <= max_temp:
        raise ValueError(f"온도는 {min_temp}에서 {max_temp} 사이여야 합니다")

    return float(temperature)


def validate_email(email: str) -> str:
    """이메일 주소 검증."""
    if not isinstance(email, str) or not email.strip():
        raise ValueError("이메일 주소는 비어 있지 않은 문자열이어야 합니다")

    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(email_pattern, email):
        raise ValueError("올바르지 않은 이메일 주소 형식입니다")

    return email


def validate_host(host: str) -> str:
    """호스트명 검증."""
    if not isinstance(host, str) or not host.strip():
        raise ValueError("호스트명은 비어 있지 않은 문자열이어야 합니다")

    # IP 주소 또는 도메인명 검증
    host_pattern = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$|^[a-zA-Z0-9.-]+$"
    if not re.match(host_pattern, host):
        raise ValueError("올바르지 않은 호스트명 형식입니다")

    return host
