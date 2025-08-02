"""
어댑터 계층을 위한 인프라 예외.

이 모듈은 각 어댑터 폴더의 기술별 예외가 상속할 수 있는
공통 인프라 예외 기본 클래스를 제공합니다.
이러한 예외는 인프라 관련 문제를 도메인 로직과 분리하여
헥사고날 아키텍처 원칙을 따릅니다.
"""

from .authentication import (
    AuthenticationException,
    AuthorizationException,
)
from .base import InfrastructureException
from .connection import (
    ConnectionException,
    DatabaseConnectionException,
    HTTPConnectionException,
)
from .data import (
    DataException,
    DataIntegrityException,
    DataParsingException,
    DataValidationException,
)
from .timeout import (
    DatabaseTimeoutException,
    HTTPTimeoutException,
    TimeoutException,
)

__all__ = [
    # Base
    "InfrastructureException",
    # Connection
    "ConnectionException",
    "DatabaseConnectionException",
    "HTTPConnectionException",
    # Timeout
    "TimeoutException",
    "DatabaseTimeoutException",
    "HTTPTimeoutException",
    # Data
    "DataException",
    "DataIntegrityException",
    "DataValidationException",
    "DataParsingException",
    # Authentication
    "AuthenticationException",
    "AuthorizationException",
]
