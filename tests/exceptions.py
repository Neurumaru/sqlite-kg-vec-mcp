"""
테스트용 예외 클래스들.

이 모듈은 테스트에서 사용되는 예외 클래스들을 정의합니다.
"""


class UnexpectedException(Exception):
    """
    테스트용 예외 클래스.

    테스트에서 예상치 못한 오류 상황을 시뮬레이션할 때 사용합니다.
    일반적인 Exception보다 구체적인 타입을 제공하여 ruff B017 경고를 방지합니다.
    """


class DatabaseConnectionException(Exception):
    """테스트용 데이터베이스 연결 예외."""


class StorageException(Exception):
    """테스트용 저장소 예외."""


class NetworkTimeoutException(Exception):
    """테스트용 네트워크 타임아웃 예외."""
