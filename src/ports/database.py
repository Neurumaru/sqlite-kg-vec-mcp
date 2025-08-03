"""
최적화된 데이터베이스 포트.

실제 사용 분석을 기반으로 불필요한 추상화를 제거하고 핵심 기능에
초점을 맞춘 경량 데이터베이스 인터페이스.
"""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from typing import Any, Optional


class Database(ABC):
    """
    최적화된 데이터베이스 포트.

    실제 사용 패턴 분석을 기반으로 필수 기능만 포함하는 경량 인터페이스.
    """

    # 핵심 작업 - 가장 자주 사용되는 필수 메서드
    @abstractmethod
    async def execute_query(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
        transaction_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        SELECT 쿼리를 실행합니다.

        가장 자주 사용되는 핵심 메서드.

        인자:
            query: 실행할 SQL 쿼리
            parameters: 선택적 쿼리 매개변수
            transaction_id: 선택적 트랜잭션 ID

        반환:
            딕셔너리 리스트 형태의 쿼리 결과
        """

    @abstractmethod
    async def execute_command(
        self,
        command: str,
        parameters: Optional[dict[str, Any]] = None,
        transaction_id: Optional[str] = None,
    ) -> int:
        """
        SELECT가 아닌 명령(INSERT, UPDATE, DELETE)을 실행합니다.

        두 번째로 가장 자주 사용되는 핵심 메서드.

        인자:
            command: 실행할 SQL 명령
            parameters: 선택적 명령 매개변수
            transaction_id: 선택적 트랜잭션 ID

        반환:
            영향을 받은 행의 수
        """

    # 트랜잭션 관리 - 컨텍스트 관리자만 유지됨
    @abstractmethod
    def transaction(self) -> AbstractAsyncContextManager[None]:
        """
        트랜잭션 컨텍스트를 생성합니다.

        대부분의 트랜잭션 사용은 컨텍스트 관리자 패턴을 따릅니다.

        반환:
            트랜잭션 컨텍스트 관리자
        """

    # 연결 관리 - 최소한의 연결 처리
    @abstractmethod
    async def connect(self) -> bool:
        """
        데이터베이스 연결을 설정합니다.

        반환:
            연결 성공 상태
        """

    @abstractmethod
    async def is_connected(self) -> bool:
        """
        데이터베이스 연결 상태를 확인합니다.

        반환:
            연결 상태
        """

    # 스키마 검사 - 필요할 때만 사용되는 메서드
    @abstractmethod
    async def table_exists(self, table_name: str) -> bool:
        """
        테이블이 존재하는지 확인합니다.

        인자:
            table_name: 테이블 이름

        반환:
            테이블 존재 여부 상태
        """

    @abstractmethod
    async def get_table_schema(self, table_name: str) -> Optional[dict[str, Any]]:
        """
        테이블 스키마를 가져옵니다.

        인자:
            table_name: 테이블 이름

        반환:
            테이블 스키마 또는 테이블이 존재하지 않으면 None
        """


class DatabaseMaintenance(ABC):
    """
    데이터베이스 유지 관리 작업을 위한 별도의 인터페이스.

    필요할 때만 사용하도록 일반 데이터베이스 작업과 분리되었습니다.
    어댑터는 선택적으로 이 인터페이스를 구현할 수 있습니다.
    """

    @abstractmethod
    async def vacuum(self) -> bool:
        """데이터베이스 VACUUM 작업을 수행합니다."""

    @abstractmethod
    async def analyze(self, table_name: Optional[str] = None) -> bool:
        """데이터베이스 통계를 분석합니다."""

    @abstractmethod
    async def create_table(
        self,
        table_name: str,
        schema: dict[str, Any],
        if_not_exists: bool = True,
    ) -> bool:
        """데이터베이스 테이블을 생성합니다."""

    @abstractmethod
    async def drop_table(self, table_name: str, if_exists: bool = True) -> bool:
        """데이터베이스 테이블을 삭제합니다."""

    @abstractmethod
    async def create_index(
        self,
        index_name: str,
        table_name: str,
        columns: list[str],
        unique: bool = False,
        if_not_exists: bool = True,
    ) -> bool:
        """데이터베이스 인덱스를 생성합니다."""

    @abstractmethod
    async def drop_index(self, index_name: str, if_exists: bool = True) -> bool:
        """데이터베이스 인덱스를 삭제합니다."""
