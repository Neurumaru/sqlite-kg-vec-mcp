"""
최적화된 데이터베이스 포트.

실제 사용 현황 분석을 바탕으로 불필요한 추상화를 제거하고
핵심 기능에 집중한 경량화된 데이터베이스 인터페이스입니다.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncContextManager, Dict, List, Optional


class Database(ABC):
    """
    최적화된 데이터베이스 포트.

    실제 사용 패턴을 분석하여 필수 기능만 남긴 경량화된 인터페이스.
    과도한 추상화를 제거하고 현재 요구사항에 집중합니다.
    """

    # Core operations - 가장 많이 사용되는 필수 메서드들
    @abstractmethod
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        transaction_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        SELECT 쿼리를 실행합니다.

        가장 많이 사용되는 핵심 메서드입니다.

        Args:
            query: 실행할 SQL 쿼리
            parameters: 선택적 쿼리 매개변수
            transaction_id: 선택적 트랜잭션 ID

        Returns:
            쿼리 결과 (딕셔너리 리스트)
        """

    @abstractmethod
    async def execute_command(
        self,
        command: str,
        parameters: Optional[Dict[str, Any]] = None,
        transaction_id: Optional[str] = None,
    ) -> int:
        """
        비-SELECT 명령(INSERT, UPDATE, DELETE)을 실행합니다.

        두 번째로 많이 사용되는 핵심 메서드입니다.

        Args:
            command: 실행할 SQL 명령
            parameters: 선택적 명령 매개변수
            transaction_id: 선택적 트랜잭션 ID

        Returns:
            영향받은 행 수
        """

    # Transaction management - 트랜잭션 컨텍스트 매니저만 유지
    @abstractmethod
    def transaction(self) -> AsyncContextManager[None]:
        """
        트랜잭션 컨텍스트를 생성합니다.

        대부분의 트랜잭션 사용이 컨텍스트 매니저 패턴으로 이루어집니다.

        Returns:
            트랜잭션 컨텍스트 매니저
        """

    # Connection management - 최소한의 연결 관리
    @abstractmethod
    async def connect(self) -> bool:
        """
        데이터베이스 연결을 설정합니다.

        Returns:
            연결 성공 여부
        """

    @abstractmethod
    async def is_connected(self) -> bool:
        """
        데이터베이스 연결 상태를 확인합니다.

        Returns:
            연결 상태
        """

    # Schema inspection - 필요시에만 사용되는 메서드들
    @abstractmethod
    async def table_exists(self, table_name: str) -> bool:
        """
        테이블 존재 여부를 확인합니다.

        Args:
            table_name: 테이블명

        Returns:
            테이블 존재 여부
        """

    @abstractmethod
    async def get_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        테이블 스키마를 가져옵니다.

        Args:
            table_name: 테이블명

        Returns:
            테이블 스키마 또는 None (테이블이 없는 경우)
        """


class DatabaseMaintenance(ABC):
    """
    데이터베이스 유지보수 작업을 위한 별도 인터페이스.

    일반적인 데이터베이스 작업과 분리하여 필요시에만 사용합니다.
    어댑터에서 이 인터페이스를 선택적으로 구현할 수 있습니다.
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
        schema: Dict[str, Any],
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
        columns: List[str],
        unique: bool = False,
        if_not_exists: bool = True,
    ) -> bool:
        """데이터베이스 인덱스를 생성합니다."""

    @abstractmethod
    async def drop_index(self, index_name: str, if_exists: bool = True) -> bool:
        """데이터베이스 인덱스를 삭제합니다."""


# 최적화 설명:
#
# 1. 핵심 Database 인터페이스: 7개 메서드만 유지 (기존 23개에서 70% 감소)
#    - execute_query, execute_command (가장 많이 사용)
#    - transaction (컨텍스트 매니저만 유지)
#    - connect, is_connected (필수 연결 관리)
#    - table_exists, get_table_schema (스키마 조회)
#
# 2. 분리된 인터페이스로 관심사 분리:
#    - DatabaseMaintenance: 스키마 관리 및 유지보수 작업
#
# 3. 장점:
#    - 인터페이스가 단순해져 구현과 테스트가 용이
#    - 불필요한 추상화 제거로 성능 향상
#    - 실제 사용 패턴에 최적화된 실용적 설계
#    - 핵심 기능에 집중하여 복잡성 최소화
