"""
데이터베이스 포트.
"""

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, Dict, List, Optional


class Database(ABC):
    """
    데이터베이스 포트.
    
    데이터베이스 연결, 트랜잭션, 쿼리 실행 등의 기능을 제공합니다.
    """

    # Connection management
    @abstractmethod
    async def connect(self) -> bool:
        """
        데이터베이스 연결을 설정합니다.

        Returns:
            연결 성공 여부
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        데이터베이스 연결을 종료합니다.

        Returns:
            연결 해제 성공 여부
        """
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """
        데이터베이스 연결 상태를 확인합니다.

        Returns:
            연결 상태
        """
        pass

    @abstractmethod
    async def ping(self) -> bool:
        """
        데이터베이스 응답을 확인합니다.

        Returns:
            응답 여부
        """
        pass

    # Transaction management
    @abstractmethod
    @asynccontextmanager
    async def transaction(self) -> AsyncContextManager[None]:
        """
        트랜잭션 컨텍스트를 생성합니다.

        Yields:
            트랜잭션 컨텍스트
        """
        pass

    @abstractmethod
    async def begin_transaction(self) -> str:
        """
        새 트랜잭션을 시작합니다.

        Returns:
            트랜잭션 ID
        """
        pass

    @abstractmethod
    async def commit_transaction(self, transaction_id: str) -> bool:
        """
        트랜잭션을 커밋합니다.

        Args:
            transaction_id: 트랜잭션 ID

        Returns:
            커밋 성공 여부
        """
        pass

    @abstractmethod
    async def rollback_transaction(self, transaction_id: str) -> bool:
        """
        트랜잭션을 롤백합니다.

        Args:
            transaction_id: 트랜잭션 ID

        Returns:
            롤백 성공 여부
        """
        pass

    # Query execution
    @abstractmethod
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        transaction_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        SELECT 쿼리를 실행합니다.

        Args:
            query: 실행할 SQL 쿼리
            parameters: 선택적 쿼리 매개변수
            transaction_id: 선택적 트랜잭션 ID

        Returns:
            쿼리 결과 (딕셔너리 리스트)
        """
        pass

    @abstractmethod
    async def execute_command(
        self,
        command: str,
        parameters: Optional[Dict[str, Any]] = None,
        transaction_id: Optional[str] = None,
    ) -> int:
        """
        비-SELECT 명령(INSERT, UPDATE, DELETE)을 실행합니다.

        Args:
            command: 실행할 SQL 명령
            parameters: 선택적 명령 매개변수
            transaction_id: 선택적 트랜잭션 ID

        Returns:
            영향받은 행 수
        """
        pass

    @abstractmethod
    async def execute_batch(
        self,
        commands: List[str],
        parameters: Optional[List[Dict[str, Any]]] = None,
        transaction_id: Optional[str] = None,
    ) -> List[int]:
        """
        여러 명령을 일괄 실행합니다.

        Args:
            commands: SQL 명령 리스트
            parameters: 각 명령에 대한 선택적 매개변수 리스트
            transaction_id: 선택적 트랜잭션 ID

        Returns:
            각 명령의 영향받은 행 수 리스트
        """
        pass

    # Schema management
    @abstractmethod
    async def create_table(
        self, table_name: str, schema: Dict[str, Any], if_not_exists: bool = True
    ) -> bool:
        """
        데이터베이스 테이블을 생성합니다.

        Args:
            table_name: 테이블명
            schema: 테이블 스키마 정의
            if_not_exists: IF NOT EXISTS 절 사용 여부

        Returns:
            테이블 생성 성공 여부
        """
        pass

    @abstractmethod
    async def drop_table(self, table_name: str, if_exists: bool = True) -> bool:
        """
        데이터베이스 테이블을 삭제합니다.

        Args:
            table_name: 테이블명
            if_exists: IF EXISTS 절 사용 여부

        Returns:
            테이블 삭제 성공 여부
        """
        pass

    @abstractmethod
    async def table_exists(self, table_name: str) -> bool:
        """
        테이블 존재 여부를 확인합니다.

        Args:
            table_name: 테이블명

        Returns:
            테이블 존재 여부
        """
        pass

    @abstractmethod
    async def get_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        테이블 스키마를 가져옵니다.

        Args:
            table_name: 테이블명

        Returns:
            테이블 스키마 또는 None (테이블이 없는 경우)
        """
        pass

    @abstractmethod
    async def create_index(
        self,
        index_name: str,
        table_name: str,
        columns: List[str],
        unique: bool = False,
        if_not_exists: bool = True,
    ) -> bool:
        """
        데이터베이스 인덱스를 생성합니다.

        Args:
            index_name: 인덱스명
            table_name: 인덱스를 생성할 테이블명
            columns: 인덱스에 포함할 컬럼들
            unique: 유니크 인덱스 여부
            if_not_exists: IF NOT EXISTS 절 사용 여부

        Returns:
            인덱스 생성 성공 여부
        """
        pass

    @abstractmethod
    async def drop_index(self, index_name: str, if_exists: bool = True) -> bool:
        """
        데이터베이스 인덱스를 삭제합니다.

        Args:
            index_name: 인덱스명
            if_exists: IF EXISTS 절 사용 여부

        Returns:
            인덱스 삭제 성공 여부
        """
        pass

    # Database maintenance
    @abstractmethod
    async def vacuum(self) -> bool:
        """
        데이터베이스 VACUUM 작업을 수행합니다.

        Returns:
            VACUUM 성공 여부
        """
        pass

    @abstractmethod
    async def analyze(self, table_name: Optional[str] = None) -> bool:
        """
        데이터베이스 통계를 분석합니다.

        Args:
            table_name: 분석할 특정 테이블명 (선택적)

        Returns:
            분석 성공 여부
        """
        pass

    @abstractmethod
    async def get_database_info(self) -> Dict[str, Any]:
        """
        데이터베이스 정보와 통계를 가져옵니다.

        Returns:
            데이터베이스 정보
        """
        pass

    @abstractmethod
    async def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        특정 테이블에 대한 정보를 가져옵니다.

        Args:
            table_name: 테이블명

        Returns:
            테이블 정보 또는 None (테이블이 없는 경우)
        """
        pass

    # Health and diagnostics
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        데이터베이스 상태 점검을 수행합니다.

        Returns:
            상태 점검 정보
        """
        pass

    @abstractmethod
    async def get_connection_info(self) -> Dict[str, Any]:
        """
        연결 정보와 상태를 가져옵니다.

        Returns:
            연결 정보
        """
        pass

    @abstractmethod
    async def get_performance_stats(self) -> Dict[str, Any]:
        """
        데이터베이스 성능 통계를 가져옵니다.

        Returns:
            성능 통계
        """
        pass