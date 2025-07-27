"""
벡터 저장소 포트.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from src.domain.value_objects.vector import Vector


class VectorStore(ABC):
    """
    벡터 저장소 포트.
    
    벡터 저장, 검색, 메타데이터 관리 등의 기능을 제공합니다.
    """

    # Store management
    @abstractmethod
    async def initialize_store(
        self,
        dimension: int,
        metric: str = "cosine",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        벡터 저장소를 초기화합니다.

        Args:
            dimension: 벡터 차원
            metric: 거리 메트릭 ("cosine", "euclidean", "dot_product")
            parameters: 선택적 저장소 매개변수

        Returns:
            초기화 성공 여부
        """
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """
        벡터 저장소에 연결합니다.

        Returns:
            연결 성공 여부
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        벡터 저장소 연결을 해제합니다.

        Returns:
            연결 해제 성공 여부
        """
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """
        벡터 저장소 연결 상태를 확인합니다.

        Returns:
            연결 상태
        """
        pass

    # Vector operations
    @abstractmethod
    async def add_vector(
        self, vector_id: str, vector: Vector, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        벡터를 저장소에 추가합니다.

        Args:
            vector_id: 벡터의 고유 식별자
            vector: 벡터 데이터
            metadata: 선택적 메타데이터

        Returns:
            추가 성공 여부
        """
        pass

    @abstractmethod
    async def add_vectors(
        self,
        vectors: Dict[str, Vector],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> bool:
        """
        여러 벡터를 일괄 추가합니다.

        Args:
            vectors: 벡터 ID와 벡터의 매핑 딕셔너리
            metadata: 각 벡터에 대한 선택적 메타데이터

        Returns:
            일괄 추가 성공 여부
        """
        pass

    @abstractmethod
    async def get_vector(self, vector_id: str) -> Optional[Vector]:
        """
        ID로 벡터를 조회합니다.

        Args:
            vector_id: 벡터 식별자

        Returns:
            벡터 (찾지 못하면 None)
        """
        pass

    @abstractmethod
    async def get_vectors(self, vector_ids: List[str]) -> Dict[str, Optional[Vector]]:
        """
        여러 ID로 벡터들을 조회합니다.

        Args:
            vector_ids: 벡터 식별자 리스트

        Returns:
            벡터 ID와 벡터의 매핑 딕셔너리 (찾지 못하면 None)
        """
        pass

    @abstractmethod
    async def update_vector(
        self, vector_id: str, vector: Vector, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        기존 벡터를 업데이트합니다.

        Args:
            vector_id: 벡터 식별자
            vector: 새로운 벡터 데이터
            metadata: 선택적 새 메타데이터

        Returns:
            업데이트 성공 여부
        """
        pass

    @abstractmethod
    async def delete_vector(self, vector_id: str) -> bool:
        """
        저장소에서 벡터를 삭제합니다.

        Args:
            vector_id: 벡터 식별자

        Returns:
            삭제 성공 여부
        """
        pass

    @abstractmethod
    async def delete_vectors(self, vector_ids: List[str]) -> int:
        """
        저장소에서 여러 벡터를 삭제합니다.

        Args:
            vector_ids: 벡터 식별자 리스트

        Returns:
            성공적으로 삭제된 벡터 수
        """
        pass

    @abstractmethod
    async def vector_exists(self, vector_id: str) -> bool:
        """
        벡터가 저장소에 존재하는지 확인합니다.

        Args:
            vector_id: 벡터 식별자

        Returns:
            벡터 존재 여부
        """
        pass

    # Search operations
    @abstractmethod
    async def search_similar(
        self,
        query_vector: Vector,
        k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """
        유사한 벡터를 검색합니다.

        Args:
            query_vector: 쿼리 벡터
            k: 반환할 결과 수
            filter_criteria: 선택적 필터 조건

        Returns:
            (벡터_ID, 유사도_점수) 튜플의 리스트
        """
        pass

    @abstractmethod
    async def search_similar_with_vectors(
        self,
        query_vector: Vector,
        k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, Vector, float]]:
        """
        유사한 벡터를 검색하고 벡터 자체도 반환합니다.

        Args:
            query_vector: 쿼리 벡터
            k: 반환할 결과 수
            filter_criteria: 선택적 필터 조건

        Returns:
            (벡터_ID, 벡터, 유사도_점수) 튜플의 리스트
        """
        pass

    @abstractmethod
    async def search_by_ids(
        self, query_vector: Vector, candidate_ids: List[str], k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        특정 벡터 ID 집합 내에서 검색합니다.

        Args:
            query_vector: 쿼리 벡터
            candidate_ids: 후보 벡터 ID 리스트
            k: 선택적 결과 제한 (기본값은 모든 후보)

        Returns:
            (벡터_ID, 유사도_점수) 튜플의 리스트
        """
        pass

    @abstractmethod
    async def batch_search(
        self,
        query_vectors: List[Vector],
        k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[List[Tuple[str, float]]]:
        """
        여러 쿼리 벡터에 대해 일괄 검색을 수행합니다.

        Args:
            query_vectors: 쿼리 벡터 리스트
            k: 쿼리당 결과 수
            filter_criteria: 선택적 필터 조건

        Returns:
            각 쿼리에 대한 검색 결과 리스트
        """
        pass

    # Metadata operations
    @abstractmethod
    async def get_metadata(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """
        벡터의 메타데이터를 가져옵니다.

        Args:
            vector_id: 벡터 식별자

        Returns:
            메타데이터 딕셔너리 (찾지 못하면 None)
        """
        pass

    @abstractmethod
    async def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        """
        벡터의 메타데이터를 업데이트합니다.

        Args:
            vector_id: 벡터 식별자
            metadata: 새 메타데이터

        Returns:
            업데이트 성공 여부
        """
        pass

    @abstractmethod
    async def search_by_metadata(
        self, filter_criteria: Dict[str, Any], limit: int = 100
    ) -> List[str]:
        """
        메타데이터 조건으로 벡터를 검색합니다.

        Args:
            filter_criteria: 메타데이터 필터 조건
            limit: 최대 결과 수

        Returns:
            조건에 맞는 벡터 ID 리스트
        """
        pass

    # Store information and maintenance
    @abstractmethod
    async def get_store_info(self) -> Dict[str, Any]:
        """
        벡터 저장소 정보를 가져옵니다.

        Returns:
            크기, 차원 등을 포함한 저장소 정보
        """
        pass

    @abstractmethod
    async def get_vector_count(self) -> int:
        """
        저장소의 총 벡터 수를 가져옵니다.

        Returns:
            벡터 수
        """
        pass

    @abstractmethod
    async def get_dimension(self) -> int:
        """
        저장소의 벡터 차원을 가져옵니다.

        Returns:
            벡터 차원
        """
        pass

    @abstractmethod
    async def optimize_store(self) -> Dict[str, Any]:
        """
        성능 향상을 위해 벡터 저장소를 최적화합니다.

        Returns:
            최적화 결과
        """
        pass

    @abstractmethod
    async def rebuild_index(self, parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        벡터 인덱스를 재구축합니다.

        Args:
            parameters: 선택적 재구축 매개변수

        Returns:
            재구축 성공 여부
        """
        pass

    @abstractmethod
    async def clear_store(self) -> bool:
        """
        저장소의 모든 벡터를 지웁니다.

        Returns:
            지우기 성공 여부
        """
        pass

    # Backup and recovery
    @abstractmethod
    async def create_snapshot(self, snapshot_path: str) -> bool:
        """
        벡터 저장소의 스냅샷을 생성합니다.

        Args:
            snapshot_path: 스냅샷을 저장할 경로

        Returns:
            스냅샷 생성 성공 여부
        """
        pass

    @abstractmethod
    async def restore_snapshot(self, snapshot_path: str) -> bool:
        """
        스냅샷에서 벡터 저장소를 복원합니다.

        Args:
            snapshot_path: 스냅샷 파일 경로

        Returns:
            복원 성공 여부
        """
        pass

    # Health and diagnostics
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        벡터 저장소 상태 점검을 수행합니다.

        Returns:
            상태 점검 정보
        """
        pass

    @abstractmethod
    async def get_performance_stats(self) -> Dict[str, Any]:
        """
        벡터 저장소의 성능 통계를 가져옵니다.

        Returns:
            성능 통계
        """
        pass