"""
벡터 유사도 검색을 위한 HNSW(Hierarchical Navigable Small World) 인덱스.
빠른 근사 최근접 이웃 검색을 위해 hnswlib 백엔드를 사용합니다.
"""

import pickle
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import hnswlib
import numpy as np

if TYPE_CHECKING:
    from ..hnsw.embeddings import EmbeddingManager


class HNSWBackend(Enum):
    """사용 가능한 HNSW 백엔드."""

    HNSWLIB = "hnswlib"


class HNSWIndex:
    """
    빠른 근사 최근접 이웃 검색을 위한 HNSW 인덱스.
    효율적인 벡터 유사도 검색을 위해 hnswlib 백엔드를 사용합니다.
    """

    def __init__(
        self,
        space: str = "cosine",
        dim: int = 128,
        ef_construction: int = 200,
        m_parameter: int = 16,
        index_dir: Optional[str | Path] = None,
        backend: str | HNSWBackend = HNSWBackend.HNSWLIB,
    ):
        """
        HNSW 인덱스를 초기화합니다.

        Args:
            space: 거리 측정 기준 ('cosine', 'ip' for inner product, or 'l2')
            dim: 벡터 차원
            ef_construction: 인덱스 구성 시 품질/속도 트레이드오프 제어
            m_parameter: 인덱스 그래프 연결성을 제어하는 매개변수
            index_dir: 인덱스 파일을 저장/로드할 디렉토리 (None은 메모리 전용)
            backend: 사용할 백엔드 ('hnswlib'만 지원)
        """
        self.space = space
        self.dim = dim
        self.ef_construction = ef_construction
        self.m_parameter = m_parameter
        self.index_dir = Path(index_dir) if index_dir else None

        # 백엔드 선택 처리
        if isinstance(backend, str):
            backend = HNSWBackend(backend.lower())
        self.backend = backend

        # 백엔드별 인덱스 초기화
        self.index: hnswlib.Index
        self._init_backend()

        # SQLite ID에서 인덱스 ID로의 매핑
        self.id_to_idx: dict[tuple[str, int], int] = {}
        self.idx_to_id: dict[int, tuple[str, int]] = {}

        # 현재 크기 추적
        self.current_size = 0
        self.current_capacity = 0
        self.is_initialized = False

    def _init_backend(self):
        """백엔드별 인덱스를 초기화합니다."""
        self.index = hnswlib.Index(space=self.space, dim=self.dim)

    def init_index(self, max_elements: int = 1000) -> None:
        """
        지정된 용량으로 새 인덱스를 초기화합니다.

        Args:
            max_elements: 인덱스의 최대 요소 수
        """
        if self.index is not None:
            self.index.init_index(
                max_elements=max_elements,
                ef_construction=self.ef_construction,
                M=self.m_parameter,
            )
            # 검색 매개변수 설정
            self.index.set_ef(max(self.ef_construction, 100))

        self.current_capacity = max_elements
        self.current_size = 0
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.is_initialized = True

    def load_index(self, filename: Optional[str] = None) -> None:
        """
        이전에 저장된 인덱스와 매핑을 로드합니다.

        Args:
            filename: 확장자가 없는 기본 파일 이름 (제공된 경우 index_dir 사용)
        """
        if self.index_dir is None and filename is None:
            raise ValueError("index_dir 또는 filename 중 하나를 제공해야 합니다.")

        if filename is None:
            # 매개변수를 기반으로 기본 파일 이름 사용
            base_name = f"hnsw_{self.backend.value}_{self.space}_{self.dim}_{self.m_parameter}"
        else:
            base_name = filename

        # 파일 경로 결정
        if self.index_dir:
            index_path = self.index_dir / f"{base_name}.bin"
            mapping_path = self.index_dir / f"{base_name}_mapping.pkl"
        else:
            index_path = Path(f"{base_name}.bin")
            mapping_path = Path(f"{base_name}_mapping.pkl")

        # 인덱스 로드
        if index_path.exists() and self.index is not None:
            self.index.load_index(str(index_path))
            self.current_capacity = self.index.get_max_elements()
            self.current_size = self.index.get_current_count()
            self.is_initialized = True

            # ID 매핑 로드
            if mapping_path.exists():
                with open(mapping_path, "rb") as file_handle:
                    mappings = pickle.load(file_handle)
                    self.id_to_idx = mappings["id_to_idx"]
                    self.idx_to_id = mappings["idx_to_id"]
            else:
                raise FileNotFoundError(f"인덱스 매핑 파일을 찾을 수 없습니다: {mapping_path}")
        else:
            raise FileNotFoundError(f"인덱스 파일을 찾을 수 없습니다: {index_path}")

    def save_index(self, filename: Optional[str] = None) -> None:
        """
        인덱스와 ID 매핑을 디스크에 저장합니다.

        Args:
            filename: 확장자가 없는 기본 파일 이름 (제공된 경우 index_dir 사용)
        """
        if not self.is_initialized:
            raise RuntimeError("인덱스가 초기화되지 않았습니다.")

        if self.index_dir is None and filename is None:
            raise ValueError("index_dir 또는 filename 중 하나를 제공해야 합니다.")

        if filename is None:
            # 매개변수를 기반으로 기본 파일 이름 사용
            base_name = f"hnsw_{self.backend.value}_{self.space}_{self.dim}_{self.m_parameter}"
        else:
            base_name = filename

        # 파일 경로 결정
        if self.index_dir:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            index_path = self.index_dir / f"{base_name}.bin"
            mapping_path = self.index_dir / f"{base_name}_mapping.pkl"
        else:
            index_path = Path(f"{base_name}.bin")
            mapping_path = Path(f"{base_name}_mapping.pkl")

        # 인덱스 저장
        if self.index is not None:
            self.index.save_index(str(index_path))

        # ID 매핑 저장
        mappings = {"id_to_idx": self.id_to_idx, "idx_to_id": self.idx_to_id}
        with open(mapping_path, "wb") as file_handle:
            pickle.dump(mappings, file_handle)

    def resize_index(self, new_size: int) -> None:
        """
        더 많은 요소를 수용하기 위해 인덱스 크기를 조정합니다.

        Args:
            new_size: 새 최대 용량
        """
        if new_size <= self.current_capacity:
            return

        if self.index is not None:
            self.index.resize_index(new_size)
        self.current_capacity = new_size

    def add_item(
        self,
        entity_type: str,
        entity_id: int,
        vector: np.ndarray,
        replace_existing: bool = True,
    ) -> int:
        """
        인덱스에 항목을 추가합니다.

        Args:
            entity_type: 엔티티 유형 ('node', 'edge', or 'hyperedge')
            entity_id: 엔티티의 ID
            vector: 임베딩 벡터
            replace_existing: 항목이 이미 존재하는 경우 교체할지 여부

        Returns:
            추가된 항목의 인덱스 ID
        """
        if not self.is_initialized:
            raise RuntimeError("인덱스가 초기화되지 않았습니다.")

        item_key = (entity_type, entity_id)

        # 항목이 이미 존재하는지 확인
        if item_key in self.id_to_idx:
            if replace_existing:
                # 먼저 이전 항목 제거
                self.remove_item(entity_type, entity_id)
            else:
                # 기존 인덱스 반환
                return self.id_to_idx[item_key]

        # 크기 조정이 필요한지 확인
        if self.current_size >= self.current_capacity:
            # 용량을 두 배로 조정
            new_capacity = max(1000, self.current_capacity * 2)
            self.resize_index(new_capacity)

        # 벡터 준비
        vector = vector.astype(np.float32)  # 올바른 데이터 유형 확인

        idx = self.current_size
        if self.index is not None:
            self.index.add_items(vector, [idx])

        # 매핑 업데이트
        self.id_to_idx[item_key] = idx
        self.idx_to_id[idx] = item_key
        self.current_size += 1

        return idx

    def add_items_batch(
        self,
        entity_types: list[str],
        entity_ids: list[int],
        vectors: np.ndarray,
        replace_existing: bool = True,
    ) -> list[int]:
        """
        배치 작업을 사용하여 여러 항목을 인덱스에 효율적으로 추가합니다.

        Args:
            entity_types: 엔티티 유형 목록
            entity_ids: 엔티티 ID 목록
            vectors: 2D numpy 배열 벡터 (n_vectors x dimension)
            replace_existing: 항목이 이미 존재하는 경우 교체할지 여부

        Returns:
            추가된 항목의 인덱스 ID 목록
        """
        if not self.is_initialized:
            raise RuntimeError("인덱스가 초기화되지 않았습니다.")

        if len(entity_types) != len(entity_ids) or len(entity_types) != len(vectors):
            raise ValueError("entity_types, entity_ids, vectors는 길이가 같아야 합니다.")

        if len(vectors) == 0:
            return []

        # 데이터 준비
        vectors = vectors.astype(np.float32)
        item_keys = list(zip(entity_types, entity_ids, strict=False))

        # 교체하지 않는 경우 기존 항목 필터링
        if not replace_existing:
            new_indices = []
            new_vectors_list = []
            new_item_keys = []

            for i, item_key in enumerate(item_keys):
                if item_key not in self.id_to_idx:
                    new_indices.append(i)
                    new_vectors_list.append(vectors[i])
                    new_item_keys.append(item_key)

            if not new_vectors_list:
                # 모든 항목이 이미 존재함
                return [self.id_to_idx[key] for key in item_keys]

            vectors = np.array(new_vectors_list)
            item_keys = new_item_keys
        else:
            # 먼저 기존 항목 제거
            for item_key in item_keys:
                if item_key in self.id_to_idx:
                    entity_type, entity_id = item_key
                    self.remove_item(entity_type, entity_id)

        n_items = len(vectors)

        # 크기 조정이 필요한지 확인
        if self.current_size + n_items > self.current_capacity:
            new_capacity = max(self.current_capacity * 2, self.current_size + n_items + 1000)
            self.resize_index(new_capacity)

        # 새 인덱스 생성
        start_idx = self.current_size
        indices = list(range(start_idx, start_idx + n_items))

        # 인덱스에 배치 추가
        if self.index is not None:
            self.index.add_items(vectors, indices)

        # 매핑 배치 업데이트
        for item_key, idx in zip(item_keys, indices, strict=False):
            self.id_to_idx[item_key] = idx
            self.idx_to_id[idx] = item_key

        self.current_size += n_items
        return indices

    def remove_item(self, entity_type: str, entity_id: int) -> bool:
        """
        인덱스에서 항목을 제거합니다.

        Args:
            entity_type: 엔티티 유형
            entity_id: 엔티티 ID

        Returns:
            항목이 제거되었으면 True, 찾을 수 없으면 False
        """
        if not self.is_initialized:
            raise RuntimeError("인덱스가 초기화되지 않았습니다.")

        item_key = (entity_type, entity_id)

        if item_key in self.id_to_idx:
            idx = self.id_to_idx[item_key]

            if self.index is not None:
                self.index.mark_deleted(idx)

            # 매핑 업데이트
            del self.id_to_idx[item_key]
            del self.idx_to_id[idx]

            return True

        return False

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        ef_search: Optional[int] = None,
        filter_entity_types: Optional[list[str]] = None,
    ) -> list[tuple[str, int, float]]:
        """
        쿼리 벡터에 가장 가까운 벡터를 검색합니다.

        Args:
            query_vector: 쿼리 임베딩 벡터
            k: 검색할 최근접 이웃 수
            ef_search: 런타임 매개변수로 쿼리 정확도/속도 트레이드오프 제어
            filter_entity_types: 결과에 포함할 엔티티 유형 목록

        Returns:
            (entity_type, entity_id, distance) 튜플 목록
        """
        if not self.is_initialized:
            raise RuntimeError("인덱스가 초기화되지 않았습니다.")

        # 빈 인덱스 처리
        if self.current_size == 0:
            return []

        # 제공된 경우 검색 매개변수 설정
        if ef_search is not None and self.index is not None:
            self.index.set_ef(ef_search)

        # 올바른 유형으로 변환
        query_vector = query_vector.astype(np.float32)

        # 인덱스 크기를 초과하지 않도록 k 조정
        adjusted_k = min(k, self.current_size)
        if adjusted_k <= 0:
            return []

        # 인덱스 검색
        if self.index is not None:
            indices, distances = self.index.knn_query(query_vector, k=adjusted_k)
        else:
            return []
        indices, distances = indices[0], distances[0]

        # 결과 처리 (리스트 컴프리헨션 및 적은 조회로 최적화)
        if filter_entity_types:
            # O(n) 리스트 조회 대신 O(1) 조회를 위해 집합으로 변환
            filter_set = set(filter_entity_types)
            results = [
                (entity_type, entity_id, float(dist))
                for idx, dist in zip(indices, distances, strict=False)
                if idx in self.idx_to_id
                and (entity_type := self.idx_to_id[idx][0]) in filter_set
                and (entity_id := self.idx_to_id[idx][1]) is not None
            ]
        else:
            results = [
                (entity_type, entity_id, float(dist))
                for idx, dist in zip(indices, distances, strict=False)
                if idx in self.idx_to_id
                and (entity_type := self.idx_to_id[idx][0]) is not None
                and (entity_id := self.idx_to_id[idx][1]) is not None
            ]

        return results

    def build_from_embeddings(
        self,
        embedding_manager: "EmbeddingManager",
        entity_types: Optional[list[str]] = None,
        model_info: Optional[str] = None,
        batch_size: int = 1000,
    ) -> int:
        """
        데이터베이스의 임베딩으로 인덱스를 빌드합니다.

        Args:
            embedding_manager: EmbeddingManager 인스턴스
            entity_types: 포함할 엔티티 유형 목록, 또는 모두에 대해 None
            model_info: model info로 필터링, 또는 모든 모델에 대해 None
            batch_size: 임베딩 처리를 위한 배치 크기

        Returns:
            인덱스에 추가된 임베딩 수
        """
        entity_types = entity_types or ["node", "edge", "hyperedge"]
        total_embeddings = 0

        # 초기 용량을 결정하기 위해 총 임베딩 수 가져오기
        count_query = """
        SELECT COUNT(*)
        FROM (
            SELECT 1 FROM node_embeddings
            WHERE {model_clause}
            UNION ALL
            SELECT 1 FROM edge_embeddings
            WHERE {model_clause}
            UNION ALL
            SELECT 1 FROM hyperedge_embeddings
            WHERE {model_clause}
        )
        """

        model_clause = "1=1" if model_info is None else "model_info = ?"
        params = [] if model_info is None else [model_info, model_info, model_info]

        cursor = embedding_manager.connection.cursor()
        cursor.execute(count_query.format(model_clause=model_clause), params)
        total_count = cursor.fetchone()[0]

        # 약간 더 큰 용량으로 초기화
        init_capacity = max(1000, int(total_count * 1.2))
        self.init_index(max_elements=init_capacity)

        # 각 엔티티 유형 처리
        for entity_type in entity_types:
            # 배치로 임베딩 가져오기
            offset = 0

            while True:
                embeddings = embedding_manager.get_all_embeddings(
                    entity_type=entity_type,
                    model_info=model_info,
                    batch_size=batch_size,
                    offset=offset,
                )

                if not embeddings:
                    break

                # 효율적인 삽입을 위한 배치 데이터 준비
                entity_types_batch = [entity_type] * len(embeddings)
                entity_ids_batch = [emb.entity_id for emb in embeddings]

                # 최적화된 벡터 배치 생성 - 리스트->배열 변환 대신 직접 스택
                if embeddings:
                    vectors_batch = np.stack([emb.embedding for emb in embeddings]).astype(
                        np.float32
                    )
                else:
                    vectors_batch = np.array([], dtype=np.float32)

                # 더 나은 성능을 위해 배치 삽입 사용
                self.add_items_batch(
                    entity_types=entity_types_batch,
                    entity_ids=entity_ids_batch,
                    vectors=vectors_batch,
                    replace_existing=False,  # 초기 빌드 중 교체 확인 방지
                )

                total_embeddings += len(embeddings)
                offset += batch_size

                # batch_size보다 적게 가져온 경우 완료
                if len(embeddings) < batch_size:
                    break

        return total_embeddings

    def sync_with_outbox(self, embedding_manager: "EmbeddingManager", batch_size: int = 100) -> int:
        """
        아웃박스에서 벡터 작업을 처리하고 인덱스를 업데이트합니다.

        Args:
            embedding_manager: EmbeddingManager 인스턴스
            batch_size: 한 번에 처리할 작업 수

        Returns:
            처리된 작업 수
        """
        # 임베딩이 최신 상태인지 확인하기 위해 먼저 아웃박스 처리
        embedding_manager.process_outbox(batch_size)

        # 이제 처리된 임베딩으로 인덱스 동기화
        cursor = embedding_manager.connection.cursor()

        # 완료된 작업 가져오기
        cursor.execute(
            """
            SELECT operation_type, entity_type, entity_id
            FROM vector_outbox
            WHERE status = 'completed'
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (batch_size,),
        )

        operations = cursor.fetchall()
        sync_count = 0

        for operation in operations:
            operation_type = operation["operation_type"]
            entity_type = operation["entity_type"]
            entity_id = operation["entity_id"]

            try:
                if operation_type == "delete":
                    # 인덱스에서 제거
                    self.remove_item(entity_type, entity_id)

                elif operation_type in ("insert", "update"):
                    # 데이터베이스에서 임베딩 가져오기
                    embedding = embedding_manager.get_embedding(entity_type, entity_id)

                    if embedding:
                        # 인덱스에 추가 또는 업데이트
                        self.add_item(
                            entity_type=entity_type,
                            entity_id=entity_id,
                            vector=embedding.embedding,
                            replace_existing=True,
                        )

                sync_count += 1

            except Exception as exception:
                # 오류를 기록하지만 다른 작업은 계속 진행
                # 지금은 간단한 print 사용 (나중에 적절한 로깅으로 대체 가능)
                print(f"엔티티 {entity_type}:{entity_id} 동기화 오류 - {exception}")

        return sync_count
