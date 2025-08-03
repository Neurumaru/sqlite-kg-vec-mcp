"""
벡터 임베딩 저장 및 관리.
"""

import json
import sqlite3
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config.embedding_config import EmbeddingConfig

# from .search import create_embedder  # 순환 임포트 방지를 위해 동적 임포트 사용

# from .transactions import UnitOfWork  # TODO: transactions 모듈 구현


@dataclass
class Embedding:
    """엔티티 또는 관계에 대한 벡터 임베딩을 나타냅니다."""

    entity_id: int
    entity_type: str  # 'node', 'edge', or 'hyperedge'
    embedding: np.ndarray
    dimensions: int
    model_info: str
    embedding_version: int
    created_at: str
    updated_at: str

    # 성능을 위한 클래스 수준 상수
    _ENTITY_ID_FIELDS = {
        "node": "node_id",
        "edge": "edge_id",
        "hyperedge": "hyperedge_id",
    }

    @classmethod
    def from_row(cls, row: sqlite3.Row, entity_type: str) -> "Embedding":
        """
        데이터베이스 행에서 Embedding을 생성합니다.

        Args:
            row: 임베딩 데이터가 있는 SQLite 행 객체
            entity_type: 엔티티 유형 ('node', 'edge', or 'hyperedge')

        Returns:
            Embedding 객체
        """
        # BLOB을 numpy 배열로 변환 (최적화 - 중간 변수 방지)
        embedding = np.frombuffer(row["embedding"], dtype=np.float32)

        # if-elif 체인 대신 빠른 사전 조회
        id_field = cls._ENTITY_ID_FIELDS.get(entity_type)
        if not id_field:
            raise ValueError(f"지원되지 않는 엔티티 유형: {entity_type}")

        entity_id = row[id_field]

        return cls(
            entity_id=entity_id,
            entity_type=entity_type,
            embedding=embedding,
            dimensions=row["dimensions"],
            model_info=row["model_info"],
            embedding_version=row["embedding_version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class EmbeddingManager:
    """
    엔티티, 관계, 하이퍼엣지에 대한 벡터 임베딩을 관리합니다.
    """

    def __init__(self, connection: sqlite3.Connection, config: EmbeddingConfig = None):
        """
        임베딩 관리자를 초기화합니다.

        Args:
            connection: SQLite 데이터베이스 연결
            config: 임베딩 설정 (없으면 기본값 사용)
        """
        self.connection = connection
        self.config = config or EmbeddingConfig()
        # self.unit_of_work = UnitOfWork(connection)  # TODO: UnitOfWork 구현

    def store_embedding(
        self,
        entity_type: str,
        entity_id: int,
        embedding: np.ndarray,
        model_info: str,
        embedding_version: int = 1,
    ) -> bool:
        """
        엔티티, 엣지 또는 하이퍼엣지에 대한 벡터 임베딩을 저장합니다.

        Args:
            entity_type: 엔티티 유형 ('node', 'edge', or 'hyperedge')
            entity_id: 엔티티의 ID
            embedding: 임베딩 값의 Numpy 배열
            model_info: 임베딩 모델에 대한 정보
            embedding_version: 임베딩의 버전 번호

        Returns:
            성공하면 True, 그렇지 않으면 False

        Raises:
            ValueError: entity_type이 잘못된 경우
        """
        if entity_type not in ("node", "edge", "hyperedge"):
            raise ValueError("엔티티 유형은 'node', 'edge' 또는 'hyperedge'여야 합니다.")

        # 엔티티 유형에 따라 테이블 이름 및 ID 열 가져오기
        if entity_type == "node":
            table = "node_embeddings"
            id_column = "node_id"
        elif entity_type == "edge":
            table = "edge_embeddings"
            id_column = "edge_id"
        else:  # hyperedge
            table = "hyperedge_embeddings"
            id_column = "hyperedge_id"

        # 저장을 위해 numpy 배열을 바이트로 변환
        embedding_blob = embedding.astype(np.float32).tobytes()
        dimensions = len(embedding)

        # TODO: 적절한 작업 단위 패턴 구현
        # with self.unit_of_work.begin() as conn:
        #     cursor = conn.cursor()
        cursor = self.connection.cursor()

        # 임베딩이 이미 존재하는지 확인
        cursor.execute(f"SELECT 1 FROM {table} WHERE {id_column} = ?", (entity_id,))

        if cursor.fetchone():
            # 기존 임베딩 업데이트
            cursor.execute(
                f"""
                UPDATE {table}
                SET embedding = ?, dimensions = ?, model_info = ?,
                    embedding_version = ?
                WHERE {id_column} = ?
                """,
                (
                    embedding_blob,
                    dimensions,
                    model_info,
                    embedding_version,
                    entity_id,
                ),
            )
        else:
            # 새 임베딩 삽입
            cursor.execute(
                f"""
                INSERT INTO {table}
                ({id_column}, embedding, dimensions, model_info, embedding_version)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    entity_id,
                    embedding_blob,
                    dimensions,
                    model_info,
                    embedding_version,
                ),
            )

        self.connection.commit()
        return cursor.rowcount > 0

    def get_embedding(self, entity_type: str, entity_id: int) -> Optional[Embedding]:
        """
        특정 엔티티에 대한 임베딩을 가져옵니다.

        Args:
            entity_type: 엔티티 유형 ('node', 'edge', or 'hyperedge')
            entity_id: 엔티티의 ID

        Returns:
            Embedding 객체 또는 찾을 수 없는 경우 None
        """
        if entity_type not in ("node", "edge", "hyperedge"):
            raise ValueError("엔티티 유형은 'node', 'edge' 또는 'hyperedge'여야 합니다.")

        # 엔티티 유형에 따라 테이블 이름 및 ID 열 가져오기
        if entity_type == "node":
            table = "node_embeddings"
            id_column = "node_id"
        elif entity_type == "edge":
            table = "edge_embeddings"
            id_column = "edge_id"
        else:  # hyperedge
            table = "hyperedge_embeddings"
            id_column = "hyperedge_id"

        cursor = self.connection.cursor()
        cursor.execute(f"SELECT * FROM {table} WHERE {id_column} = ?", (entity_id,))

        row = cursor.fetchone()
        return Embedding.from_row(row, entity_type) if row else None

    def delete_embedding(self, entity_type: str, entity_id: int) -> bool:
        """
        특정 엔티티에 대한 임베딩을 삭제합니다.

        Args:
            entity_type: 엔티티 유형 ('node', 'edge', or 'hyperedge')
            entity_id: 엔티티의 ID

        Returns:
            성공하면 True, 찾을 수 없으면 False
        """
        if entity_type not in ("node", "edge", "hyperedge"):
            raise ValueError("엔티티 유형은 'node', 'edge' 또는 'hyperedge'여야 합니다.")

        # 엔티티 유형에 따라 테이블 이름 및 ID 열 가져오기
        if entity_type == "node":
            table = "node_embeddings"
            id_column = "node_id"
        elif entity_type == "edge":
            table = "edge_embeddings"
            id_column = "edge_id"
        else:  # hyperedge
            table = "hyperedge_embeddings"
            id_column = "hyperedge_id"

        cursor = self.connection.cursor()
        cursor.execute(f"DELETE FROM {table} WHERE {id_column} = ?", (entity_id,))

        return cursor.rowcount > 0

    def get_all_embeddings(
        self,
        entity_type: str,
        model_info: Optional[str] = None,
        batch_size: int = 1000,
        offset: int = 0,
    ) -> list[Embedding]:
        """
        특정 유형의 모든 임베딩을 가져오고, 선택적으로 model_info로 필터링합니다.

        Args:
            entity_type: 엔티티 유형 ('node', 'edge', or 'hyperedge')
            model_info: 특정 모델에 대한 선택적 필터
            batch_size: 배치당 가져올 임베딩 수

        Returns:
            Embedding 객체 목록
        """
        if entity_type not in ("node", "edge", "hyperedge"):
            raise ValueError("엔티티 유형은 'node', 'edge' 또는 'hyperedge'여야 합니다.")

        # 엔티티 유형에 따라 테이블 이름 및 ID 열 가져오기
        if entity_type == "node":
            table = "node_embeddings"
            id_column = "node_id"
        elif entity_type == "edge":
            table = "edge_embeddings"
            id_column = "edge_id"
        else:  # hyperedge
            table = "hyperedge_embeddings"
            id_column = "hyperedge_id"

        # 쿼리 빌드
        query = f"SELECT * FROM {table}"
        params = []

        if model_info:
            query += " WHERE model_info = ?"
            params.append(model_info)

        query += f" ORDER BY {id_column}"

        # 메모리 문제를 피하기 위해 배치로 임베딩 가져오기
        cursor = self.connection.cursor()
        offset = 0
        all_embeddings = []

        while True:
            batch_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
            cursor.execute(batch_query, params)

            batch = cursor.fetchall()
            if not batch:
                break

            # 행을 임베딩으로 변환 (리스트 컴프리헨션으로 최적화)
            batch_embeddings = [Embedding.from_row(row, entity_type) for row in batch]
            all_embeddings.extend(batch_embeddings)

            offset += batch_size

        return all_embeddings

    def get_outdated_embeddings(
        self, entity_type: str, current_version: int, batch_size: int = 1000
    ) -> list[int]:
        """
        오래된 임베딩(버전 < 현재 버전)이 있는 엔티티의 ID를 가져옵니다.

        Args:
            entity_type: 엔티티 유형 ('node', 'edge', or 'hyperedge')
            current_version: 현재 임베딩 버전
            batch_size: 배치당 가져올 ID 수

        Returns:
            오래된 임베딩이 있는 엔티티 ID 목록
        """
        if entity_type not in ("node", "edge", "hyperedge"):
            raise ValueError("엔티티 유형은 'node', 'edge' 또는 'hyperedge'여야 합니다.")

        # 엔티티 유형에 따라 테이블 이름 및 ID 열 가져오기
        if entity_type == "node":
            table = "node_embeddings"
            id_column = "node_id"
        elif entity_type == "edge":
            table = "edge_embeddings"
            id_column = "edge_id"
        else:  # hyperedge
            table = "hyperedge_embeddings"
            id_column = "hyperedge_id"

        # 오래된 임베딩에 대한 쿼리
        query = f"""
        SELECT {id_column} FROM {table}
        WHERE embedding_version < ?
        ORDER BY {id_column}
        """

        # 배치로 ID 가져오기
        cursor = self.connection.cursor()
        offset = 0
        all_ids = []

        while True:
            batch_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
            cursor.execute(batch_query, (current_version,))

            batch = cursor.fetchall()
            if not batch:
                break

            for row in batch:
                all_ids.append(row[0])

            offset += batch_size

        return all_ids

    def process_outbox(self, batch_size: int = 100) -> int:
        """
        아웃박스에서 보류 중인 벡터 작업을 처리합니다.

        Args:
            batch_size: 한 번에 처리할 작업 수

        Returns:
            처리된 작업 수
        """
        cursor = self.connection.cursor()

        # 보류 중인 작업 가져오기
        cursor.execute(
            """
            SELECT id, operation_type, entity_type, entity_id, model_info
            FROM vector_outbox
            WHERE status = 'pending'
            ORDER BY created_at
            LIMIT ?
            """,
            (batch_size,),
        )

        operations = cursor.fetchall()
        processed_count = 0

        for operation in operations:
            outbox_id = operation["id"]
            operation_type = operation["operation_type"]
            entity_type = operation["entity_type"]
            entity_id = operation["entity_id"]
            model_info = operation["model_info"]

            try:
                # 처리 중으로 표시
                cursor.execute(
                    "UPDATE vector_outbox SET status = 'processing' WHERE id = ?",
                    (outbox_id,),
                )

                if operation_type == "delete":
                    # 삭제 처리
                    self.delete_embedding(entity_type, entity_id)

                elif operation_type in ("insert", "update"):
                    # 엔티티 내용을 기반으로 실제 임베딩 생성
                    embedding = self._generate_embedding_for_entity(
                        entity_type, entity_id, model_info
                    )

                    # 임베딩 저장
                    self.store_embedding(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        embedding=embedding,
                        model_info=model_info or "default_model",
                        embedding_version=1,
                    )

                # 완료됨으로 표시
                cursor.execute(
                    "UPDATE vector_outbox SET status = 'completed' WHERE id = ?",
                    (outbox_id,),
                )

                processed_count += 1

            except Exception as exception:
                # 오류 기록 및 실패로 표시
                cursor.execute(
                    """
                    UPDATE vector_outbox
                    SET status = 'failed',
                        retry_count = retry_count + 1,
                        last_error = ?
                    WHERE id = ?
                    """,
                    (str(exception), outbox_id),
                )

                # sync_failures 테이블에 기록
                cursor.execute(
                    """
                    INSERT INTO sync_failures
                    (outbox_id, entity_type, entity_id, operation_type,
                     error_message, retry_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        outbox_id,
                        entity_type,
                        entity_id,
                        operation_type,
                        str(exception),
                        cursor.execute(
                            "SELECT retry_count FROM vector_outbox WHERE id = ?",
                            (outbox_id,),
                        ).fetchone()[0],
                    ),
                )

        return processed_count

    def _generate_embedding_for_entity(
        self, entity_type: str, entity_id: int, model_info: Optional[str] = None
    ) -> np.ndarray:
        """
        엔티티의 텍스트 콘텐츠를 추출하여 임베딩을 생성합니다.

        Args:
            entity_type: 엔티티 유형 ('node' or 'edge')
            entity_id: 엔티티의 ID
            model_info: 임베딩 생성을 위한 모델 정보

        Returns:
            생성된 임베딩 벡터
        """
        try:
            # 임베딩을 위한 엔티티 콘텐츠 가져오기
            text_content = self._extract_entity_text(entity_type, entity_id)

            # 텍스트 임베더가 있으면 사용
            if hasattr(self, "text_embedder") and self.text_embedder is not None:
                result = self.text_embedder.embed(text_content)
                return np.asarray(result, dtype=np.float32)

            # 대체: 기본 임베더 생성 시도
            # model_info에서 임베딩 차원을 결정하거나 기본값 사용
            if model_info and "dim" in model_info:
                try:
                    int(model_info.split("dim=")[1].split(",")[0])
                except Exception:
                    pass

            # 기본 문장-변환기 임베더 생성
            from .embedder_factory import create_embedder  # pylint: disable=import-outside-toplevel

            embedder = create_embedder(
                embedder_type="sentence-transformers", model_name="all-MiniLM-L6-v2"
            )

            result = embedder.embed(text_content)
            return np.asarray(result, dtype=np.float32)

        except Exception as exception:
            # 경고와 함께 랜덤 임베딩으로 대체
            warnings.warn(
                f"{entity_type} {entity_id}에 대한 임베딩 생성 실패: {exception}. 랜덤 임베딩을 사용합니다.",
                stacklevel=2,
            )
            return np.random.rand(384).astype(np.float32)

    def _extract_entity_text(self, entity_type: str, entity_id: int) -> str:
        """
        임베딩 생성을 위해 엔티티에서 텍스트 콘텐츠를 추출합니다.

        Args:
            entity_type: 엔티티 유형 ('node' or 'edge')
            entity_id: 엔티티의 ID

        Returns:
            엔티티의 텍스트 표현
        """
        cursor = self.connection.cursor()

        if entity_type == "node":
            # 엔티티에서 텍스트 추출
            cursor.execute(
                """
                SELECT name, type, properties
                FROM entities
                WHERE id = ?
            """,
                (entity_id,),
            )

            result = cursor.fetchone()
            if not result:
                return f"엔티티 {entity_id}를 찾을 수 없습니다."

            name, ent_type, properties = result

            # 이름, 유형 및 관련 속성 결합
            text_parts = []
            if name:
                text_parts.append(f"이름: {name}")
            if ent_type:
                text_parts.append(f"유형: {ent_type}")

            # 속성 JSON에서 텍스트 추출
            if properties:
                try:
                    props = json.loads(properties) if isinstance(properties, str) else properties
                    for key, value in props.items():
                        if isinstance(value, str | int | float):
                            text_parts.append(f"{key}: {value}")
                except Exception:
                    text_parts.append(f"속성: {properties}")

            return " | ".join(text_parts)

        if entity_type == "edge":
            # 관계에서 텍스트 추출
            cursor.execute(
                """
                SELECT r.relation_type, r.properties,
                       e1.name as source_name, e1.type as source_type,
                       e2.name as target_name, e2.type as target_type
                FROM edges r
                JOIN entities e1 ON r.source_id = e1.id
                JOIN entities e2 ON r.target_id = e2.id
                WHERE r.id = ?
            """,
                (entity_id,),
            )

            result = cursor.fetchone()
            if not result:
                return f"엣지 {entity_id}를 찾을 수 없습니다."

            relation_type, properties, src_name, src_type, tgt_name, tgt_type = result

            # 관계 텍스트 생성
            text_parts = [
                f"관계: {relation_type}",
                f"출발: {src_name or src_type}",
                f"도착: {tgt_name or tgt_type}",
            ]

            # 속성이 있으면 추가
            if properties:
                try:
                    props = json.loads(properties) if isinstance(properties, str) else properties
                    for key, value in props.items():
                        if isinstance(value, str | int | float):
                            text_parts.append(f"{key}: {value}")
                except Exception:
                    text_parts.append(f"속성: {properties}")

            return " | ".join(text_parts)

        return f"알 수 없는 엔티티 유형: {entity_type}"
