"""
sqlite-vec 확장을 사용한 VectorStore 포트의 SQLite 구현.
"""

import json
import math
import shutil
import sqlite3
import struct
from pathlib import Path
from sqlite3 import Connection
from typing import Any, Optional

from langchain_core.documents import Document

from src.domain import Vector
from src.ports.vector_store import VectorStore

from .connection import DatabaseConnection


class SQLiteVectorStore(VectorStore):
    """
    VectorStore 포트의 SQLite 구현.
    이 어댑터는 벡터 저장 및 검색을 위해 sqlite-vec 확장을 사용하여
    SQLite로 벡터 연산의 구체적인 구현을 제공합니다.
    """

    def __init__(self, db_path: str, table_name: str = "vectors", optimize: bool = True):
        """
        SQLite 벡터 저장소 어댑터를 초기화합니다.
        Args:
            db_path: SQLite 데이터베이스 파일 경로
            table_name: 벡터를 저장할 테이블 이름
            optimize: 최적화 PRAGMA 적용 여부
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.optimize = optimize
        self._connection_manager = DatabaseConnection(db_path, optimize)
        self._connection: sqlite3.Optional[Connection] = None
        self._dimension: Optional[int] = None
        self._metric: str = "cosine"

    # 저장소 관리
    async def initialize_store(
        self,
        dimension: int,
        metric: str = "cosine",
        parameters: dict[str, Any] | None = None,
    ) -> bool:
        """
        벡터 저장소를 초기화합니다.
        Args:
            dimension: 벡터 차원
            metric: 거리 메트릭 ("cosine", "euclidean", "dot_product")
            parameters: 선택적 저장소 매개변수
        Returns:
            초기화 성공 시 True
        """
        try:
            if not self._connection:
                await self.connect()
            if not self._connection:
                raise RuntimeError("데이터베이스 연결 설정 실패")
            self._dimension = dimension
            self._metric = metric
            # 벡터 테이블이 없으면 생성
            cursor = self._connection.cursor()
            # 기본 벡터 테이블 생성
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    vector BLOB NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            # 메타데이터 인덱스 생성
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_metadata
                ON {self.table_name} (metadata)
            """
            )
            # 메타데이터 테이블에 설정 저장
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_store_config (
                    table_name TEXT PRIMARY KEY,
                    dimension INTEGER NOT NULL,
                    metric TEXT NOT NULL,
                    parameters TEXT
                )
            """
            )
            # 설정 삽입 또는 업데이트
            config_data = json.dumps(parameters) if parameters else None
            cursor.execute(
                """
                INSERT OR REPLACE INTO vector_store_config
                (table_name, dimension, metric, parameters)
                VALUES (?, ?, ?, ?)
            """,
                (self.table_name, dimension, metric, config_data),
            )
            self._connection.commit()
            cursor.close()
            return True
        except Exception:
            return False

    async def connect(self) -> bool:
        """
        벡터 저장소에 연결합니다.
        Returns:
            연결 성공 시 True
        """
        try:
            self._connection = self._connection_manager.connect()
            # 설정이 존재하면 로드
            await self._load_config()
            return True
        except Exception:
            return False

    async def disconnect(self) -> bool:
        """
        벡터 저장소에서 연결을 끊습니다.
        Returns:
            연결 끊기 성공 시 True
        """
        try:
            self._connection_manager.close()
            self._connection = None
            return True
        except Exception:
            return False

    async def is_connected(self) -> bool:
        """
        벡터 저장소에 연결되었는지 확인합니다.
        Returns:
            연결된 경우 True
        """
        if not self._connection:
            return False
        try:
            self._connection.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    # 벡터 연산
    async def add_vector(
        self, vector_id: str, vector: Vector, metadata: dict[str, Any] | None = None
    ) -> bool:
        """
        저장소에 벡터를 추가합니다.
        Args:
            vector_id: 벡터의 고유 식별자
            vector: 벡터 데이터
            metadata: 선택적 메타데이터
        Returns:
            추가 성공 시 True
        """
        try:
            if not self._connection:
                return False
            cursor = self._connection.cursor()
            # 벡터를 blob으로 변환
            vector_blob = self._vector_to_blob(vector)
            metadata_json = json.dumps(metadata) if metadata else None
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {self.table_name}
                (id, vector, metadata, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (vector_id, vector_blob, metadata_json),
            )
            self._connection.commit()
            cursor.close()
            return True
        except Exception:
            return False

    async def add_vectors(
        self,
        vectors: dict[str, Vector],
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> bool:
        """
        저장소에 여러 벡터를 일괄 추가합니다.
        Args:
            vectors: 벡터 ID를 벡터에 매핑하는 사전
            metadata: 각 벡터에 대한 선택적 메타데이터
        Returns:
            일괄 추가 성공 시 True
        """
        try:
            if not self._connection:
                return False
            cursor = self._connection.cursor()
            # 일괄 데이터 준비
            batch_data = []
            for vector_id, vector in vectors.items():
                vector_blob = self._vector_to_blob(vector)
                vector_metadata = metadata.get(vector_id) if metadata else None
                metadata_json = json.dumps(vector_metadata) if vector_metadata else None
                batch_data.append((vector_id, vector_blob, metadata_json))
            cursor.executemany(
                f"""
                INSERT OR REPLACE INTO {self.table_name}
                (id, vector, metadata, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
                batch_data,
            )
            self._connection.commit()
            cursor.close()
            return True
        except Exception:
            return False

    async def get_vector(self, vector_id: str) -> Optional[Vector]:
        """
        ID로 벡터를 검색합니다.
        Args:
            vector_id: 벡터 식별자
        Returns:
            찾은 경우 벡터, 그렇지 않으면 None
        """
        try:
            if not self._connection:
                return None
            cursor = self._connection.cursor()
            cursor.execute(
                f"""
                SELECT vector FROM {self.table_name} WHERE id = ?
            """,
                (vector_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            if row:
                return self._blob_to_vector(row[0])
            return None
        except Exception:
            return None

    async def get_vectors(self, vector_ids: list[str]) -> dict[str, Optional[Vector]]:
        """
        ID로 여러 벡터를 검색합니다.
        Args:
            vector_ids: 벡터 식별자 목록
        Returns:
            벡터 ID를 벡터에 매핑하는 사전 (찾지 못한 경우 None)
        """
        try:
            if not self._connection:
                return dict.fromkeys(vector_ids)
            cursor = self._connection.cursor()
            # IN 절에 대한 플레이스홀더 생성
            placeholders = ", ".join("?" * len(vector_ids))
            cursor.execute(
                f"""
                SELECT id, vector FROM {self.table_name}
                WHERE id IN ({placeholders})
            """,
                vector_ids,
            )
            rows = cursor.fetchall()
            cursor.close()
            # 결과 사전 빌드
            result: dict[str, Optional[Vector]] = dict.fromkeys(vector_ids)
            for row in rows:
                vector_id, vector_blob = row
                result[vector_id] = self._blob_to_vector(vector_blob)
            return result
        except Exception:
            return dict.fromkeys(vector_ids)

    async def update_vector(
        self, vector_id: str, vector: Vector, metadata: dict[str, Any] | None = None
    ) -> bool:
        """
        기존 벡터를 업데이트합니다.
        Args:
            vector_id: 벡터 식별자
            vector: 새 벡터 데이터
            metadata: 선택적 새 메타데이터
        Returns:
            업데이트 성공 시 True
        """
        # SQLite의 경우 업데이트는 REPLACE를 사용한 추가와 동일합니다.
        return await self.add_vector(vector_id, vector, metadata)

    async def delete_vector(self, vector_id: str) -> bool:
        """
        저장소에서 벡터를 삭제합니다.
        Args:
            vector_id: 벡터 식별자
        Returns:
            삭제 성공 시 True
        """
        try:
            if not self._connection:
                return False
            cursor = self._connection.cursor()
            cursor.execute(
                f"""
                DELETE FROM {self.table_name} WHERE id = ?
            """,
                (vector_id,),
            )
            success = cursor.rowcount > 0
            self._connection.commit()
            cursor.close()
            return success
        except Exception:
            return False

    async def delete_vectors(self, vector_ids: list[str]) -> int:
        """
        저장소에서 여러 벡터를 삭제합니다.
        Args:
            vector_ids: 벡터 식별자 목록
        Returns:
            성공적으로 삭제된 벡터 수
        """
        try:
            if not self._connection:
                return 0
            cursor = self._connection.cursor()
            placeholders = ", ".join("?" * len(vector_ids))
            cursor.execute(
                f"""
                DELETE FROM {self.table_name} WHERE id IN ({placeholders})
            """,
                vector_ids,
            )
            deleted_count = cursor.rowcount
            self._connection.commit()
            cursor.close()
            return deleted_count
        except Exception:
            return 0

    async def vector_exists(self, vector_id: str) -> bool:
        """
        저장소에 벡터가 존재하는지 확인합니다.
        Args:
            vector_id: 벡터 식별자
        Returns:
            벡터가 존재하면 True
        """
        try:
            if not self._connection:
                return False
            cursor = self._connection.cursor()
            cursor.execute(
                f"""
                SELECT 1 FROM {self.table_name} WHERE id = ? LIMIT 1
            """,
                (vector_id,),
            )
            exists = cursor.fetchone() is not None
            cursor.close()
            return exists
        except Exception:
            return False

    # 검색 작업 (sqlite-vec 없는 기본 구현)
    async def search_similar(
        self,
        query_vector: Vector,
        k: int = 10,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """
        기본 유사도 계산을 사용하여 유사한 벡터를 검색합니다.
        참고: 이것은 sqlite-vec 확장 없는 기본 구현입니다.
        프로덕션 환경에서는 더 나은 성능을 위해 sqlite-vec 사용을 고려하세요.
        Args:
            query_vector: 쿼리 벡터
            k: 반환할 결과 수
            filter_criteria: 선택적 필터 기준
        Returns:
            (vector_id, similarity_score) 튜플 목록
        """
        try:
            if not self._connection:
                return []
            cursor = self._connection.cursor()
            # 필터에 대한 WHERE 절 빌드
            where_clause = ""
            params = []
            if filter_criteria:
                conditions = []
                for key, value in filter_criteria.items():
                    conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                    params.append(value)
                if conditions:
                    where_clause = f"WHERE {' AND '.join(conditions)}"
            cursor.execute(
                f"""
                SELECT id, vector FROM {self.table_name} {where_clause}
            """,
                params,
            )
            rows = cursor.fetchall()
            cursor.close()
            # 유사도 계산
            similarities = []
            for row in rows:
                vector_id, vector_blob = row
                stored_vector = self._blob_to_vector(vector_blob)
                if stored_vector:
                    similarity = self._calculate_similarity(query_vector, stored_vector)
                    similarities.append((vector_id, similarity))
            # 유사도에 따라 정렬하고 상위 k개 반환
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]
        except Exception:
            return []

    async def search_similar_with_vectors(
        self,
        query_vector: Vector,
        k: int = 10,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[tuple[str, Vector, float]]:
        """
        유사한 벡터를 검색하고 벡터 자체를 반환합니다.
        Args:
            query_vector: 쿼리 벡터
            k: 반환할 결과 수
            filter_criteria: 선택적 필터 기준
        Returns:
            (vector_id, vector, similarity_score) 튜플 목록
        """
        try:
            # 유사한 벡터 ID 및 점수 가져오기
            similar = await self.search_similar(query_vector, k, filter_criteria)
            # 실제 벡터 가져오기
            vector_ids = [item[0] for item in similar]
            vectors = await self.get_vectors(vector_ids)
            # 결과 결합
            result = []
            for vector_id, score in similar:
                vector = vectors.get(vector_id)
                if vector:
                    result.append((vector_id, vector, score))
            return result
        except Exception:
            return []

    async def search_by_ids(
        self, query_vector: Vector, candidate_ids: list[str], k: Optional[int] = None
    ) -> list[tuple[str, float]]:
        """
        특정 벡터 ID 집합 내에서 검색합니다.
        Args:
            query_vector: 쿼리 벡터
            candidate_ids: 후보 벡터 ID 목록
            k: 선택적 결과 제한 (기본값은 모든 후보)
        Returns:
            (vector_id, similarity_score) 튜플 목록
        """
        try:
            # 후보 ID에 대한 벡터 가져오기
            vectors = await self.get_vectors(candidate_ids)
            # 유사도 계산
            similarities = []
            for vector_id, vector in vectors.items():
                if vector:
                    similarity = self._calculate_similarity(query_vector, vector)
                    similarities.append((vector_id, similarity))
            # 정렬 및 제한
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k] if k else similarities
        except Exception:
            return []

    async def batch_search(
        self,
        query_vectors: list[Vector],
        k: int = 10,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[list[tuple[str, float]]]:
        """
        여러 쿼리 벡터에 대해 일괄 검색을 수행합니다.
        Args:
            query_vectors: 쿼리 벡터 목록
            k: 쿼리당 결과 수
            filter_criteria: 선택적 필터 기준
        Returns:
            각 쿼리에 대한 검색 결과 목록
        """
        try:
            results = []
            for query_vector in query_vectors:
                result = await self.search_similar(query_vector, k, filter_criteria)
                results.append(result)
            return results
        except Exception:
            return [[] for _ in query_vectors]

    # 메타데이터 작업
    async def get_metadata(self, vector_id: str) -> dict[str, Any] | None:
        """
        벡터에 대한 메타데이터를 가져옵니다.
        Args:
            vector_id: 벡터 식별자
        Returns:
            찾은 경우 메타데이터 사전, 그렇지 않으면 None
        """
        try:
            if not self._connection:
                return None
            cursor = self._connection.cursor()
            cursor.execute(
                f"""
                SELECT metadata FROM {self.table_name} WHERE id = ?
            """,
                (vector_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            if row and row[0]:
                return json.loads(row[0])  # type: ignore
            return None
        except Exception:
            return None

    async def update_metadata(self, vector_id: str, metadata: dict[str, Any]) -> bool:
        """
        벡터에 대한 메타데이터를 업데이트합니다.
        Args:
            vector_id: 벡터 식별자
            metadata: 새 메타데이터
        Returns:
            업데이트 성공 시 True
        """
        try:
            if not self._connection:
                return False
            cursor = self._connection.cursor()
            metadata_json = json.dumps(metadata)
            cursor.execute(
                f"""
                UPDATE {self.table_name}
                SET metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (metadata_json, vector_id),
            )
            success = cursor.rowcount > 0
            self._connection.commit()
            cursor.close()
            return success
        except Exception:
            return False

    async def search_by_metadata(
        self, filter_criteria: dict[str, Any], limit: int = 100
    ) -> list[str]:
        """
        메타데이터 기준으로 벡터를 검색합니다.
        Args:
            filter_criteria: 메타데이터 필터 기준
            limit: 최대 결과 수
        Returns:
            기준과 일치하는 벡터 ID 목록
        """
        try:
            if not self._connection:
                return []
            cursor = self._connection.cursor()
            # WHERE 절 빌드
            conditions = []
            params = []
            for key, value in filter_criteria.items():
                conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                params.append(value)
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            cursor.execute(
                f"""
                SELECT id FROM {self.table_name} {where_clause} LIMIT ?
            """,
                params + [limit],
            )
            rows = cursor.fetchall()
            cursor.close()
            return [row[0] for row in rows]
        except Exception:
            return []

    # 저장소 정보 및 유지보수
    async def get_store_info(self) -> dict[str, Any]:
        """
        벡터 저장소에 대한 정보를 가져옵니다.
        Returns:
            크기, 차원 등을 포함한 저장소 정보
        """
        try:
            info = {
                "table_name": self.table_name,
                "dimension": self._dimension,
                "metric": self._metric,
                "db_path": str(self.db_path),
            }
            if self._connection:
                cursor = self._connection.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                row = cursor.fetchone()
                info["vector_count"] = row[0] if row else 0
                cursor.close()
            return info
        except Exception:
            return {"error": "저장소 정보를 가져오는 데 실패했습니다"}

    async def get_vector_count(self) -> int:
        """
        저장소의 총 벡터 수를 가져옵니다.
        Returns:
            벡터 수
        """
        try:
            if not self._connection:
                return 0
            cursor = self._connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            row = cursor.fetchone()
            cursor.close()
            return row[0] if row else 0
        except Exception:
            return 0

    async def get_dimension(self) -> int:
        """
        저장소의 벡터 차원을 가져옵니다.
        Returns:
            벡터 차원
        """
        return self._dimension or 0

    async def optimize_store(self) -> dict[str, Any]:
        """
        더 나은 성능을 위해 벡터 저장소를 최적화합니다.
        Returns:
            최적화 결과
        """
        try:
            if not self._connection:
                return {"error": "연결되지 않음"}
            cursor = self._connection.cursor()
            # 공간 회수를 위해 VACUUM 실행
            cursor.execute("VACUUM")
            # 더 나은 쿼리 계획을 위해 테이블 분석
            cursor.execute(f"ANALYZE {self.table_name}")
            cursor.close()
            return {"status": "optimized", "operations": ["vacuum", "analyze"]}
        except Exception as exception:
            return {"error": f"최적화 실패: {str(exception)}"}

    async def rebuild_index(self, parameters: dict[str, Any] | None = None) -> bool:
        """
        벡터 인덱스를 다시 빌드합니다.
        Args:
            parameters: 선택적 리빌드 매개변수
        Returns:
            리빌드 성공 시 True
        """
        try:
            if not self._connection:
                return False
            cursor = self._connection.cursor()
            # 메타데이터 인덱스 삭제 및 재생성
            cursor.execute(f"DROP INDEX IF EXISTS idx_{self.table_name}_metadata")
            cursor.execute(
                f"""
                CREATE INDEX idx_{self.table_name}_metadata
                ON {self.table_name} (metadata)
            """
            )
            self._connection.commit()
            cursor.close()
            return True
        except Exception:
            return False

    async def clear_store(self) -> bool:
        """
        저장소에서 모든 벡터를 지웁니다.
        Returns:
            지우기 성공 시 True
        """
        try:
            if not self._connection:
                return False
            cursor = self._connection.cursor()
            cursor.execute(f"DELETE FROM {self.table_name}")
            self._connection.commit()
            cursor.close()
            return True
        except Exception:
            return False

    # 백업 및 복구
    async def create_snapshot(self, snapshot_path: str) -> bool:
        """
        벡터 저장소의 스냅샷을 생성합니다.
        Args:
            snapshot_path: 스냅샷을 저장할 경로
        Returns:
            스냅샷 생성 성공 시 True
        """
        try:
            if not self._connection:
                return False
            # 일시적으로 연결 닫기
            self._connection.close()
            # 데이터베이스 파일 복사
            shutil.copy2(self.db_path, snapshot_path)
            # 다시 연결
            await self.connect()
            return True
        except Exception:
            return False

    async def restore_snapshot(self, snapshot_path: str) -> bool:
        """
        스냅샷에서 벡터 저장소를 복원합니다.
        Args:
            snapshot_path: 스냅샷 파일 경로
        Returns:
            복원 성공 시 True
        """
        try:
            if not Path(snapshot_path).exists():
                return False
            # 연결 닫기
            if self._connection:
                self._connection.close()
            # 스냅샷을 데이터베이스 위치로 복사
            shutil.copy2(snapshot_path, self.db_path)
            # 다시 연결하고 설정 다시 로드
            await self.connect()
            return True
        except Exception:
            return False

    # 상태 및 진단
    async def health_check(self) -> dict[str, Any]:
        """
        벡터 저장소에 대한 상태 확인을 수행합니다.
        Returns:
            상태 정보
        """
        health: dict[str, Any] = {
            "connected": await self.is_connected(),
            "table_exists": False,
            "config_loaded": self._dimension is not None,
        }
        if health["connected"] and self._connection:
            try:
                cursor = self._connection.cursor()
                cursor.execute(
                    f"""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='{self.table_name}'
                """
                )
                health["table_exists"] = cursor.fetchone() is not None
                cursor.close()
            except Exception:
                pass
        health["status"] = (
            "healthy"
            if all([health["connected"], health["table_exists"], health["config_loaded"]])
            else "unhealthy"
        )
        return health

    async def get_performance_stats(self) -> dict[str, Any]:
        """
        벡터 저장소의 성능 통계를 가져옵니다.
        Returns:
            성능 통계
        """
        try:
            stats = await self.get_store_info()
            if self._connection:
                cursor = self._connection.cursor()
                # 테이블 정보 가져오기
                cursor.execute(f"PRAGMA table_info({self.table_name})")
                table_info = cursor.fetchall()
                stats["columns"] = len(table_info)
                cursor.close()
            return stats
        except Exception:
            return {"error": "성능 통계를 가져오는 데 실패했습니다"}

    # 헬퍼 메서드
    def _vector_to_blob(self, vector: Vector | list[float]) -> bytes:
        """저장을 위해 벡터를 바이트로 변환합니다."""
        if hasattr(vector, "values"):
            values = vector.values
        else:
            values = vector
        return struct.pack(f"{len(values)}f", *values)

    def _blob_to_vector(self, blob: bytes) -> Vector:
        """바이트를 다시 벡터로 변환합니다."""
        values = list(struct.unpack(f"{len(blob)//4}f", blob))
        return Vector(values)

    def _calculate_similarity(
        self, vector1: Vector | list[float], vector2: Vector | list[float]
    ) -> float:
        """두 벡터 간의 코사인 유사도를 계산합니다."""
        try:
            # 벡터 값 가져오기
            v1_values = vector1.values if hasattr(vector1, "values") else vector1
            v2_values = vector2.values if hasattr(vector2, "values") else vector2
            # 벡터 차원이 동일한지 확인
            if len(v1_values) != len(v2_values):
                return 0.0
            # 내적 및 크기 계산
            dot_product = sum(a * b for a, b in zip(v1_values, v2_values, strict=False))
            magnitude1 = math.sqrt(sum(a * a for a in v1_values))
            magnitude2 = math.sqrt(sum(a * a for a in v2_values))
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            return float(dot_product / (magnitude1 * magnitude2))
        except Exception:
            return 0.0

    async def _load_config(self) -> None:
        """데이터베이스에서 설정을 로드합니다."""
        try:
            if not self._connection:
                return
            cursor = self._connection.cursor()
            cursor.execute(
                """
                SELECT dimension, metric, parameters
                FROM vector_store_config
                WHERE table_name = ?
            """,
                (self.table_name,),
            )
            row = cursor.fetchone()
            if row:
                self._dimension = row[0]
                self._metric = row[1]
                # 매개변수는 JSON으로 저장되지만 이 기본 구현에서는 사용되지 않습니다
            cursor.close()
        except Exception:
            # 설정 테이블이 아직 존재하지 않을 수 있습니다
            pass

    # VectorStore의 추상 메서드 구현 (LangChain 호환성)
    async def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        raise NotImplementedError("add_documents는 SQLiteVectorStore에 구현되지 않았습니다")

    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        # 이것은 쿼리를 임베딩하고 search_similar를 호출하여 구현할 수 있습니다
        raise NotImplementedError("similarity_search는 SQLiteVectorStore에 구현되지 않았습니다")

    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        # 이것은 쿼리를 임베딩하고 search_similar를 호출하여 구현할 수 있습니다
        raise NotImplementedError(
            "similarity_search_with_score는 SQLiteVectorStore에 구현되지 않았습니다"
        )

    async def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        # 이것은 search_similar를 직접 호출하여 구현할 수 있습니다
        raise NotImplementedError(
            "similarity_search_by_vector는 SQLiteVectorStore에 구현되지 않았습니다"
        )

    async def delete(self, ids: list[str] | None = None, **kwargs: Any) -> Optional[bool]:
        # 이것은 delete_vectors를 호출하여 구현할 수 있습니다
        raise NotImplementedError("delete는 SQLiteVectorStore에 구현되지 않았습니다")

    @classmethod
    async def from_documents(
        cls,
        documents: list[Document],
        embedding: Any,
        **kwargs: Any,
    ) -> "VectorStore":
        raise NotImplementedError("from_documents는 SQLiteVectorStore에 구현되지 않았습니다")

    @classmethod
    async def from_texts(
        cls,
        texts: list[str],
        embedding: Any,
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> "VectorStore":
        raise NotImplementedError("from_texts는 SQLiteVectorStore에 구현되지 않았습니다")

    def as_retriever(self, **kwargs: Any) -> Any:
        raise NotImplementedError("as_retriever는 SQLiteVectorStore에 구현되지 않았습니다")
