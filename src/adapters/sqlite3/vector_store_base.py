"""
SQLite 벡터 저장소 어댑터들의 공통 기본 클래스.
"""

import json
import math
import struct

# ABC 제거 - 공통 베이스 클래스로 사용
from pathlib import Path
from sqlite3 import Connection
from typing import Any

from src.domain import Vector

from .connection import DatabaseConnection


class SQLiteVectorStoreBase:
    """
    SQLite 벡터 저장소 어댑터들의 공통 기본 클래스.

    연결 관리, 트랜잭션, 유틸리티 메서드 등 공통 기능을 제공합니다.
    """

    def __init__(self, db_path: str, table_name: str = "vectors", optimize: bool = True):
        """
        SQLite 벡터 저장소 베이스를 초기화합니다.

        Args:
            db_path: SQLite 데이터베이스 파일 경로
            table_name: 벡터를 저장할 테이블 이름
            optimize: 최적화 PRAGMA 적용 여부
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.optimize = optimize
        self._connection_manager = DatabaseConnection(db_path, optimize)
        self._connection: Connection | None = None
        self._dimension: int | None = None
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
            연결되어 있으면 True
        """
        return self._connection is not None

    # 유틸리티 메서드
    def _serialize_vector(self, vector: Vector) -> bytes:
        """벡터를 바이너리 형식으로 직렬화합니다."""
        # 바이트 순서 + 차원 + 벡터 데이터
        return struct.pack("<I", len(vector.values)) + struct.pack(
            f"<{len(vector.values)}f", *vector.values
        )

    def _deserialize_vector(self, blob: bytes) -> Vector:
        """바이너리 데이터에서 벡터를 역직렬화합니다."""
        dimension = struct.unpack("<I", blob[:4])[0]
        values = list(struct.unpack(f"<{dimension}f", blob[4 : 4 + dimension * 4]))
        return Vector(values=values)

    def _serialize_metadata(self, metadata: dict[str, Any]) -> str:
        """메타데이터를 JSON 문자열로 직렬화합니다."""
        return json.dumps(metadata, ensure_ascii=False)

    def _deserialize_metadata(self, json_str: str) -> dict[str, Any]:
        """JSON 문자열에서 메타데이터를 역직렬화합니다."""
        return json.loads(json_str) if json_str else {}

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """두 벡터 간의 코사인 유사도를 계산합니다."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(a * a for a in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _euclidean_distance(self, vec1: list[float], vec2: list[float]) -> float:
        """두 벡터 간의 유클리드 거리를 계산합니다."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2, strict=True)))

    def _get_dimension(self) -> int:
        """저장소의 벡터 차원을 반환합니다."""
        if self._dimension is None:
            raise ValueError("벡터 저장소가 초기화되지 않았습니다")
        return self._dimension

    def _get_metric(self) -> str:
        """저장소의 거리 메트릭을 반환합니다."""
        return self._metric

    async def _load_config(self) -> None:
        """저장된 설정을 로드합니다."""
        if not self._connection:
            return

        cursor = self._connection.cursor()
        try:
            cursor.execute(
                "SELECT dimension, metric FROM vector_store_config WHERE table_name = ?",
                (self.table_name,),
            )
            result = cursor.fetchone()
            if result:
                self._dimension, self._metric = result
        except Exception:
            # 설정 테이블이 없거나 오류가 발생한 경우 무시
            pass
        finally:
            cursor.close()

    def _ensure_connected(self) -> Connection:
        """연결이 설정되어 있는지 확인하고 Connection 객체를 반환합니다."""
        if not self._connection:
            raise RuntimeError("데이터베이스에 연결되지 않았습니다")
        return self._connection

    def _validate_vector_dimension(self, vector: Vector) -> None:
        """벡터 차원이 올바른지 검증합니다."""
        expected_dim = self._get_dimension()
        if len(vector.values) != expected_dim:
            raise ValueError(
                f"벡터 차원이 일치하지 않습니다: 예상 {expected_dim}, 실제 {len(vector.values)}"
            )
