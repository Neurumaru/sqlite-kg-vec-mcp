"""
SQLite 벡터 저장소의 쓰기 작업 구현체.
"""

import shutil
from typing import Any, Optional

from src.domain.value_objects.document_metadata import DocumentMetadata
from src.domain.value_objects.vector import Vector
from src.ports.vector_writer import VectorWriter

from .vector_store_base import SQLiteVectorStoreBase


class SQLiteVectorWriter(SQLiteVectorStoreBase, VectorWriter):
    """
    SQLite를 사용한 VectorWriter 포트의 구현체.

    벡터와 문서의 추가, 수정, 삭제 작업을 담당합니다.
    """

    async def add_documents(self, documents: list[DocumentMetadata], **kwargs: Any) -> list[str]:
        """
        문서를 벡터 저장소에 추가합니다.

        Args:
            documents: 추가할 DocumentMetadata 객체 목록
            **kwargs: 추가 옵션

        Returns:
            추가된 문서 ID 목록
        """
        try:
            conn = self._ensure_connected()
            cursor = conn.cursor()

            document_ids: list[str] = []
            for i, doc in enumerate(documents):
                # 문서 ID가 없으면 생성
                document_id = f"doc_{i}"

                # 문서를 저장 (벡터 없이)
                metadata_json = self._serialize_metadata(
                    {
                        "source": doc.source,
                        "content": doc.content,
                        "created_at": None,
                        "updated_at": None,
                        **doc.metadata,
                    }
                )

                # 더미 벡터 (실제 벡터는 add_vectors에서 업데이트)
                dummy_vector = Vector(values=[0.0] * self._get_dimension())
                vector_blob = self._serialize_vector(dummy_vector)

                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO {self.table_name}
                    (id, vector, metadata, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                    (document_id, vector_blob, metadata_json),
                )

                document_ids.append(document_id)

            conn.commit()
            cursor.close()
            return document_ids
        except Exception as e:
            raise RuntimeError(f"문서 추가 실패: {e}") from e

    async def add_vectors(
        self, vectors: list[Vector], documents: list[DocumentMetadata], **kwargs: Any
    ) -> list[str]:
        """
        벡터와 문서를 함께 저장소에 추가합니다.

        Args:
            vectors: 추가할 Vector 객체 목록
            documents: 벡터에 대응하는 DocumentMetadata 객체 목록
            **kwargs: 추가 옵션

        Returns:
            추가된 문서 ID 목록
        """
        if len(vectors) != len(documents):
            raise ValueError("벡터와 문서의 개수가 일치하지 않습니다")

        try:
            conn = self._ensure_connected()
            cursor = conn.cursor()

            document_ids: list[str] = []
            for i, (vector, doc) in enumerate(zip(vectors, documents, strict=True)):
                # 벡터 차원 검증
                self._validate_vector_dimension(vector)

                # 문서 ID가 없으면 생성
                document_id = f"doc_{i}"

                # 벡터 직렬화
                vector_blob = self._serialize_vector(vector)

                # 메타데이터 직렬화
                metadata_json = self._serialize_metadata(
                    {
                        "source": doc.source,
                        "content": doc.content,
                        "created_at": None,
                        "updated_at": None,
                        **doc.metadata,
                    }
                )

                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO {self.table_name}
                    (id, vector, metadata, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                    (document_id, vector_blob, metadata_json),
                )

                document_ids.append(document_id)

            conn.commit()
            cursor.close()
            return document_ids
        except Exception as e:
            raise RuntimeError(f"벡터 추가 실패: {e}") from e

    async def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> bool:
        """
        문서를 삭제합니다.

        Args:
            ids: 삭제할 문서 ID 목록
            **kwargs: 추가 삭제 옵션

        Returns:
            삭제 성공 여부
        """
        try:
            conn = self._ensure_connected()
            cursor = conn.cursor()

            if ids is None:
                # 모든 문서 삭제
                cursor.execute(f"DELETE FROM {self.table_name}")
            else:
                # 특정 ID들 삭제
                placeholders = ",".join("?" * len(ids))
                cursor.execute(
                    f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})",
                    ids,
                )

            deleted_count = cursor.rowcount
            conn.commit()
            cursor.close()

            return deleted_count > 0
        except Exception as e:
            raise RuntimeError(f"문서 삭제 실패: {e}") from e

    async def update_document(
        self,
        document_id: str,
        document: DocumentMetadata,
        vector: Optional[Vector] = None,
        **kwargs: Any,
    ) -> bool:
        """
        문서와 벡터를 업데이트합니다.

        Args:
            document_id: 업데이트할 문서 ID
            document: 새로운 DocumentMetadata
            vector: 새로운 Vector (선택사항)
            **kwargs: 추가 업데이트 옵션

        Returns:
            업데이트 성공 여부
        """
        try:
            conn = self._ensure_connected()
            cursor = conn.cursor()

            # 기존 문서 확인
            cursor.execute(
                f"SELECT vector, metadata FROM {self.table_name} WHERE id = ?",
                (document_id,),
            )
            existing = cursor.fetchone()

            if not existing:
                cursor.close()
                return False

            # 벡터 처리
            if vector is not None:
                self._validate_vector_dimension(vector)
                vector_blob = self._serialize_vector(vector)
            else:
                # 기존 벡터 유지
                vector_blob = existing[0]

            # 메타데이터 업데이트
            metadata_json = self._serialize_metadata(
                {
                    "source": document.source,
                    "content": document.content,
                    "created_at": None,
                    "updated_at": None,
                    **document.metadata,
                }
            )

            cursor.execute(
                f"""
                UPDATE {self.table_name}
                SET vector = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (vector_blob, metadata_json, document_id),
            )

            updated = cursor.rowcount > 0
            conn.commit()
            cursor.close()

            return updated
        except Exception as e:
            raise RuntimeError(f"문서 업데이트 실패: {e}") from e

    # 백업 및 유지보수 작업
    async def backup_store(self, backup_path: str) -> bool:
        """
        벡터 저장소를 백업합니다.

        Args:
            backup_path: 백업 파일 경로

        Returns:
            백업 성공 여부
        """
        try:
            shutil.copy2(self.db_path, backup_path)
            return True
        except Exception:
            return False

    async def restore_store(self, backup_path: str) -> bool:
        """
        백업에서 벡터 저장소를 복원합니다.

        Args:
            backup_path: 백업 파일 경로

        Returns:
            복원 성공 여부
        """
        try:
            # 기존 연결 닫기
            await self.disconnect()
            # 백업에서 복원
            shutil.copy2(backup_path, self.db_path)
            # 다시 연결
            await self.connect()
            return True
        except Exception:
            return False

    async def optimize_store(self) -> bool:
        """
        벡터 저장소를 최적화합니다.

        Returns:
            최적화 성공 여부
        """
        try:
            conn = self._ensure_connected()
            cursor = conn.cursor()

            # SQLite 최적화 작업
            cursor.execute("VACUUM")
            cursor.execute("ANALYZE")
            cursor.execute(f"REINDEX {self.table_name}")

            conn.commit()
            cursor.close()
            return True
        except Exception:
            return False
