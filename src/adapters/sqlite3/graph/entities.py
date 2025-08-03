"""
지식 그래프의 엔티티(노드) 관리.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any

from src.adapters.sqlite3.transactions import UnitOfWork


@dataclass
class Entity:
    """지식 그래프의 노드를 나타냅니다."""

    id: int
    uuid: str
    name: str | None
    type: str
    properties: dict[str, Any]
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Entity:
        """
        데이터베이스 행에서 엔티티를 생성합니다.
        Args:
            row: 엔티티 데이터가 있는 SQLite Row 객체
        Returns:
            엔티티 객체
        """
        # 필요한 경우 JSON 속성 파싱
        properties = row["properties"]
        if isinstance(properties, str):
            properties = json.loads(properties)
        elif properties is None:
            properties = {}
        return cls(
            id=row["id"],
            uuid=row["uuid"],
            name=row["name"],
            type=row["type"],
            properties=properties,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class EntityManager:
    """
    지식 그래프의 엔티티(노드) 작업을 관리합니다.
    """

    def __init__(self, connection: sqlite3.Connection):
        """
        엔티티 관리자를 초기화합니다.
        Args:
            connection: SQLite 데이터베이스 연결
        """
        self.connection = connection
        self.unit_of_work = UnitOfWork(connection)

    def create_entity(
        self,
        entity_type: str,
        name: str | None = None,
        properties: dict[str, Any] | None = None,
        custom_uuid: str | None = None,
    ) -> Entity:
        """
        지식 그래프에 새 엔티티를 생성합니다.
        Args:
            entity_type: 엔티티 유형
            name: 엔티티의 선택적 이름
            properties: 선택적 속성 사전
            custom_uuid: 선택적 사용자 지정 UUID (제공되지 않으면 생성됨)
        Returns:
            생성된 엔티티 객체
        """
        entity_uuid = custom_uuid or str(uuid.uuid4())
        props = properties or {}
        with self.unit_of_work.begin() as conn:
            cursor = conn.cursor()
            # 엔티티 삽입
            cursor.execute(
                """
            INSERT INTO entities (uuid, name, type, properties)
            VALUES (?, ?, ?, ?)
            """,
                (entity_uuid, name, entity_type, json.dumps(props)),
            )
            entity_id = cursor.lastrowid
            if entity_id is None:
                raise RuntimeError("엔티티 삽입 실패")
            # 필요한 경우 벡터 처리를 위해 등록
            self.unit_of_work.register_vector_operation(
                entity_type="node", entity_id=entity_id, operation_type="insert"
            )
            # 생성된 엔티티 가져오기
            cursor.execute(
                """
            SELECT * FROM entities WHERE id = ?
            """,
                (entity_id,),
            )
            return Entity.from_row(cursor.fetchone())

    def get_entity(self, entity_id: int) -> Entity | None:
        """
        ID로 엔티티를 가져옵니다.
        Args:
            entity_id: 엔티티 ID
        Returns:
            엔티티 객체 또는 찾을 수 없는 경우 None
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
        row = cursor.fetchone()
        return Entity.from_row(row) if row else None

    def get_entity_by_uuid(self, entity_uuid: str) -> Entity | None:
        """
        UUID로 엔티티를 가져옵니다.
        Args:
            entity_uuid: 엔티티 UUID
        Returns:
            엔티티 객체 또는 찾을 수 없는 경우 None
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM entities WHERE uuid = ?", (entity_uuid,))
        row = cursor.fetchone()
        return Entity.from_row(row) if row else None

    def update_entity(
        self,
        entity_id: int,
        name: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> Entity | None:
        """
        엔티티의 속성을 업데이트합니다.
        Args:
            entity_id: 엔티티 ID
            name: 새 이름 (None이면 변경되지 않음)
            properties: 새 속성 (None이면 변경되지 않음)
        Returns:
            업데이트된 엔티티 객체 또는 찾을 수 없는 경우 None
        """
        # 속성을 병합하기 위해 현재 엔티티를 가져옵니다.
        current_entity = self.get_entity(entity_id)
        if not current_entity:
            return None
        # 업데이트 준비
        updates = []
        params = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if properties is not None:
            # 필요한 경우 기존 속성과 병합
            merged_props = {**current_entity.properties, **properties}
            updates.append("properties = ?")
            params.append(json.dumps(merged_props))
        if not updates:
            return current_entity  # 업데이트할 내용 없음
        # 업데이트 실행
        with self.unit_of_work.begin() as conn:
            cursor = conn.cursor()
            query = f"UPDATE entities SET {', '.join(updates)} WHERE id = ?"
            params.append(str(entity_id))
            cursor.execute(query, params)
            if cursor.rowcount > 0:
                # 벡터 처리를 위해 등록
                self.unit_of_work.register_vector_operation(
                    entity_type="node", entity_id=entity_id, operation_type="update"
                )
                # 업데이트된 엔티티 가져오기
                cursor.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
                return Entity.from_row(cursor.fetchone())
        return None

    def delete_entity(self, entity_id: int) -> bool:
        """
        지식 그래프에서 엔티티를 삭제합니다.
        Args:
            entity_id: 엔티티 ID
        Returns:
            삭제된 경우 True, 찾을 수 없는 경우 False
        """
        with self.unit_of_work.begin() as conn:
            cursor = conn.cursor()
            # 삭제 전 벡터 처리를 위해 등록
            self.unit_of_work.register_vector_operation(
                entity_type="node", entity_id=entity_id, operation_type="delete"
            )
            # 엔티티 삭제 (관련 테이블로 전파됨)
            cursor.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
            return cursor.rowcount > 0

    def find_entities(
        self,
        entity_type: str | None = None,
        name_pattern: str | None = None,
        property_filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Entity], int]:
        """
        주어진 기준과 일치하는 엔티티를 찾습니다.
        Args:
            entity_type: 선택적 엔티티 유형 필터
            name_pattern: 선택적 이름 패턴 (SQL LIKE 패턴)
            property_filters: 선택적 속성 필터
            limit: 최대 결과 수
            offset: 페이지네이션을 위한 쿼리 오프셋
        Returns:
            (엔티티 객체 목록, 총 개수) 튜플
        """
        # 쿼리 조건 빌드
        conditions = []
        params = []
        if entity_type:
            conditions.append("type = ?")
            params.append(entity_type)
        if name_pattern:
            conditions.append("name LIKE ?")
            params.append(name_pattern)
        # 속성 필터는 JSON으로 특별한 처리가 필요합니다.
        # 이것은 단순화되었으며 프로덕션 환경에서는 최적화가 필요할 수 있습니다.
        property_clauses = []
        if property_filters:
            for key, value in property_filters.items():
                # JSON 속성을 쿼리하기 위해 JSON_EXTRACT 사용
                property_clauses.append(f"JSON_EXTRACT(properties, '$.{key}') = ?")
                params.append(value)
        if property_clauses:
            conditions.extend(property_clauses)
        # 최종 쿼리 빌드
        query = "SELECT DISTINCT * FROM entities"
        count_query = "SELECT COUNT(DISTINCT id) FROM entities"
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
            query += where_clause
            count_query += where_clause
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        # SQL 매개변수를 위해 문자열로 변환
        params.extend([str(limit), str(offset)])
        # 쿼리 실행
        cursor = self.connection.cursor()
        # 총 개수 가져오기
        cursor.execute(count_query, params[:-2] if params else [])
        total_count = cursor.fetchone()[0]
        # 엔티티 가져오기
        cursor.execute(query, params)
        entities = [Entity.from_row(row) for row in cursor.fetchall()]
        return entities, total_count
