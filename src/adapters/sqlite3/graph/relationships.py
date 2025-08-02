"""
지식 그래프의 관계(엣지) 관리.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Optional

from ..transactions import UnitOfWork
from .entities import Entity


@dataclass
class Relationship:
    """지식 그래프의 이진 관계(엣지)를 나타냅니다."""

    id: int
    source_id: int
    target_id: int
    relation_type: str
    properties: dict[str, Any]
    created_at: str
    updated_at: str
    # 이 필드들은 상세 정보를 로드할 때 채워집니다.
    source: Optional[Entity] = None
    target: Optional[Entity] = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Relationship:
        """
        데이터베이스 행에서 관계를 생성합니다.
        Args:
            row: 관계 데이터가 있는 SQLite Row 객체
        Returns:
            관계 객체
        """
        # 필요한 경우 JSON 속성 파싱
        properties = row["properties"]
        if isinstance(properties, str):
            properties = json.loads(properties)
        elif properties is None:
            properties = {}
        return cls(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation_type=row["relation_type"],
            properties=properties,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class RelationshipManager:
    """
    지식 그래프의 관계(엣지) 작업을 관리합니다.
    """

    def __init__(self, connection: sqlite3.Connection):
        """
        관계 관리자를 초기화합니다.
        Args:
            connection: SQLite 데이터베이스 연결
        """
        self.connection = connection
        self.unit_of_work = UnitOfWork(connection)

    def create_relationship(
        self,
        source_id: int,
        target_id: int,
        relation_type: str,
        properties: dict[str, Any] | None = None,
    ) -> Relationship:
        """
        두 엔티티 사이에 새로운 관계(엣지)를 생성합니다.
        Args:
            source_id: 소스 엔티티 ID
            target_id: 대상 엔티티 ID
            relation_type: 관계 유형
            properties: 선택적 속성 사전
        Returns:
            생성된 관계 객체
        Raises:
            ValueError: 소스 또는 대상 엔티티가 존재하지 않는 경우
        """
        # 소스 및 대상 엔티티가 존재하는지 확인
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM entities WHERE id IN (?, ?)", (source_id, target_id))
        if cursor.fetchone()[0] != 2:
            raise ValueError("소스 또는 대상 엔티티가 존재하지 않습니다")
        props = properties or {}
        with self.unit_of_work.begin() as conn:
            cursor = conn.cursor()
            # 관계 삽입
            cursor.execute(
                """
            INSERT INTO edges (source_id, target_id, relation_type, properties)
            VALUES (?, ?, ?, ?)
            """,
                (source_id, target_id, relation_type, json.dumps(props)),
            )
            edge_id = cursor.lastrowid
            if edge_id is None:
                raise RuntimeError("엣지 삽입 실패")
            # 필요한 경우 벡터 처리를 위해 등록
            self.unit_of_work.register_vector_operation(
                entity_type="edge", entity_id=edge_id, operation_type="insert"
            )
            # 생성된 관계 가져오기
            cursor.execute(
                """
            SELECT * FROM edges WHERE id = ?
            """,
                (edge_id,),
            )
            return Relationship.from_row(cursor.fetchone())

    def get_relationship(
        self, relationship_id: int, include_entities: bool = False
    ) -> Optional[Relationship]:
        """
        ID로 관계를 가져옵니다.
        Args:
            relationship_id: 관계 ID
            include_entities: 소스 및 대상 엔티티 포함 여부
        Returns:
            관계 객체 또는 찾을 수 없는 경우 None
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM edges WHERE id = ?", (relationship_id,))
        row = cursor.fetchone()
        if not row:
            return None
        relationship = Relationship.from_row(row)
        # 요청된 경우 소스 및 대상 엔티티 포함
        if include_entities:
            self._load_relationship_entities(relationship)
        return relationship

    def _load_relationship_entities(self, relationship: Relationship) -> None:
        """
        관계에 대한 소스 및 대상 엔티티를 로드합니다.
        Args:
            relationship: 채울 관계 객체
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM entities WHERE id IN (?, ?)",
            (relationship.source_id, relationship.target_id),
        )
        entities = {row["id"]: Entity.from_row(row) for row in cursor.fetchall()}
        relationship.source = entities.get(relationship.source_id)
        relationship.target = entities.get(relationship.target_id)

    def update_relationship(
        self, relationship_id: int, properties: dict[str, Any]
    ) -> Optional[Relationship]:
        """
        관계의 속성을 업데이트합니다.
        Args:
            relationship_id: 관계 ID
            properties: 기존 속성과 병합할 새 속성
        Returns:
            업데이트된 관계 객체 또는 찾을 수 없는 경우 None
        """
        # 속성을 병합하기 위해 현재 관계를 가져옵니다.
        current = self.get_relationship(relationship_id)
        if not current:
            return None
        # 기존 속성과 병합
        merged_props = {**current.properties, **properties}
        with self.unit_of_work.begin() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE edges SET properties = ? WHERE id = ?",
                (json.dumps(merged_props), relationship_id),
            )
            if cursor.rowcount > 0:
                # 벡터 처리를 위해 등록
                self.unit_of_work.register_vector_operation(
                    entity_type="edge",
                    entity_id=relationship_id,
                    operation_type="update",
                )
                # 업데이트된 관계 가져오기
                cursor.execute("SELECT * FROM edges WHERE id = ?", (relationship_id,))
                return Relationship.from_row(cursor.fetchone())
        return None

    def delete_relationship(self, relationship_id: int) -> bool:
        """
        지식 그래프에서 관계를 삭제합니다.
        Args:
            relationship_id: 관계 ID
        Returns:
            삭제된 경우 True, 찾을 수 없는 경우 False
        """
        with self.unit_of_work.begin() as conn:
            cursor = conn.cursor()
            # 삭제 전 벡터 처리를 위해 등록
            self.unit_of_work.register_vector_operation(
                entity_type="edge", entity_id=relationship_id, operation_type="delete"
            )
            # 관계 삭제
            cursor.execute("DELETE FROM edges WHERE id = ?", (relationship_id,))
            return cursor.rowcount > 0

    def find_relationships(
        self,
        source_id: Optional[int] = None,
        target_id: Optional[int] = None,
        relation_type: Optional[str] = None,
        property_filters: dict[str, Any] | None = None,
        include_entities: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Relationship], int]:
        """
        주어진 기준과 일치하는 관계를 찾습니다.
        Args:
            source_id: 선택적 소스 엔티티 ID 필터
            target_id: 선택적 대상 엔티티 ID 필터
            relation_type: 선택적 관계 유형 필터
            property_filters: 선택적 속성 필터
            include_entities: 소스 및 대상 엔티티 포함 여부
            limit: 최대 결과 수
            offset: 페이지네이션을 위한 쿼리 오프셋
        Returns:
            (관계 객체 목록, 총 개수) 튜플
        """
        # 쿼리 조건 빌드
        conditions = []
        params = []
        if source_id is not None:
            conditions.append("source_id = ?")
            params.append(source_id)
        if target_id is not None:
            conditions.append("target_id = ?")
            params.append(target_id)
        if relation_type is not None:
            conditions.append("relation_type = ?")
            params.append(relation_type)  # type: ignore
        # 속성 필터는 JSON으로 특별한 처리가 필요합니다.
        property_clauses = []
        if property_filters:
            for key, value in property_filters.items():
                property_clauses.append(f"JSON_EXTRACT(properties, '$.{key}') = ?")
                params.append(value)
        if property_clauses:
            conditions.extend(property_clauses)
        # 최종 쿼리 빌드
        query = "SELECT * FROM edges"
        count_query = "SELECT COUNT(*) FROM edges"
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
            query += where_clause
            count_query += where_clause
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        # 쿼리 실행
        cursor = self.connection.cursor()
        # 총 개수 가져오기
        cursor.execute(count_query, params[:-2] if params else [])
        total_count = cursor.fetchone()[0]
        # 관계 가져오기
        cursor.execute(query, params)
        relationships = [Relationship.from_row(row) for row in cursor.fetchall()]
        # 요청된 경우 엔티티 로드
        if include_entities and relationships:
            for relationship in relationships:
                self._load_relationship_entities(relationship)
        return relationships, total_count

    def bulk_create_relationships(
        self, relationships_data: list[tuple[int, int, str, dict[str, Any] | None]]
    ) -> list[int]:
        """
        관계를 대량으로 생성합니다.
        Args:
            relationships_data: (source_id, target_id, relation_type, properties) 튜플 목록
        Returns:
            생성된 관계 ID 목록
        """
        if not relationships_data:
            return []

        created_ids = []
        with self.unit_of_work.begin() as conn:
            cursor = conn.cursor()
            for source_id, target_id, relation_type, properties in relationships_data:
                props = properties or {}
                cursor.execute(
                    """
                INSERT INTO edges (source_id, target_id, relation_type, properties)
                VALUES (?, ?, ?, ?)
                """,
                    (source_id, target_id, relation_type, json.dumps(props)),
                )
                edge_id = cursor.lastrowid
                if edge_id is None:
                    raise RuntimeError("대량 생성 중 엣지 삽입 실패")
                self.unit_of_work.register_vector_operation(
                    entity_type="edge", entity_id=edge_id, operation_type="insert"
                )
                created_ids.append(edge_id)
        return created_ids

    def get_relationship_count(self) -> int:
        """
        그래프의 총 관계 수를 가져옵니다.
        Returns:
            총 관계 수
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM edges")
        result = cursor.fetchone()
        return int(result[0]) if result and result[0] is not None else 0

    def get_relationship_count_by_type(self, relation_type: str) -> int:
        """
        특정 유형의 관계 수를 가져옵니다.
        Args:
            relation_type: 관계 유형
        Returns:
            지정된 유형의 관계 수
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM edges WHERE relation_type = ?", (relation_type,))
        result = cursor.fetchone()
        return int(result[0]) if result and result[0] is not None else 0

    def get_entity_relationships(
        self,
        entity_id: int,
        direction: str = "both",
        relation_types: list[str] | None = None,
        include_entities: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Relationship], int]:
        """
        특정 엔티티의 관계를 가져옵니다.
        Args:
            entity_id: 엔티티 ID
            direction: 'outgoing', 'incoming', 또는 'both'
            relation_types: 필터링할 관계 유형의 선택적 목록
            include_entities: 관련 엔티티 포함 여부
            limit: 최대 결과 수
            offset: 페이지네이션을 위한 쿼리 오프셋
        Returns:
            (관계 객체 목록, 총 개수) 튜플
        Raises:
            ValueError: 방향이 잘못된 경우
        """
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError("방향은 'outgoing', 'incoming', 또는 'both'여야 합니다")
        # 방향에 따라 조건 빌드
        conditions = []
        params = []
        if direction == "outgoing":
            conditions.append("source_id = ?")
            params.append(entity_id)
        elif direction == "incoming":
            conditions.append("target_id = ?")
            params.append(entity_id)
        else:  # 'both'
            conditions.append("(source_id = ? OR target_id = ?)")
            params.extend([entity_id, entity_id])
        # 제공된 경우 관계 유형 필터 추가
        if relation_types:
            placeholders = ", ".join(["?"] * len(relation_types))
            conditions.append(f"relation_type IN ({placeholders})")
            params.extend(relation_types)  # type: ignore
        # 최종 쿼리 빌드
        query = "SELECT * FROM edges"
        count_query = "SELECT COUNT(*) FROM edges"
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
            query += where_clause
            count_query += where_clause
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        # 쿼리 실행
        cursor = self.connection.cursor()
        # 총 개수 가져오기
        cursor.execute(count_query, params[:-2])
        total_count = cursor.fetchone()[0]
        # 관계 가져오기
        cursor.execute(query, params)
        relationships = [Relationship.from_row(row) for row in cursor.fetchall()]
        # 요청된 경우 엔티티 로드
        if include_entities and relationships:
            for relationship in relationships:
                self._load_relationship_entities(relationship)
        return relationships, total_count
