"""
지식 그래프 탐색을 위한 그래프 순회 알고리즘.
"""

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Optional

from src.common.observability import get_observable_logger

from .entities import Entity
from .relationships import Relationship


@dataclass
class PathNode:
    """그래프 순회 중 경로의 노드를 나타냅니다."""

    entity: Entity
    relationship: Optional[Relationship] = None
    parent: Optional["PathNode"] = None
    depth: int = 0

    @property
    def path_to_root(self) -> list["PathNode"]:
        """이 노드에서 루트까지의 경로를 가져옵니다."""
        result = [self]
        current = self
        while current.parent:
            current = current.parent
            result.insert(0, current)
        return result


class GraphTraversal:
    """
    지식 그래프 탐색을 위한 그래프 순회 알고리즘을 구현합니다.
    """

    def __init__(self, connection: sqlite3.Connection):
        """
        그래프 순회 유틸리티를 초기화합니다.
        Args:
            connection: SQLite 데이터베이스 연결
        """
        self.connection = connection

    def get_neighbors(
        self,
        entity_id: int,
        direction: str = "both",
        relation_types: Optional[list[str]] = None,
        entity_types: Optional[list[str]] = None,
        limit: int = 100,
    ) -> list[tuple[Entity, Relationship]]:
        """
        주어진 엔티티에 연결된 이웃 엔티티를 가져옵니다.
        Args:
            entity_id: 루트 엔티티 ID
            direction: 'outgoing'(나가는), 'incoming'(들어오는), 또는 'both'(양방향)
            relation_types: 필터링할 관계 유형의 선택적 목록
            entity_types: 필터링할 엔티티 유형의 선택적 목록
            limit: 최대 결과 수
        Returns:
            (엔티티, 관계) 튜플 목록
        """
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError("방향은 'outgoing', 'incoming', 또는 'both'여야 합니다")
        queries = []
        params: list[Any] = []
        # 나가는 관계 (entity_id -> neighbor)
        if direction in ("outgoing", "both"):
            conditions = ["r.source_id = ?"]
            query_params: list[Any] = [entity_id]
            if relation_types:
                placeholders = ", ".join(["?"] * len(relation_types))
                conditions.append(f"r.relation_type IN ({placeholders})")
                query_params.extend(relation_types)
            if entity_types:
                placeholders = ", ".join(["?"] * len(entity_types))
                conditions.append(f"e.type IN ({placeholders})")
                query_params.extend(entity_types)
            query = f"""
            SELECT e.*, r.id as rel_id, r.source_id, r.target_id,
                   r.relation_type, r.properties as rel_properties,
                   r.created_at as rel_created_at, r.updated_at as rel_updated_at
            FROM entities e
            JOIN edges r ON e.id = r.target_id
            WHERE {" AND ".join(conditions)}
            """
            queries.append(query)
            params.extend(query_params)
        # 들어오는 관계 (neighbor -> entity_id)
        if direction in ("incoming", "both"):
            conditions = ["r.target_id = ?"]
            query_params_inc: list[Any] = [entity_id]
            if relation_types:
                placeholders = ", ".join(["?"] * len(relation_types))
                conditions.append(f"r.relation_type IN ({placeholders})")
                query_params_inc.extend(relation_types)
            if entity_types:
                placeholders = ", ".join(["?"] * len(entity_types))
                conditions.append(f"e.type IN ({placeholders})")
                query_params_inc.extend(entity_types)
            query = f"""
            SELECT e.*, r.id as rel_id, r.source_id, r.target_id,
                   r.relation_type, r.properties as rel_properties,
                   r.created_at as rel_created_at, r.updated_at as rel_updated_at
            FROM entities e
            JOIN edges r ON e.id = r.source_id
            WHERE {" AND ".join(conditions)}
            """
            queries.append(query)
            params.extend(query_params_inc)
        # 필요한 경우 UNION으로 쿼리 결합
        if len(queries) > 1:
            # SQLite는 UNION에서 전체 SELECT 문을 괄호로 묶는 것을 좋아하지 않습니다
            final_query = f"{queries[0]} UNION {queries[1]} LIMIT ?"
            params.append(limit)
        else:
            final_query = f"{queries[0]} LIMIT ?"
            params.append(limit)
        # 쿼리 실행
        cursor = self.connection.cursor()
        cursor.execute(final_query, params)
        # 결과 처리
        results = []
        for row in cursor.fetchall():
            # 엔티티 필드 추출
            entity = Entity(
                id=row["id"],
                uuid=row["uuid"],
                name=row["name"],
                type=row["type"],
                properties=json.loads(row["properties"]) if row["properties"] else {},
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            # 관계 필드 추출
            rel_props = row["rel_properties"]
            relationship = Relationship(
                id=row["rel_id"],
                source_id=row["source_id"],
                target_id=row["target_id"],
                relation_type=row["relation_type"],
                properties=json.loads(rel_props) if rel_props else {},
                created_at=row["rel_created_at"],
                updated_at=row["rel_updated_at"],
            )
            results.append((entity, relationship))
        return results

    def breadth_first_search(
        self,
        start_id: int,
        max_depth: int = 5,
        relation_types: Optional[list[str]] = None,
        entity_types: Optional[list[str]] = None,
    ) -> list[PathNode]:
        """너비 우선 탐색(BFS)을 사용하여 start_id에서 max_depth 내에서 도달 가능한 모든 노드를 찾습니다."""
        if max_depth < 0:
            return []

        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM entities WHERE id = ?", (start_id,))
        start_row = cursor.fetchone()
        if not start_row:
            return []

        start_entity = Entity(
            id=start_row["id"],
            uuid=start_row["uuid"],
            name=start_row["name"],
            type=start_row["type"],
            properties=(json.loads(start_row["properties"]) if start_row["properties"] else {}),
            created_at=start_row["created_at"],
            updated_at=start_row["updated_at"],
        )

        queue = [PathNode(entity=start_entity, depth=0)]
        visited: dict[int, PathNode] = {start_id: queue[0]}
        results: list[PathNode] = []

        while queue:
            current = queue.pop(0)
            results.append(current)

            if current.depth >= max_depth:
                continue

            try:
                neighbors = self.get_neighbors(
                    current.entity.id,
                    direction="both",
                    relation_types=relation_types,
                    entity_types=entity_types,
                )

                for neighbor_entity, relationship in neighbors:
                    if neighbor_entity.id not in visited:
                        new_node = PathNode(
                            entity=neighbor_entity,
                            relationship=relationship,
                            parent=current,
                            depth=current.depth + 1,
                        )
                        queue.append(new_node)
                        visited[neighbor_entity.id] = new_node
            except Exception as exception:
                logger = get_observable_logger("graph_traversal", "adapter")
                logger.error(
                    "neighbor_query_failed_bfs",
                    entity_id=current.entity.id,
                    error_type=type(exception).__name__,
                    error_message=str(exception),
                )

        return results

    def find_shortest_path(
        self,
        start_id: int,
        end_id: int,
        max_depth: int = 5,
        relation_types: Optional[list[str]] = None,
        entity_types: Optional[list[str]] = None,
    ) -> Optional[list[PathNode]]:
        """너비 우선 탐색(BFS)을 사용하여 두 엔티티 간의 최단 경로를 찾습니다."""
        if start_id == end_id:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM entities WHERE id = ?", (start_id,))
            start_row = cursor.fetchone()
            if start_row:
                start_entity = Entity(
                    id=start_row["id"],
                    uuid=start_row["uuid"],
                    name=start_row["name"],
                    type=start_row["type"],
                    properties=(
                        json.loads(start_row["properties"]) if start_row["properties"] else {}
                    ),
                    created_at=start_row["created_at"],
                    updated_at=start_row["updated_at"],
                )
                return [PathNode(entity=start_entity)]
            return None

        # BFS를 사용하여 도달 가능한 모든 노드를 찾고, 그중에 end_id가 있는지 확인합니다.
        reachable_nodes = self.breadth_first_search(
            start_id=start_id,
            max_depth=max_depth,
            relation_types=relation_types,
            entity_types=entity_types,
        )

        # end_id에 해당하는 PathNode를 찾습니다.
        end_node = next((node for node in reachable_nodes if node.entity.id == end_id), None)

        if end_node:
            return end_node.path_to_root
        return None

    def recursive_query(
        self,
        start_id: int,
        direction: str = "outgoing",
        relation_types: Optional[list[str]] = None,
        entity_types: Optional[list[str]] = None,
        max_depth: int = 3,
        limit: int = 100,
    ) -> list[dict]:
        """
        SQLite의 재귀적 CTE를 사용하여 하위 그래프를 찾기 위한 재귀적 쿼리를 수행합니다.
        Args:
            start_id: 시작 엔티티 ID
            direction: 'outgoing', 'incoming', 또는 'both'
            relation_types: 필터링할 관계 유형의 선택적 목록
            entity_types: 필터링할 엔티티 유형의 선택적 목록
            max_depth: 최대 순회 깊이
            limit: 최대 결과 수
        Returns:
            엔티티 및 관계 정보가 포함된 결과 행 목록
        """
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError("방향은 'outgoing', 'incoming', 또는 'both'여야 합니다.")
        relation_filter = ""
        entity_filter = ""
        # 각 쿼리 부분에 대해 필터 절과 매개변수를 별도로 빌드합니다.
        relation_filter_params = []
        entity_filter_params = []
        # 관계 유형 필터 추가
        if relation_types:
            placeholders = ", ".join(["?"] * len(relation_types))
            relation_filter = f"AND r.relation_type IN ({placeholders})"
            relation_filter_params = list(relation_types)
        # 엔티티 유형 필터 추가
        if entity_types:
            placeholders = ", ".join(["?"] * len(entity_types))
            entity_filter = f"AND e.type IN ({placeholders})"
            entity_filter_params = list(entity_types)
        # 방향에 따라 매개변수 빌드
        if direction == "both":
            # 양방향은 start_id가 두 번 필요합니다 (나가는 및 들어오는 기본 사례).
            base_params = [start_id] + relation_filter_params + [start_id] + relation_filter_params
            recursive_params = [max_depth] + relation_filter_params
        else:
            # 단방향은 start_id가 한 번 필요합니다.
            base_params = [start_id] + relation_filter_params
            recursive_params = [max_depth] + relation_filter_params
        final_params = entity_filter_params + [limit]
        # 모든 매개변수를 순서대로 결합
        params = base_params + recursive_params + final_params
        # 방향에 따라 재귀적 CTE 쿼리 빌드
        if direction == "outgoing":
            recursive_query = f"""
            WITH RECURSIVE
            graph_path(source_id, target_id, relation_id, depth) AS (
                -- 기본 사례: start_id에서 직접 나가는 관계
                SELECT r.source_id, r.target_id, r.id, 1
                FROM edges r
                WHERE r.source_id = ? {relation_filter}
                UNION ALL
                -- 재귀 사례: 나가는 엣지 따라가기
                SELECT r.source_id, r.target_id, r.id, gp.depth + 1
                FROM graph_path gp
                JOIN edges r ON gp.target_id = r.source_id
                WHERE gp.depth < ? {relation_filter}
            )
            SELECT e.*, r.id as rel_id, r.source_id, r.target_id,
                   r.relation_type, r.properties as rel_properties,
                   gp.depth, s.id as source_entity_id, s.name as source_name,
                   s.type as source_type
            FROM graph_path gp
            JOIN edges r ON r.id = gp.relation_id
            JOIN entities e ON e.id = gp.target_id
            JOIN entities s ON s.id = gp.source_id
            WHERE 1=1 {entity_filter}
            ORDER BY gp.depth
            LIMIT ?
            """
        elif direction == "incoming":
            recursive_query = f"""
            WITH RECURSIVE
            graph_path(source_id, target_id, relation_id, depth) AS (
                -- 기본 사례: start_id로 직접 들어오는 관계
                SELECT r.source_id, r.target_id, r.id, 1
                FROM edges r
                WHERE r.target_id = ? {relation_filter}
                UNION ALL
                -- 재귀 사례: 들어오는 엣지 따라가기
                SELECT r.source_id, r.target_id, r.id, gp.depth + 1
                FROM graph_path gp
                JOIN edges r ON gp.source_id = r.target_id
                WHERE gp.depth < ? {relation_filter}
            )
            SELECT e.*, r.id as rel_id, r.source_id, r.target_id,
                   r.relation_type, r.properties as rel_properties,
                   gp.depth, t.id as target_entity_id, t.name as target_name,
                   t.type as target_type
            FROM graph_path gp
            JOIN edges r ON r.id = gp.relation_id
            JOIN entities e ON e.id = gp.source_id
            JOIN entities t ON t.id = gp.target_id
            WHERE 1=1 {entity_filter}
            ORDER BY gp.depth
            LIMIT ?
            """
        else:  # 'both'
            recursive_query = f"""
            WITH RECURSIVE
            graph_path(entity_id, related_id, relation_id, direction, depth) AS (
                -- 기본 사례: start_id에서 직접 나가는 관계
                SELECT r.source_id, r.target_id, r.id, 'outgoing', 1
                FROM edges r
                WHERE r.source_id = ? {relation_filter}
                UNION ALL
                -- 기본 사례: start_id로 직접 들어오는 관계
                SELECT r.target_id, r.source_id, r.id, 'incoming', 1
                FROM edges r
                WHERE r.target_id = ? {relation_filter}
                UNION ALL
                -- 재귀 사례: 양방향으로 관계 따라가기
                SELECT
                    CASE
                        WHEN gp.direction = 'outgoing' THEN r.source_id
                        ELSE r.target_id
                    END,
                    CASE
                        WHEN gp.direction = 'outgoing' THEN r.target_id
                        ELSE r.source_id
                    END,
                    r.id,
                    gp.direction,
                    gp.depth + 1
                FROM graph_path gp
                JOIN edges r ON
                    (gp.direction = 'outgoing' AND gp.related_id = r.source_id) OR
                    (gp.direction = 'incoming' AND gp.related_id = r.target_id)
                WHERE gp.depth < ? {relation_filter}
            )
            SELECT e.*, r.id as rel_id, r.source_id, r.target_id,
                   r.relation_type, r.properties as rel_properties,
                   gp.depth, gp.direction
            FROM graph_path gp
            JOIN edges r ON r.id = gp.relation_id
            JOIN entities e ON e.id = gp.related_id
            WHERE 1=1 {entity_filter}
            ORDER BY gp.depth
            LIMIT ?
            """
            # 'both' 방향의 경우, start_id가 두 번 사용됩니다.
            both_params: list[Any] = [start_id, start_id, max_depth] + params[2:]
            params = both_params
        params.append(limit)
        # 재귀 쿼리 실행
        cursor = self.connection.cursor()
        cursor.execute(recursive_query, params)
        # 결과 처리
        results = []
        for row in cursor.fetchall():
            # 쉬운 조작을 위해 dict로 변환
            result_dict = dict(row)
            # JSON 속성 파싱
            if result_dict.get("properties"):
                result_dict["properties"] = json.loads(result_dict["properties"])
            else:
                result_dict["properties"] = {}
            if result_dict.get("rel_properties"):
                result_dict["rel_properties"] = json.loads(result_dict["rel_properties"])
            else:
                result_dict["rel_properties"] = {}
            results.append(result_dict)
        return results

    def depth_first_search(
        self,
        start_id: int,
        max_depth: int = 5,
        relation_types: Optional[list[str]] = None,
        entity_types: Optional[list[str]] = None,
    ) -> list[PathNode]:
        """깊이 우선 탐색(DFS)을 사용하여 start_id에서 max_depth 내에서 도달 가능한 모든 노드를 찾습니다."""
        if max_depth < 0:
            return []

        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM entities WHERE id = ?", (start_id,))
        start_row = cursor.fetchone()
        if not start_row:
            return []

        start_entity = Entity(
            id=start_row["id"],
            uuid=start_row["uuid"],
            name=start_row["name"],
            type=start_row["type"],
            properties=(json.loads(start_row["properties"]) if start_row["properties"] else {}),
            created_at=start_row["created_at"],
            updated_at=start_row["updated_at"],
        )

        stack = [PathNode(entity=start_entity, depth=0)]
        visited: dict[int, PathNode] = {start_id: stack[0]}
        results: list[PathNode] = []

        while stack:
            current = stack.pop()
            results.append(current)

            if current.depth >= max_depth:
                continue

            try:
                # DFS를 시뮬레이션하기 위해 이웃을 역순으로 가져옵니다 (스택 동작).
                neighbors = self.get_neighbors(
                    current.entity.id,
                    direction="both",
                    relation_types=relation_types,
                    entity_types=entity_types,
                )
                # 자연스러운 순서로 처리하기 위해 이웃을 역순으로 스택에 추가합니다.
                for neighbor_entity, relationship in reversed(neighbors):
                    if neighbor_entity.id not in visited:
                        new_node = PathNode(
                            entity=neighbor_entity,
                            relationship=relationship,
                            parent=current,
                            depth=current.depth + 1,
                        )
                        stack.append(new_node)
                        visited[neighbor_entity.id] = new_node
            except Exception as exception:
                logger = get_observable_logger("graph_traversal", "adapter")
                logger.error(
                    "neighbor_query_failed_dfs",
                    entity_id=current.entity.id,
                    error_type=type(exception).__name__,
                    error_message=str(exception),
                )

        return results

    def get_connected_components(self) -> list[list[int]]:
        """그래프의 모든 연결된 구성 요소를 찾습니다 (무방향)."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM entities")
        all_entity_ids = [row[0] for row in cursor.fetchall()]

        visited_all = set()
        components = []

        for entity_id in all_entity_ids:
            if entity_id not in visited_all:
                component_nodes = []
                queue = [entity_id]
                visited_component = {entity_id}

                while queue:
                    current_id = queue.pop(0)
                    component_nodes.append(current_id)
                    visited_all.add(current_id)

                    # 이웃 가져오기 (연결된 구성 요소의 경우 무방향)
                    neighbors = self.get_neighbors(current_id, direction="both")
                    for neighbor_entity, _ in neighbors:
                        if neighbor_entity.id not in visited_component:
                            queue.append(neighbor_entity.id)
                            visited_component.add(neighbor_entity.id)

                components.append(component_nodes)

        return components
