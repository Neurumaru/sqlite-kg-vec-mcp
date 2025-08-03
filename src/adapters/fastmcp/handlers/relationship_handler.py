"""
MCP 작업을 위한 관계 관리 핸들러.
"""

from typing import Any

from fastmcp import Context

from src.domain.entities.relationship import RelationshipType
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.use_cases.relationship import RelationshipManagementUseCase

from ..exceptions import MCPServerException
from .base import BaseHandler


class RelationshipHandler(BaseHandler):
    """관계 관련 MCP 작업을 위한 핸들러."""

    def __init__(self, relationship_use_case: RelationshipManagementUseCase, config):
        """
        관계 핸들러를 초기화합니다.

        Args:
            relationship_use_case: 관계 관리 유스케이스
            config: FastMCP 설정
        """
        super().__init__(config)
        self.relationship_use_case = relationship_use_case

    async def create_edge(
        self,
        source_node_id: int,
        target_node_id: int,
        relation_type: str,
        label: str,
        properties: dict[str, Any]] = None,
        weight: Optional[float] = None,
        ctx: Optional[Context] = None,
    ) -> dict[str, Any]:
        """
        노드 간에 새 관계를 생성합니다.

        Args:
            source_node_id: 소스 노드의 ID
            target_node_id: 대상 노드의 ID
            relation_type: 관계의 유형
            label: 관계의 레이블
            properties: 관계의 사용자 지정 속성 (선택 사항)
            weight: 관계의 가중치 (제공되지 않으면 설정 기본값 사용)
            ctx: MCP 컨텍스트 객체

        Returns:
            생성된 관계 데이터
        """
        self.logger.info("%s에서 %s로 엣지를 생성합니다.", source_node_id, target_node_id)

        try:
            # 도메인 객체로 변환
            source_id = NodeId(str(source_node_id))
            target_id = NodeId(str(target_node_id))
            domain_relation_type = RelationshipType(relation_type)

            # 제공되지 않은 경우 설정 기본 가중치 사용
            actual_weight = (
                weight if weight is not None else self.config.default_relationship_weight
            )

            # 유스케이스 호출
            relationship = await self.relationship_use_case.create_relationship(
                source_node_id=source_id,
                target_node_id=target_id,
                relationship_type=domain_relation_type,
                label=label,
                properties=properties,
                weight=actual_weight,
            )

            # MCP 응답 형식으로 변환
            return self._relationship_to_mcp_response(relationship)

        except ValueError as e:
            error_msg = f"잘못된 관계 매개변수: {e}"
            self.logger.error(error_msg)
            return self._create_error_response(error_msg, "INVALID_PARAMETERS")
        except Exception as e:
            self.logger.error("관계 생성 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="create_edge",
                message=str(e),
                original_error=e,
            ) from e

    async def get_edge(
        self,
        edge_id: int,
        ctx: Optional[Context] = None,
    ) -> dict[str, Any]:
        """
        지식 그래프에서 관계를 가져옵니다.

        Args:
            edge_id: 검색할 관계의 ID
            ctx: MCP 컨텍스트 객체

        Returns:
            관계 데이터 또는 오류
        """
        self.logger.info("ID %s로 엣지를 검색합니다.", edge_id)

        try:
            domain_relationship_id = RelationshipId(str(edge_id))

            # 유스케이스 호출
            relationship = await self.relationship_use_case.get_relationship(domain_relationship_id)

            if not relationship:
                error_msg = "관계를 찾을 수 없습니다."
                self.logger.error(error_msg)
                return self._create_error_response(error_msg, "RELATIONSHIP_NOT_FOUND")

            # MCP 응답 형식으로 변환
            return self._relationship_to_mcp_response(relationship)

        except Exception as e:
            self.logger.error("관계 가져오기 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="get_edge",
                message=str(e),
                original_error=e,
            ) from e

    async def update_edge(
        self,
        edge_id: int,
        label: Optional[str] = None,
        properties: dict[str, Any]] = None,
        weight: Optional[float] = None,
        ctx: Optional[Context] = None,
    ) -> dict[str, Any]:
        """
        지식 그래프의 관계를 업데이트합니다.

        Args:
            edge_id: 업데이트할 관계의 ID
            label: 관계의 새 레이블 (선택 사항)
            properties: 관계의 새 속성 (선택 사항)
            weight: 관계의 새 가중치 (선택 사항)
            ctx: MCP 컨텍스트 객체

        Returns:
            업데이트된 관계 데이터
        """
        self.logger.info("ID %s의 엣지를 업데이트합니다.", edge_id)

        try:
            domain_relationship_id = RelationshipId(str(edge_id))

            # 유스케이스 호출
            relationship = await self.relationship_use_case.update_relationship(
                relationship_id=domain_relationship_id,
                label=label,
                properties=properties,
                weight=weight,
            )

            # MCP 응답 형식으로 변환
            return self._relationship_to_mcp_response(relationship)

        except Exception as e:
            self.logger.error("관계 업데이트 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="update_edge",
                message=str(e),
                original_error=e,
            ) from e

    async def delete_edge(
        self,
        edge_id: int,
        ctx: Optional[Context] = None,
    ) -> dict[str, Any]:
        """
        지식 그래프에서 관계를 삭제합니다.

        Args:
            edge_id: 삭제할 관계의 ID
            ctx: MCP 컨텍스트 객체

        Returns:
            성공 또는 오류 메시지
        """
        self.logger.info("ID %s의 엣지를 삭제합니다.", edge_id)

        try:
            domain_relationship_id = RelationshipId(str(edge_id))

            # 유스케이스 호출
            await self.relationship_use_case.delete_relationship(domain_relationship_id)

            return self._create_success_response(
                {"message": f"관계 {edge_id}가 성공적으로 삭제되었습니다."}
            )

        except Exception as e:
            self.logger.error("관계 삭제 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="delete_edge",
                message=str(e),
                original_error=e,
            ) from e

    async def find_edges(
        self,
        relation_type: Optional[str] = None,
        source_node_id: Optional[int] = None,
        target_node_id: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
        ctx: Optional[Context] = None,
    ) -> dict[str, Any]:
        """
        지식 그래프에서 관계를 찾습니다.

        Args:
            relation_type: 관계 유형으로 필터링 (선택 사항)
            source_node_id: 소스 노드 ID로 필터링 (선택 사항)
            target_node_id: 대상 노드 ID로 필터링 (선택 사항)
            limit: 반환할 최대 결과 수 (기본값 100)
            offset: 건너뛸 결과 수 (기본값 0)
            ctx: MCP 컨텍스트 객체

        Returns:
            조건과 일치하는 관계 목록
        """
        self.logger.info(
            "유형=%s, 소스=%s, 대상=%s으로 엣지를 찾습니다.",
            relation_type,
            source_node_id,
            target_node_id,
        )

        try:
            # 도메인 객체로 변환
            domain_relation_type = RelationshipType(relation_type) if relation_type else None
            domain_source_id = NodeId(str(source_node_id)) if source_node_id else None
            domain_target_id = NodeId(str(target_node_id)) if target_node_id else None

            # 유스케이스 호출
            relationships = await self.relationship_use_case.list_relationships(
                relationship_type=domain_relation_type,
                source_node_id=domain_source_id,
                target_node_id=domain_target_id,
                limit=limit,
                offset=offset,
            )

            # MCP 응답 형식으로 변환
            result_relationships = [
                self._relationship_to_mcp_response(rel) for rel in relationships
            ]

            self.logger.info("%s개의 관계를 찾았습니다.", len(result_relationships))

            return {
                "edges": result_relationships,
                "count": len(result_relationships),
                "limit": limit,
                "offset": offset,
            }

        except Exception as e:
            self.logger.error("관계 찾기 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="find_edges",
                message=str(e),
                original_error=e,
            ) from e

    def _relationship_to_mcp_response(self, relationship) -> dict[str, Any]:
        """도메인 관계를 MCP 응답 형식으로 변환합니다."""
        return {
            "edge_id": str(relationship.id),
            "source_id": str(relationship.source_node_id),
            "target_id": str(relationship.target_node_id),
            "relation_type": relationship.relationship_type.value,
            "label": relationship.label,
            "properties": relationship.properties or {},
            "confidence": relationship.confidence,
            "created_at": (
                relationship.created_at.isoformat() if hasattr(relationship, "created_at") else None
            ),
        }
