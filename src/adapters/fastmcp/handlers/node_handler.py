"""
MCP 작업을 위한 노드 관리 핸들러.
"""

import sqlite3
from typing import Any, Optional

from fastmcp import Context

from src.domain.entities.node import NodeType
from src.domain.value_objects.node_id import NodeId
from src.use_cases.node import NodeManagementUseCase

from ..exceptions import MCPServerException
from .base import BaseHandler


class NodeHandler(BaseHandler):
    """노드 관련 MCP 작업을 위한 핸들러."""

    def __init__(self, node_use_case: NodeManagementUseCase, config):
        """
        노드 핸들러를 초기화합니다.

        Args:
            node_use_case: 노드 관리 유스케이스
            config: FastMCP 설정
        """
        super().__init__(config)
        self.node_use_case = node_use_case

    async def create_node(
        self,
        node_type: str,
        name: Optional[str] = None,
        properties: Optional[dict[str, Any]] = None,
        node_uuid: Optional[str] = None,
        ctx: Optional[Context] = None,
    ) -> dict[str, Any]:
        """
        지식 그래프에 새 노드를 생성합니다.

        Args:
            node_type: 생성할 노드의 유형
            name: 노드의 이름 (선택 사항)
            properties: 노드의 사용자 지정 속성 (선택 사항)
            node_uuid: 노드의 사용자 지정 UUID (선택 사항)
            ctx: MCP 컨텍스트 객체

        Returns:
            생성된 노드 데이터
        """
        self.logger.info("'%s' 유형의 노드를 생성합니다.", node_type)

        try:
            # 도메인 객체로 변환
            domain_node_type = NodeType(node_type)

            # 유스케이스 호출
            node = await self.node_use_case.create_node(
                name=name or f"Node_{node_type}",
                node_type=domain_node_type,
                properties=properties,
            )

            # MCP 응답 형식으로 변환
            return self._node_to_mcp_response(node)

        except ValueError as e:
            error_msg = f"잘못된 노드 유형 또는 매개변수: {e}"
            self.logger.error(error_msg)
            return self._create_error_response(error_msg, "INVALID_PARAMETERS")
        except (KeyError, TypeError, sqlite3.Error, RuntimeError) as e:
            self.logger.error("노드 생성 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="create_node",
                message=str(e),
                original_error=e,
            ) from e

    async def get_node(
        self,
        node_id: Optional[int] = None,
        node_uuid: Optional[str] = None,
        ctx: Optional[Context] = None,
    ) -> dict[str, Any]:
        """
        지식 그래프에서 노드를 가져옵니다.

        Args:
            node_id: 검색할 노드의 ID (uuid가 제공되면 선택 사항)
            node_uuid: 검색할 노드의 UUID (id가 제공되면 선택 사항)
            ctx: MCP 컨텍스트 객체

        Returns:
            노드 데이터 또는 오류
        """
        if node_id is None and node_uuid is None:
            error_msg = "필수 매개변수 누락: id 또는 node_uuid 중 하나를 제공해야 합니다."
            self.logger.error(error_msg)
            return self._create_error_response(error_msg, "MISSING_PARAMETERS")

        try:
            if node_id is not None:
                domain_node_id = NodeId(str(node_id))
                self.logger.info("ID %s로 노드를 검색합니다.", node_id)
            else:
                # 위에서 둘 다 None이 아닌지 확인했으므로 node_uuid는 여기서 str임이 보장됨
                assert node_uuid is not None, "이 시점에서 node_uuid는 None이 아니어야 합니다."
                domain_node_id = NodeId(node_uuid)
                self.logger.info("UUID %s로 노드를 검색합니다.", node_uuid)

            # 유스케이스 호출
            node = await self.node_use_case.get_node(domain_node_id)

            if not node:
                error_msg = "노드를 찾을 수 없습니다."
                self.logger.error(error_msg)
                return self._create_error_response(error_msg, "NODE_NOT_FOUND")

            # MCP 응답 형식으로 변환
            return self._node_to_mcp_response(node)

        except (ValueError, KeyError, sqlite3.Error, RuntimeError) as e:
            self.logger.error("노드 가져오기 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="get_node",
                message=str(e),
                original_error=e,
            ) from e

    async def update_node(
        self,
        node_id: int,
        name: Optional[str] = None,
        properties: Optional[dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> dict[str, Any]:
        """
        지식 그래프의 노드를 업데이트합니다.

        Args:
            node_id: 업데이트할 노드의 ID
            name: 노드의 새 이름 (선택 사항)
            properties: 노드의 새 속성 (선택 사항)
            ctx: MCP 컨텍스트 객체

        Returns:
            업데이트된 노드 데이터
        """
        self.logger.info("ID %s의 노드를 업데이트합니다.", node_id)

        try:
            domain_node_id = NodeId(str(node_id))

            # 유스케이스 호출
            node = await self.node_use_case.update_node(
                node_id=domain_node_id,
                name=name,
                properties=properties,
            )

            # MCP 응답 형식으로 변환
            return self._node_to_mcp_response(node)

        except (ValueError, KeyError, TypeError, sqlite3.Error, RuntimeError) as e:
            self.logger.error("노드 업데이트 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="update_node",
                message=str(e),
                original_error=e,
            ) from e

    async def delete_node(
        self,
        node_id: int,
        ctx: Optional[Context] = None,
    ) -> dict[str, Any]:
        """
        지식 그래프에서 노드를 삭제합니다.

        Args:
            node_id: 삭제할 노드의 ID
            ctx: MCP 컨텍스트 객체

        Returns:
            성공 또는 오류 메시지
        """
        self.logger.info("ID %s의 노드를 삭제합니다.", node_id)

        try:
            domain_node_id = NodeId(str(node_id))

            # 유스케이스 호출
            await self.node_use_case.delete_node(domain_node_id)

            return self._create_success_response(
                {"message": f"노드 {node_id}가 성공적으로 삭제되었습니다."}
            )

        except (ValueError, sqlite3.Error, RuntimeError) as e:
            self.logger.error("노드 삭제 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="delete_node",
                message=str(e),
                original_error=e,
            ) from e

    async def find_nodes(
        self,
        node_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        ctx: Optional[Context] = None,
    ) -> dict[str, Any]:
        """
        지식 그래프에서 노드를 찾습니다.

        Args:
            node_type: 노드 유형으로 필터링 (선택 사항)
            limit: 반환할 최대 결과 수 (기본값 100)
            offset: 건너뛸 결과 수 (기본값 0)
            ctx: MCP 컨텍스트 객체

        Returns:
            조건과 일치하는 노드 목록
        """
        self.logger.info(
            "유형=%s, 제한=%s, 오프셋=%s으로 노드를 찾습니다.", node_type, limit, offset
        )

        try:
            domain_node_type = NodeType(node_type) if node_type else None

            # 유스케이스 호출
            nodes = await self.node_use_case.list_nodes(
                node_type=domain_node_type,
                limit=limit,
                offset=offset,
            )

            # MCP 응답 형식으로 변환
            result_nodes = [self._node_to_mcp_response(node) for node in nodes]

            self.logger.info("%s개의 노드를 찾았습니다.", len(result_nodes))

            return {
                "nodes": result_nodes,
                "count": len(result_nodes),
                "limit": limit,
                "offset": offset,
            }

        except (ValueError, sqlite3.Error, RuntimeError) as e:
            self.logger.error("노드 찾기 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="find_nodes",
                message=str(e),
                original_error=e,
            ) from e

    def _node_to_mcp_response(self, node) -> dict[str, Any]:
        """도메인 노드를 MCP 응답 형식으로 변환합니다."""
        return {
            "node_id": str(node.id),
            "uuid": str(node.id),  # 지금은 node.id를 UUID로 사용
            "name": node.name,
            "type": node.node_type.value,
            "properties": node.properties or {},
            "created_at": node.created_at.isoformat() if hasattr(node, "created_at") else None,
        }
