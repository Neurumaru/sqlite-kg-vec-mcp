"""
MCP 서버 인터페이스를 위한 API 엔드포인트 및 핸들러.
"""

import logging
from typing import Any

from fastmcp import Context, FastMCP

from src.domain.entities.node import NodeType
from src.domain.entities.relationship import RelationshipType
from src.domain.services.knowledge_search import SearchStrategy
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.use_cases.knowledge_search import KnowledgeSearchUseCase
from src.use_cases.node import NodeManagementUseCase
from src.use_cases.relationship import RelationshipManagementUseCase

from .config import FastMCPConfig
from .exceptions import MCPServerException


class KnowledgeGraphServer:
    """
    지식 그래프 API를 제공하는 MCP 서버.

    이 어댑터는 MCP 프로토콜 메시지를 도메인 유스케이스 호출로 변환하고
    응답을 다시 MCP 형식으로 포맷합니다.
    """

    def __init__(
        self,
        node_use_case: NodeManagementUseCase,
        relationship_use_case: RelationshipManagementUseCase,
        knowledge_search_use_case: KnowledgeSearchUseCase,
        config: FastMCPConfig,
    ):
        """
        지식 그래프 MCP 서버를 초기화합니다.

        Args:
            node_use_case: 노드 관리 유스케이스
            relationship_use_case: 관계 관리 유스케이스
            knowledge_search_use_case: 지식 검색 유스케이스
            config: MCP 서버 설정
        """
        self.node_use_case = node_use_case
        self.relationship_use_case = relationship_use_case
        self.knowledge_search_use_case = knowledge_search_use_case
        self.config = config

        # 로깅 설정
        self.logger = logging.getLogger("kg_server")
        self.logger.setLevel(getattr(logging, config.log_level))

        # FastMCP로 MCP 서버 생성
        self.mcp_server: FastMCP = FastMCP(
            name="Knowledge Graph Server",
            instructions="SQLite-based knowledge graph with vector search capabilities",
        )

        # 모든 도구 등록
        self._register_tools()

        self.logger.info("지식 그래프 서버가 초기화되었습니다.")

    def _register_tools(self):
        """모든 API 엔드포인트 도구를 등록합니다."""
        # 엔티티 도구
        self.mcp_server.tool()(self.create_node)
        self.mcp_server.tool()(self.get_node)
        self.mcp_server.tool()(self.update_node)
        self.mcp_server.tool()(self.delete_node)
        self.mcp_server.tool()(self.find_nodes)

        # 관계 도구
        self.mcp_server.tool()(self.create_edge)
        self.mcp_server.tool()(self.get_edge)
        self.mcp_server.tool()(self.update_edge)
        self.mcp_server.tool()(self.delete_edge)
        self.mcp_server.tool()(self.find_edges)

        # 그래프 순회 도구
        self.mcp_server.tool()(self.get_neighbors)
        self.mcp_server.tool()(self.find_paths)

        # 벡터 검색 도구
        self.mcp_server.tool()(self.search_similar_nodes)
        self.mcp_server.tool()(self.search_by_text)

    # === 노드 관리 메서드 ===

    async def create_node(
        self,
        node_type: str,
        name: Optional[str] = None,
        properties: dict[str, Any]] = None,
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
            return {"error": error_msg}
        except Exception as e:
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
            return {"error": error_msg}

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
                if ctx:
                    self.logger.error(error_msg)
                return {"error": error_msg}

            # MCP 응답 형식으로 변환
            return self._node_to_mcp_response(node)

        except Exception as e:
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
        properties: dict[str, Any]] = None,
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

        except Exception as e:
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

            return {"success": True, "message": f"노드 {node_id}가 성공적으로 삭제되었습니다."}

        except Exception as e:
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

            if ctx:
                self.logger.info("%s개의 노드를 찾았습니다.", len(result_nodes))

            return {
                "nodes": result_nodes,
                "count": len(result_nodes),
                "limit": limit,
                "offset": offset,
            }

        except Exception as e:
            self.logger.error("노드 찾기 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="find_nodes",
                message=str(e),
                original_error=e,
            ) from e

    # === 검색 메서드 ===

    async def search_by_text(
        self,
        query: str,
        limit: int = 10,
        include_documents: bool = True,
        include_nodes: bool = True,
        include_relationships: bool = True,
        ctx: Optional[Context] = None,
    ) -> dict[str, Any]:
        """
        텍스트 쿼리로 지식 그래프를 검색합니다.

        Args:
            query: 검색 쿼리 문자열
            limit: 반환할 최대 결과 수 (기본값 10)
            include_documents: 문서 결과를 포함할지 여부 (기본값 True)
            include_nodes: 노드 결과를 포함할지 여부 (기본값 True)
            include_relationships: 관계 결과를 포함할지 여부 (기본값 True)
            ctx: MCP 컨텍스트 객체

        Returns:
            검색 결과
        """
        self.logger.info("텍스트로 검색: '%s'", query)

        try:
            # 유스케이스 호출
            results = await self.knowledge_search_use_case.search_knowledge(
                query=query,
                strategy=SearchStrategy.SEMANTIC,
                limit=limit,
                include_documents=include_documents,
                include_nodes=include_nodes,
                include_relationships=include_relationships,
            )

            # MCP 응답 형식으로 변환
            result_items = []
            for result in results.results:
                if result.document:
                    result_items.append(
                        {
                            "type": "document",
                            "id": str(result.document.id),
                            "title": result.document.title,
                            "content": (
                                result.document.content[: self.config.content_summary_length]
                                + "..."
                                if len(result.document.content) > self.config.content_summary_length
                                else result.document.content
                            ),
                            "similarity": result.score,
                        }
                    )
                elif result.node:
                    result_items.append(
                        {
                            "type": "node",
                            "id": str(result.node.id),
                            "name": result.node.name,
                            "node_type": result.node.node_type.value,
                            "similarity": result.score,
                        }
                    )
                elif result.relationship:
                    result_items.append(
                        {
                            "type": "relationship",
                            "id": str(result.relationship.id),
                            "label": result.relationship.label,
                            "relationship_type": result.relationship.relationship_type.value,
                            "similarity": result.score,
                        }
                    )

            if ctx:
                self.logger.info("%s개의 결과를 찾았습니다.", len(result_items))

            return {"results": result_items, "count": len(result_items)}

        except Exception as e:
            self.logger.error("텍스트 검색 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="search_by_text",
                message=str(e),
                original_error=e,
            ) from e

    # === 도우미 메서드 ===

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

    # === 나머지 기능에 대한 플레이스홀더 메서드 ===
    # 이들은 동일한 패턴을 따라 구현될 것입니다.

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
            return {"error": error_msg}
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
                if ctx:
                    self.logger.error(error_msg)
                return {"error": error_msg}

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

            return {"success": True, "message": f"관계 {edge_id}가 성공적으로 삭제되었습니다."}

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

            if ctx:
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

    async def get_neighbors(
        self,
        node_id: int,
        depth: int = 1,
        ctx: Optional[Context] = None,
    ) -> dict[str, Any]:
        """
        지식 그래프에서 이웃 노드를 가져옵니다.

        Args:
            node_id: 중앙 노드의 ID
            depth: 검색할 이웃의 깊이 (기본값 1)
            ctx: MCP 컨텍스트 객체

        Returns:
            이웃 노드 목록
        """
        self.logger.info("깊이 %s에서 노드 %s의 이웃을 가져옵니다.", depth, node_id)

        try:
            domain_node_id = NodeId(str(node_id))

            # 노드의 관계 가져오기
            relationships = await self.relationship_use_case.get_node_relationships(
                node_id=domain_node_id, direction="both"
            )

            # 이웃 노드 ID 추출
            neighbor_ids = set()
            for rel in relationships:
                if str(rel.source_node_id) != str(node_id):
                    neighbor_ids.add(rel.source_node_id)
                if str(rel.target_node_id) != str(node_id):
                    neighbor_ids.add(rel.target_node_id)

            # 이웃 노드 가져오기
            neighbors = []
            for neighbor_id in neighbor_ids:
                neighbor = await self.node_use_case.get_node(neighbor_id)
                if neighbor:
                    neighbors.append(self._node_to_mcp_response(neighbor))

            if ctx:
                self.logger.info("%s개의 이웃을 찾았습니다.", len(neighbors))

            return {
                "neighbors": neighbors,
                "count": len(neighbors),
                "depth": depth,
            }

        except Exception as e:
            self.logger.error("이웃 가져오기 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="get_neighbors",
                message=str(e),
                original_error=e,
            ) from e

    async def find_paths(
        self,
        source_node_id: int,
        target_node_id: int,
        max_depth: int = 5,
        ctx: Optional[Context] = None,
    ) -> dict[str, Any]:
        """
        지식 그래프에서 두 노드 간의 경로를 찾습니다.

        Args:
            source_node_id: 소스 노드의 ID
            target_node_id: 대상 노드의 ID
            max_depth: 검색할 최대 경로 깊이 (기본값 5)
            ctx: MCP 컨텍스트 객체

        Returns:
            노드 간 최단 경로 또는 경로가 없는 경우 None
        """
        self.logger.info("%s에서 %s로의 경로를 찾습니다.", source_node_id, target_node_id)

        try:
            # RelationshipAnalysisUseCase 사용 가능 여부 확인
            if not hasattr(self.relationship_use_case, "find_shortest_path"):
                # 사용 가능한 메서드를 사용한 기본 구현으로 대체
                return {
                    "path": None,
                    "length": 0,
                    "message": "경로 찾기는 RelationshipAnalysisUseCase 구현이 필요합니다.",
                }

            source_id = NodeId(str(source_node_id))
            target_id = NodeId(str(target_node_id))

            # 유스케이스 호출 (사용 가능한 경우)
            path_relationships = await self.relationship_use_case.find_shortest_path(
                source_node_id=source_id,
                target_node_id=target_id,
                max_depth=max_depth,
            )

            if not path_relationships:
                if ctx:
                    self.logger.info("경로를 찾을 수 없습니다.")
                return {
                    "path": None,
                    "length": 0,
                    "message": "지정된 노드 간에 경로를 찾을 수 없습니다.",
                }

            # 경로를 MCP 응답 형식으로 변환
            path_edges = [self._relationship_to_mcp_response(rel) for rel in path_relationships]

            if ctx:
                self.logger.info("%s개의 엣지가 있는 경로를 찾았습니다.", len(path_edges))

            return {
                "path": path_edges,
                "length": len(path_edges),
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
            }

        except Exception as e:
            self.logger.error("경로 찾기 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="find_paths",
                message=str(e),
                original_error=e,
            ) from e

    async def search_similar_nodes(
        self,
        node_id: int,
        limit: int = 10,
        threshold: Optional[float] = None,
        ctx: Optional[Context] = None,
    ) -> dict[str, Any]:
        """
        주어진 노드와 유사한 노드를 검색합니다.

        Args:
            node_id: 참조 노드의 ID
            limit: 반환할 최대 유사 노드 수 (기본값 10)
            threshold: 유사도 임계값 (제공되지 않으면 설정 기본값 사용)
            ctx: MCP 컨텍스트 객체

        Returns:
            유사도 점수와 함께 유사한 노드 목록
        """
        self.logger.info("%s와 유사한 노드를 검색합니다.", node_id)

        try:
            domain_node_id = NodeId(str(node_id))

            # 제공되지 않은 경우 설정 기본 임계값 사용
            actual_threshold = (
                threshold if threshold is not None else self.config.similarity_threshold
            )

            # NodeEmbeddingUseCase 사용 가능 여부 확인
            if not hasattr(self.node_use_case, "find_similar_nodes"):
                # 대체: 노드 이름으로 지식 검색 사용
                node = await self.node_use_case.get_node(domain_node_id)
                if not node:
                    return {"error": f"노드 {node_id}를 찾을 수 없습니다."}

                # 텍스트 검색을 대체로 사용
                results = await self.knowledge_search_use_case.search_knowledge(
                    query=node.name,
                    strategy=SearchStrategy.SEMANTIC,
                    limit=limit,
                    include_documents=False,
                    include_nodes=True,
                    include_relationships=False,
                )

                # 유사 노드 형식으로 변환
                similar_nodes = []
                for result in results.results:
                    if result.node and str(result.node.id) != str(node_id):
                        similar_nodes.append(
                            {
                                **self._node_to_mcp_response(result.node),
                                "similarity": result.score,
                            }
                        )

                if ctx:
                    self.logger.info(
                        "%s개의 유사한 노드를 찾았습니다 (텍스트 검색 사용).", len(similar_nodes)
                    )

                return {
                    "similar_nodes": similar_nodes,
                    "count": len(similar_nodes),
                    "reference_node_id": node_id,
                    "method": "text_search_fallback",
                }

            # 전용 유사도 검색 사용 가능 시 사용
            similar_nodes_with_scores = await self.node_use_case.find_similar_nodes(
                node_id=domain_node_id,
                limit=limit,
                threshold=actual_threshold,
            )

            # MCP 응답 형식으로 변환
            similar_nodes = []
            for node, similarity in similar_nodes_with_scores:
                similar_nodes.append(
                    {
                        **self._node_to_mcp_response(node),
                        "similarity": similarity,
                    }
                )

            if ctx:
                self.logger.info("%s개의 유사한 노드를 찾았습니다.", len(similar_nodes))

            return {
                "similar_nodes": similar_nodes,
                "count": len(similar_nodes),
                "reference_node_id": node_id,
                "method": "embedding_similarity",
            }

        except Exception as e:
            self.logger.error("유사 노드 검색 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="search_similar_nodes",
                message=str(e),
                original_error=e,
            ) from e

    # === 서버 생명주기 ===

    def start(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        """
        MCP 서버를 시작합니다.

        Args:
            host: 서버 호스트 (제공되지 않으면 설정 기본값 사용)
            port: 서버 포트 (제공되지 않으면 설정 기본값 사용)
        """
        actual_host = host or self.config.host
        actual_port = port or self.config.port

        self.logger.info("%s:%s에서 지식 그래프 MCP 서버를 시작합니다.", actual_host, actual_port)

        # FastMCP의 내장 메서드를 사용하여 서버 시작
        self.mcp_server.run()

    def close(self) -> None:
        """서버를 닫습니다."""
        self.logger.info("MCP 서버를 닫습니다.")
        # FastMCP가 서버 생명주기를 자동으로 처리합니다.
