"""
MCP 작업을 위한 검색 및 순회 핸들러.
"""

import sqlite3
from typing import Any, Optional

from fastmcp import Context

from src.domain.services.knowledge_search import SearchStrategy
from src.domain.value_objects.node_id import NodeId
from src.use_cases.knowledge_search import KnowledgeSearchUseCase
from src.use_cases.node import NodeManagementUseCase
from src.use_cases.relationship import RelationshipManagementUseCase

from ..exceptions import MCPServerException
from .base import BaseHandler


class SearchHandler(BaseHandler):
    """검색 및 그래프 순회 MCP 작업을 위한 핸들러."""

    def __init__(
        self,
        node_use_case: NodeManagementUseCase,
        relationship_use_case: RelationshipManagementUseCase,
        knowledge_search_use_case: KnowledgeSearchUseCase,
        config,
    ):
        """
        검색 핸들러를 초기화합니다.

        Args:
            node_use_case: 노드 관리 유스케이스
            relationship_use_case: 관계 관리 유스케이스
            knowledge_search_use_case: 지식 검색 유스케이스
            config: FastMCP 설정
        """
        super().__init__(config)
        self.node_use_case = node_use_case
        self.relationship_use_case = relationship_use_case
        self.knowledge_search_use_case = knowledge_search_use_case

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

            self.logger.info("%s개의 결과를 찾았습니다.", len(result_items))

            return {"results": result_items, "count": len(result_items)}

        except (ValueError, sqlite3.Error, RuntimeError) as e:
            self.logger.error("텍스트 검색 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="search_by_text",
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
                    return self._create_error_response(
                        f"노드 {node_id}를 찾을 수 없습니다.", "NODE_NOT_FOUND"
                    )

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

            self.logger.info("%s개의 유사한 노드를 찾았습니다.", len(similar_nodes))

            return {
                "similar_nodes": similar_nodes,
                "count": len(similar_nodes),
                "reference_node_id": node_id,
                "method": "embedding_similarity",
            }

        except (ValueError, sqlite3.Error, RuntimeError) as e:
            self.logger.error("유사 노드 검색 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="search_similar_nodes",
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

            self.logger.info("%s개의 이웃을 찾았습니다.", len(neighbors))

            return {
                "neighbors": neighbors,
                "count": len(neighbors),
                "depth": depth,
            }

        except (ValueError, sqlite3.Error, RuntimeError) as e:
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
                self.logger.info("경로를 찾을 수 없습니다.")
                return {
                    "path": None,
                    "length": 0,
                    "message": "지정된 노드 간에 경로를 찾을 수 없습니다.",
                }

            # 경로를 MCP 응답 형식으로 변환
            path_edges = [self._relationship_to_mcp_response(rel) for rel in path_relationships]

            self.logger.info("%s개의 엣지가 있는 경로를 찾았습니다.", len(path_edges))

            return {
                "path": path_edges,
                "length": len(path_edges),
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
            }

        except (ValueError, sqlite3.Error, RuntimeError) as e:
            self.logger.error("경로 찾기 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="find_paths",
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
