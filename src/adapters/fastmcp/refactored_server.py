"""
MCP 서버 인터페이스를 위한 리팩토링된 API 엔드포인트 및 핸들러.

이 버전은 큰 KnowledgeGraphServer 클래스를 단일 책임 원칙에 따라
더 작고 집중된 핸들러 클래스로 분할합니다.
"""

import logging
from typing import Optional

from fastmcp import FastMCP

from src.use_cases.knowledge_search import KnowledgeSearchUseCase
from src.use_cases.node import NodeManagementUseCase
from src.use_cases.relationship import RelationshipManagementUseCase

from .config import FastMCPConfig
from .exceptions import MCPServerException
from .handlers import NodeHandler, RelationshipHandler, SearchHandler


class KnowledgeGraphServer:
    """
    지식 그래프 API를 제공하는 리팩토링된 MCP 서버.

    이 서버는 MCP 클라이언트와 전문 핸들러 간의 요청을 조정하며,
    각 핸들러는 특정 작업 도메인을 담당합니다.
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
        self.config = config

        # 로깅 설정
        self.logger = logging.getLogger("kg_server")
        self.logger.setLevel(getattr(logging, config.log_level))

        # 전문 핸들러 초기화
        self.node_handler = NodeHandler(node_use_case, config)
        self.relationship_handler = RelationshipHandler(relationship_use_case, config)
        self.search_handler = SearchHandler(
            node_use_case, relationship_use_case, knowledge_search_use_case, config
        )

        # FastMCP로 MCP 서버 생성
        self.mcp_server: FastMCP = FastMCP(
            name="Knowledge Graph Server",
            instructions="SQLite-based knowledge graph with vector search capabilities",
        )

        # 모든 도구 등록
        self._register_tools()

        self.logger.info("전문 핸들러로 지식 그래프 서버가 초기화되었습니다.")

    def _register_tools(self):
        """각 핸들러에 모든 API 엔드포인트 도구를 등록합니다."""
        # 노드 관리 도구
        self.mcp_server.tool()(self.node_handler.create_node)
        self.mcp_server.tool()(self.node_handler.get_node)
        self.mcp_server.tool()(self.node_handler.update_node)
        self.mcp_server.tool()(self.node_handler.delete_node)
        self.mcp_server.tool()(self.node_handler.find_nodes)

        # 관계 관리 도구
        self.mcp_server.tool()(self.relationship_handler.create_edge)
        self.mcp_server.tool()(self.relationship_handler.get_edge)
        self.mcp_server.tool()(self.relationship_handler.update_edge)
        self.mcp_server.tool()(self.relationship_handler.delete_edge)
        self.mcp_server.tool()(self.relationship_handler.find_edges)

        # 검색 및 순회 도구
        self.mcp_server.tool()(self.search_handler.get_neighbors)
        self.mcp_server.tool()(self.search_handler.find_paths)
        self.mcp_server.tool()(self.search_handler.search_similar_nodes)
        self.mcp_server.tool()(self.search_handler.search_by_text)

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

        try:
            # FastMCP의 내장 메서드를 사용하여 서버 시작
            self.mcp_server.run()
        except Exception as e:
            self.logger.error("MCP 서버 시작 실패: %s", e)
            raise MCPServerException(
                server_state="starting",
                operation="start",
                message=str(e),
                host=actual_host,
                port=actual_port,
                original_error=e,
            ) from e

    def close(self) -> None:
        """서버를 닫고 리소스를 정리합니다."""
        self.logger.info("MCP 서버를 닫고 리소스를 정리합니다.")
        try:
            # FastMCP가 서버 생명주기를 자동으로 처리합니다.
            # 필요한 경우 여기에 추가 정리 코드를 추가할 수 있습니다.
            self.logger.info("MCP 서버가 성공적으로 닫혔습니다.")
        except Exception as e:
            self.logger.error("서버 종료 중 오류 발생: %s", e)
            raise MCPServerException(
                server_state="stopping",
                operation="close",
                message=str(e),
                original_error=e,
            ) from e
