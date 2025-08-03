"""
LLM 기반 Interactive Knowledge Graph Search 모듈.
Note: Langfuse integration has been removed. This is a simplified version
with basic search functionality.
"""

import logging
import uuid
from typing import Any


class SearchContext:
    """검색 컨텍스트를 관리하는 클래스."""

    def __init__(self, original_query: str):
        self.original_query = original_query
        self.entities: list[dict] = []
        self.relationships: list[dict] = []
        self.history: list[dict] = []
        self.current_step = 0
        self.metadata: dict[str, Any] = {}

    def add_findings(self, entities: list[dict], relationships: list[dict]):
        """새로운 발견 사항을 추가합니다."""
        self.entities.extend(entities)
        self.relationships.extend(relationships)

    def add_history_step(self, step_info: dict):
        """탐색 히스토리에 단계를 추가합니다."""
        self.history.append(step_info)
        self.current_step += 1

    def get_history_summary(self) -> str:
        """탐색 히스토리 요약을 반환합니다."""
        if not self.history:
            return "탐색 시작"
        summary_parts = []
        for i, step in enumerate(self.history[-3:]):  # 최근 3단계만
            action = step.get("action", "unknown")
            result_count = step.get("result_count", 0)
            summary_parts.append(f"Step {i+1}: {action} -> {result_count}개 결과")
        return " | ".join(summary_parts)

    def get_entity_names(self) -> list[str]:
        """현재 발견한 엔티티 이름 목록을 반환합니다."""
        return [entity.get("name", entity.get("id", "unknown")) for entity in self.entities]


class InteractiveSearchEngine:
    """Simplified Interactive 검색 엔진 (Langfuse integration removed)."""

    def __init__(
        self,
        knowledge_graph,
        llm_client,
        max_steps: int = 5,  # Reduced from 10
        enable_langfuse: bool = False,  # Deprecated parameter
    ):
        """
        Interactive 검색 엔진 초기화.
        Args:
            knowledge_graph: 지식 그래프 인스턴스
            llm_client: LLM 클라이언트
            max_steps: 최대 탐색 단계 수
            enable_langfuse: Deprecated - Langfuse integration removed
        """
        self.kg = knowledge_graph
        self.llm = llm_client
        self.max_steps = max_steps
        self.logger = logging.getLogger(__name__)
        if enable_langfuse:
            self.logger.warning("Langfuse integration has been removed")

    async def search(
        self,
        query: str,
        user_id: str | None = None,
        session_metadata: dict | None = None,
    ) -> dict[str, Any]:
        """
        Simplified interactive 검색을 수행합니다.
        Args:
            query: 검색 쿼리
            user_id: 사용자 ID (unused)
            session_metadata: 세션 메타데이터 (unused)
        Returns:
            검색 결과 및 메타데이터
        """
        session_id = str(uuid.uuid4())
        context = SearchContext(query)
        try:
            # Simplified search: just perform basic semantic search
            self.logger.info("Performing simplified search for: %s", query)
            # Basic semantic search
            if hasattr(self.kg, "search_by_text"):
                results = await self.kg.search_by_text(query, limit=20)
                entities = [r.entity.to_dict() if hasattr(r, "entity") else r for r in results]
                context.add_findings(entities, [])
            else:
                self.logger.warning("Knowledge graph does not support text search")
            # Prepare final results
            final_results = self._prepare_final_results(context)
            return {
                "session_id": session_id,
                "original_query": query,
                "final_results": final_results,
                "total_steps": 1,  # Simplified to single step
                "search_metadata": context.metadata,
                "success": True,
                "note": "Simplified search - Langfuse integration removed",
            }
        except Exception as exception:
            self.logger.error("Search failed: %s", exception)
            return {
                "session_id": session_id,
                "original_query": query,
                "final_results": {"entities": [], "relationships": []},
                "total_steps": 0,
                "error": str(exception),
                "success": False,
            }

    def _prepare_final_results(self, context: SearchContext) -> dict[str, Any]:
        """최종 결과를 정리합니다."""
        # 중복 제거 및 정렬
        unique_entities = {}
        for entity in context.entities:
            entity_id = entity.get("id")
            if entity_id and entity_id not in unique_entities:
                unique_entities[entity_id] = entity
        unique_relationships = {}
        for rel in context.relationships:
            rel_key = f"{rel.get('source')}_{rel.get('target')}_{rel.get('type')}"
            if rel_key not in unique_relationships:
                unique_relationships[rel_key] = rel
        return {
            "entities": list(unique_entities.values()),
            "relationships": list(unique_relationships.values()),
            "metadata": {
                "total_entities": len(unique_entities),
                "total_relationships": len(unique_relationships),
                "search_steps": context.current_step,
            },
        }
