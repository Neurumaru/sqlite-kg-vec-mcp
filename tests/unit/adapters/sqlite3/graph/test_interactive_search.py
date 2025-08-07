"""
InteractiveSearchEngine 및 SearchContext 단위 테스트.
"""

# pylint: disable=protected-access

import unittest
from unittest.mock import AsyncMock, Mock

from src.adapters.sqlite3.graph.interactive_search import InteractiveSearchEngine, SearchContext


class TestSearchContext(unittest.TestCase):
    """SearchContext 클래스 테스트."""

    def test_init(self):
        """Given: 원본 쿼리가 제공될 때
        When: SearchContext를 초기화하면
        Then: 모든 속성이 초기화된다
        """
        # Given
        original_query = "테스트 쿼리"

        # When
        context = SearchContext(original_query)

        # Then
        self.assertEqual(context.original_query, original_query)
        self.assertEqual(context.entities, [])
        self.assertEqual(context.relationships, [])
        self.assertEqual(context.history, [])
        self.assertEqual(context.current_step, 0)
        self.assertEqual(context.metadata, {})

    def test_add_findings(self):
        """Given: 새로운 엔티티와 관계가 발견될 때
        When: add_findings를 호출하면
        Then: 결과가 컨텍스트에 추가된다
        """
        # Given
        context = SearchContext("테스트")
        entities = [{"id": "1", "name": "Entity 1"}, {"id": "2", "name": "Entity 2"}]
        relationships = [{"id": "rel1", "source": "1", "target": "2"}]

        # When
        context.add_findings(entities, relationships)

        # Then
        self.assertEqual(len(context.entities), 2)
        self.assertEqual(len(context.relationships), 1)
        self.assertEqual(context.entities[0]["name"], "Entity 1")
        self.assertEqual(context.relationships[0]["source"], "1")

    def test_add_findings_multiple_calls(self):
        """Given: 여러 번 결과가 추가될 때
        When: add_findings를 여러 번 호출하면
        Then: 모든 결과가 누적된다
        """
        # Given
        context = SearchContext("테스트")

        # When
        context.add_findings([{"id": "1"}], [{"id": "rel1"}])
        context.add_findings([{"id": "2"}], [{"id": "rel2"}])

        # Then
        self.assertEqual(len(context.entities), 2)
        self.assertEqual(len(context.relationships), 2)

    def test_add_history_step(self):
        """Given: 탐색 단계 정보가 있을 때
        When: add_history_step을 호출하면
        Then: 히스토리에 추가되고 단계가 증가한다
        """
        # Given
        context = SearchContext("테스트")
        step_info = {"action": "search", "result_count": 5}

        # When
        context.add_history_step(step_info)

        # Then
        self.assertEqual(len(context.history), 1)
        self.assertEqual(context.current_step, 1)
        self.assertEqual(context.history[0], step_info)

    def test_get_history_summary_empty(self):
        """Given: 히스토리가 비어있을 때
        When: get_history_summary를 호출하면
        Then: 시작 메시지를 반환한다
        """
        # Given
        context = SearchContext("테스트")

        # When
        summary = context.get_history_summary()

        # Then
        self.assertEqual(summary, "탐색 시작")

    def test_get_history_summary_with_steps(self):
        """Given: 여러 탐색 단계가 있을 때
        When: get_history_summary를 호출하면
        Then: 최근 3단계의 요약을 반환한다
        """
        # Given
        context = SearchContext("테스트")
        for i in range(5):
            context.add_history_step({"action": f"action_{i}", "result_count": i + 1})

        # When
        summary = context.get_history_summary()

        # Then
        # 최근 3단계만 포함되어야 함
        self.assertIn("action_2", summary)
        self.assertIn("action_3", summary)
        self.assertIn("action_4", summary)
        self.assertNotIn("action_0", summary)
        self.assertNotIn("action_1", summary)

    def test_get_entity_names(self):
        """Given: 엔티티들이 발견되었을 때
        When: get_entity_names를 호출하면
        Then: 엔티티 이름 목록을 반환한다
        """
        # Given
        context = SearchContext("테스트")
        entities = [
            {"id": "1", "name": "Entity 1"},
            {"id": "2", "name": "Entity 2"},
            {"id": "3"},  # name이 없는 경우
        ]
        context.add_findings(entities, [])

        # When
        names = context.get_entity_names()

        # Then
        self.assertEqual(len(names), 3)
        self.assertEqual(names[0], "Entity 1")
        self.assertEqual(names[1], "Entity 2")
        self.assertEqual(names[2], "3")  # id가 name 대신 사용됨


class TestInteractiveSearchEngine(unittest.IsolatedAsyncioTestCase):
    """InteractiveSearchEngine 클래스 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.mock_knowledge_graph = Mock()
        self.mock_llm_client = Mock()
        self.search_engine = InteractiveSearchEngine(
            knowledge_graph=self.mock_knowledge_graph, llm_client=self.mock_llm_client, max_steps=3
        )

    def test_init(self):
        """Given: 초기화 파라미터가 제공될 때
        When: InteractiveSearchEngine을 초기화하면
        Then: 속성이 올바르게 설정된다
        """
        # Given & When - suppress expected deprecation warning
        with self.assertLogs(level="WARNING"):
            engine = InteractiveSearchEngine(
                knowledge_graph=self.mock_knowledge_graph,
                llm_client=self.mock_llm_client,
                max_steps=5,
                enable_langfuse=True,  # deprecated parameter
            )

        # Then
        self.assertEqual(engine.kg, self.mock_knowledge_graph)
        self.assertEqual(engine.llm, self.mock_llm_client)
        self.assertEqual(engine.max_steps, 5)

    def test_init_with_langfuse_warning(self):
        """Given: enable_langfuse가 True일 때
        When: InteractiveSearchEngine을 초기화하면
        Then: 경고 로그가 출력된다
        """
        # Given & When
        with self.assertLogs(level="WARNING") as log:
            InteractiveSearchEngine(
                knowledge_graph=self.mock_knowledge_graph,
                llm_client=self.mock_llm_client,
                enable_langfuse=True,
            )

        # Then
        self.assertIn("Langfuse integration has been removed", log.output[0])

    async def test_search_with_text_search_capability(self):
        """Given: 지식 그래프가 텍스트 검색을 지원할 때
        When: search를 호출하면
        Then: 텍스트 검색 결과를 반환한다
        """
        # Given
        query = "테스트 검색"
        mock_results = [
            Mock(entity=Mock(to_dict=Mock(return_value={"id": "1", "name": "Entity 1"}))),
            Mock(entity=Mock(to_dict=Mock(return_value={"id": "2", "name": "Entity 2"}))),
        ]
        self.mock_knowledge_graph.search_by_text = AsyncMock(return_value=mock_results)

        # When
        result = await self.search_engine.search(query)

        # Then
        self.assertTrue(result["success"])
        self.assertEqual(result["original_query"], query)
        self.assertEqual(result["total_steps"], 1)
        self.assertEqual(len(result["final_results"]["entities"]), 2)
        self.assertIn("session_id", result)
        self.mock_knowledge_graph.search_by_text.assert_called_once_with(query, limit=20)

    async def test_search_without_text_search_capability(self):
        """Given: 지식 그래프가 텍스트 검색을 지원하지 않을 때
        When: search를 호출하면
        Then: 경고 로그와 함께 빈 결과를 반환한다
        """
        # Given
        query = "테스트 검색"
        # search_by_text 메서드가 없는 경우
        if hasattr(self.mock_knowledge_graph, "search_by_text"):
            delattr(self.mock_knowledge_graph, "search_by_text")

        # When
        with self.assertLogs(level="WARNING") as log:
            result = await self.search_engine.search(query)

        # Then
        self.assertTrue(result["success"])
        self.assertEqual(len(result["final_results"]["entities"]), 0)
        self.assertIn("does not support text search", log.output[0])

    async def test_search_with_dict_results(self):
        """Given: 검색 결과가 딕셔너리 형태일 때
        When: search를 호출하면
        Then: 딕셔너리 결과를 그대로 사용한다
        """
        # Given
        query = "테스트 검색"
        mock_results = [{"id": "1", "name": "Entity 1"}, {"id": "2", "name": "Entity 2"}]
        self.mock_knowledge_graph.search_by_text = AsyncMock(return_value=mock_results)

        # When
        result = await self.search_engine.search(query)

        # Then
        self.assertTrue(result["success"])
        self.assertEqual(len(result["final_results"]["entities"]), 2)
        self.assertEqual(result["final_results"]["entities"][0]["name"], "Entity 1")

    async def test_search_with_exception(self):
        """Given: 검색 중 예외가 발생할 때
        When: search를 호출하면
        Then: 실패 결과를 반환한다
        """
        # Given
        query = "테스트 검색"
        exception_message = "검색 실패"
        self.mock_knowledge_graph.search_by_text = AsyncMock(
            side_effect=Exception(exception_message)
        )

        # When
        with self.assertLogs(level="ERROR") as log:
            result = await self.search_engine.search(query)

        # Then
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], exception_message)
        self.assertEqual(result["total_steps"], 0)
        self.assertEqual(len(result["final_results"]["entities"]), 0)
        self.assertIn("Search failed", log.output[0])

    async def test_search_with_user_metadata(self):
        """Given: 사용자 ID와 세션 메타데이터가 제공될 때
        When: search를 호출하면
        Then: 메타데이터가 무시되고 결과를 반환한다 (simplified version)
        """
        # Given
        query = "테스트 검색"
        user_id = "user123"
        session_metadata = {"key": "value"}
        mock_results = [{"id": "1", "name": "Entity 1"}]
        self.mock_knowledge_graph.search_by_text = AsyncMock(return_value=mock_results)

        # When
        result = await self.search_engine.search(
            query, user_id=user_id, session_metadata=session_metadata
        )

        # Then
        self.assertTrue(result["success"])
        self.assertEqual(result["original_query"], query)
        # user_id와 session_metadata는 사용되지 않음 (simplified version)

    def test_prepare_final_results_with_unique_entities(self):
        """Given: 중복된 엔티티가 있는 컨텍스트가 있을 때
        When: _prepare_final_results를 호출하면
        Then: 중복이 제거된 결과를 반환한다
        """
        # Given
        context = SearchContext("테스트")
        entities = [
            {"id": "1", "name": "Entity 1"},
            {"id": "2", "name": "Entity 2"},
            {"id": "1", "name": "Entity 1 Updated"},  # 중복, 첫 번째만 유지됨
        ]
        relationships = [
            {"id": "rel1", "source": "1", "target": "2", "type": "connected"},
            {"id": "rel2", "source": "2", "target": "1", "type": "related"},
        ]
        context.add_findings(entities, relationships)

        # When
        result = self.search_engine._prepare_final_results(context)

        # Then
        self.assertEqual(len(result["entities"]), 2)  # 중복 제거됨
        self.assertEqual(len(result["relationships"]), 2)
        self.assertEqual(result["metadata"]["total_entities"], 2)
        self.assertEqual(result["metadata"]["total_relationships"], 2)
        self.assertEqual(result["metadata"]["search_steps"], 0)

    def test_prepare_final_results_with_unique_relationships(self):
        """Given: 중복된 관계가 있는 컨텍스트가 있을 때
        When: _prepare_final_results를 호출하면
        Then: 중복이 제거된 결과를 반환한다
        """
        # Given
        context = SearchContext("테스트")
        entities = [{"id": "1", "name": "Entity 1"}]
        relationships = [
            {"source": "1", "target": "2", "type": "connected"},
            {"source": "1", "target": "2", "type": "connected"},  # 중복
            {"source": "2", "target": "3", "type": "related"},
        ]
        context.add_findings(entities, relationships)

        # When
        result = self.search_engine._prepare_final_results(context)

        # Then
        self.assertEqual(len(result["relationships"]), 2)  # 중복 제거됨

    def test_prepare_final_results_empty_context(self):
        """Given: 빈 컨텍스트가 있을 때
        When: _prepare_final_results를 호출하면
        Then: 빈 결과를 반환한다
        """
        # Given
        context = SearchContext("테스트")

        # When
        result = self.search_engine._prepare_final_results(context)

        # Then
        self.assertEqual(len(result["entities"]), 0)
        self.assertEqual(len(result["relationships"]), 0)
        self.assertEqual(result["metadata"]["total_entities"], 0)
        self.assertEqual(result["metadata"]["total_relationships"], 0)

    def test_prepare_final_results_with_steps(self):
        """Given: 여러 단계를 거친 컨텍스트가 있을 때
        When: _prepare_final_results를 호출하면
        Then: 단계 수가 포함된 메타데이터를 반환한다
        """
        # Given
        context = SearchContext("테스트")
        context.add_history_step({"action": "step1"})
        context.add_history_step({"action": "step2"})
        context.add_findings([{"id": "1", "name": "Entity"}], [])

        # When
        result = self.search_engine._prepare_final_results(context)

        # Then
        self.assertEqual(result["metadata"]["search_steps"], 2)


if __name__ == "__main__":
    unittest.main()
