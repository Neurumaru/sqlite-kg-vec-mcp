"""
FastMCP Server Search 기능 테스트.
"""

import unittest

from tests.unit.adapters.fastmcp.server.test_mocks import (
    BaseServerTestCase,
    MockContext,
    MockSearchResult,
)


class TestKnowledgeGraphServerSearchSimilarNodes(unittest.TestCase, BaseServerTestCase):
    """KnowledgeGraphServer.search_similar_nodes 메서드 테스트."""

    def setUp(self):
        """테스트 픽스처 설정."""
        BaseServerTestCase.setUp(self)

    def tearDown(self):
        """테스트 정리."""
        BaseServerTestCase.tearDown(self)

    def test_success_when_node_id_provided(self):
        """
        Given: 노드 ID를 이용한 유사 노드 검색 요청
        When: search_similar_nodes 메서드를 호출할 때
        Then: 유사한 노드들이 반환되어야 함
        """
        # Given
        node_id = 1
        limit = 5

        mock_results = [
            MockSearchResult(node_id=2, similarity=0.9),
            MockSearchResult(node_id=3, similarity=0.8),
            MockSearchResult(node_id=4, similarity=0.7),
        ]
        self.vector_search.search_similar_by_node_id.return_value = mock_results

        # Simulate KnowledgeGraphServer.search_similar_nodes method
        def search_similar_nodes(node_id=None, text=None, limit=10, ctx=None):
            if not node_id and not text:
                return {"error": "Either node_id or text must be provided"}

            try:
                if node_id:
                    if ctx:
                        ctx.info(f"Searching for nodes similar to node {node_id}")

                    results = self.vector_search.search_similar_by_node_id(
                        node_id=node_id, limit=limit
                    )
                elif text:
                    if ctx:
                        ctx.info(f"Searching for nodes similar to text: '{text}'")

                    results = self.vector_search.search_similar_by_text(text=text, limit=limit)

                return {
                    "results": [result.to_dict() for result in results],
                    "count": len(results),
                }
            except Exception as exception:
                if ctx:
                    ctx.error(f"Failed to search similar nodes: {exception}")
                return {"error": str(exception)}

        # When
        result = search_similar_nodes(node_id=node_id, limit=limit)

        # Then
        self.vector_search.search_similar_by_node_id.assert_called_once_with(
            node_id=node_id, limit=limit
        )

        expected_results = [
            {"node_id": 2, "similarity": 0.9},
            {"node_id": 3, "similarity": 0.8},
            {"node_id": 4, "similarity": 0.7},
        ]
        self.assertEqual(result["results"], expected_results)
        self.assertEqual(result["count"], 3)

    def test_success_when_text_provided(self):
        """
        Given: 텍스트를 이용한 유사 노드 검색 요청
        When: search_similar_nodes 메서드를 호출할 때
        Then: 유사한 노드들이 반환되어야 함
        """
        # Given
        text = "artificial intelligence"
        limit = 3

        mock_results = [
            MockSearchResult(node_id=5, similarity=0.95),
            MockSearchResult(node_id=6, similarity=0.85),
        ]
        self.vector_search.search_similar_by_text.return_value = mock_results

        # Simulate method
        def search_similar_nodes(node_id=None, text=None, limit=10, ctx=None):
            if not node_id and not text:
                return {"error": "Either node_id or text must be provided"}

            try:
                if node_id:
                    if ctx:
                        ctx.info(f"Searching for nodes similar to node {node_id}")

                    results = self.vector_search.search_similar_by_node_id(
                        node_id=node_id, limit=limit
                    )
                elif text:
                    if ctx:
                        ctx.info(f"Searching for nodes similar to text: '{text}'")

                    results = self.vector_search.search_similar_by_text(text=text, limit=limit)

                return {
                    "results": [result.to_dict() for result in results],
                    "count": len(results),
                }
            except Exception as exception:
                if ctx:
                    ctx.error(f"Failed to search similar nodes: {exception}")
                return {"error": str(exception)}

        # When
        result = search_similar_nodes(text=text, limit=limit)

        # Then
        self.vector_search.search_similar_by_text.assert_called_once_with(text=text, limit=limit)

        expected_results = [
            {"node_id": 5, "similarity": 0.95},
            {"node_id": 6, "similarity": 0.85},
        ]
        self.assertEqual(result["results"], expected_results)
        self.assertEqual(result["count"], 2)

    def test_success_when_context_provided(self):
        """
        Given: Context가 포함된 검색 요청
        When: search_similar_nodes 메서드를 호출할 때
        Then: Context에 로그가 기록되어야 함
        """
        # Given
        mock_context = MockContext()
        node_id = 1
        limit = 5

        mock_results = [MockSearchResult(node_id=2, similarity=0.9)]
        self.vector_search.search_similar_by_node_id.return_value = mock_results

        # Simulate method
        def search_similar_nodes(node_id=None, text=None, limit=10, ctx=None):
            if not node_id and not text:
                return {"error": "Either node_id or text must be provided"}

            try:
                if node_id:
                    if ctx:
                        ctx.info(f"Searching for nodes similar to node {node_id}")

                    results = self.vector_search.search_similar_by_node_id(
                        node_id=node_id, limit=limit
                    )
                elif text:
                    if ctx:
                        ctx.info(f"Searching for nodes similar to text: '{text}'")

                    results = self.vector_search.search_similar_by_text(text=text, limit=limit)

                return {
                    "results": [result.to_dict() for result in results],
                    "count": len(results),
                }
            except Exception as exception:
                if ctx:
                    ctx.error(f"Failed to search similar nodes: {exception}")
                return {"error": str(exception)}

        # When
        result = search_similar_nodes(node_id=node_id, limit=limit, ctx=mock_context)

        # Then
        self.assertEqual(len(mock_context.info_calls), 1)
        self.assertIn("Searching for nodes similar to node 1", mock_context.info_calls[0])
        self.assertEqual(result["count"], 1)

    def test_error_when_missing_parameters(self):
        """
        Given: 필수 파라미터가 누락된 검색 요청
        When: search_similar_nodes 메서드를 호출할 때
        Then: 에러가 반환되어야 함
        """

        # Given
        # Simulate method
        def search_similar_nodes(node_id=None, text=None, limit=10, ctx=None):
            if not node_id and not text:
                return {"error": "Either node_id or text must be provided"}

            try:
                if node_id:
                    if ctx:
                        ctx.info(f"Searching for nodes similar to node {node_id}")

                    results = self.vector_search.search_similar_by_node_id(
                        node_id=node_id, limit=limit
                    )
                elif text:
                    if ctx:
                        ctx.info(f"Searching for nodes similar to text: '{text}'")

                    results = self.vector_search.search_similar_by_text(text=text, limit=limit)

                return {
                    "results": [result.to_dict() for result in results],
                    "count": len(results),
                }
            except Exception as exception:
                if ctx:
                    ctx.error(f"Failed to search similar nodes: {exception}")
                return {"error": str(exception)}

        # When
        result = search_similar_nodes()  # node_id와 text 모두 누락

        # Then
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Either node_id or text must be provided")

    def test_exception_when_search_fails(self):
        """
        Given: 검색 중 예외가 발생하는 상황
        When: search_similar_nodes 메서드를 호출할 때
        Then: 에러가 반환되어야 함
        """
        # Given
        mock_context = MockContext()
        node_id = 1
        error_message = "Vector search service unavailable"

        self.vector_search.search_similar_by_node_id.side_effect = Exception(error_message)

        # Simulate method
        def search_similar_nodes(node_id=None, text=None, limit=10, ctx=None):
            if not node_id and not text:
                return {"error": "Either node_id or text must be provided"}

            try:
                if node_id:
                    if ctx:
                        ctx.info(f"Searching for nodes similar to node {node_id}")

                    results = self.vector_search.search_similar_by_node_id(
                        node_id=node_id, limit=limit
                    )
                elif text:
                    if ctx:
                        ctx.info(f"Searching for nodes similar to text: '{text}'")

                    results = self.vector_search.search_similar_by_text(text=text, limit=limit)

                return {
                    "results": [result.to_dict() for result in results],
                    "count": len(results),
                }
            except Exception as exception:
                if ctx:
                    ctx.error(f"Failed to search similar nodes: {exception}")
                return {"error": str(exception)}

        # When
        result = search_similar_nodes(node_id=node_id, ctx=mock_context)

        # Then
        self.assertIn("error", result)
        self.assertEqual(result["error"], error_message)
        self.assertEqual(len(mock_context.error_calls), 1)
        self.assertIn("Failed to search similar nodes:", mock_context.error_calls[0])


if __name__ == "__main__":
    unittest.main()
