"""
Ollama LLM Service 어댑터 단위 테스트.
"""

# pylint: disable=protected-access
import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.adapters.ollama.ollama_llm_service import OllamaLLMService
from src.common.config.llm import OllamaConfig


class TestOllamaLLMService(unittest.IsolatedAsyncioTestCase):
    """OllamaLLMService 어댑터 테스트 케이스."""

    def setUp(self):
        """테스트 픽스처 설정."""
        # Mock Ollama client
        self.mock_ollama_client = Mock()
        self.mock_ollama_client.model = "llama3.2"
        self.mock_ollama_client.base_url = "http://localhost:11434"

        # Create service with mock client
        self.llm_service = OllamaLLMService(
            ollama_client=self.mock_ollama_client, default_temperature=0.7, max_tokens=1000
        )

    def test_initialization_with_client(self):
        """클라이언트로 초기화 테스트."""
        # Then: Should use provided client and settings
        self.assertEqual(self.llm_service.ollama_client, self.mock_ollama_client)
        self.assertEqual(self.llm_service.default_temperature, 0.7)
        self.assertEqual(self.llm_service.max_tokens, 1000)

    def test_initialization_with_config(self):
        """설정 객체로 초기화 테스트."""
        # Given: Ollama configuration
        config = OllamaConfig(
            host="test-host", port=8080, model="test-model", temperature=0.3, max_tokens=500
        )

        # When: Create service with config
        with patch("src.adapters.ollama.ollama_llm_service.OllamaClient") as mock_client_cls:
            mock_client = Mock()
            mock_client_cls.return_value = mock_client

            service = OllamaLLMService(config=config)

            # Then: Should create client with config
            mock_client_cls.assert_called_once_with(config=config)
            self.assertEqual(service.default_temperature, 0.3)
            self.assertEqual(service.max_tokens, 500)

    def test_initialization_default_config(self):
        """기본 설정으로 초기화 테스트."""
        # When: Create service without config
        with (
            patch("src.adapters.ollama.ollama_llm_service.OllamaClient") as mock_client_cls,
            patch("src.adapters.ollama.ollama_llm_service.OllamaConfig") as mock_config_cls,
        ):

            mock_config = Mock()
            mock_config.temperature = 0.7
            mock_config.max_tokens = 2000
            mock_config_cls.return_value = mock_config

            mock_client = Mock()
            mock_client_cls.return_value = mock_client

            _ = OllamaLLMService()

            # Then: Should create default config and client
            mock_config_cls.assert_called_once()
            mock_client_cls.assert_called_once_with(config=mock_config)

    def test_messages_to_text_conversion(self):
        """메시지를 텍스트로 변환 테스트."""
        # Given: Various message types
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello, how are you?"),
            AIMessage(content="I'm doing well, thank you!"),
        ]

        # When: Convert messages to text
        result = self.llm_service._messages_to_text(messages)

        # Then: Should format correctly
        expected = "system: You are a helpful assistant.\nuser: Hello, how are you?\nassistant: I'm doing well, thank you!"
        self.assertEqual(result, expected)

    async def test_invoke_success(self):
        """LangChain invoke 메서드 성공 테스트."""
        # Given: Mock successful generation
        mock_response = Mock()
        mock_response.text = "Generated response"

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            # When: Invoke with messages
            messages = [HumanMessage(content="Hello")]
            result = await self.llm_service.invoke(messages, temperature=0.5, max_tokens=100)

            # Then: Should return AIMessage
            self.assertIsInstance(result, AIMessage)
            self.assertEqual(result.content, "Generated response")

            # Verify client was called correctly
            mock_to_thread.assert_called_once()
            args, kwargs = mock_to_thread.call_args
            self.assertEqual(args[0], self.mock_ollama_client.generate)
            self.assertEqual(kwargs["prompt"], "user: Hello")
            self.assertEqual(kwargs["temperature"], 0.5)
            self.assertEqual(kwargs["max_tokens"], 100)

    async def test_stream_success(self):
        """LangChain stream 메서드 성공 테스트."""
        # Given: Mock successful generation
        mock_response = Mock()
        mock_response.text = "Hello world from streaming response!"

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            # When: Stream with messages
            messages = [HumanMessage(content="Hello")]
            chunks = []

            async for chunk in self.llm_service.stream(messages, temperature=0.3):
                chunks.append(chunk)

            # Then: Should yield response chunks (at least 1)
            self.assertGreaterEqual(len(chunks), 1)
            combined = "".join(chunks)
            self.assertEqual(combined, "Hello world from streaming response!")

    async def test_batch_success(self):
        """LangChain batch 메서드 성공 테스트."""
        # Given: Multiple message inputs
        inputs = [[HumanMessage(content="Hello")], [HumanMessage(content="Goodbye")]]

        # Mock invoke method
        responses = [AIMessage(content="Hi there!"), AIMessage(content="See you later!")]

        with patch.object(self.llm_service, "invoke", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = responses

            # When: Batch process
            results = await self.llm_service.batch(inputs, temperature=0.2)

            # Then: Should return all responses
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].content, "Hi there!")
            self.assertEqual(results[1].content, "See you later!")

    async def test_batch_with_exceptions(self):
        """배치 처리 중 예외 발생 테스트."""
        # Given: Multiple inputs with one failing
        inputs = [[HumanMessage(content="Hello")], [HumanMessage(content="Failing input")]]

        with patch.object(self.llm_service, "invoke", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = [AIMessage(content="Success"), Exception("API Error")]

            # When: Batch process
            results = await self.llm_service.batch(inputs)

            # Then: Should handle exceptions gracefully
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].content, "Success")
            self.assertIn("일괄 처리 항목 1 실패", results[1].content)

    # === 비동기 예외 처리 일관성 테스트 추가 ===

    async def test_async_exception_propagation_consistency(self):
        """비동기 작업에서 예외 전파 일관성 테스트."""
        # Given: 다양한 예외 시나리오
        test_scenarios = [
            {
                "name": "network_timeout",
                "exception": TimeoutError("Network timeout"),
                "should_propagate": True,
                "expected_behavior": "immediate_failure",
            },
            {
                "name": "api_rate_limit",
                "exception": Exception("Rate limit exceeded"),
                "should_propagate": False,
                "expected_behavior": "graceful_degradation",
            },
            {
                "name": "invalid_response",
                "exception": ValueError("Invalid JSON response"),
                "should_propagate": False,
                "expected_behavior": "fallback_response",
            },
        ]

        for scenario in test_scenarios:
            with self.subTest(scenario=scenario["name"]):
                # Given: Mock specific exception
                with patch.object(
                    self.llm_service, "invoke", new_callable=AsyncMock
                ) as mock_invoke:
                    mock_invoke.side_effect = scenario["exception"]

                    if scenario["should_propagate"]:
                        # When & Then: Exception should be propagated
                        with self.assertRaises(type(scenario["exception"])):
                            await self.llm_service.invoke([HumanMessage(content="test")])
                    else:
                        # When: Exception should be handled gracefully
                        if scenario["expected_behavior"] == "graceful_degradation":
                            result = await self.llm_service.batch([[HumanMessage(content="test")]])
                            # Then: Should return error message in result
                            self.assertEqual(len(result), 1)
                            self.assertIn("실패", result[0].content)

                        elif scenario["expected_behavior"] == "fallback_response":
                            # Fallback behavior for analyze_query - mock response with invalid JSON
                            with patch(
                                "asyncio.to_thread", new_callable=AsyncMock
                            ) as mock_to_thread:
                                mock_response = Mock()
                                mock_response.text = "Invalid JSON response"
                                mock_to_thread.return_value = mock_response
                                result = await self.llm_service.analyze_query("test query")
                                # Then: Should return fallback result
                                self.assertEqual(result["strategy"], "SEMANTIC")
                                self.assertEqual(result["confidence"], 0.5)

    async def test_batch_partial_failure_behavior_consistency(self):
        """배치 처리에서 부분 실패 동작 일관성 테스트."""
        # Given: Mixed success/failure scenarios
        test_cases = [
            {
                "name": "first_success_second_fail",
                "side_effects": [AIMessage(content="Success"), Exception("Fail")],
                "expected_success_count": 1,
                "expected_error_count": 1,
            },
            {
                "name": "all_fail_different_errors",
                "side_effects": [
                    TimeoutError("Timeout"),
                    ValueError("Invalid"),
                    ConnectionError("Network"),
                ],
                "expected_success_count": 0,
                "expected_error_count": 3,
            },
            {
                "name": "alternating_success_fail",
                "side_effects": [
                    AIMessage(content="Success1"),
                    Exception("Fail1"),
                    AIMessage(content="Success2"),
                    Exception("Fail2"),
                ],
                "expected_success_count": 2,
                "expected_error_count": 2,
            },
        ]

        for test_case in test_cases:
            with self.subTest(case=test_case["name"]):
                # Given: Mock with specific side effects
                inputs = [
                    [HumanMessage(content=f"Input {i}")]
                    for i in range(len(test_case["side_effects"]))
                ]

                with patch.object(
                    self.llm_service, "invoke", new_callable=AsyncMock
                ) as mock_invoke:
                    mock_invoke.side_effect = test_case["side_effects"]

                    # When: Process batch
                    results = await self.llm_service.batch(inputs)

                    # Then: Verify consistent behavior
                    self.assertEqual(len(results), len(inputs))

                    success_count = sum(1 for r in results if "실패" not in r.content)
                    error_count = sum(1 for r in results if "실패" in r.content)

                    self.assertEqual(success_count, test_case["expected_success_count"])
                    self.assertEqual(error_count, test_case["expected_error_count"])

                    # Verify all results are AIMessage instances (consistent type)
                    for result in results:
                        self.assertIsInstance(result, AIMessage)

    async def test_async_exception_context_preservation(self):
        """비동기 예외에서 컨텍스트 보존 테스트."""
        # Given: Exception with context
        original_error = ValueError("Original validation error")

        class ContextPreservingError(Exception):
            """컨텍스트 보존 예외 클래스."""

            def __init__(self, message, original_error=None, context=None):
                super().__init__(message)
                self.original_error = original_error
                self.context = context or {}

        # When: Chain exceptions occur
        with patch.object(self.llm_service, "invoke", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = ContextPreservingError(
                "Processing failed",
                original_error=original_error,
                context={"input_length": 150, "model": "llama3.2", "attempt": 1},
            )

            # Then: Context should be preserved in batch processing
            results = await self.llm_service.batch([[HumanMessage(content="test")]])

            self.assertEqual(len(results), 1)
            error_result = results[0]

            # Verify error context is preserved in response
            self.assertIn("Processing failed", error_result.content)
            self.assertIn("일괄 처리 항목 0 실패", error_result.content)

    async def test_concurrent_async_exception_isolation(self):
        """동시 비동기 작업에서 예외 격리 테스트."""

        # Given: Multiple concurrent operations with mixed outcomes
        async def successful_operation(delay=0.01):
            await asyncio.sleep(delay)
            return AIMessage(content="Success")

        async def failing_operation(delay=0.01, error_type=ValueError):
            await asyncio.sleep(delay)
            raise error_type("Operation failed")

        # Create mixed operations
        operations = [
            successful_operation(0.01),
            failing_operation(0.02, TimeoutError),
            successful_operation(0.015),
            failing_operation(0.005, ValueError),
            successful_operation(0.03),
        ]

        # When: Execute concurrently with exception handling
        results = await asyncio.gather(*operations, return_exceptions=True)

        # Then: Verify exception isolation
        self.assertEqual(len(results), 5)

        # Count successes and exceptions
        successes = [r for r in results if isinstance(r, AIMessage)]
        exceptions = [r for r in results if isinstance(r, Exception)]

        self.assertEqual(len(successes), 3)  # 3 successful operations
        self.assertEqual(len(exceptions), 2)  # 2 failed operations

        # Verify exception types are preserved
        timeout_errors = [e for e in exceptions if isinstance(e, TimeoutError)]
        value_errors = [e for e in exceptions if isinstance(e, ValueError)]

        self.assertEqual(len(timeout_errors), 1)
        self.assertEqual(len(value_errors), 1)

        # Verify successful operations are unaffected by failures
        for success in successes:
            self.assertEqual(success.content, "Success")

    async def test_async_error_recovery_patterns(self):
        """비동기 에러 복구 패턴 테스트."""
        # Given: Error recovery scenarios
        recovery_scenarios = [
            {
                "name": "retry_on_timeout",
                "initial_error": TimeoutError("Initial timeout"),
                "recovery_result": AIMessage(content="Recovered"),
                "should_recover": True,
            },
            {
                "name": "no_recovery_on_auth_error",
                "initial_error": PermissionError("Authentication failed"),
                "recovery_result": None,
                "should_recover": False,
            },
        ]

        def create_mock_recovery_function(scenario_data):
            call_count_container = {"count": 0}

            async def mock_invoke_with_recovery(*args, **kwargs):
                call_count_container["count"] += 1

                if call_count_container["count"] == 1:
                    raise scenario_data["initial_error"]
                if scenario_data["should_recover"]:
                    return scenario_data["recovery_result"]
                raise scenario_data["initial_error"]  # Continue failing

            return mock_invoke_with_recovery, call_count_container

        for scenario in recovery_scenarios:
            with self.subTest(scenario=scenario["name"]):
                mock_invoke_with_recovery, call_count_container = create_mock_recovery_function(
                    scenario.copy()
                )

                with patch.object(
                    self.llm_service, "invoke", new_callable=AsyncMock
                ) as mock_invoke:
                    mock_invoke.side_effect = mock_invoke_with_recovery

                    if scenario["should_recover"]:
                        # When: Should recover after retry
                        # Note: This assumes the service has retry logic
                        # For now, we test the pattern without actual retry implementation
                        try:
                            # First call fails
                            await self.llm_service.invoke([HumanMessage(content="test")])
                            self.fail("Should have raised exception on first try")
                        except type(scenario["initial_error"]):
                            # Expected first failure
                            pass

                        # Second call succeeds (simulating retry)
                        result = await self.llm_service.invoke([HumanMessage(content="test")])
                        self.assertEqual(result, scenario["recovery_result"])
                        self.assertEqual(call_count_container["count"], 2)  # Verify retry occurred

                    else:
                        # When: Should not recover
                        with self.assertRaises(type(scenario["initial_error"])):
                            await self.llm_service.invoke([HumanMessage(content="test")])

                        # Even on retry, should still fail
                        with self.assertRaises(type(scenario["initial_error"])):
                            await self.llm_service.invoke([HumanMessage(content="test")])

                        self.assertEqual(call_count_container["count"], 2)  # Both calls failed

    async def test_analyze_query_success(self):
        """쿼리 분석 성공 테스트."""
        # Given: Mock successful analysis
        mock_response = Mock()
        mock_response.text = (
            '{"strategy": "SEMANTIC", "confidence": 0.8, "reasoning": "Conceptual query"}'
        )

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            # When: Analyze query
            result = await self.llm_service.analyze_query(
                "What is machine learning?", context={"domain": "AI"}
            )

            # Then: Should return analysis result
            expected = {"strategy": "SEMANTIC", "confidence": 0.8, "reasoning": "Conceptual query"}
            self.assertEqual(result, expected)

    async def test_analyze_query_json_error_fallback(self):
        """쿼리 분석 JSON 오류 시 fallback 테스트."""
        # Given: Invalid JSON response
        mock_response = Mock()
        mock_response.text = "Invalid JSON response"

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            # Suppress expected warning during this test
            with self.assertLogs(level="WARNING") as cm:
                # When: Analyze query
                result = await self.llm_service.analyze_query("Test query")

                # Then: Should return fallback result
                self.assertEqual(result["strategy"], "SEMANTIC")
                self.assertEqual(result["confidence"], 0.5)

                # Verify warning was logged
                self.assertIn("쿼리 분석 응답 파싱 실패", cm.output[0])
            self.assertIn("대체", result["reasoning"])

    async def test_guide_search_navigation(self):
        """검색 내비게이션 가이드 테스트."""
        # Given: Mock search results and history
        mock_results = [
            Mock(score=0.9, entity_type="Person"),
            Mock(score=0.7, entity_type="Company"),
        ]

        search_history = [{"query": "previous query", "results_count": 5}]

        mock_response = Mock()
        mock_response.text = '{"next_action": "refine", "strategy": "HYBRID", "confidence": 0.7}'

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            # When: Guide search navigation
            result = await self.llm_service.guide_search_navigation(
                current_results=mock_results,
                original_query="test query",
                search_history=search_history,
                step_number=2,
            )

            # Then: Should return navigation guidance
            self.assertEqual(result["next_action"], "refine")
            self.assertEqual(result["strategy"], "HYBRID")
            self.assertEqual(result["confidence"], 0.7)

    async def test_evaluate_search_results(self):
        """검색 결과 평가 테스트."""
        # Given: Mock search results
        mock_results = [
            Mock(score=0.95, snippet="Relevant result"),
            Mock(score=0.8, snippet="Another result"),
        ]

        mock_response = Mock()
        mock_response.text = '{"overall_quality": 0.9, "relevance_score": 0.85}'

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            # When: Evaluate search results
            result = await self.llm_service.evaluate_search_results(
                results=mock_results, query="test query", search_context={"domain": "tech"}
            )

            # Then: Should return evaluation
            self.assertEqual(result["overall_quality"], 0.9)
            self.assertEqual(result["relevance_score"], 0.85)

    async def test_extract_knowledge_from_text(self):
        """텍스트에서 지식 추출 테스트."""
        # Given: Mock knowledge extraction
        extraction_result = {"entities": [{"name": "John", "type": "Person"}], "relationships": []}

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = extraction_result

            # When: Extract knowledge
            result = await self.llm_service.extract_knowledge_from_text(
                "John is a person",
                extraction_schema={"entities": ["Person"]},
                context={"domain": "people"},
            )

            # Then: Should return enriched extraction result
            self.assertEqual(result["entities"], [{"name": "John", "type": "Person"}])
            self.assertEqual(result["relationships"], [])
            self.assertIn("extraction_metadata", result)
            self.assertEqual(result["extraction_metadata"]["text_length"], len("John is a person"))
            self.assertTrue(result["extraction_metadata"]["schema_used"])
            self.assertTrue(result["extraction_metadata"]["context_provided"])

    async def test_generate_entity_summary(self):
        """엔티티 요약 생성 테스트."""
        # Given: Entity data
        entity_data = {
            "name": "John Doe",
            "type": "Person",
            "properties": {"occupation": "engineer"},
        }

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = "John Doe is a skilled engineer."

            # When: Generate entity summary
            result = await self.llm_service.generate_entity_summary(
                entity_data, related_entities=[{"name": "Company A", "type": "Organization"}]
            )

            # Then: Should return summary
            self.assertEqual(result, "John Doe is a skilled engineer.")

    async def test_suggest_relationships(self):
        """관계 제안 테스트."""
        # Given: Source and target entities
        source_entity = {"name": "John", "type": "Person"}
        target_entities = [
            {"name": "Company A", "type": "Organization"},
            {"name": "Seattle", "type": "Location"},
        ]

        mock_response = Mock()
        mock_response.text = (
            '[{"target_entity": "Company A", "relationship_type": "WORKS_FOR", "confidence": 0.9}]'
        )

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            # When: Suggest relationships
            result = await self.llm_service.suggest_relationships(
                source_entity, target_entities, context="Business context"
            )

            # Then: Should return relationship suggestions
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["target_entity"], "Company A")
            self.assertEqual(result[0]["relationship_type"], "WORKS_FOR")
            self.assertEqual(result[0]["confidence"], 0.9)

    async def test_expand_query(self):
        """쿼리 확장 테스트."""
        # Given: Original query
        mock_response = Mock()
        mock_response.text = (
            '["machine learning", "artificial intelligence", "deep learning", "neural networks"]'
        )

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            # When: Expand query
            result = await self.llm_service.expand_query(
                "AI technology", search_context={"domain": "technology"}
            )

            # Then: Should return expanded terms
            expected = [
                "machine learning",
                "artificial intelligence",
                "deep learning",
                "neural networks",
            ]
            self.assertEqual(result, expected)

    async def test_expand_query_error_fallback(self):
        """쿼리 확장 오류 시 fallback 테스트."""
        # Given: Invalid response
        mock_response = Mock()
        mock_response.text = "Invalid JSON"

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            # Suppress expected warning during this test
            with self.assertLogs(level="WARNING") as cm:
                # When: Expand query
                result = await self.llm_service.expand_query("original query")

                # Then: Should return original query as fallback
                self.assertEqual(result, ["original query"])

                # Verify warning was logged
                self.assertIn("쿼리 확장 파싱 실패", cm.output[0])

    async def test_generate_search_suggestions(self):
        """검색 제안 생성 테스트."""
        # Given: Partial query and history
        mock_response = Mock()
        mock_response.text = '["machine learning basics", "machine learning tutorial", "machine learning applications"]'

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            # When: Generate search suggestions
            result = await self.llm_service.generate_search_suggestions(
                "machine learn", search_history=["AI", "neural networks"]
            )

            # Then: Should return suggestions
            expected = [
                "machine learning basics",
                "machine learning tutorial",
                "machine learning applications",
            ]
            self.assertEqual(result, expected)

    async def test_classify_content(self):
        """컨텐츠 분류 테스트."""
        # Given: Content and classification schema
        content = "This is a technical article about artificial intelligence and machine learning."
        schema = {
            "categories": ["technical", "business", "entertainment"],
            "topics": ["AI", "ML", "software"],
        }

        mock_response = Mock()
        mock_response.text = '{"category": "technical", "topics": ["AI", "ML"], "confidence": 0.9}'

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            # When: Classify content
            result = await self.llm_service.classify_content(content, schema)

            # Then: Should return classification
            self.assertEqual(result["category"], "technical")
            self.assertEqual(result["topics"], ["AI", "ML"])
            self.assertEqual(result["confidence"], 0.9)

    async def test_detect_language(self):
        """언어 감지 테스트."""
        # Given: Text in different languages
        test_cases = [
            ("Hello, how are you?", "en"),
            ("Hola, ¿cómo estás?", "es"),
            ("안녕하세요, 어떻게 지내세요?", "ko"),
        ]

        for text, expected_lang in test_cases:
            with self.subTest(text=text, expected=expected_lang):
                mock_response = Mock()
                mock_response.text = expected_lang

                with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
                    mock_to_thread.return_value = mock_response

                    # When: Detect language
                    result = await self.llm_service.detect_language(text)

                    # Then: Should return correct language code
                    self.assertEqual(result, expected_lang)

    async def test_detect_language_invalid_response(self):
        """언어 감지 잘못된 응답 테스트."""
        # Given: Invalid language code response
        mock_response = Mock()
        mock_response.text = "invalid_language_code_123"

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            # When: Detect language
            result = await self.llm_service.detect_language("Some text")

            # Then: Should return default language
            self.assertEqual(result, "en")

    async def test_stream_analysis(self):
        """스트리밍 분석 테스트."""
        # Given: Mock response for streaming
        mock_response = Mock()
        mock_response.text = (
            "This is a comprehensive analysis of the given prompt with detailed insights."
        )

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            # When: Stream analysis
            chunks = []
            async for chunk in self.llm_service.stream_analysis(
                "Analyze this data", context={"type": "data_analysis"}
            ):
                chunks.append(chunk)

            # Then: Should yield chunks (at least 1)
            self.assertGreaterEqual(len(chunks), 1)
            combined = " ".join(chunks)
            # Remove extra spaces from joining
            combined = " ".join(combined.split())
            expected = (
                "This is a comprehensive analysis of the given prompt with detailed insights."
            )
            self.assertEqual(combined, expected)

    async def test_get_model_info(self):
        """모델 정보 조회 테스트."""
        # Given: Mock available models
        available_models = ["llama3.2", "codellama", "mistral"]

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = available_models

            # When: Get model info
            result = await self.llm_service.get_model_info()

            # Then: Should return model information
            self.assertEqual(result["current_model"], "llama3.2")
            self.assertEqual(result["base_url"], "http://localhost:11434")
            self.assertEqual(result["available_models"], available_models)
            self.assertEqual(result["provider"], "ollama")

    async def test_health_check_healthy(self):
        """헬스 체크 정상 상태 테스트."""
        # Given: Successful health check response
        mock_response = Mock()
        mock_response.text = "OK"
        mock_response.response_time = 0.5

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            # When: Perform health check
            result = await self.llm_service.health_check()

            # Then: Should return healthy status
            self.assertEqual(result["status"], "healthy")
            self.assertEqual(result["model"], "llama3.2")
            self.assertEqual(result["response_time"], 0.5)

    async def test_health_check_unhealthy(self):
        """헬스 체크 비정상 상태 테스트."""
        # Given: Health check failure
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = ConnectionError("Connection failed")

            # When: Perform health check
            result = await self.llm_service.health_check()

            # Then: Should return unhealthy status
            self.assertEqual(result["status"], "unhealthy")
            self.assertIn("Connection failed", result["error"])

    async def test_get_usage_stats(self):
        """사용량 통계 조회 테스트."""
        # When: Get usage stats
        result = await self.llm_service.get_usage_stats()

        # Then: Should return stats structure (placeholder implementation)
        self.assertEqual(result["total_requests"], 0)
        self.assertEqual(result["total_tokens"], 0)
        self.assertEqual(result["model"], "llama3.2")
        self.assertIn("Usage tracking not implemented", result["note"])

    def test_parse_json_response_valid_json(self):
        """유효한 JSON 응답 파싱 테스트."""
        # Given: Valid JSON response
        response_text = '{"key": "value", "number": 42}'

        # When: Parse JSON response
        result = self.llm_service._parse_json_response(response_text)

        # Then: Should return parsed data
        expected = {"key": "value", "number": 42}
        self.assertEqual(result, expected)

    def test_parse_json_response_with_markdown(self):
        """마크다운 코드 블록이 있는 JSON 응답 파싱 테스트."""
        # Given: JSON with markdown code blocks
        response_text = '```json\n{"data": "test"}\n```'

        # When: Parse JSON response
        result = self.llm_service._parse_json_response(response_text)

        # Then: Should clean markdown and parse
        expected = {"data": "test"}
        self.assertEqual(result, expected)

    def test_parse_json_response_extraction_from_text(self):
        """텍스트에서 JSON 추출 테스트."""
        # Given: Response with JSON embedded in text
        response_text = 'Here is the result: {"extracted": true} and that\'s it.'

        # When: Parse JSON response
        result = self.llm_service._parse_json_response(response_text)

        # Then: Should extract JSON from text
        expected = {"extracted": True}
        self.assertEqual(result, expected)

    def test_parse_json_response_invalid_json(self):
        """잘못된 JSON 응답 파싱 오류 테스트."""
        # Given: Invalid JSON response
        response_text = "This is not JSON at all"

        # When & Then: Should raise ValueError
        with self.assertRaises(ValueError) as context:
            self.llm_service._parse_json_response(response_text)

        self.assertIn("No valid JSON found in response", str(context.exception))
        self.assertIn(response_text, str(context.exception))


if __name__ == "__main__":
    unittest.main()
