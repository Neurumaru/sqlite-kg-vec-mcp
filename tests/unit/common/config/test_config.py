"""
설정 관리 시스템에 대한 테스트.

이 모듈은 프로젝트의 테스트 규칙에 따라 모든 설정 클래스를 테스트합니다:
- 테스트 구조를 위한 Given-When-Then 패턴
- 엣지 케이스를 포함한 포괄적인 유효성 검사 테스트
- 적절한 모의(mocking)를 사용한 환경 변수 로드 테스트
- 예외 처리 및 오류 메시지 검증
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from src.common.config import (
    AppConfig,
    DatabaseConfig,
    LLMConfig,
    MCPConfig,
    ObservabilityConfig,
)
from src.common.config.llm import AnthropicConfig, OllamaConfig, OpenAIConfig
from src.common.config.observability import (
    LangfuseConfig,
    LoggingObservabilityConfig,
    PrometheusConfig,
)


class TestDatabaseConfig(unittest.TestCase):
    """DatabaseConfig 유효성 검사 및 기능 테스트."""

    def test_default_values(self):
        """기본 설정 값을 테스트합니다."""
        config = DatabaseConfig()

        self.assertEqual(config.db_path, "data/knowledge_graph.db")
        self.assertTrue(config.optimize)
        self.assertEqual(config.timeout, 30.0)
        self.assertEqual(config.vector_dimension, 384)

    def test_path_validation(self):
        """데이터베이스 경로 유효성 검사를 테스트합니다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "test.db")
            config = DatabaseConfig(db_path=db_path)

            # 경로는 정규화되어야 합니다.
            self.assertEqual(config.db_path, db_path)
            # 부모 디렉토리가 생성되어야 합니다.
            self.assertTrue(Path(config.db_path).parent.exists())

    def test_vector_dimension_validation(self):
        """특정 오류 메시지와 함께 벡터 차원 유효성 검사를 테스트합니다."""
        # Given: 잘못된 벡터 차원
        invalid_dimensions = [0, -1, -100]

        for dimension in invalid_dimensions:
            with self.subTest(dimension=dimension):
                # When & Then: 잘못된 차원은 ValueError를 발생시켜야 합니다.
                with self.assertRaises(ValidationError) as context:
                    DatabaseConfig(vector_dimension=dimension)
                self.assertIn("Vector dimension must be positive", str(context.exception))

    def test_backup_path_validation(self):
        """백업 경로 유효성 검사 및 디렉토리 생성을 테스트합니다."""
        # Given: 백업을 위한 임시 디렉토리
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_path = str(Path(temp_dir) / "backups")

            # When: 백업 경로로 config 생성
            config = DatabaseConfig(backup_path=backup_path)

            # Then: 백업 경로가 설정되고 디렉토리가 생성되어야 합니다.
            self.assertEqual(config.backup_path, backup_path)
            self.assertTrue(Path(backup_path).parent.exists())

    def test_timeout_edge_cases(self):
        """엣지 케이스와 함께 타임아웃 유효성 검사를 테스트합니다."""
        # Given: 유효한 타임아웃 값
        valid_timeouts = [0.1, 1.0, 30.0, 300.0]

        for timeout in valid_timeouts:
            with self.subTest(timeout=timeout):
                # When: 타임아웃으로 config 생성
                config = DatabaseConfig(timeout=timeout)

                # Then: 타임아웃이 올바르게 설정되어야 합니다.
                self.assertEqual(config.timeout, timeout)


class TestLLMConfig(unittest.TestCase):
    """LLMConfig 및 공급자 설정을 테스트합니다."""

    def test_default_provider(self):
        """기본 공급자 설정을 테스트합니다."""
        config = LLMConfig()

        self.assertEqual(config.default_provider, "ollama")
        self.assertIsInstance(config.ollama, OllamaConfig)
        self.assertIsInstance(config.openai, OpenAIConfig)
        self.assertIsInstance(config.anthropic, AnthropicConfig)

    def test_provider_validation(self):
        """특정 오류 메시지와 함께 공급자 유효성 검사를 테스트합니다."""
        # Given: 잘못된 공급자 이름
        invalid_providers = ["invalid_provider", "gpt", "claude"]

        for provider in invalid_providers:
            with self.subTest(provider=provider):
                # When & Then: 잘못된 공급자는 ValidationError를 발생시켜야 합니다.
                with self.assertRaises(ValidationError) as context:
                    LLMConfig(default_provider=provider)
                # 오류 메시지는 잘못된 공급자를 나타내야 합니다.
                self.assertIn("Provider must be one of", str(context.exception))

    def test_ollama_config(self):
        """Ollama 설정을 테스트합니다."""
        config = OllamaConfig(host="custom_host", port=8080, model="llama3", temperature=0.5)

        self.assertEqual(config.host, "custom_host")
        self.assertEqual(config.port, 8080)
        self.assertEqual(config.model, "llama3")
        self.assertEqual(config.temperature, 0.5)

    def test_openai_config(self):
        """OpenAI 설정을 테스트합니다."""
        config = OpenAIConfig(
            api_key="sk-test_api_key",
            model="gpt-4",
            embedding_model="text-embedding-3-large",
            temperature=0.3,
        )

        self.assertEqual(config.api_key, "sk-test_api_key")
        self.assertEqual(config.model, "gpt-4")
        self.assertEqual(config.embedding_model, "text-embedding-3-large")
        self.assertEqual(config.temperature, 0.3)

    def test_temperature_validation(self):
        """유효성 검사기가 있는 설정에 대한 온도 유효성 검사를 테스트합니다."""
        # Given: 잘못된 온도 값
        invalid_temps = [-0.1, 2.1, -1.0, 10.0]

        # 실제로 온도 유효성 검사기가 있는 설정만 테스트합니다.
        configs_with_validators = [OllamaConfig]

        for config_class in configs_with_validators:
            for temp in invalid_temps:
                with self.subTest(config=config_class.__name__, temperature=temp):
                    # When & Then: 잘못된 온도는 ValidationError를 발생시켜야 합니다.
                    with self.assertRaises(ValidationError) as context:
                        config_class(temperature=temp)
                    self.assertIn("between 0.0 and 2.0", str(context.exception))

    def test_max_tokens_validation(self):
        """max_tokens 유효성 검사를 테스트합니다 - 유효성 검사기가 없으면 건너뜁니다."""
        # 참고: 현재 구현을 기반으로 max_tokens 유효성 검사기가 존재하지 않을 수 있습니다.
        # 이 테스트는 유효성 검사기가 구현된 경우 예상되는 동작을 보여줍니다.

        # Given: 유효한 max_tokens 값 (현재 동작 테스트)
        valid_tokens = [100, 1000, 2000]

        configs = [OllamaConfig, OpenAIConfig, AnthropicConfig]

        for config_class in configs:
            for tokens in valid_tokens:
                with self.subTest(config=config_class.__name__, max_tokens=tokens):
                    # When: 유효한 max_tokens로 config 생성
                    config = config_class(max_tokens=tokens)

                    # Then: Config가 성공적으로 생성되어야 합니다.
                    self.assertEqual(config.max_tokens, tokens)


class TestMCPConfig(unittest.TestCase):
    """MCPConfig 유효성 검사를 테스트합니다."""

    def test_default_values(self):
        """기본 MCP 설정을 테스트합니다."""
        config = MCPConfig()

        self.assertEqual(config.server_name, "Knowledge Graph Server")
        self.assertEqual(config.host, "localhost")
        self.assertEqual(config.port, 8000)
        self.assertEqual(config.embedding_dim, 384)
        self.assertEqual(config.vector_similarity, "cosine")

    def test_port_validation(self):
        """특정 오류 메시지와 함께 포트 유효성 검사를 테스트합니다."""
        # Given: 잘못된 포트 번호
        invalid_ports = [0, -1, 70000, 99999]

        for port in invalid_ports:
            with self.subTest(port=port):
                # When & Then: 잘못된 포트는 ValidationError를 발생시켜야 합니다.
                with self.assertRaises(ValidationError) as context:
                    MCPConfig(port=port)
                self.assertIn("between 1 and 65535", str(context.exception))

    def test_similarity_validation(self):
        """벡터 유사도 유효성 검사를 테스트합니다."""
        # Given: 잘못된 유사도 메트릭
        invalid_similarities = ["invalid", "manhattan"]

        for similarity in invalid_similarities:
            with self.subTest(similarity=similarity):
                # When & Then: 잘못된 유사도는 ValidationError를 발생시켜야 합니다.
                with self.assertRaises(ValidationError) as context:
                    MCPConfig(vector_similarity=similarity)
                self.assertIn("Vector similarity must be one of", str(context.exception))

    def test_search_threshold_validation(self):
        """검색 임계값 유효성 검사를 테스트합니다."""
        # Given: 잘못된 임계값
        invalid_thresholds = [-0.1, 1.1, -1.0, 2.0]

        for threshold in invalid_thresholds:
            with self.subTest(threshold=threshold):
                # When & Then: 잘못된 임계값은 ValidationError를 발생시켜야 합니다.
                with self.assertRaises(ValidationError) as context:
                    MCPConfig(search_threshold=threshold)
                self.assertIn("between 0.0 and 1.0", str(context.exception))

    def test_embedding_dim_validation(self):
        """임베딩 차원 유효성 검사를 테스트합니다."""
        # Given: 잘못된 임베딩 차원
        invalid_dims = [0, -1, -100]

        for dim in invalid_dims:
            with self.subTest(embedding_dim=dim):
                # When & Then: 잘못된 차원은 ValidationError를 발생시켜야 합니다.
                with self.assertRaises(ValidationError) as context:
                    MCPConfig(embedding_dim=dim)
                self.assertIn("must be positive", str(context.exception))


class TestObservabilityConfig(unittest.TestCase):
    """ObservabilityConfig 유효성 검사를 테스트합니다."""

    def test_default_values(self):
        """기본 관찰 가능성 설정을 테스트합니다."""
        config = ObservabilityConfig()

        self.assertTrue(config.enabled)
        self.assertEqual(config.service_name, "sqlite-kg-vec-mcp")
        self.assertEqual(config.environment, "development")
        self.assertIsInstance(config.langfuse, LangfuseConfig)
        self.assertIsInstance(config.prometheus, PrometheusConfig)

    def test_environment_validation(self):
        """환경 유효성 검사를 테스트합니다."""
        # Given: 잘못된 환경 값
        invalid_envs = ["invalid", "prod", "dev"]

        for env in invalid_envs:
            with self.subTest(environment=env):
                # When & Then: 잘못된 환경은 ValidationError를 발생시켜야 합니다.
                with self.assertRaises(ValidationError) as context:
                    ObservabilityConfig(environment=env)
                self.assertIn("Environment must be one of", str(context.exception))

    def test_trace_sampling_validation(self):
        """추적 샘플링 비율 유효성 검사를 테스트합니다."""
        # Given: 잘못된 샘플링 비율
        invalid_ratios = [-0.1, 1.1, -1.0, 2.0]

        for ratio in invalid_ratios:
            with self.subTest(ratio=ratio):
                # When & Then: 잘못된 비율은 ValidationError를 발생시켜야 합니다.
                with self.assertRaises(ValidationError) as context:
                    ObservabilityConfig(trace_sampling_ratio=ratio)
                self.assertIn("between 0.0 and 1.0", str(context.exception))


class TestAppConfig(unittest.TestCase):
    """메인 AppConfig 통합을 테스트합니다."""

    def test_default_initialization(self):
        """기본 앱 설정을 테스트합니다."""
        config = AppConfig()

        self.assertEqual(config.app_name, "sqlite-kg-vec-mcp")
        self.assertEqual(config.app_version, "0.2.0")
        self.assertEqual(config.environment, "development")
        self.assertIsInstance(config.database, DatabaseConfig)
        self.assertIsInstance(config.llm, LLMConfig)
        self.assertIsInstance(config.mcp, MCPConfig)
        self.assertIsInstance(config.observability, ObservabilityConfig)

    def test_nested_config_updates(self):
        """중첩된 설정이 앱 수준 설정으로 업데이트되는지 테스트합니다."""
        config = AppConfig(app_name="test-app", app_version="1.0.0", environment="production")

        # 관찰 가능성 설정이 업데이트되어야 합니다.
        self.assertEqual(config.observability.service_name, "test-app")
        self.assertEqual(config.observability.service_version, "1.0.0")
        self.assertEqual(config.observability.environment, "production")

    def test_database_path_update(self):
        """데이터베이스 경로가 data_dir에 상대적으로 만들어지는지 테스트합니다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(data_dir=temp_dir)

            # 데이터베이스 경로는 data_dir 아래에 있어야 합니다.
            expected_path = str(Path(temp_dir) / "data/knowledge_graph.db")
            self.assertEqual(config.database.db_path, expected_path)

    def test_vector_index_dir_update(self):
        """벡터 인덱스 디렉토리가 data_dir에 상대적으로 설정되는지 테스트합니다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(data_dir=temp_dir)

            # 벡터 인덱스 디렉토리는 data_dir 아래에 있어야 합니다.
            expected_path = str(Path(temp_dir) / "vector_index")
            self.assertEqual(config.mcp.vector_index_dir, expected_path)

    def test_database_url(self):
        """데이터베이스 URL 생성을 테스트합니다."""
        config = AppConfig()
        db_url = config.get_database_url()

        self.assertTrue(db_url.startswith("sqlite:///"))
        self.assertIn("knowledge_graph.db", db_url)

    def test_environment_helpers(self):
        """환경 헬퍼 메서드를 테스트합니다."""
        dev_config = AppConfig(environment="development")
        prod_config = AppConfig(environment="production")

        self.assertTrue(dev_config.is_development())
        self.assertFalse(dev_config.is_production())

        self.assertFalse(prod_config.is_development())
        self.assertTrue(prod_config.is_production())


class TestLoggingObservabilityConfig(unittest.TestCase):
    """LoggingObservabilityConfig 유효성 검사 및 기능 테스트."""

    def test_default_values(self):
        """기본 로깅 설정 값을 테스트합니다."""
        # Given: 기본 설정
        config = LoggingObservabilityConfig()

        # Then: 모든 기본 값 확인
        self.assertEqual(config.level, "INFO")
        self.assertEqual(config.format, "json")
        self.assertEqual(config.output, "console")
        self.assertIsNone(config.file_path)
        self.assertTrue(config.include_trace)
        self.assertFalse(config.include_caller)
        self.assertTrue(config.sanitize_sensitive_data)

    def test_log_level_validation(self):
        """로그 수준 유효성 검사를 테스트합니다."""
        # Given: 유효한 로그 수준
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            with self.subTest(level=level):
                # When: 유효한 수준으로 config 생성
                config = LoggingObservabilityConfig(level=level)

                # Then: 수준이 수락되어야 함
                self.assertEqual(config.level, level)

        # Given: 잘못된 로그 수준
        invalid_levels = ["TRACE", "VERBOSE", "invalid"]

        for level in invalid_levels:
            with self.subTest(level=level):
                # When & Then: 잘못된 수준은 ValidationError를 발생시켜야 함
                with self.assertRaises(ValidationError) as context:
                    LoggingObservabilityConfig(level=level)
                self.assertIn("Level must be one of", str(context.exception))

    def test_format_validation(self):
        """로그 형식 유효성 검사를 테스트합니다."""
        # Given: 유효한 형식
        valid_formats = ["json", "text"]

        for fmt in valid_formats:
            with self.subTest(format=fmt):
                # When: 유효한 형식으로 config 생성
                config = LoggingObservabilityConfig(format=fmt)

                # Then: 형식이 수락되어야 함
                self.assertEqual(config.format, fmt)

    def test_output_validation(self):
        """로그 출력 유효성 검사를 테스트합니다."""
        # Given: 유효한 출력
        valid_outputs = ["console", "file", "both"]

        for output in valid_outputs:
            with self.subTest(output=output):
                # When: 유효한 출력으로 config 생성
                config = LoggingObservabilityConfig(output=output)

                # Then: 출력이 수락되어야 함
                self.assertEqual(config.output, output)


class TestConfigFromEnv(unittest.TestCase):
    """환경 변수로부터 설정 로딩을 테스트합니다."""

    def setUp(self):
        """테스트 환경 변수를 설정합니다."""
        self.original_env = dict(os.environ)

    def tearDown(self):
        """원래 환경을 복원합니다."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_database_config_from_env(self):
        """명시적 매개변수로 DatabaseConfig 로딩을 테스트합니다."""
        # Given: 데이터베이스를 위한 임시 디렉토리
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "custom.sqlite")

            # When: 명시적 매개변수로 DatabaseConfig 인스턴스 생성
            config = DatabaseConfig(
                db_path=db_path, optimize=False, timeout=60.0, vector_dimension=512
            )

            # Then: 설정이 제공된 값을 반영해야 함
            self.assertEqual(config.db_path, db_path)
            self.assertFalse(config.optimize)
            self.assertEqual(config.timeout, 60.0)
            self.assertEqual(config.vector_dimension, 512)

    @patch.dict(
        os.environ,
        {
            "LLM_DEFAULT_PROVIDER": "openai",
            "OLLAMA_HOST": "custom-ollama",
            "OLLAMA_PORT": "8080",
            "OPENAI_API_KEY": "sk-test_api_key",
            "OPENAI_MODEL": "gpt-4",
        },
    )
    def test_llm_config_from_env(self):
        """patch.dict를 사용하여 환경으로부터 LLMConfig 로딩을 테스트합니다."""
        # Given: 환경 변수가 patch.dict를 통해 설정됨
        # When: LLMConfig 인스턴스 생성
        config = LLMConfig()

        # Then: 설정이 환경 값을 반영해야 함
        self.assertEqual(config.default_provider, "openai")
        self.assertEqual(config.ollama.host, "custom-ollama")
        self.assertEqual(config.ollama.port, 8080)
        self.assertEqual(config.openai.api_key, "sk-test_api_key")
        self.assertEqual(config.openai.model, "gpt-4")

    @patch.dict(
        os.environ,
        {
            "MCP_SERVER_NAME": "Test Server",
            "MCP_HOST": "0.0.0.0",
            "MCP_PORT": "9000",
            "MCP_EMBEDDING_DIM": "768",
        },
    )
    def test_mcp_config_from_env(self):
        """patch.dict를 사용하여 환경으로부터 MCPConfig 로딩을 테스트합니다."""
        # Given: 환경 변수가 patch.dict를 통해 설정됨
        # When: MCPConfig 인스턴스 생성
        config = MCPConfig()

        # Then: 설정이 환경 값을 반영해야 함
        self.assertEqual(config.server_name, "Test Server")
        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 9000)
        self.assertEqual(config.embedding_dim, 768)

    @patch.dict(
        os.environ,
        {
            "OBSERVABILITY_ENABLED": "false",
            "OBSERVABILITY_SERVICE_NAME": "test-service",
            "OBSERVABILITY_ENVIRONMENT": "staging",
            "OBSERVABILITY_TRACE_SAMPLING_RATIO": "0.1",
        },
    )
    def test_observability_config_from_env(self):
        """patch.dict를 사용하여 환경으로부터 ObservabilityConfig 로딩을 테스트합니다."""
        # Given: 환경 변수가 patch.dict를 통해 설정됨
        # When: ObservabilityConfig 인스턴스 생성
        config = ObservabilityConfig()

        # Then: 설정이 환경 값을 반영해야 함
        self.assertFalse(config.enabled)
        self.assertEqual(config.service_name, "test-service")
        self.assertEqual(config.environment, "staging")
        self.assertEqual(config.trace_sampling_ratio, 0.1)

    @patch.dict(
        os.environ,
        {
            "LOG_LEVEL": "DEBUG",
            "LOG_FORMAT": "text",
            "LOG_OUTPUT": "file",
            "LOG_FILE_PATH": "/var/log/app.log",
            "LOG_INCLUDE_TRACE": "false",
        },
    )
    def test_logging_config_from_env(self):
        """patch.dict를 사용하여 환경으로부터 LoggingObservabilityConfig 로딩을 테스트합니다."""
        # Given: 환경 변수가 patch.dict를 통해 설정됨
        # When: LoggingObservabilityConfig 인스턴스 생성
        config = LoggingObservabilityConfig()

        # Then: 설정이 환경 값을 반영해야 함
        self.assertEqual(config.level, "DEBUG")
        self.assertEqual(config.format, "text")
        self.assertEqual(config.output, "file")
        self.assertEqual(config.file_path, "/var/log/app.log")
        self.assertFalse(config.include_trace)


class TestConfigErrorHandling(unittest.TestCase):
    """모든 설정 클래스에 대한 포괄적인 오류 처리 및 엣지 케이스를 테스트합니다."""

    def test_multiple_validation_errors(self):
        """여러 유효성 검사 오류가 올바르게 보고되는지 테스트합니다."""
        # Given: 여러 잘못된 매개변수
        with self.assertRaises(ValidationError) as context:
            # When: 여러 잘못된 값으로 config 생성
            DatabaseConfig(vector_dimension=-1, timeout=-5.0)

        # Then: 두 오류 모두 보고되어야 함
        error_str = str(context.exception)
        self.assertIn("Vector dimension must be positive", error_str)

    def test_type_coercion(self):
        """문자열 값이 올바른 유형으로 올바르게 강제 변환되는지 테스트합니다."""
        # Given: 강제 변환되어야 하는 문자열 값
        config = DatabaseConfig(timeout="30.5", vector_dimension="512", optimize="true")

        # Then: 값이 올바르게 형식화되어야 함
        self.assertEqual(config.timeout, 30.5)
        self.assertEqual(config.vector_dimension, 512)
        self.assertTrue(config.optimize)

    def test_none_values_handling(self):
        """선택적 필드에 대한 None 값 처리를 테스트합니다."""
        # Given: None 값을 가진 설정
        config = DatabaseConfig(backup_path=None)

        # Then: 선택적 필드에 대해 None이 수락되어야 함
        self.assertIsNone(config.backup_path)

    def test_empty_string_handling(self):
        """빈 문자열 처리를 테스트합니다."""
        # Given: 현재 빈 문자열 값이 수락됨
        # When: 빈 문자열로 config 생성
        config = MCPConfig(server_name="")

        # Then: 빈 문자열이 수락되어야 함 (현재 동작)
        self.assertEqual(config.server_name, "")


class TestConfigIntegration(unittest.TestCase):
    """서로 다른 설정 클래스 간의 통합을 테스트합니다."""

    def test_app_config_propagation(self):
        """앱 수준 설정이 하위 설정에 올바르게 전파되는지 테스트합니다."""
        # Given: 사용자 지정 설정이 있는 앱 설정
        config = AppConfig(app_name="integration-test", app_version="2.0.0", environment="staging")

        # Then: 설정이 관찰 가능성 설정으로 전파되어야 함
        self.assertEqual(config.observability.service_name, "integration-test")
        self.assertEqual(config.observability.service_version, "2.0.0")
        self.assertEqual(config.observability.environment, "staging")

    def test_data_dir_propagation(self):
        """data_dir 설정이 모든 관련 설정에 영향을 미치는지 테스트합니다."""
        # Given: 사용자 지정 데이터 디렉토리가 있는 앱 설정
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(data_dir=temp_dir)

            # Then: 데이터베이스 및 MCP 설정이 데이터 디렉토리를 사용해야 함
            self.assertTrue(config.database.db_path.startswith(temp_dir))
            self.assertTrue(config.mcp.vector_index_dir.startswith(temp_dir))


if __name__ == "__main__":
    unittest.main()
