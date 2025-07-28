"""
Tests for configuration management system.
"""

import os
import tempfile
import unittest
from pathlib import Path

from src.common.config import (
    AppConfig,
    DatabaseConfig,
    LLMConfig,
    MCPConfig,
    ObservabilityConfig,
)
from src.common.config.llm import AnthropicConfig, OllamaConfig, OpenAIConfig
from src.common.config.observability import LangfuseConfig, PrometheusConfig


class TestDatabaseConfig(unittest.TestCase):
    """Test DatabaseConfig validation and functionality."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DatabaseConfig()
        
        self.assertEqual(config.db_path, "data/knowledge_graph.db")
        self.assertTrue(config.optimize)
        self.assertEqual(config.timeout, 30.0)
        self.assertEqual(config.vector_dimension, 384)
    
    def test_path_validation(self):
        """Test database path validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "test.db")
            config = DatabaseConfig(db_path=db_path)
            
            # Path should be normalized
            self.assertEqual(config.db_path, db_path)
            # Parent directory should be created
            self.assertTrue(Path(config.db_path).parent.exists())
    
    def test_vector_dimension_validation(self):
        """Test vector dimension validation."""
        with self.assertRaises(ValueError):
            DatabaseConfig(vector_dimension=0)
        
        with self.assertRaises(ValueError):
            DatabaseConfig(vector_dimension=-1)


class TestLLMConfig(unittest.TestCase):
    """Test LLMConfig and provider configs."""
    
    def test_default_provider(self):
        """Test default provider configuration."""
        config = LLMConfig()
        
        self.assertEqual(config.default_provider, "ollama")
        self.assertIsInstance(config.ollama, OllamaConfig)
        self.assertIsInstance(config.openai, OpenAIConfig)
        self.assertIsInstance(config.anthropic, AnthropicConfig)
    
    def test_provider_validation(self):
        """Test provider validation."""
        with self.assertRaises(ValueError):
            LLMConfig(default_provider="invalid_provider")
    
    def test_ollama_config(self):
        """Test Ollama configuration."""
        config = OllamaConfig(
            host="custom_host",
            port=8080,
            model="llama3",
            temperature=0.5
        )
        
        self.assertEqual(config.host, "custom_host")
        self.assertEqual(config.port, 8080)
        self.assertEqual(config.model, "llama3")
        self.assertEqual(config.temperature, 0.5)
    
    def test_openai_config(self):
        """Test OpenAI configuration."""
        config = OpenAIConfig(
            api_key="test_key",
            model="gpt-4",
            embedding_model="text-embedding-3-large",
            temperature=0.3
        )
        
        self.assertEqual(config.api_key, "test_key")
        self.assertEqual(config.model, "gpt-4")
        self.assertEqual(config.embedding_model, "text-embedding-3-large")
        self.assertEqual(config.temperature, 0.3)


class TestMCPConfig(unittest.TestCase):
    """Test MCPConfig validation."""
    
    def test_default_values(self):
        """Test default MCP configuration."""
        config = MCPConfig()
        
        self.assertEqual(config.server_name, "Knowledge Graph Server")
        self.assertEqual(config.host, "localhost")
        self.assertEqual(config.port, 8000)
        self.assertEqual(config.embedding_dim, 384)
        self.assertEqual(config.vector_similarity, "cosine")
    
    def test_port_validation(self):
        """Test port validation."""
        with self.assertRaises(ValueError):
            MCPConfig(port=0)
        
        with self.assertRaises(ValueError):
            MCPConfig(port=70000)
    
    def test_similarity_validation(self):
        """Test vector similarity validation."""
        with self.assertRaises(ValueError):
            MCPConfig(vector_similarity="invalid")
    
    def test_search_threshold_validation(self):
        """Test search threshold validation."""
        with self.assertRaises(ValueError):
            MCPConfig(search_threshold=-0.1)
        
        with self.assertRaises(ValueError):
            MCPConfig(search_threshold=1.1)


class TestObservabilityConfig(unittest.TestCase):
    """Test ObservabilityConfig validation."""
    
    def test_default_values(self):
        """Test default observability configuration."""
        config = ObservabilityConfig()
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.service_name, "sqlite-kg-vec-mcp")
        self.assertEqual(config.environment, "development")
        self.assertIsInstance(config.langfuse, LangfuseConfig)
        self.assertIsInstance(config.prometheus, PrometheusConfig)
    
    def test_environment_validation(self):
        """Test environment validation."""
        with self.assertRaises(ValueError):
            ObservabilityConfig(environment="invalid")
    
    def test_trace_sampling_validation(self):
        """Test trace sampling ratio validation."""
        with self.assertRaises(ValueError):
            ObservabilityConfig(trace_sampling_ratio=-0.1)
        
        with self.assertRaises(ValueError):
            ObservabilityConfig(trace_sampling_ratio=1.1)


class TestAppConfig(unittest.TestCase):
    """Test main AppConfig integration."""
    
    def test_default_initialization(self):
        """Test default app configuration."""
        config = AppConfig()
        
        self.assertEqual(config.app_name, "sqlite-kg-vec-mcp")
        self.assertEqual(config.app_version, "0.2.0")
        self.assertEqual(config.environment, "development")
        self.assertIsInstance(config.database, DatabaseConfig)
        self.assertIsInstance(config.llm, LLMConfig)
        self.assertIsInstance(config.mcp, MCPConfig)
        self.assertIsInstance(config.observability, ObservabilityConfig)
    
    def test_nested_config_updates(self):
        """Test that nested configs are updated with app-level settings."""
        config = AppConfig(
            app_name="test-app",
            app_version="1.0.0",
            environment="production"
        )
        
        # Observability config should be updated
        self.assertEqual(config.observability.service_name, "test-app")
        self.assertEqual(config.observability.service_version, "1.0.0")
        self.assertEqual(config.observability.environment, "production")
    
    def test_database_path_update(self):
        """Test database path is made relative to data_dir."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(data_dir=temp_dir)
            
            # Database path should be under data_dir
            expected_path = str(Path(temp_dir) / "data/knowledge_graph.db")
            self.assertEqual(config.database.db_path, expected_path)
    
    def test_vector_index_dir_update(self):
        """Test vector index directory is set relative to data_dir."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(data_dir=temp_dir)
            
            # Vector index dir should be under data_dir
            expected_path = str(Path(temp_dir) / "vector_index")
            self.assertEqual(config.mcp.vector_index_dir, expected_path)
    
    def test_database_url(self):
        """Test database URL generation."""
        config = AppConfig()
        db_url = config.get_database_url()
        
        self.assertTrue(db_url.startswith("sqlite:///"))
        self.assertIn("knowledge_graph.db", db_url)
    
    def test_environment_helpers(self):
        """Test environment helper methods."""
        dev_config = AppConfig(environment="development")
        prod_config = AppConfig(environment="production")
        
        self.assertTrue(dev_config.is_development())
        self.assertFalse(dev_config.is_production())
        
        self.assertFalse(prod_config.is_development())
        self.assertTrue(prod_config.is_production())


class TestConfigFromEnv(unittest.TestCase):
    """Test configuration loading from environment variables."""
    
    def setUp(self):
        """Set up test environment variables."""
        self.original_env = dict(os.environ)
    
    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_database_config_from_env(self):
        """Test DatabaseConfig loading from environment."""
        os.environ.update({
            "DB_PATH": "/custom/path/db.sqlite",
            "DB_OPTIMIZE": "false",
            "DB_TIMEOUT": "60.0",
            "DB_VECTOR_DIMENSION": "512"
        })
        
        config = DatabaseConfig()
        
        self.assertEqual(config.db_path, "/custom/path/db.sqlite")
        self.assertFalse(config.optimize)
        self.assertEqual(config.timeout, 60.0)
        self.assertEqual(config.vector_dimension, 512)
    
    def test_llm_config_from_env(self):
        """Test LLMConfig loading from environment."""
        os.environ.update({
            "LLM_DEFAULT_PROVIDER": "openai",
            "OLLAMA_HOST": "custom-ollama",
            "OLLAMA_PORT": "8080",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "gpt-4"
        })
        
        config = LLMConfig()
        
        self.assertEqual(config.default_provider, "openai")
        self.assertEqual(config.ollama.host, "custom-ollama")
        self.assertEqual(config.ollama.port, 8080)
        self.assertEqual(config.openai.api_key, "test-key")
        self.assertEqual(config.openai.model, "gpt-4")


if __name__ == "__main__":
    unittest.main()