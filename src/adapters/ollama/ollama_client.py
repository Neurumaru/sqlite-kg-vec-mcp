"""
Ollama client for LLM integration with SQLite KG Vec MCP.
"""

import json
import time
from dataclasses import dataclass
from typing import Any, cast

import requests

from src.common.config.llm import OllamaConfig
from src.common.observability import get_observable_logger, with_observability

from .exceptions import (
    OllamaConnectionException,
    OllamaGenerationException,
    OllamaResponseException,
    OllamaTimeoutException,
)

# Langfuse integration removed - using fallback prompts


@dataclass
class LLMResponse:
    """Response from LLM."""

    text: str
    model: str
    tokens_used: int
    response_time: float
    metadata: dict[str, Any] | None = None


class OllamaClient:
    """Client for interacting with Ollama LLM models."""

    def __init__(
        self,
        config: OllamaConfig | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ):
        """
        Initialize Ollama client.

        Args:
            config: Ollama configuration object
            base_url: Ollama server URL (deprecated, use config instead)
            model: Model name to use (deprecated, use config instead)
            timeout: Request timeout in seconds (deprecated, use config instead)
        """
        if config is None:
            config = OllamaConfig()

        # Override config with individual parameters if provided (for backward compatibility)
        self.base_url = (base_url or f"http://{config.host}:{config.port}").rstrip("/")
        self.model = model or config.model
        self.timeout = timeout or int(config.timeout)
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.session = requests.Session()
        self.logger = get_observable_logger("ollama_client", "adapter")

        # Test connection
        self._test_connection()

    def _test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return True
        except requests.ConnectionError as exception:
            self.logger.warning(
                "ollama_connection_failed",
                base_url=self.base_url,
                error_type="connection_error",
                error_message=str(exception),
            )
            return False
        except requests.Timeout as exception:
            self.logger.warning(
                "ollama_connection_timeout",
                base_url=self.base_url,
                timeout_duration=5,
                error_message=str(exception),
            )
            return False
        except requests.HTTPError as exception:
            status_code = getattr(exception.response, "status_code", None)
            self.logger.warning(
                "ollama_http_error",
                base_url=self.base_url,
                status_code=status_code,
                error_message=str(exception),
            )
            return False
        except (requests.RequestException, ValueError, KeyError) as exception:
            self.logger.warning(
                "ollama_unexpected_error",
                base_url=self.base_url,
                error_type=type(exception).__name__,
                error_message=str(exception),
            )
            return False

    @with_observability(operation="llm_generate", include_args=True, include_result=True)
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        """
        Generate text using Ollama model.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response

        Returns:
            LLMResponse with generated text and metadata
        """
        start_time = time.time()

        # Prepare request data
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens

        data: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": options,
        }

        if system_prompt:
            data["system"] = system_prompt

        # Initialize response_text to handle potential exceptions
        response_text = ""

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate", json=data, timeout=self.timeout
            )
            response.raise_for_status()
            response_text = response.text

            # Parse response
            generated_text = ""
            total_tokens = 0

            if stream:
                # Handle streaming response
                text_chunks = []
                for line in response.iter_lines():
                    if line:
                        chunk_data = json.loads(line.decode("utf-8"))
                        if "response" in chunk_data:
                            text_chunks.append(chunk_data["response"])

                        if chunk_data.get("done", False):
                            total_tokens = chunk_data.get("eval_count", 0)
                            break

                generated_text = "".join(text_chunks)
            else:
                # Handle non-streaming response
                result = response.json()
                generated_text = result.get("response", "")
                total_tokens = result.get("eval_count", 0)

            response_time = time.time() - start_time

            return LLMResponse(
                text=generated_text,
                model=self.model,
                tokens_used=total_tokens,
                response_time=response_time,
                metadata={"temperature": temperature},
            )

        except requests.ConnectionError as exception:
            raise OllamaConnectionException.from_requests_error(
                self.base_url, exception
            ) from exception
        except requests.Timeout as exception:
            raise OllamaTimeoutException(
                base_url=self.base_url,
                operation="text generation",
                timeout_duration=self.timeout,
                original_error=exception,
            ) from exception
        except requests.HTTPError as exception:
            status_code = getattr(exception.response, "status_code", None)
            raise OllamaConnectionException(
                base_url=self.base_url,
                message=f"HTTP {status_code} error during generation",
                status_code=status_code,
                original_error=exception,
            ) from exception
        except json.JSONDecodeError as exception:
            raise OllamaResponseException(
                response_text=response_text, parsing_error=str(exception), original_error=exception
            ) from exception
        except (ValueError, KeyError, TypeError) as exception:
            raise OllamaGenerationException(
                model_name=self.model,
                prompt=prompt,
                message=f"Data processing error during generation: {exception}",
                generation_params={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                original_error=exception,
            ) from exception

    def extract_entities_and_relationships(self, text: str) -> dict[str, Any]:
        """
        Extract entities and relationships from text using LLM.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary containing extracted entities and relationships
        """
        # Use default prompt (Langfuse removed)
        system_prompt = """You are an expert knowledge graph extraction system.
            Analyze the given text and extract entities and relationships in JSON format.
            Return a JSON object with this structure:
            {
                "entities": [
                    {
                        "id": "unique_id",
                        "name": "entity_name",
                        "type": "entity_type",
                        "properties": {"key": "value"}
                    }
                ],
                "relationships": [
                    {
                        "source": "source_entity_id",
                        "target": "target_entity_id",
                        "type": "relationship_type",
                        "properties": {"key": "value"}
                    }
                ]
            }
            Focus on extracting:
            - People, organizations, places, concepts, events
            - Clear relationships between entities
            - Important properties and attributes
            Be precise and only extract information explicitly mentioned in the text."""

        prompt = f"""Extract entities and relationships from this text:

{text}

Return only valid JSON, no additional text or explanation."""

        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,  # Lower temperature for more consistent extraction
            max_tokens=2000,
        )

        try:
            # Clean the response text (remove markdown code blocks if present)
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith("```"):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove closing ```
            response_text = response_text.strip()

            # Parse the JSON response
            result = cast(dict[str, Any], json.loads(response_text))

            # Validate structure
            if "entities" not in result:
                result["entities"] = []
            if "relationships" not in result:
                result["relationships"] = []

            # Prompt usage logging removed (Langfuse integration removed)

            return result

        except json.JSONDecodeError as exception:
            self.logger.error(
                "llm_response_parse_failed",
                error_message=str(exception),
                response_text=response.text,
            )

            # Return empty structure on parse failure
            return {"entities": [], "relationships": []}

    def generate_embeddings_description(self, entity: dict[str, Any]) -> str:
        """
        Generate a rich description for an entity to improve embeddings.

        Args:
            entity: Entity dictionary with name, type, and properties

        Returns:
            Enhanced description string
        """
        system_prompt = """You are an expert at creating rich, informative descriptions for knowledge graph entities.
        Given an entity with its name, type, and properties, create a comprehensive but concise description
        that would be ideal for vector embeddings and semantic search.
        The description should:
        - Include the entity's main characteristics
        - Mention key relationships and context
        - Be 1-3 sentences long
        - Be informative for search and matching
        Return only the description, no additional text."""

        entity_info = f"""Entity Name: {entity.get('name', 'Unknown')}
Entity Type: {entity.get('type', 'Unknown')}
Properties: {json.dumps(entity.get('properties', {}), indent=2)}"""

        prompt = f"Create a rich description for this entity:\n\n{entity_info}"

        response = self.generate(
            prompt=prompt, system_prompt=system_prompt, temperature=0.3, max_tokens=150
        )

        return str(response.text).strip()

    def list_available_models(self) -> list[str]:
        """Get list of available models from Ollama."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()

            data = response.json()
            return [model["name"] for model in data.get("models", [])]

        except (requests.RequestException, ValueError, KeyError) as exception:
            self.logger.error("model_list_fetch_failed", error_message=str(exception))
            return []

    def pull_model(self, model_name: str) -> bool:
        """
        Pull/download a model from Ollama.

        Args:
            model_name: Name of the model to pull

        Returns:
            True if successful, False otherwise
        """
        try:
            data = {"name": model_name}
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json=data,
                timeout=300,  # 5 minutes for model download
            )
            response.raise_for_status()

            # Check if pull was successful
            for line in response.iter_lines():
                if line:
                    chunk_data = json.loads(line.decode("utf-8"))
                    if chunk_data.get("status") == "success":
                        return True

            return False

        except (requests.RequestException, ValueError, json.JSONDecodeError) as exception:
            self.logger.error(
                "model_pull_failed", model_name=model_name, error_message=str(exception)
            )
            return False
