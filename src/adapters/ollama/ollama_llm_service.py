"""
Ollama-based implementation of LLMService interface.
"""

import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage

from src.common.config.llm import OllamaConfig
from src.ports.llm import LLM

from .ollama_client import OllamaClient

# Temporary type alias to avoid circular dependency
SearchResult = Any


class OllamaLLMService(LLM):
    """
    Ollama-based implementation of the LLM port.

    This adapter provides LLM capabilities using Ollama models
    for search guidance, knowledge extraction, and analysis.
    """

    def __init__(
        self,
        ollama_client: OllamaClient | None = None,
        config: OllamaConfig | None = None,
        default_temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """
        Initialize Ollama LLM service.

        Args:
            ollama_client: Configured Ollama client (deprecated, use config instead)
            config: Ollama configuration object
            default_temperature: Default sampling temperature (deprecated, use config instead)
            max_tokens: Default maximum tokens for responses (deprecated, use config instead)
        """
        if config is None:
            config = OllamaConfig()

        if ollama_client is None:
            ollama_client = OllamaClient(config=config)

        self.ollama_client = ollama_client
        self.default_temperature = default_temperature or config.temperature
        self.max_tokens = max_tokens or config.max_tokens

    # LangChain compatible methods

    async def invoke(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> BaseMessage:
        """
        Generate response based on messages (LangChain invoke style).

        Args:
            input: LangChain BaseMessage list
            **kwargs: Model parameters (temperature, max_tokens, stop, etc.)

        Returns:
            AIMessage response
        """
        # Convert messages to text
        prompt = self._messages_to_text(messages)

        # Extract parameters
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Generate using Ollama client
        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return AIMessage(content=response.text)

    async def stream(  # type: ignore[override]
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Generate response in streaming mode (LangChain stream style).

        Args:
            input: LangChain BaseMessage list
            **kwargs: Model parameters

        Yields:
            Response text chunks
        """
        # Convert messages to text
        prompt = self._messages_to_text(messages)

        # Extract parameters
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Generate using Ollama client (currently simulating streaming)
        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Split response into chunks for streaming simulation
        text = response.text
        chunk_size = 50
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            yield chunk
            await asyncio.sleep(0.05)  # Simulate streaming delay

    async def batch(
        self,
        inputs: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> list[BaseMessage]:
        """
        Process multiple message sequences in batch (LangChain batch style).

        Args:
            inputs: List of message sequences
            **kwargs: Model parameters

        Returns:
            List of AIMessage responses
        """
        # Create tasks for concurrent processing
        tasks = []
        for message_list in inputs:
            task = asyncio.create_task(self.invoke(message_list, **kwargs))
            tasks.append(task)

        # Wait for all tasks to complete concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results: list[BaseMessage] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error message for failed requests
                error_msg = f"Batch item {i} failed: {str(result)}"
                processed_results.append(AIMessage(content=error_msg))
            elif isinstance(result, BaseMessage):
                processed_results.append(result)
            else:
                # Handle unexpected result types
                processed_results.append(
                    AIMessage(content=f"Unexpected result type: {type(result)}")
                )

        return processed_results

    def _messages_to_text(self, messages: list[BaseMessage]) -> str:
        """Convert BaseMessage list to text."""
        text_parts = []
        for message in messages:
            role = message.__class__.__name__.replace("Message", "").lower()
            if role == "ai":
                role = "assistant"
            elif role == "human":
                role = "user"

            text_parts.append(f"{role}: {message.content}")

        return "\n".join(text_parts)

    # Interactive search guidance

    async def analyze_query(
        self, query: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Analyze a search query to determine optimal search strategy."""
        system_prompt = """You are an expert search analyst. Analyze the given query and recommend the best search strategy.

        Classify the query intent and recommend one of these strategies:
        - SEMANTIC: For conceptual, meaning-based searches
        - STRUCTURAL: For specific entity/relationship queries
        - HYBRID: For complex queries needing both approaches
        - STOP: If the query is unclear or invalid

        Return JSON with this structure:
        {
            "strategy": "SEMANTIC|STRUCTURAL|HYBRID|STOP",
            "confidence": 0.0-1.0,
            "reasoning": "explanation",
            "suggested_filters": ["filter1", "filter2"],
            "query_type": "factual|exploratory|navigational|transactional"
        }"""

        context_str = f"Context: {json.dumps(context)}\n" if context else ""
        prompt = f"{context_str}Query to analyze: {query}"

        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=500,
        )

        try:
            result = self._parse_json_response(response.text)
            if isinstance(result, dict):
                return result
            raise ValueError("Expected dict response") from None
        except Exception as exception:
            logging.warning("Failed to parse query analysis response: %s", exception)
            return {
                "strategy": "SEMANTIC",
                "confidence": 0.5,
                "reasoning": "Fallback to semantic search due to parsing error",
                "suggested_filters": [],
                "query_type": "exploratory",
            }

    async def guide_search_navigation(
        self,
        current_results: list[SearchResult],
        original_query: str,
        search_history: list[dict[str, Any]],
        step_number: int,
    ) -> dict[str, Any]:
        """Guide the next step in interactive search navigation."""
        system_prompt = """You are a search navigation guide. Based on current results and search history, recommend the next search action.

        Return JSON with this structure:
        {
            "next_action": "refine|expand|pivot|stop",
            "strategy": "SEMANTIC|STRUCTURAL|HYBRID|STOP",
            "suggested_query": "new query or refinement",
            "reasoning": "explanation for recommendation",
            "focus_areas": ["area1", "area2"],
            "confidence": 0.0-1.0
        }"""

        # Summarize current results
        results_summary = []
        for i, result in enumerate(current_results[:5]):  # Limit to top 5
            results_summary.append(
                {
                    "rank": i + 1,
                    "score": result.score,
                    "type": (result.entity_type if hasattr(result, "entity_type") else "unknown"),
                }
            )

        prompt = f"""Original query: {original_query}
        Step number: {step_number}
        Current results summary: {json.dumps(results_summary)}
        Search history: {json.dumps(search_history[-3:])}  # Last 3 steps

        What should be the next search action?"""

        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=400,
        )

        try:
            result = self._parse_json_response(response.text)
            if isinstance(result, dict):
                return result
            raise ValueError("Expected dict response") from None
        except Exception as exception:
            logging.warning("Failed to parse navigation guidance: %s", exception)
            return {
                "next_action": "stop",
                "strategy": "STOP",
                "suggested_query": original_query,
                "reasoning": "Unable to generate guidance due to parsing error",
                "focus_areas": [],
                "confidence": 0.0,
            }

    async def evaluate_search_results(
        self, results: list[SearchResult], query: str, search_context: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate search results quality and relevance."""
        system_prompt = """You are a search quality evaluator. Assess the relevance and quality of search results.

        Return JSON with this structure:
        {
            "overall_quality": 0.0-1.0,
            "relevance_score": 0.0-1.0,
            "coverage_score": 0.0-1.0,
            "diversity_score": 0.0-1.0,
            "recommendations": ["suggestion1", "suggestion2"],
            "best_result_index": 0,
            "quality_issues": ["issue1", "issue2"]
        }"""

        # Summarize results for evaluation
        results_data = []
        for i, result in enumerate(results[:10]):  # Limit to top 10
            results_data.append(
                {
                    "index": i,
                    "score": result.score,
                    "snippet": (
                        getattr(result, "snippet", "")[:100] + "..."
                        if hasattr(result, "snippet")
                        else ""
                    ),
                }
            )

        prompt = f"""Query: {query}
        Context: {json.dumps(search_context)}
        Results to evaluate: {json.dumps(results_data)}

        Evaluate the quality and relevance of these search results."""

        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=600,
        )

        try:
            result = self._parse_json_response(response.text)
            if isinstance(result, dict):
                return result
            raise ValueError("Expected dict response") from None
        except Exception as exception:
            logging.warning("Failed to parse result evaluation: %s", exception)
            return {
                "overall_quality": 0.5,
                "relevance_score": 0.5,
                "coverage_score": 0.5,
                "diversity_score": 0.5,
                "recommendations": ["Try refining your search query"],
                "best_result_index": 0,
                "quality_issues": ["Unable to evaluate due to parsing error"],
            }

    # Knowledge extraction

    async def extract_knowledge_from_text(
        self,
        text: str,
        extraction_schema: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract structured knowledge from unstructured text."""
        # Use existing Ollama client method
        result = await asyncio.to_thread(
            self.ollama_client.extract_entities_and_relationships, text
        )

        result["extraction_metadata"] = {
            "text_length": len(text),
            "schema_used": extraction_schema is not None,
            "context_provided": context is not None,
            "model": self.ollama_client.model,
        }

        return result

    async def generate_entity_summary(
        self,
        entity_data: dict[str, Any],
        related_entities: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a summary for an entity based on its data and relationships."""
        # Use existing Ollama client method
        summary = await asyncio.to_thread(
            self.ollama_client.generate_embeddings_description, entity_data
        )

        return summary

    async def suggest_relationships(
        self,
        source_entity: dict[str, Any],
        target_entities: list[dict[str, Any]],
        context: str | None = None,
    ) -> list[dict[str, Any]]:
        """Suggest potential relationships between entities."""
        system_prompt = """You are a knowledge graph relationship expert. Suggest potential relationships between entities.

        Return JSON array with this structure for each relationship:
        {
            "target_entity": "target_name",
            "relationship_type": "relationship",
            "confidence": 0.0-1.0,
            "reasoning": "explanation",
            "properties": {"key": "value"}
        }"""

        target_names = [entity.get("name", "Unknown") for entity in target_entities[:10]]
        context_str = f"Context: {context}\n" if context else ""

        prompt = f"""{context_str}Source entity: {json.dumps(source_entity)}
        Target entities: {json.dumps(target_names)}

        Suggest meaningful relationships between the source entity and target entities."""

        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=800,
        )

        try:
            result = self._parse_json_response(response.text)
            if isinstance(result, list):
                return result
            return []
        except Exception as exception:
            logging.warning("Failed to parse relationship suggestions: %s", exception)
            return []

    # Query enhancement

    async def expand_query(
        self, original_query: str, search_context: dict[str, Any] | None = None
    ) -> list[str]:
        """Expand a query with related terms and concepts."""
        system_prompt = """You are a query expansion expert. Generate related terms and concepts for the given query.

        Return JSON array of expanded query terms:
        ["term1", "term2", "term3", ...]

        Focus on synonyms, related concepts, and contextually relevant terms."""

        context_str = f"Context: {json.dumps(search_context)}\n" if search_context else ""
        prompt = f"{context_str}Original query: {original_query}\n\nGenerate 5-10 related terms."

        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=300,
        )

        try:
            result = self._parse_json_response(response.text)
            if isinstance(result, list):
                return result
            return [original_query]
        except Exception as exception:
            logging.warning("Failed to parse query expansion: %s", exception)
            return [original_query]

    async def generate_search_suggestions(
        self, partial_query: str, search_history: list[str] | None = None
    ) -> list[str]:
        """Generate search suggestions for partial queries."""
        system_prompt = """You are a search suggestion generator. Complete and suggest variations of the partial query.

        Return JSON array of search suggestions:
        ["suggestion1", "suggestion2", "suggestion3", ...]

        Consider common search patterns and user intent."""

        history_str = (
            f"Recent searches: {json.dumps(search_history[-5:])}\n" if search_history else ""
        )
        prompt = f"{history_str}Partial query: {partial_query}\n\nGenerate 3-7 search suggestions."

        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=200,
        )

        try:
            result = self._parse_json_response(response.text)
            if isinstance(result, list):
                return result
            return [partial_query]
        except Exception as exception:
            logging.warning("Failed to parse search suggestions: %s", exception)
            return [partial_query]

    # Content analysis

    async def classify_content(
        self, content: str, classification_schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Classify content according to a given schema."""
        system_prompt = f"""You are a content classifier. Classify the given content according to the provided schema.

        Classification schema: {json.dumps(classification_schema)}

        Return JSON with classification results and confidence scores."""

        prompt = f"Content to classify: {content[:1000]}..."  # Limit content length

        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=400,
        )

        try:
            result = self._parse_json_response(response.text)
            if isinstance(result, dict):
                return result
            raise ValueError("Expected dict response") from None
        except Exception as exception:
            logging.warning("Failed to parse content classification: %s", exception)
            return {"error": "Classification failed", "confidence": 0.0}

    async def detect_language(self, text: str) -> str:
        """Detect the language of given text."""
        system_prompt = """Detect the language of the given text. Return only the ISO 639-1 language code (e.g., 'en', 'es', 'fr', 'de', 'ko', etc.)."""

        prompt = f"Text: {text[:500]}..."  # Limit text length

        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=10,
        )

        # Extract language code from response
        lang_code = str(response.text).strip().lower()
        # Validate it's a reasonable language code
        if len(lang_code) == 2 and lang_code.isalpha():
            return lang_code
        return "en"  # Default to English

    # Streaming responses

    async def stream_analysis(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream analysis results for real-time processing."""
        context_str = f"Context: {json.dumps(context)}\n" if context else ""
        full_prompt = f"{context_str}{prompt}"

        # Try to use streaming if supported, fallback to chunked simulation
        try:
            # Attempt streaming generation (if Ollama client supports it in the future)
            response = await asyncio.to_thread(
                self.ollama_client.generate,
                prompt=full_prompt,
                temperature=self.default_temperature,
                max_tokens=self.max_tokens,
                stream=True,  # Try streaming first
            )

            # If streaming is supported, this would yield real chunks
            # For now, we simulate with better chunking logic
            text = response.text

            # Dynamic chunk sizing based on content
            words = text.split()
            current_chunk = ""
            word_count = 0

            for word in words:
                current_chunk += word + " "
                word_count += 1

                # Yield chunk when it reaches optimal size or at natural breaks
                if word_count >= 5 and (
                    word.endswith(".")
                    or word.endswith("!")
                    or word.endswith("?")
                    or len(current_chunk) > 100
                ):
                    yield current_chunk.strip()
                    current_chunk = ""
                    word_count = 0
                    await asyncio.sleep(0.05)  # Natural typing delay

            # Yield remaining content
            if current_chunk.strip():
                yield current_chunk.strip()

        except Exception:
            # Fallback to simple chunking on any error
            response = await asyncio.to_thread(
                self.ollama_client.generate,
                prompt=full_prompt,
                temperature=self.default_temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )

            text = response.text
            chunk_size = 50
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                yield chunk
                await asyncio.sleep(0.1)

    # Configuration and health

    async def get_model_info(self) -> dict[str, Any]:
        """Get information about the current LLM model."""
        available_models = await asyncio.to_thread(self.ollama_client.list_available_models)

        return {
            "current_model": self.ollama_client.model,
            "base_url": self.ollama_client.base_url,
            "available_models": available_models,
            "default_temperature": self.default_temperature,
            "max_tokens": self.max_tokens,
            "provider": "ollama",
        }

    async def health_check(self) -> dict[str, Any]:
        """Check the health status of the LLM service."""
        try:
            # Test connection with a simple request
            test_response = await asyncio.to_thread(
                self.ollama_client.generate, prompt="Hello", max_tokens=10
            )

            return {
                "status": "healthy",
                "model": self.ollama_client.model,
                "base_url": self.ollama_client.base_url,
                "response_time": test_response.response_time,
                "last_check": "now",
            }
        except Exception as exception:
            return {
                "status": "unhealthy",
                "error": str(exception),
                "model": self.ollama_client.model,
                "base_url": self.ollama_client.base_url,
                "last_check": "now",
            }

    async def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics for the LLM service."""
        # Note: This would require tracking usage in a real implementation
        return {
            "total_requests": 0,
            "total_tokens": 0,
            "average_response_time": 0.0,
            "error_rate": 0.0,
            "model": self.ollama_client.model,
            "note": "Usage tracking not implemented yet",
        }

    # Helper methods

    def _parse_json_response(self, response_text: str) -> dict[str, Any] | list[Any]:
        """Parse JSON response from LLM, handling common formatting issues."""
        response_text = response_text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]

        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

        try:
            parsed = json.loads(response_text)
            return cast(dict[str, Any] | list[Any], parsed)
        except json.JSONDecodeError as exception:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return cast(dict[str, Any] | list[Any], parsed)
                except json.JSONDecodeError as inner_exception:
                    raise ValueError(
                        f"Could not parse extracted JSON from response: {response_text}"
                    ) from inner_exception
            raise ValueError(f"No valid JSON found in response: {response_text}") from exception
