"""
Ollama-based implementation of LLMService interface.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.domain.services.knowledge_search import SearchStrategy
from src.ports.llm_service import LLMService

# Temporary type alias to avoid circular dependency
SearchResult = Any
from .ollama_client import OllamaClient


class OllamaLLMService(LLMService):
    """
    Ollama-based implementation of the LLMService port.

    This adapter provides LLM capabilities using Ollama models
    for search guidance, knowledge extraction, and analysis.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        default_temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        """
        Initialize Ollama LLM service.

        Args:
            ollama_client: Configured Ollama client
            default_temperature: Default sampling temperature
            max_tokens: Default maximum tokens for responses
        """
        self.ollama_client = ollama_client
        self.default_temperature = default_temperature
        self.max_tokens = max_tokens

    # Interactive search guidance

    async def analyze_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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
            return self._parse_json_response(response.text)
        except Exception as e:
            logging.warning(f"Failed to parse query analysis response: {e}")
            return {
                "strategy": "SEMANTIC",
                "confidence": 0.5,
                "reasoning": "Fallback to semantic search due to parsing error",
                "suggested_filters": [],
                "query_type": "exploratory",
            }

    async def guide_search_navigation(
        self,
        current_results: List[SearchResult],
        original_query: str,
        search_history: List[Dict[str, Any]],
        step_number: int,
    ) -> Dict[str, Any]:
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
                    "type": (
                        result.entity_type
                        if hasattr(result, "entity_type")
                        else "unknown"
                    ),
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
            return self._parse_json_response(response.text)
        except Exception as e:
            logging.warning(f"Failed to parse navigation guidance: {e}")
            return {
                "next_action": "stop",
                "strategy": "STOP",
                "suggested_query": original_query,
                "reasoning": "Unable to generate guidance due to parsing error",
                "focus_areas": [],
                "confidence": 0.0,
            }

    async def evaluate_search_results(
        self, results: List[SearchResult], query: str, search_context: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            return self._parse_json_response(response.text)
        except Exception as e:
            logging.warning(f"Failed to parse result evaluation: {e}")
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
        extraction_schema: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract structured knowledge from unstructured text."""
        # Use existing Ollama client method
        result = await asyncio.to_thread(
            self.ollama_client.extract_entities_and_relationships, text
        )

        # Add extraction metadata
        result["extraction_metadata"] = {
            "text_length": len(text),
            "schema_used": extraction_schema is not None,
            "context_provided": context is not None,
            "model": self.ollama_client.model,
        }

        return result

    async def generate_entity_summary(
        self,
        entity_data: Dict[str, Any],
        related_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a summary for an entity based on its data and relationships."""
        # Use existing Ollama client method
        summary = await asyncio.to_thread(
            self.ollama_client.generate_embeddings_description, entity_data
        )

        return summary

    async def suggest_relationships(
        self,
        source_entity: Dict[str, Any],
        target_entities: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
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

        target_names = [
            entity.get("name", "Unknown") for entity in target_entities[:10]
        ]
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
            return result if isinstance(result, list) else []
        except Exception as e:
            logging.warning(f"Failed to parse relationship suggestions: {e}")
            return []

    # Query enhancement

    async def expand_query(
        self, original_query: str, search_context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Expand a query with related terms and concepts."""
        system_prompt = """You are a query expansion expert. Generate related terms and concepts for the given query.

        Return JSON array of expanded query terms:
        ["term1", "term2", "term3", ...]
        
        Focus on synonyms, related concepts, and contextually relevant terms."""

        context_str = (
            f"Context: {json.dumps(search_context)}\n" if search_context else ""
        )
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
            return result if isinstance(result, list) else [original_query]
        except Exception as e:
            logging.warning(f"Failed to parse query expansion: {e}")
            return [original_query]

    async def generate_search_suggestions(
        self, partial_query: str, search_history: Optional[List[str]] = None
    ) -> List[str]:
        """Generate search suggestions for partial queries."""
        system_prompt = """You are a search suggestion generator. Complete and suggest variations of the partial query.

        Return JSON array of search suggestions:
        ["suggestion1", "suggestion2", "suggestion3", ...]
        
        Consider common search patterns and user intent."""

        history_str = (
            f"Recent searches: {json.dumps(search_history[-5:])}\n"
            if search_history
            else ""
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
            return result if isinstance(result, list) else [partial_query]
        except Exception as e:
            logging.warning(f"Failed to parse search suggestions: {e}")
            return [partial_query]

    # Content analysis

    async def classify_content(
        self, content: str, classification_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            return self._parse_json_response(response.text)
        except Exception as e:
            logging.warning(f"Failed to parse content classification: {e}")
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
        lang_code = response.text.strip().lower()
        # Validate it's a reasonable language code
        if len(lang_code) == 2 and lang_code.isalpha():
            return lang_code
        else:
            return "en"  # Default to English

    # Streaming responses

    async def stream_analysis(
        self, prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream analysis results for real-time processing."""
        context_str = f"Context: {json.dumps(context)}\n" if context else ""
        full_prompt = f"{context_str}{prompt}"

        # Note: This is a simplified implementation
        # Real streaming would require Ollama streaming API support
        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=full_prompt,
            temperature=self.default_temperature,
            max_tokens=self.max_tokens,
            stream=False,  # Ollama client would need stream=True support
        )

        # Simulate streaming by yielding chunks
        text = response.text
        chunk_size = 50
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            yield chunk
            await asyncio.sleep(0.1)  # Simulate streaming delay

    # Configuration and health

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current LLM model."""
        available_models = await asyncio.to_thread(
            self.ollama_client.list_available_models
        )

        return {
            "current_model": self.ollama_client.model,
            "base_url": self.ollama_client.base_url,
            "available_models": available_models,
            "default_temperature": self.default_temperature,
            "max_tokens": self.max_tokens,
            "provider": "ollama",
        }

    async def health_check(self) -> Dict[str, Any]:
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
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.ollama_client.model,
                "base_url": self.ollama_client.base_url,
                "last_check": "now",
            }

    async def get_usage_stats(self) -> Dict[str, Any]:
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

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
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
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse JSON from response: {response_text}")
