"""
LLM service port for language model interactions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator
from enum import Enum

from ...domain import SearchResult


class SearchStrategy(Enum):
    """Search strategy options for LLM guidance."""
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"
    STOP = "stop"


class LLMService(ABC):
    """
    Secondary port for LLM service interactions.

    This interface defines how the domain interacts with language models
    for search guidance, knowledge extraction, and analysis.
    """

    # Interactive search guidance
    @abstractmethod
    async def analyze_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a search query to determine optimal search strategy.

        Args:
            query: User's search query
            context: Optional context information

        Returns:
            Analysis results with search strategy recommendations
        """
        pass

    @abstractmethod
    async def guide_search_navigation(
        self,
        current_results: List[SearchResult],
        original_query: str,
        search_history: List[Dict[str, Any]],
        step_number: int
    ) -> Dict[str, Any]:
        """
        Guide the next step in interactive search navigation.

        Args:
            current_results: Current search results
            original_query: Original user query
            search_history: History of search steps
            step_number: Current step number

        Returns:
            Navigation guidance with next search strategy
        """
        pass

    @abstractmethod
    async def evaluate_search_results(
        self,
        results: List[SearchResult],
        query: str,
        search_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate search results quality and relevance.

        Args:
            results: Search results to evaluate
            query: Original search query
            search_context: Search context information

        Returns:
            Evaluation results with quality metrics
        """
        pass

    # Knowledge extraction
    @abstractmethod
    async def extract_knowledge_from_text(
        self,
        text: str,
        extraction_schema: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract structured knowledge from unstructured text.

        Args:
            text: Text to extract knowledge from
            extraction_schema: Optional schema for extraction
            context: Optional context information

        Returns:
            Extracted knowledge with entities and relationships
        """
        pass

    @abstractmethod
    async def generate_entity_summary(
        self,
        entity_data: Dict[str, Any],
        related_entities: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate a summary for an entity based on its data and relationships.

        Args:
            entity_data: Entity data and properties
            related_entities: Optional related entity information

        Returns:
            Generated entity summary
        """
        pass

    @abstractmethod
    async def suggest_relationships(
        self,
        source_entity: Dict[str, Any],
        target_entities: List[Dict[str, Any]],
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest potential relationships between entities.

        Args:
            source_entity: Source entity data
            target_entities: Potential target entities
            context: Optional context for relationship suggestion

        Returns:
            List of suggested relationships with confidence scores
        """
        pass

    # Query enhancement
    @abstractmethod
    async def expand_query(
        self,
        original_query: str,
        search_context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Expand a query with related terms and concepts.

        Args:
            original_query: Original search query
            search_context: Optional context information

        Returns:
            List of expanded query terms
        """
        pass

    @abstractmethod
    async def generate_search_suggestions(
        self,
        partial_query: str,
        search_history: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate search suggestions for partial queries.

        Args:
            partial_query: Partial search query
            search_history: Optional search history

        Returns:
            List of search suggestions
        """
        pass

    # Content analysis
    @abstractmethod
    async def classify_content(
        self,
        content: str,
        classification_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify content according to a given schema.

        Args:
            content: Content to classify
            classification_schema: Classification schema

        Returns:
            Classification results with confidence scores
        """
        pass

    @abstractmethod
    async def detect_language(self, text: str) -> str:
        """
        Detect the language of given text.

        Args:
            text: Text to analyze

        Returns:
            Detected language code
        """
        pass

    # Streaming responses
    @abstractmethod
    async def stream_analysis(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream analysis results for real-time processing.

        Args:
            prompt: Analysis prompt
            context: Optional context information

        Yields:
            Analysis results in chunks
        """
        pass

    # Configuration and health
    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current LLM model.

        Returns:
            Model information and capabilities
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the LLM service.

        Returns:
            Health status information
        """
        pass

    @abstractmethod
    async def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for the LLM service.

        Returns:
            Usage statistics and metrics
        """
        pass
