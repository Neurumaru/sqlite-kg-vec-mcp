"""
Knowledge extraction port interface.

This module defines the abstract interface for extracting knowledge graphs
from text, following the hexagonal architecture pattern.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

from src.domain.entities.entity import Entity
from src.domain.entities.relationship import Relationship


@dataclass
class ExtractionResult:
    """Result of knowledge extraction from text."""
    entities_created: int
    relationships_created: int
    errors: List[str]
    processing_time: float


class KnowledgeExtractor(ABC):
    """
    Port interface for knowledge extraction from text.
    
    This interface defines how the domain layer interacts with
    knowledge extraction services (LLMs, NLP models, etc.).
    """

    @abstractmethod
    async def extract_knowledge(self, text: str) -> ExtractionResult:
        """
        Extract entities and relationships from text.
        
        Args:
            text: Input text to extract knowledge from
            
        Returns:
            ExtractionResult containing summary of extraction
        """
        pass

    @abstractmethod
    async def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of extracted entities
        """
        pass

    @abstractmethod
    async def extract_relationships(
        self, 
        text: str, 
        entities: List[Entity]
    ) -> List[Relationship]:
        """
        Extract relationships from text given existing entities.
        
        Args:
            text: Input text to analyze
            entities: List of entities to find relationships between
            
        Returns:
            List of extracted relationships
        """
        pass

    @abstractmethod
    async def validate_extraction(self, text: str, result: ExtractionResult) -> bool:
        """
        Validate the quality of extraction results.
        
        Args:
            text: Original input text
            result: Extraction result to validate
            
        Returns:
            True if extraction quality is acceptable
        """
        pass

    @abstractmethod
    async def get_extraction_confidence(self, text: str) -> float:
        """
        Get confidence score for extraction capability on given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass