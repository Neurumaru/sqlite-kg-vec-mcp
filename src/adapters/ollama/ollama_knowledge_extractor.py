"""
Ollama-based knowledge extraction adapter.

This module provides a concrete implementation of the KnowledgeExtractor port
using Ollama LLM for automatic knowledge graph construction.
"""

import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.adapters.hnsw.embeddings import EmbeddingManager
from src.adapters.sqlite3.graph.entities import EntityManager
from src.adapters.sqlite3.graph.relationships import RelationshipManager
from src.domain.entities.node import Node
from src.domain.entities.relationship import Relationship, RelationshipType
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.ports.knowledge_extractor import KnowledgeExtractor

from .ollama_client import OllamaClient


@dataclass
class ExtractionResult:
    """지식 추출 결과를 담는 데이터클래스."""

    entities_created: int = 0
    relationships_created: int = 0
    errors: Optional[List[str]] = None
    processing_time: float = 0.0

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class OllamaKnowledgeExtractor(KnowledgeExtractor):
    """Ollama LLM-based implementation of knowledge extraction adapter."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        ollama_client: OllamaClient,
        auto_embed: bool = True,
    ):
        """
        Initialize knowledge extractor.

        Args:
            connection: SQLite database connection
            ollama_client: Ollama client for LLM operations
            auto_embed: Whether to automatically generate embeddings
        """
        self.connection = connection
        self.ollama_client = ollama_client
        self.auto_embed = auto_embed

        # Initialize managers
        self.entity_manager = EntityManager(connection)
        self.relationship_manager = RelationshipManager(connection)
        self.embedding_manager = EmbeddingManager(connection) if auto_embed else None

        # Entity ID mapping for batch processing
        self.entity_id_mapping: Dict[str, int] = {}

    def extract_from_text(
        self,
        text: str,
        source_id: Optional[str] = None,
        enhance_descriptions: bool = True,
    ) -> ExtractionResult:
        """
        Extract knowledge graph from text.

        Args:
            text: Input text to process
            source_id: Optional source identifier for tracking
            enhance_descriptions: Whether to enhance entity descriptions with LLM

        Returns:
            ExtractionResult with statistics and errors
        """
        start_time = time.time()

        entities_created = 0
        relationships_created = 0
        errors: List[str] = []

        try:
            # Extract entities and relationships using LLM
            logging.info("Extracting knowledge from text (%s characters)...", len(text))
            extraction_data = self.ollama_client.extract_entities_and_relationships(text)

            # Process entities
            entities_created = self._process_entities(
                extraction_data.get("entities", []),
                source_id,
                enhance_descriptions,
                errors,
            )

            # Process relationships
            relationships_created = self._process_relationships(
                extraction_data.get("relationships", []), errors
            )

            # Generate embeddings if enabled
            if self.auto_embed and self.embedding_manager:
                try:
                    processed_count = self.embedding_manager.process_outbox()
                    logging.info("Generated embeddings for %s entities", processed_count)
                except Exception as exception:
                    error_msg = "Embedding generation failed: %s" % exception
                    logging.error(error_msg)
                    errors.append(error_msg)

        except Exception as exception:
            error_msg = "Knowledge extraction failed: %s" % exception
            logging.error(error_msg)
            errors.append(error_msg)

        processing_time = time.time() - start_time

        result = ExtractionResult(
            entities_created=entities_created,
            relationships_created=relationships_created,
            errors=errors,
            processing_time=processing_time,
        )

        logging.info(
            "Extraction completed: %s entities, %s relationships in %.2fs",
            entities_created,
            relationships_created,
            processing_time,
        )

        return result

    def _process_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: Optional[str],
        enhance_descriptions: bool,
        errors: List[str],
    ) -> int:
        """Process and create entities from extraction data."""
        created_count = 0

        for entity_data in entities:
            try:
                # Validate required fields
                if "name" not in entity_data or "type" not in entity_data:
                    errors.append(f"Entity missing required fields: {entity_data}")
                    continue

                # Prepare properties
                properties = entity_data.get("properties", {})
                if source_id:
                    properties["source_id"] = source_id

                # Enhance description if requested
                if enhance_descriptions:
                    try:
                        enhanced_desc = self.ollama_client.generate_embeddings_description(
                            entity_data
                        )
                        properties["llm_description"] = enhanced_desc
                    except Exception as exception:
                        logging.warning(
                            "Failed to enhance description for %s: %s",
                            entity_data["name"],
                            exception,
                        )

                # Create entity
                entity = self.entity_manager.create_entity(
                    type=entity_data["type"],
                    name=entity_data["name"],
                    properties=properties,
                    custom_uuid=entity_data.get("id"),
                )

                # Store mapping for relationship processing
                extraction_id = entity_data.get("id", entity_data["name"])
                self.entity_id_mapping[extraction_id] = entity.id

                created_count += 1
                logging.debug("Created entity: %s (%s)", entity.name, entity.type)

            except Exception as exception:
                error_msg = "Failed to create entity %s: %s" % (
                    entity_data.get("name", "unknown"),
                    exception,
                )
                logging.error(error_msg)
                errors.append(error_msg)

        return created_count

    def _process_relationships(self, relationships: List[Dict[str, Any]], errors: List[str]) -> int:
        """Process and create relationships from extraction data."""
        created_count = 0

        for rel_data in relationships:
            try:
                # Validate required fields
                if not all(field in rel_data for field in ["source", "target", "type"]):
                    errors.append(f"Relationship missing required fields: {rel_data}")
                    continue

                # Resolve entity IDs
                source_id = self.entity_id_mapping.get(rel_data["source"])
                target_id = self.entity_id_mapping.get(rel_data["target"])

                if source_id is None:
                    errors.append(f"Source entity not found: {rel_data['source']}")
                    continue

                if target_id is None:
                    errors.append(f"Target entity not found: {rel_data['target']}")
                    continue

                # Create relationship
                self.relationship_manager.create_relationship(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=rel_data["type"],
                    properties=rel_data.get("properties", {}),
                )

                created_count += 1
                logging.debug(
                    "Created relationship: %s --%s--> %s",
                    rel_data["source"],
                    rel_data["type"],
                    rel_data["target"],
                )

            except Exception as exception:
                error_msg = "Failed to create relationship %s: %s" % (
                    rel_data.get("type", "unknown"),
                    exception,
                )
                logging.error(error_msg)
                errors.append(error_msg)

        return created_count

    def extract_from_documents(
        self, documents: List[Dict[str, str]], batch_size: int = 10
    ) -> List[ExtractionResult]:
        """
        Extract knowledge from multiple documents in batches.

        Args:
            documents: List of documents with 'text' and optional 'id' fields
            batch_size: Number of documents to process in each batch

        Returns:
            List of ExtractionResult for each document
        """
        results = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            logging.info(
                "Processing batch %s/%s",
                i // batch_size + 1,
                (len(documents) + batch_size - 1) // batch_size,
            )

            for doc in batch:
                doc_id = doc.get("id", f"doc_{i}")
                text = doc.get("text", "")

                if not text.strip():
                    continue

                result = self.extract_from_text(text, source_id=doc_id)
                results.append(result)

                # Log progress
                if result.errors:
                    logging.warning("Document %s had %s errors", doc_id, len(result.errors))

        return results

    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        cursor = self.connection.cursor()

        # Entity statistics
        cursor.execute("SELECT type, COUNT(*) FROM entities GROUP BY type")
        entity_stats = dict(cursor.fetchall())

        cursor.execute("SELECT COUNT(*) FROM entities")
        total_entities = cursor.fetchone()[0]

        # Relationship statistics
        cursor.execute("SELECT relation_type, COUNT(*) FROM edges GROUP BY relation_type")
        relationship_stats = dict(cursor.fetchall())

        cursor.execute("SELECT COUNT(*) FROM edges")
        total_relationships = cursor.fetchone()[0]

        # Embedding statistics if available
        embedding_stats = {}
        if self.embedding_manager:
            try:
                cursor.execute("SELECT COUNT(*) FROM vector_embeddings")
                total_embeddings = cursor.fetchone()[0]
                embedding_stats["total_embeddings"] = total_embeddings
            except Exception:
                embedding_stats["total_embeddings"] = 0

        return {
            "entities": {"total": total_entities, "by_type": entity_stats},
            "relationships": {
                "total": total_relationships,
                "by_type": relationship_stats,
            },
            "embeddings": embedding_stats,
            "model": self.ollama_client.model,
        }

    # Port interface implementation methods

    async def extract_knowledge(self, text: str) -> ExtractionResult:
        """Extract entities and relationships from text using Ollama LLM."""
        return self.extract_from_text(text)

    async def extract_entities(self, text: str) -> List[Node]:
        """Extract only entities from text."""
        # Extract entities using the existing synchronous method
        extraction_data = await asyncio.to_thread(
            self.ollama_client.extract_entities_and_relationships, text
        )
        entities = []

        for entity_data in extraction_data.get("entities", []):
            try:
                entity = Node(
                    id=NodeId.generate(),
                    name=entity_data.get("name", "Unknown"),
                    node_type=entity_data.get("type", "Unknown"),
                    properties=entity_data.get("properties", {}),
                )
                entities.append(entity)
            except Exception as exception:
                logging.warning("Failed to create entity from data %s: %s", entity_data, exception)

        return entities

    async def extract_relationships(self, text: str, entities: List[Node]) -> List[Relationship]:
        """Extract relationships from text given existing entities."""
        # Extract relationships using the existing synchronous method
        extraction_data = await asyncio.to_thread(
            self.ollama_client.extract_entities_and_relationships, text
        )
        relationships = []

        # Create entity name to ID mapping for reference
        entity_map = {entity.name: entity.id for entity in entities}

        for rel_data in extraction_data.get("relationships", []):
            try:
                # Try to resolve entity references
                source_id = entity_map.get(rel_data.get("source"))
                target_id = entity_map.get(rel_data.get("target"))

                if source_id and target_id:
                    rel_type_str = rel_data.get("type", "RELATED_TO").upper()
                    rel_type = getattr(RelationshipType, rel_type_str, RelationshipType.OTHER)

                    relationship = Relationship(
                        id=RelationshipId.generate(),
                        source_node_id=source_id,
                        target_node_id=target_id,
                        relationship_type=rel_type,
                        label=rel_data.get("type", "RELATED_TO"),
                        properties=rel_data.get("properties", {}),
                    )
                    relationships.append(relationship)
            except Exception as exception:
                logging.warning(
                    "Failed to create relationship from data %s: %s", rel_data, exception
                )

        return relationships

    async def validate_extraction(self, text: str, result: ExtractionResult) -> bool:
        """Validate the quality of extraction results."""
        # Simple validation based on extraction success
        if result.errors:
            error_ratio = len(result.errors) / max(
                1, result.entities_created + result.relationships_created
            )
            return error_ratio < 0.5  # Less than 50% error rate

        # Check if we extracted something meaningful
        return result.entities_created > 0 or result.relationships_created > 0

    async def get_extraction_confidence(self, text: str) -> float:
        """Get confidence score for extraction capability on given text."""
        # Simple heuristic based on text length and content
        if not text or len(text.strip()) < 10:
            return 0.0

        # Longer texts generally provide better extraction opportunities
        length_score = min(len(text) / 1000, 1.0)  # Normalize to 1000 chars

        # Check for structured content indicators
        structure_indicators = [
            ".",
            ":",
            ";",
            ",",  # Punctuation
            "is",
            "was",
            "are",
            "were",  # Linking verbs
            "the",
            "a",
            "an",  # Articles
        ]

        structure_score = sum(
            1 for indicator in structure_indicators if indicator in text.lower()
        ) / len(structure_indicators)

        # Combine scores
        confidence = length_score * 0.3 + structure_score * 0.7
        return min(confidence, 1.0)
