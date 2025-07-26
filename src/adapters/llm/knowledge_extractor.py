"""
Knowledge extraction service using LLM for automatic knowledge graph construction.
"""

import logging
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .ollama_client import OllamaClient
from src.adapters.graph.entities import EntityManager
from src.adapters.graph.relationships import RelationshipManager
from src.adapters.vector.embeddings import EmbeddingManager


@dataclass
class ExtractionResult:
    """Result of knowledge extraction from text."""
    entities_created: int
    relationships_created: int
    errors: List[str]
    processing_time: float


class KnowledgeExtractor:
    """Service for extracting knowledge graphs from text using LLM."""
    
    def __init__(
        self,
        connection: sqlite3.Connection,
        ollama_client: OllamaClient,
        auto_embed: bool = True
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
        enhance_descriptions: bool = True
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
        import time
        start_time = time.time()
        
        entities_created = 0
        relationships_created = 0
        errors = []
        
        try:
            # Extract entities and relationships using LLM
            logging.info(f"Extracting knowledge from text ({len(text)} characters)...")
            extraction_data = self.ollama_client.extract_entities_and_relationships(text)
            
            # Process entities
            entities_created = self._process_entities(
                extraction_data.get("entities", []),
                source_id,
                enhance_descriptions,
                errors
            )
            
            # Process relationships
            relationships_created = self._process_relationships(
                extraction_data.get("relationships", []),
                errors
            )
            
            # Generate embeddings if enabled
            if self.auto_embed and self.embedding_manager:
                try:
                    processed_count = self.embedding_manager.process_outbox()
                    logging.info(f"Generated embeddings for {processed_count} entities")
                except Exception as e:
                    error_msg = f"Embedding generation failed: {e}"
                    logging.error(error_msg)
                    errors.append(error_msg)
            
        except Exception as e:
            error_msg = f"Knowledge extraction failed: {e}"
            logging.error(error_msg)
            errors.append(error_msg)
        
        processing_time = time.time() - start_time
        
        result = ExtractionResult(
            entities_created=entities_created,
            relationships_created=relationships_created,
            errors=errors,
            processing_time=processing_time
        )
        
        logging.info(f"Extraction completed: {entities_created} entities, "
                    f"{relationships_created} relationships in {processing_time:.2f}s")
        
        return result
    
    def _process_entities(
        self, 
        entities: List[Dict[str, Any]], 
        source_id: Optional[str],
        enhance_descriptions: bool,
        errors: List[str]
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
                        enhanced_desc = self.ollama_client.generate_embeddings_description(entity_data)
                        properties["llm_description"] = enhanced_desc
                    except Exception as e:
                        logging.warning(f"Failed to enhance description for {entity_data['name']}: {e}")
                
                # Create entity
                entity = self.entity_manager.create_entity(
                    type=entity_data["type"],
                    name=entity_data["name"],
                    properties=properties,
                    custom_uuid=entity_data.get("id")
                )
                
                # Store mapping for relationship processing
                extraction_id = entity_data.get("id", entity_data["name"])
                self.entity_id_mapping[extraction_id] = entity.id
                
                created_count += 1
                logging.debug(f"Created entity: {entity.name} ({entity.type})")
                
            except Exception as e:
                error_msg = f"Failed to create entity {entity_data.get('name', 'unknown')}: {e}"
                logging.error(error_msg)
                errors.append(error_msg)
        
        return created_count
    
    def _process_relationships(
        self, 
        relationships: List[Dict[str, Any]], 
        errors: List[str]
    ) -> int:
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
                relationship = self.relationship_manager.create_relationship(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=rel_data["type"],
                    properties=rel_data.get("properties", {})
                )
                
                created_count += 1
                logging.debug(f"Created relationship: {rel_data['source']} --{rel_data['type']}--> {rel_data['target']}")
                
            except Exception as e:
                error_msg = f"Failed to create relationship {rel_data.get('type', 'unknown')}: {e}"
                logging.error(error_msg)
                errors.append(error_msg)
        
        return created_count
    
    def extract_from_documents(
        self, 
        documents: List[Dict[str, str]], 
        batch_size: int = 10
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
            batch = documents[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            for doc in batch:
                doc_id = doc.get("id", f"doc_{i}")
                text = doc.get("text", "")
                
                if not text.strip():
                    continue
                
                result = self.extract_from_text(text, source_id=doc_id)
                results.append(result)
                
                # Log progress
                if result.errors:
                    logging.warning(f"Document {doc_id} had {len(result.errors)} errors")
        
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
            "entities": {
                "total": total_entities,
                "by_type": entity_stats
            },
            "relationships": {
                "total": total_relationships,
                "by_type": relationship_stats
            },
            "embeddings": embedding_stats,
            "model": self.ollama_client.model
        }