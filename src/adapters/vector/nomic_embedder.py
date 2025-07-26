"""
Nomic Embed Text integration for vector embeddings.
"""

import logging
import requests
import json
from typing import List, Optional, Union
import numpy as np
from .text_embedder import VectorTextEmbedder


class NomicEmbedder(VectorTextEmbedder):
    """Nomic Embed Text embedder using Ollama."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model_name: str = "nomic-embed-text",
        timeout: int = 60
    ):
        """
        Initialize Nomic embedder.
        
        Args:
            base_url: Ollama server URL
            model_name: Nomic model name
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.session = requests.Session()
        
        # Test connection and ensure model is available
        self._ensure_model_available()
    
    def _ensure_model_available(self) -> bool:
        """Ensure the Nomic model is available."""
        try:
            # Check if model is already available
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            available_models = [model["name"] for model in response.json().get("models", [])]
            
            if self.model_name in available_models:
                logging.info(f"Nomic model {self.model_name} is available")
                return True
            
            # Try to pull the model if not available
            logging.info(f"Pulling Nomic model {self.model_name}...")
            pull_response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=300  # 5 minutes for model download
            )
            
            if pull_response.status_code == 200:
                logging.info(f"Successfully pulled {self.model_name}")
                return True
            else:
                logging.error(f"Failed to pull {self.model_name}: {pull_response.text}")
                return False
                
        except Exception as e:
            logging.error(f"Error ensuring Nomic model availability: {e}")
            return False
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text using Nomic Embed.
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            numpy array of embeddings
        """
        # Handle single text input
        if isinstance(text, str):
            texts = [text]
            return_single = True
        else:
            texts = text
            return_single = False
        
        embeddings = []
        
        for text_item in texts:
            try:
                # Prepare request for Ollama embeddings endpoint
                data = {
                    "model": self.model_name,
                    "prompt": text_item
                }
                
                response = self.session.post(
                    f"{self.base_url}/api/embeddings",
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                embedding = np.array(result["embedding"], dtype=np.float32)
                embeddings.append(embedding)
                
            except Exception as e:
                logging.error(f"Error generating Nomic embedding for text: {e}")
                # Return zero vector on error
                embeddings.append(np.zeros(768, dtype=np.float32))  # Nomic default dimension
        
        embeddings_array = np.array(embeddings)
        
        if return_single:
            return embeddings_array[0]
        else:
            return embeddings_array
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Convert batch of texts to embedding vectors."""
        embeddings = []
        for text in texts:
            embedding = self.embed(text)
            embeddings.append(embedding)
        return embeddings
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        # Test with a small text to get actual dimension
        try:
            test_embedding = self.embed("test")
            return len(test_embedding)
        except Exception:
            # Default dimension for nomic-embed-text
            return 768
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for Nomic Embed Text."""
        return self.dimension
    
    def batch_embed(
        self, 
        texts: List[str], 
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            
        Returns:
            numpy array of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed(batch)
            
            if len(batch) == 1:
                all_embeddings.append(batch_embeddings)
            else:
                all_embeddings.extend(batch_embeddings)
            
            # Log progress for large batches
            if len(texts) > 100:
                logging.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
        
        return np.array(all_embeddings)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def most_similar_texts(
        self, 
        query_text: str, 
        candidate_texts: List[str], 
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar texts to a query.
        
        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            top_k: Number of top results to return
            
        Returns:
            List of (text, similarity_score) tuples
        """
        # Generate embeddings
        query_embedding = self.embed(query_text)
        candidate_embeddings = self.batch_embed(candidate_texts)
        
        # Calculate similarities
        similarities = []
        for i, candidate_embedding in enumerate(candidate_embeddings):
            sim_score = self.similarity(query_embedding, candidate_embedding)
            similarities.append((candidate_texts[i], sim_score))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def create_nomic_embedder(
    base_url: str = "http://localhost:11434",
    model_name: str = "nomic-embed-text"
) -> NomicEmbedder:
    """
    Factory function to create a Nomic embedder.
    
    Args:
        base_url: Ollama server URL
        model_name: Nomic model name
        
    Returns:
        Configured NomicEmbedder instance
    """
    return NomicEmbedder(base_url=base_url, model_name=model_name)