"""
Tests for text embedding functionality.
"""
import pytest
import numpy as np

from sqlite_kg_vec_mcp.vector.text_embedder import (
    TextEmbedder,
    RandomEmbedder,
    SentenceTransformerEmbedder,
    OpenAIEmbedder,
    create_embedder
)


class TestRandomEmbedder:
    """Tests for the RandomEmbedder class."""
    
    def test_initialization(self):
        """Test RandomEmbedder initialization."""
        embedder = RandomEmbedder(dimension=128)
        assert embedder.dimension == 128
    
    def test_embed_single(self):
        """Test embedding a single text."""
        embedder = RandomEmbedder(dimension=64)
        embedding = embedder.embed("Hello world")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (64,)
        assert embedding.dtype == np.float32
        
        # Check normalization for cosine similarity
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6
    
    def test_embed_batch(self):
        """Test embedding multiple texts."""
        embedder = RandomEmbedder(dimension=32)
        texts = ["Hello", "World", "Test"]
        embeddings = embedder.embed_batch(texts)
        
        assert len(embeddings) == 3
        for embedding in embeddings:
            assert embedding.shape == (32,)
            assert embedding.dtype == np.float32
    
    def test_deterministic(self):
        """Test that embeddings are deterministic for the same text."""
        embedder = RandomEmbedder(dimension=128)
        
        text = "Test text"
        embedding1 = embedder.embed(text)
        embedding2 = embedder.embed(text)
        
        assert np.allclose(embedding1, embedding2)


class TestSentenceTransformerEmbedder:
    """Tests for the SentenceTransformerEmbedder class."""
    
    @pytest.mark.skipif(True, reason="sentence-transformers requires large downloads")
    def test_initialization(self):
        """Test SentenceTransformerEmbedder initialization."""
        try:
            embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
            assert embedder.dimension == 384  # Expected dimension for this model
        except ImportError:
            pytest.skip("sentence-transformers not installed")
    
    @pytest.mark.skipif(True, reason="sentence-transformers requires large downloads")
    def test_embed(self):
        """Test embedding with sentence-transformers."""
        try:
            embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
            embedding = embedder.embed("This is a test sentence")
            
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (384,)
        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestCreateEmbedder:
    """Tests for the create_embedder factory function."""
    
    def test_create_random_embedder(self):
        """Test creating a random embedder."""
        embedder = create_embedder('random', dimension=128)
        assert isinstance(embedder, RandomEmbedder)
        assert embedder.dimension == 128
    
    def test_create_unknown_embedder(self):
        """Test creating an unknown embedder type raises error."""
        with pytest.raises(ValueError, match="Unknown embedder type"):
            create_embedder('unknown')
    
    @pytest.mark.skipif(True, reason="sentence-transformers requires large downloads")
    def test_create_sentence_transformer_embedder(self):
        """Test creating a sentence-transformers embedder."""
        try:
            embedder = create_embedder('sentence-transformers')
            assert isinstance(embedder, SentenceTransformerEmbedder)
        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestTextEmbedderIntegration:
    """Integration tests with VectorSearch."""
    
    def test_vector_search_with_custom_embedder(self):
        """Test that VectorSearch works with custom embedder."""
        import sqlite3
        from sqlite_kg_vec_mcp.vector.search import VectorSearch
        
        # Create in-memory database
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        
        # Initialize with custom embedder
        custom_embedder = RandomEmbedder(dimension=64)
        vector_search = VectorSearch(
            connection=conn,
            embedding_dim=64,
            text_embedder=custom_embedder
        )
        
        # The text embedder should be used
        assert vector_search.text_embedder == custom_embedder
        
        # Test that build_text_embedding uses our embedder
        embedding = vector_search.build_text_embedding("Test")
        assert embedding.shape == (64,)
        
        conn.close()
    
    def test_dimension_mismatch_error(self):
        """Test error when embedder dimension doesn't match index dimension."""
        import sqlite3
        from sqlite_kg_vec_mcp.vector.search import VectorSearch
        
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        
        # Create embedder with dimension 64
        embedder = RandomEmbedder(dimension=64)
        
        # Try to create VectorSearch with different dimension
        with pytest.raises(ValueError, match="dimension .* does not match"):
            VectorSearch(
                connection=conn,
                embedding_dim=128,  # Different from embedder
                text_embedder=embedder
            )
        
        conn.close()