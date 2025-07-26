"""
Example demonstrating different text embedding options.
"""
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import KnowledgeGraph, create_embedder

def main():
    """
    Demonstrate text embedding with different models.
    """
    print("Text Embedding Examples")
    print("======================")
    
    # Example 1: Using sentence-transformers (default)
    print("\n1. Using sentence-transformers")
    db_path = "embedding_example.db"
    
    # Remove existing database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    try:
        kg = KnowledgeGraph(
            db_path,
            embedding_dim=384,  # MiniLM-L6-v2 has 384 dimensions
            embedder_type='sentence-transformers',
            embedder_kwargs={'model_name': 'all-MiniLM-L6-v2'}
        )
        
        # Create some nodes
        alice = kg.create_node(
            type="Person",
            name="Alice",
            properties={"bio": "A data scientist working on machine learning"}
        )
        
        bob = kg.create_node(
            type="Person",
            name="Bob",
            properties={"bio": "A software engineer building cloud applications"}
        )
        
        # Search by text
        print("\nSearching for 'machine learning':")
        results = kg.search_by_text("machine learning", limit=5)
        for result in results:
            entity = result.entity
            print(f"- {entity.name}: {entity.properties.get('bio', '')} (score: {result.distance:.4f})")
        
        print("\nSearching for 'cloud computing':")
        results = kg.search_by_text("cloud computing", limit=5)
        for result in results:
            entity = result.entity
            print(f"- {entity.name}: {entity.properties.get('bio', '')} (score: {result.distance:.4f})")
            
    except ImportError:
        print("sentence-transformers not installed. Install with: pip install sentence-transformers")
    
    # Example 2: Using OpenAI embeddings (requires API key)
    print("\n\n2. Using OpenAI embeddings")
    print("(Requires OPENAI_API_KEY environment variable)")
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            kg_openai = KnowledgeGraph(
                "openai_example.db",
                embedding_dim=1536,  # OpenAI ada-002 dimension
                embedder_type='openai',
                embedder_kwargs={'model': 'text-embedding-ada-002'}
            )
            print("Successfully initialized with OpenAI embeddings")
        except ImportError:
            print("openai package not installed. Install with: pip install openai")
    else:
        print("OPENAI_API_KEY not found in environment")
    
    # Example 3: Using random embeddings (for testing)
    print("\n\n3. Using random embeddings (for testing)")
    kg_random = KnowledgeGraph(
        "random_example.db",
        embedding_dim=128,
        embedder_type='random'
    )
    print("Random embeddings initialized (for testing only)")
    
    # Example 4: Custom text embedder
    print("\n\n4. Using custom text embedder")
    
    # You can create a custom embedder by implementing the VectorTextEmbedder interface
    custom_embedder = create_embedder('random', dimension=256)
    kg_custom = KnowledgeGraph(
        "custom_example.db",
        embedding_dim=256,
        text_embedder=custom_embedder
    )
    print("Custom embedder initialized")
    
    print("\nText embedding integration complete!")

if __name__ == "__main__":
    main()