"""
Example script to run the Knowledge Graph MCP server.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# TODO: Fix imports - sqlite_kg_vec_mcp module doesn't exist
# from sqlite_kg_vec_mcp.db.schema import SchemaManager  # noqa: E402
# from sqlite_kg_vec_mcp.server.api import KnowledgeGraphServer  # noqa: E402

# Use correct import paths
from src.adapters.sqlite3.schema import SchemaManager  # noqa: E402

try:
    from src.adapters.fastmcp.server import KnowledgeGraphServer  # noqa: E402
except ImportError:
    # Fallback if MCP server is not available
    class KnowledgeGraphServer:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "KnowledgeGraphServer는 현재 사용할 수 없습니다. "
                "fastmcp 의존성을 확인하거나 src.adapters.fastmcp.server를 확인하세요."
            )


def main():
    """Run the Knowledge Graph MCP server."""
    parser = argparse.ArgumentParser(description="Run the Knowledge Graph MCP server")
    parser.add_argument(
        "--db-path",
        type=str,
        default="knowledge_graph.db",
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--vector-dir",
        type=str,
        default="vector_indexes",
        help="Directory for vector index files",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument(
        "--transport",
        type=str,
        default="sse",
        choices=["sse", "stdio", "streamable-http"],
        help="Transport protocol to use",
    )
    parser.add_argument(
        "--init-schema",
        action="store_true",
        help="Initialize the database schema if it doesn't exist",
    )
    parser.add_argument("--dimension", type=int, default=128, help="Dimension of embedding vectors")
    parser.add_argument(
        "--similarity",
        type=str,
        default="cosine",
        choices=["cosine", "inner_product", "l2"],
        help="Similarity metric for vector search",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="Knowledge Graph Server",
        help="Name of the MCP server",
    )
    parser.add_argument(
        "--server-instructions",
        type=str,
        default="SQLite-based knowledge graph with vector search capabilities",
        help="Instructions for the MCP server",
    )

    args = parser.parse_args()

    # Create vector directory if it doesn't exist
    os.makedirs(args.vector_dir, exist_ok=True)

    # Initialize schema if requested
    if args.init_schema:
        print(f"Initializing database schema at {args.db_path}")
        schema_manager = SchemaManager(args.db_path)
        schema_manager.initialize_schema()

    # Create and start the server
    print(f"Starting Knowledge Graph MCP server at {args.host}:{args.port}")
    server = KnowledgeGraphServer(
        db_path=args.db_path,
        vector_index_dir=args.vector_dir,
        embedding_dim=args.dimension,
        vector_similarity=args.similarity,
        server_name=args.server_name,
        server_instructions=args.server_instructions,
    )

    try:
        server.start(host=args.host, port=args.port, transport=args.transport)
    except KeyboardInterrupt:
        print("Server stopped by user")
    finally:
        server.close()
        print("Server resources cleaned up")


if __name__ == "__main__":
    main()
