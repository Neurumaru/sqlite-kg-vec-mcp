"""
Example MCP server for the SQLite KG Vec MCP library.
"""

import argparse
import os
import sys

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# TODO: Fix KnowledgeGraphServer import - module structure changed
# from src import KnowledgeGraphServer  # noqa: E402


# Temporary replacement with error message
class KnowledgeGraphServer:
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "KnowledgeGraphServer는 현재 import 오류로 인해 사용할 수 없습니다. "
            "src.adapters.fastmcp.server에서 직접 import하거나 MCP 의존성을 확인하세요."
        )


def main():
    """
    Run an MCP server for the knowledge graph.
    """
    parser = argparse.ArgumentParser(description="Run a Knowledge Graph MCP server")
    parser.add_argument("--db", default="kg.db", help="Database file path")
    parser.add_argument("--index-dir", default="./vector_index", help="Vector index directory")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument(
        "--similarity",
        default="cosine",
        choices=["cosine", "ip", "l2"],
        help="Vector similarity metric",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Create index directory if it doesn't exist
    os.makedirs(args.index_dir, exist_ok=True)

    print("Starting Knowledge Graph MCP server...")
    print(f"Database: {args.db}")
    print(f"Vector index directory: {args.index_dir}")
    print(f"Server: {args.host}:{args.port}")

    # Initialize the server
    server = KnowledgeGraphServer(
        db_path=args.db,
        vector_index_dir=args.index_dir,
        embedding_dim=args.dim,
        vector_similarity=args.similarity,
        log_level=args.log_level,
    )

    # Start the server
    try:
        print(f"Server running at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop")
        server.start(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("Stopping server...")
    finally:
        server.close()


if __name__ == "__main__":
    main()
