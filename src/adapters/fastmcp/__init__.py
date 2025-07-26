"""
FastMCP server adapter for knowledge graph API.

This module contains FastMCP-specific implementations for serving
the knowledge graph through the Model Context Protocol (MCP).
"""

from .server import KnowledgeGraphServer

__all__ = [
    "KnowledgeGraphServer"
]