"""
SQLite3 database adapters.

This module contains SQLite3-specific implementations for database operations
including connection management, schema operations, and transactions.
"""

from .connection import DatabaseConnection
from .schema import SchemaManager
from .database import SQLiteDatabase
from .vector_store import SQLiteVectorStore
from .transactions import TransactionManager

# Graph domain classes
from .graph.entities import Entity, EntityManager
from .graph.relationships import Relationship, RelationshipManager
from .graph.traversal import GraphTraversal, PathNode
from .graph.interactive_search import InteractiveSearchEngine

__all__ = [
    "DatabaseConnection",
    "SchemaManager", 
    "SQLiteDatabase",
    "SQLiteVectorStore",
    "TransactionManager",
    # Graph classes
    "Entity",
    "EntityManager",
    "Relationship", 
    "RelationshipManager",
    "GraphTraversal",
    "PathNode",
    "InteractiveSearchEngine"
]