"""
Primary ports (driving ports) for the knowledge graph system.

These are interfaces that define how external systems can interact with
the domain (inbound/left-side ports in hexagonal architecture).
"""

from .knowledge_graph_use_cases import KnowledgeGraphUseCases
from .search_use_cases import SearchUseCases
from .admin_use_cases import AdminUseCases

__all__ = [
    "KnowledgeGraphUseCases",
    "SearchUseCases",
    "AdminUseCases",
]
