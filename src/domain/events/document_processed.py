"""
문서 처리 완료 이벤트.
"""

from dataclasses import dataclass
from typing import List

from .base import DomainEvent


@dataclass
class DocumentProcessed(DomainEvent):
    """
    문서 처리가 완료되었을 때 발생하는 이벤트.
    """
    
    document_id: str
    extracted_node_count: int
    extracted_relationship_count: int
    processing_time_seconds: float
    
    @classmethod
    def create(cls, document_id: str, extracted_node_count: int, 
               extracted_relationship_count: int, processing_time_seconds: float) -> "DocumentProcessed":
        """문서 처리 완료 이벤트 생성."""
        return super().create(
            aggregate_id=document_id,
            document_id=document_id,
            extracted_node_count=extracted_node_count,
            extracted_relationship_count=extracted_relationship_count,
            processing_time_seconds=processing_time_seconds
        )