"""
문서 저장소 포트.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.value_objects.document_id import DocumentId


class DocumentRepository(ABC):
    """
    문서 저장소 포트.
    
    문서의 영속성을 담당하는 인터페이스입니다.
    """
    
    @abstractmethod
    async def save(self, document: Document) -> Document:
        """
        문서를 저장합니다.
        
        Args:
            document: 저장할 문서
            
        Returns:
            저장된 문서
        """
        pass
    
    @abstractmethod
    async def find_by_id(self, document_id: DocumentId) -> Optional[Document]:
        """
        ID로 문서를 찾습니다.
        
        Args:
            document_id: 문서 ID
            
        Returns:
            찾은 문서 또는 None
        """
        pass
    
    @abstractmethod
    async def find_by_title(self, title: str) -> List[Document]:
        """
        제목으로 문서를 찾습니다.
        
        Args:
            title: 문서 제목
            
        Returns:
            매칭되는 문서들
        """
        pass
    
    @abstractmethod
    async def find_by_status(self, status: DocumentStatus) -> List[Document]:
        """
        상태로 문서를 찾습니다.
        
        Args:
            status: 문서 상태
            
        Returns:
            해당 상태의 문서들
        """
        pass
    
    @abstractmethod
    async def find_by_type(self, doc_type: DocumentType) -> List[Document]:
        """
        타입으로 문서를 찾습니다.
        
        Args:
            doc_type: 문서 타입
            
        Returns:
            해당 타입의 문서들
        """
        pass
    
    @abstractmethod
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[Document]:
        """
        모든 문서를 조회합니다.
        
        Args:
            limit: 최대 반환 개수
            offset: 건너뛸 개수
            
        Returns:
            문서들
        """
        pass
    
    @abstractmethod
    async def find_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Document]:
        """
        날짜 범위로 문서를 찾습니다.
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            해당 기간의 문서들
        """
        pass
    
    @abstractmethod
    async def search_content(self, query: str, limit: int = 10) -> List[Document]:
        """
        문서 내용을 검색합니다.
        
        Args:
            query: 검색 쿼리
            limit: 최대 반환 개수
            
        Returns:
            검색 결과 문서들
        """
        pass
    
    @abstractmethod
    async def update(self, document: Document) -> Document:
        """
        문서를 업데이트합니다.
        
        Args:
            document: 업데이트할 문서
            
        Returns:
            업데이트된 문서
        """
        pass
    
    @abstractmethod
    async def delete(self, document_id: DocumentId) -> bool:
        """
        문서를 삭제합니다.
        
        Args:
            document_id: 삭제할 문서 ID
            
        Returns:
            삭제 성공 여부
        """
        pass
    
    @abstractmethod
    async def exists(self, document_id: DocumentId) -> bool:
        """
        문서가 존재하는지 확인합니다.
        
        Args:
            document_id: 확인할 문서 ID
            
        Returns:
            존재 여부
        """
        pass
    
    @abstractmethod
    async def count_by_status(self, status: DocumentStatus) -> int:
        """
        상태별 문서 개수를 반환합니다.
        
        Args:
            status: 문서 상태
            
        Returns:
            해당 상태의 문서 개수
        """
        pass
    
    @abstractmethod
    async def count_total(self) -> int:
        """
        전체 문서 개수를 반환합니다.
        
        Returns:
            전체 문서 개수
        """
        pass
    
    @abstractmethod
    async def find_with_connected_elements(self) -> List[Document]:
        """
        연결된 노드나 관계가 있는 문서들을 찾습니다.
        
        Returns:
            연결된 요소가 있는 문서들
        """
        pass
    
    @abstractmethod
    async def find_unprocessed(self, limit: int = 100) -> List[Document]:
        """
        처리되지 않은 문서들을 찾습니다.
        
        Args:
            limit: 최대 반환 개수
            
        Returns:
            미처리 문서들
        """
        pass
    
    @abstractmethod
    async def bulk_update_status(self, document_ids: List[DocumentId], 
                                status: DocumentStatus) -> int:
        """
        여러 문서의 상태를 일괄 업데이트합니다.
        
        Args:
            document_ids: 업데이트할 문서 ID 목록
            status: 새로운 상태
            
        Returns:
            업데이트된 문서 개수
        """
        pass