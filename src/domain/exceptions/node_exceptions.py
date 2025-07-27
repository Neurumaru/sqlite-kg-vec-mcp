"""
노드 관련 예외들.
"""

from .base import DomainException


class NodeNotFoundException(DomainException):
    """노드를 찾을 수 없을 때 발생하는 예외."""
    
    def __init__(self, node_id: str):
        super().__init__(
            message=f"Node with ID '{node_id}' not found",
            error_code="NODE_NOT_FOUND"
        )
        self.node_id = node_id


class NodeAlreadyExistsException(DomainException):
    """노드가 이미 존재할 때 발생하는 예외."""
    
    def __init__(self, node_id: str):
        super().__init__(
            message=f"Node with ID '{node_id}' already exists",
            error_code="NODE_ALREADY_EXISTS"
        )
        self.node_id = node_id


class InvalidNodeException(DomainException):
    """유효하지 않은 노드일 때 발생하는 예외."""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"Invalid node: {reason}",
            error_code="INVALID_NODE"
        )
        self.reason = reason


class NodeEmbeddingException(DomainException):
    """노드 임베딩 관련 오류."""
    
    def __init__(self, node_id: str, reason: str):
        super().__init__(
            message=f"Node embedding error for '{node_id}': {reason}",
            error_code="NODE_EMBEDDING_ERROR"
        )
        self.node_id = node_id
        self.reason = reason