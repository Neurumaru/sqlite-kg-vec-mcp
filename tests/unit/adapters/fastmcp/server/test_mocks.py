"""
FastMCP Server 테스트용 공통 Mock 클래스들.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

from src.dto.node import NodeData, NodeType
from src.dto.relationship import RelationshipData, RelationshipType


def create_test_entity(
    entity_id="1", name="Test", entity_type=NodeType.CONCEPT, properties=None
) -> NodeData:
    """NodeData DTO를 사용하여 테스트 엔티티를 생성합니다."""
    return NodeData(id=entity_id, name=name, node_type=entity_type, properties=properties or {})


# 이전 버전과의 호환성을 위한 별칭
MockEntity = create_test_entity


def create_test_relationship(
    relationship_id="1",
    source_id="1",
    target_id="2",
    relation_type=RelationshipType.RELATES_TO,
    properties=None,
) -> RelationshipData:
    """RelationshipData DTO를 사용하여 테스트 관계를 생성합니다."""
    return RelationshipData(
        id=relationship_id,
        source_node_id=source_id,
        target_node_id=target_id,
        relationship_type=relation_type,
        properties=properties or {},
    )


# 이전 버전과의 호환성을 위한 별칭
MockRelationship = create_test_relationship


class MockSearchResult:
    """테스트용 모의 검색 결과입니다."""

    def __init__(self, node_id=1, similarity=0.8):
        self.node_id = node_id
        self.similarity = similarity

    def to_dict(self):
        """검색 결과를 딕셔너리 형식으로 변환합니다."""
        return {"node_id": self.node_id, "similarity": self.similarity}


class MockContext:
    """테스트용 모의 컨텍스트입니다."""

    def __init__(self):
        self.info_calls = []
        self.error_calls = []

    def info(self, message):
        """정보 메시지를 기록합니다."""
        self.info_calls.append(message)

    def error(self, message):
        """오류 메시지를 기록합니다."""
        self.error_calls.append(message)


class BaseServerTestCase:
    """FastMCP Server 테스트용 공통 기본 클래스."""

    def setUp(self):
        """각 테스트 메서드 전에 테스트 픽스처를 설정합니다."""
        # 임시 데이터베이스 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as temp_db:
            self.db_path = temp_db.name

        # 관리자 모의 객체
        self.entity_manager = Mock()
        self.relationship_manager = Mock()
        self.vector_search = Mock()

    def tearDown(self):
        """각 테스트 메서드 후에 정리합니다."""
        # 임시 데이터베이스 파일 삭제
        try:
            Path(self.db_path).unlink()
        except FileNotFoundError:
            pass
