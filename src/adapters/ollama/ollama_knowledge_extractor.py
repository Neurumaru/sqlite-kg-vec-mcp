"""
Ollama 기반 지식 추출 어댑터.

이 모듈은 자동 지식 그래프 구축을 위해 Ollama LLM을 사용하는
KnowledgeExtractor 포트의 구체적인 구현을 제공합니다.
"""

import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Optional, cast

from src.adapters.hnsw.embeddings import EmbeddingManager
from src.adapters.sqlite3.graph.entities import EntityManager
from src.adapters.sqlite3.graph.relationships import RelationshipManager
from src.config.embedding_config import EmbeddingConfig
from src.config.search_config import SearchConfig
from src.domain.entities.node import Node
from src.domain.entities.relationship import Relationship, RelationshipType
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.dto import DocumentData
from src.dto.node import NodeData, NodeType
from src.dto.relationship import (
    RelationshipData,
)
from src.dto.relationship import RelationshipType as DTORelationshipType
from src.ports.knowledge_extractor import KnowledgeExtractor

from .ollama_client import OllamaClient


@dataclass
class ExtractionResult:
    """지식 추출 결과를 담는 데이터클래스."""

    entities_created: int = 0
    relationships_created: int = 0
    errors: Optional[list[str]] = None
    processing_time: float = 0.0

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class OllamaKnowledgeExtractor(KnowledgeExtractor):
    """Ollama LLM 기반 지식 추출 어댑터 구현."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        ollama_client: OllamaClient,
        auto_embed: bool = True,
        search_config: Optional[SearchConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
    ):
        """
        지식 추출기를 초기화합니다.

        Args:
            connection: SQLite 데이터베이스 연결
            ollama_client: LLM 작업을 위한 Ollama 클라이언트
            auto_embed: 임베딩을 자동으로 생성할지 여부
            search_config: 검색 설정 (없으면 기본값 사용)
            embedding_config: 임베딩 및 배치 설정 (없으면 기본값 사용)
        """
        self.connection = connection
        self.ollama_client = ollama_client
        self.auto_embed = auto_embed
        self.search_config = search_config or SearchConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()

        # 관리자 초기화
        self.entity_manager = EntityManager(connection)
        self.relationship_manager = RelationshipManager(connection)
        self.embedding_manager = EmbeddingManager(connection) if auto_embed else None

        # 일괄 처리를 위한 엔티티 ID 매핑
        self.entity_id_mapping: dict[str, int] = {}

    def extract_from_text(
        self,
        text: str,
        source_id: Optional[str] = None,
        enhance_descriptions: bool = True,
    ) -> ExtractionResult:
        """
        텍스트에서 지식 그래프를 추출합니다.

        Args:
            text: 처리할 입력 텍스트
            source_id: 추적을 위한 선택적 소스 식별자
            enhance_descriptions: LLM으로 엔티티 설명을 향상시킬지 여부

        Returns:
            통계 및 오류가 포함된 ExtractionResult
        """
        start_time = time.time()

        entities_created = 0
        relationships_created = 0
        errors: list[str] = []

        try:
            # LLM을 사용하여 엔티티 및 관계 추출
            logging.info("텍스트에서 지식 추출 중 (%s 문자)...", len(text))
            extraction_data = self.ollama_client.extract_entities_and_relationships(text)

            # 엔티티 처리
            entities_created = self._process_entities(
                extraction_data.get("entities", []),
                source_id,
                enhance_descriptions,
                errors,
            )

            # 관계 처리
            relationships_created = self._process_relationships(
                extraction_data.get("relationships", []), errors
            )

            # 활성화된 경우 임베딩 생성
            if self.auto_embed and self.embedding_manager:
                try:
                    processed_count = self.embedding_manager.process_outbox()
                    logging.info("%s개 엔티티에 대한 임베딩 생성됨", processed_count)
                except (ValueError, RuntimeError, sqlite3.Error) as exception:
                    error_msg = f"임베딩 생성 실패: {exception}"
                    logging.error(error_msg)
                    errors.append(error_msg)

        except (
            ValueError,
            RuntimeError,
            sqlite3.Error,
            ConnectionError,
            TimeoutError,
        ) as exception:
            error_msg = f"지식 추출 실패: {exception}"
            logging.error(error_msg)
            errors.append(error_msg)

        processing_time = time.time() - start_time

        result = ExtractionResult(
            entities_created=entities_created,
            relationships_created=relationships_created,
            errors=errors,
            processing_time=processing_time,
        )

        logging.info(
            "추출 완료: 엔티티 %s개, 관계 %s개, 처리 시간 %.2fs",
            entities_created,
            relationships_created,
            processing_time,
        )

        return result

    def _process_entities(
        self,
        entities: list[dict[str, Any]],
        source_id: Optional[str],
        enhance_descriptions: bool,
        errors: list[str],
    ) -> int:
        """추출 데이터에서 엔티티를 처리하고 생성합니다."""
        created_count = 0

        for entity_data in entities:
            try:
                # 필수 필드 유효성 검사
                if "name" not in entity_data or "type" not in entity_data:
                    errors.append(f"엔티티에 필수 필드가 없습니다: {entity_data}")
                    continue

                # 속성 준비
                properties = entity_data.get("properties", {})
                if source_id:
                    properties["source_id"] = source_id

                # 요청된 경우 설명 향상
                if enhance_descriptions:
                    try:
                        enhanced_desc = self.ollama_client.generate_embeddings_description(
                            entity_data
                        )
                        properties["llm_description"] = enhanced_desc
                    except (ConnectionError, TimeoutError, ValueError) as exception:
                        logging.warning(
                            "%s에 대한 설명 향상 실패: %s",
                            entity_data["name"],
                            exception,
                        )

                # 엔티티 생성
                entity = self.entity_manager.create_entity(
                    entity_type=entity_data["type"],
                    name=entity_data["name"],
                    properties=properties,
                    custom_uuid=entity_data.get("id"),
                )

                # 관계 처리를 위한 매핑 저장
                extraction_id = entity_data.get("id", entity_data["name"])
                self.entity_id_mapping[extraction_id] = entity.id

                created_count += 1
                logging.debug("생성된 엔티티: %s (%s)", entity.name, entity.type)

            except (sqlite3.Error, ValueError, KeyError) as exception:
                error_msg = f"엔티티 {entity_data.get('name', 'unknown')} 생성 실패: {exception}"
                logging.error(error_msg)
                errors.append(error_msg)

        return created_count

    def _process_relationships(self, relationships: list[dict[str, Any]], errors: list[str]) -> int:
        """추출 데이터에서 관계를 처리하고 생성합니다."""
        created_count = 0

        for rel_data in relationships:
            try:
                # 필수 필드 유효성 검사
                if not all(field in rel_data for field in ["source", "target", "type"]):
                    errors.append(f"관계에 필수 필드가 없습니다: {rel_data}")
                    continue

                # 엔티티 ID 확인
                source_id = self.entity_id_mapping.get(rel_data["source"])
                target_id = self.entity_id_mapping.get(rel_data["target"])

                current_rel_errors = []
                if source_id is None:
                    current_rel_errors.append(
                        f"관계에 대한 소스 엔티티를 찾을 수 없습니다: {rel_data['source']}"
                    )

                if target_id is None:
                    current_rel_errors.append(
                        f"관계에 대한 대상 엔티티를 찾을 수 없습니다: {rel_data['target']}"
                    )

                if current_rel_errors:
                    errors.extend(current_rel_errors)
                    continue

                # 관계 생성
                self.relationship_manager.create_relationship(
                    source_id=cast(int, source_id),
                    target_id=cast(int, target_id),
                    relation_type=rel_data["type"],
                    properties=rel_data.get("properties", {}),
                )

                created_count += 1
                logging.debug(
                    "생성된 관계: %s --%s--> %s",
                    rel_data["source"],
                    rel_data["type"],
                    rel_data["target"],
                )

            except (sqlite3.Error, ValueError, KeyError) as exception:
                error_msg = f"관계 {rel_data.get('type', 'unknown')} 생성 실패: {exception}"
                logging.error(error_msg)
                errors.append(error_msg)

        return created_count

    def extract_from_documents(
        self, documents: list[dict[str, str]], batch_size: Optional[int] = None
    ) -> list[ExtractionResult]:
        """
        여러 문서를 배치로 처리하여 지식을 추출합니다.

        Args:
            documents: 'text' 및 선택적 'id' 필드가 있는 문서 목록
            batch_size: 각 배치에서 처리할 문서 수 (None이면 설정값 사용)

        Returns:
            각 문서에 대한 ExtractionResult 목록
        """
        if batch_size is None:
            batch_size = self.embedding_config.knowledge_extraction_batch_size

        results = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            logging.info(
                "배치 처리 중 %s/%s",
                i // batch_size + 1,
                (len(documents) + batch_size - 1) // batch_size,
            )

            for doc in batch:
                doc_id = doc.get("id", f"doc_{i}")
                text = doc.get("text", "")

                if not text.strip():
                    continue

                result = self.extract_from_text(text, source_id=doc_id)
                results.append(result)

                # 진행 상황 기록
                if result.errors:
                    logging.warning("문서 %s에 %s개의 오류가 있습니다.", doc_id, len(result.errors))

        return results

    def get_extraction_statistics(self) -> dict[str, Any]:
        """지식 그래프에 대한 통계를 가져옵니다."""
        cursor = self.connection.cursor()

        # 엔티티 통계
        cursor.execute("SELECT type, COUNT(*) FROM entities GROUP BY type")
        entity_stats = dict(cursor.fetchall())

        cursor.execute("SELECT COUNT(*) FROM entities")
        total_entities = cursor.fetchone()[0]

        # 관계 통계
        cursor.execute("SELECT relation_type, COUNT(*) FROM edges GROUP BY relation_type")
        relationship_stats = dict(cursor.fetchall())

        cursor.execute("SELECT COUNT(*) FROM edges")
        total_relationships = cursor.fetchone()[0]

        # 사용 가능한 경우 임베딩 통계
        embedding_stats = {}
        if self.embedding_manager:
            try:
                cursor.execute("SELECT COUNT(*) FROM vector_embeddings")
                total_embeddings = cursor.fetchone()[0]
                embedding_stats["total_embeddings"] = total_embeddings
            except sqlite3.Error:
                embedding_stats["total_embeddings"] = 0

        return {
            "entities": {"total": total_entities, "by_type": entity_stats},
            "relationships": {
                "total": total_relationships,
                "by_type": relationship_stats,
            },
            "embeddings": embedding_stats,
            "model": self.ollama_client.model,
        }

    # 포트 인터페이스 구현 메서드

    async def extract_knowledge(self, text: str) -> ExtractionResult:
        """Ollama LLM을 사용하여 텍스트에서 엔티티와 관계를 추출합니다."""
        return self.extract_from_text(text)

    async def extract_entities(self, text: str) -> list[Node]:
        """텍스트에서 엔티티만 추출합니다."""
        # 기존 동기 메서드를 사용하여 엔티티 추출
        extraction_data = await asyncio.to_thread(
            self.ollama_client.extract_entities_and_relationships, text
        )
        entities = []

        for entity_data in extraction_data.get("entities", []):
            try:
                entity = Node(
                    id=NodeId.generate(),
                    name=entity_data.get("name", "Unknown"),
                    node_type=entity_data.get("type", "Unknown"),
                    properties=entity_data.get("properties", {}),
                )
                entities.append(entity)
            except (ValueError, KeyError) as exception:
                logging.warning("데이터 %s에서 엔티티 생성 실패: %s", entity_data, exception)

        return entities

    async def extract_relationships(self, text: str, entities: list[Node]) -> list[Relationship]:
        """기존 엔티티가 주어진 텍스트에서 관계를 추출합니다."""
        # 기존 동기 메서드를 사용하여 관계 추출
        extraction_data = await asyncio.to_thread(
            self.ollama_client.extract_entities_and_relationships, text
        )
        relationships = []

        # 참조를 위한 엔티티 이름 대 ID 매핑 생성
        entity_map = {entity.name: entity.id for entity in entities}

        for rel_data in extraction_data.get("relationships", []):
            try:
                # 엔티티 참조 확인 시도
                source_id = entity_map.get(rel_data.get("source"))
                target_id = entity_map.get(rel_data.get("target"))

                if source_id is not None and target_id is not None:
                    rel_type_str = rel_data.get("type", "RELATED_TO").upper()
                    rel_type = getattr(RelationshipType, rel_type_str, RelationshipType.OTHER)

                    try:
                        relationship = Relationship(
                            id=RelationshipId.generate(),
                            source_node_id=NodeId(str(cast(int, source_id))),
                            target_node_id=NodeId(str(cast(int, target_id))),
                            relationship_type=rel_type,
                            label=rel_data.get("type", "RELATED_TO"),
                            properties=rel_data.get("properties", {}),
                        )
                        relationships.append(relationship)
                    except (ValueError, KeyError) as exception:
                        logging.warning("데이터 %s에서 관계 생성 실패: %s", rel_data, exception)

            except (ValueError, KeyError) as exception:
                logging.warning("데이터 %s에서 관계 처리 실패: %s", rel_data, exception)

        return relationships

    async def validate_extraction(self, text: str, result: ExtractionResult) -> bool:
        """추출 결과의 품질을 검증합니다."""
        # 추출 성공 여부에 기반한 간단한 검증
        if result.errors:
            error_ratio = len(result.errors) / max(
                1, result.entities_created + result.relationships_created
            )
            return error_ratio < 0.5  # 50% 미만의 오류율

        # 의미 있는 것을 추출했는지 확인
        return result.entities_created > 0 or result.relationships_created > 0

    async def get_extraction_confidence(self, text: str) -> float:
        """주어진 텍스트에 대한 추출 능력 신뢰도 점수를 가져옵니다."""
        # 텍스트 길이 및 내용에 기반한 간단한 휴리스틱
        if not text or len(text.strip()) < 10:
            return 0.0

        # 긴 텍스트는 일반적으로 더 나은 추출 기회를 제공합니다.
        length_score = min(len(text) / 1000, 1.0)  # 1000자로 정규화

        # 구조화된 콘텐츠 지표 확인
        structure_indicators = [
            ".",
            ":",
            ";",
            ",",  # 구두점
            "is",
            "was",
            "are",
            "were",  # 연결 동사
            "the",
            "a",
            "an",  # 관사
        ]

        structure_score = sum(
            1 for indicator in structure_indicators if indicator in text.lower()
        ) / len(structure_indicators)

        # 점수 결합
        confidence = (
            length_score * self.search_config.confidence_length_weight
            + structure_score * self.search_config.confidence_structure_weight
        )
        return min(confidence, 1.0)

    # KnowledgeExtractor 추상 메서드 구현

    async def extract(
        self, document: DocumentData
    ) -> tuple[list[NodeData], list[RelationshipData]]:
        """
        문서에서 지식(노드와 관계)을 추출합니다.

        Args:
            document: 분석할 문서 데이터

        Returns:
            (노드 데이터 리스트, 관계 데이터 리스트) 튜플
        """
        # 문서 텍스트에서 지식 추출
        extraction_data = await asyncio.to_thread(
            self.ollama_client.extract_entities_and_relationships, document.content
        )

        # 추출된 엔티티를 NodeData로 변환
        nodes: list[NodeData] = []
        for entity_data in extraction_data.get("entities", []):
            node_data = NodeData(
                id=entity_data.get("id", f"node_{len(nodes)}"),
                name=entity_data.get("name", "Unknown"),
                node_type=NodeType.CONCEPT,  # 기본값으로 CONCEPT 타입 사용
                properties=entity_data.get("properties", {}),
            )
            nodes.append(node_data)

        # 추출된 관계를 RelationshipData로 변환
        relationships: list[RelationshipData] = []
        for rel_data in extraction_data.get("relationships", []):
            relationship_data = RelationshipData(
                id=rel_data.get("id", f"rel_{len(relationships)}"),
                source_node_id=rel_data.get("source", ""),
                target_node_id=rel_data.get("target", ""),
                relationship_type=DTORelationshipType.RELATES_TO,  # 기본값으로 RELATES_TO 타입 사용
                properties=rel_data.get("properties", {}),
            )
            relationships.append(relationship_data)

        return nodes, relationships

    async def is_available(self) -> bool:
        """
        지식 추출 서비스가 사용 가능한지 확인합니다.

        Returns:
            사용 가능 여부
        """
        try:
            # 간단한 요청으로 테스트
            test_response = await asyncio.to_thread(
                self.ollama_client.generate, prompt="Test", max_tokens=5
            )
            return test_response is not None
        except (ConnectionError, TimeoutError):
            return False
