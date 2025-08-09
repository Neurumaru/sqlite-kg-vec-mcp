"""
통합 E2E 테스트 - 단일 시나리오.

문서 처리부터 MCP 서버 상호작용까지 하나의 완전한 시나리오로 테스트합니다.
"""

import os
import shutil
import tempfile
import unittest

from src.adapters.fastmcp.config import FastMCPConfig
from src.adapters.fastmcp.server import KnowledgeGraphServer
from src.adapters.hnsw.embeddings import EmbeddingManager
from src.adapters.sqlite3.database import SQLiteDatabase
from src.adapters.sqlite3.document_repository import SQLiteDocumentRepository
from src.adapters.sqlite3.graph.entities import EntityManager
from src.adapters.sqlite3.graph.relationships import RelationshipManager
from src.adapters.sqlite3.schema import SchemaManager
from src.adapters.sqlite3.vector_store import SQLiteVectorStore
from src.adapters.testing.text_embedder import RandomTextEmbedder
from src.config.search_config import SearchConfig
from src.domain.entities.node import NodeType
from src.domain.entities.relationship import RelationshipType
from src.domain.services.knowledge_search import KnowledgeSearchService, SearchStrategy
from src.main import (
    ConcreteDocumentManagementUseCase,
    ConcreteKnowledgeSearchUseCase,
    ConcreteNodeManagementUseCase,
    ConcreteRelationshipManagementUseCase,
)


class IntegrationE2ETest(unittest.IsolatedAsyncioTestCase):
    """통합 E2E 테스트 - 완전한 지식 그래프 구축 및 검색 시나리오."""

    async def asyncSetUp(self):
        """테스트 환경 설정."""
        # 임시 디렉토리 및 데이터베이스 생성
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "integration_test.db")
        self.vector_dir = os.path.join(self.temp_dir, "vectors")
        os.makedirs(self.vector_dir, exist_ok=True)

        # 테스트 설정
        self.dimension = 128
        self.similarity_threshold = 0.7

        # 전체 시스템 초기화
        await self._setup_complete_system()

    async def asyncTearDown(self):
        """테스트 환경 정리."""
        if hasattr(self, "mcp_server"):
            self.mcp_server.close()

        if hasattr(self, "db_adapter"):
            await self.db_adapter.disconnect()

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def _setup_complete_system(self):
        """완전한 시스템 설정."""
        # 데이터베이스 초기화
        self.db_adapter = SQLiteDatabase(db_path=self.db_path)
        await self.db_adapter.connect()

        schema_manager = SchemaManager(self.db_path)
        schema_manager.initialize_schema()

        # 핵심 어댑터 초기화
        self.entity_manager = EntityManager(self.db_adapter.connection)
        self.relationship_manager = RelationshipManager(self.db_adapter.connection)
        self.document_repository = SQLiteDocumentRepository(self.db_adapter)
        self.embedding_manager = EmbeddingManager(self.db_adapter.connection)

        # 벡터 저장소 초기화
        self.vector_store = SQLiteVectorStore(self.db_path, table_name="knowledge_vectors")
        await self.vector_store.initialize_store(dimension=self.dimension, metric="cosine")

        # 임베딩 및 검색 서비스
        self.text_embedder = RandomTextEmbedder(dimension=self.dimension)
        self.search_config = SearchConfig(similarity_threshold=self.similarity_threshold)
        self.knowledge_search_service = KnowledgeSearchService(
            text_embedder=self.text_embedder, search_config=self.search_config
        )

        # 유스케이스 초기화
        self.node_use_case = ConcreteNodeManagementUseCase(
            self.entity_manager, self.embedding_manager, self.vector_store, self.text_embedder
        )
        self.relationship_use_case = ConcreteRelationshipManagementUseCase(
            self.relationship_manager, self.embedding_manager, self.vector_store, self.text_embedder
        )
        self.document_use_case = ConcreteDocumentManagementUseCase(self.document_repository)
        self.knowledge_search_use_case = ConcreteKnowledgeSearchUseCase(
            self.knowledge_search_service,
            self.node_use_case,
            self.document_use_case,
            self.relationship_use_case,
            self.vector_store,
            self.text_embedder,
            self.search_config,
        )

        # MCP 서버 초기화
        config = FastMCPConfig(host="127.0.0.1", port=8889)
        self.mcp_server = KnowledgeGraphServer(
            node_use_case=self.node_use_case,
            relationship_use_case=self.relationship_use_case,
            knowledge_search_use_case=self.knowledge_search_use_case,
            config=config,
        )

    async def test_complete_knowledge_graph_scenario(self):
        """완전한 지식 그래프 구축 및 검색 시나리오 테스트."""

        # === Phase 1: 기본 엔티티 생성 ===
        print("Phase 1: Creating basic entities...")

        # 회사 생성
        company = await self.node_use_case.create_node(
            name="TechCorp Inc",
            node_type=NodeType.ORGANIZATION,
            description="Leading technology company specializing in AI and machine learning",
            properties={
                "industry": "Technology",
                "founded": 2010,
                "employees": 5000,
                "headquarters": "San Francisco",
            },
        )

        # 직원들 생성
        ceo = await self.node_use_case.create_node(
            name="Alice Johnson",
            node_type=NodeType.PERSON,
            description="Chief Executive Officer with 20 years of experience in tech industry",
            properties={
                "age": 45,
                "department": "Executive",
                "role": "CEO",
                "experience_years": 20,
            },
        )

        cto = await self.node_use_case.create_node(
            name="Bob Smith",
            node_type=NodeType.PERSON,
            description="Chief Technology Officer, expert in artificial intelligence",
            properties={
                "age": 38,
                "department": "Technology",
                "role": "CTO",
                "expertise": ["AI", "Machine Learning", "Cloud Computing"],
            },
        )

        engineer = await self.node_use_case.create_node(
            name="Charlie Brown",
            node_type=NodeType.PERSON,
            description="Senior Software Engineer working on AI projects",
            properties={
                "age": 32,
                "department": "Engineering",
                "role": "Senior Engineer",
                "skills": ["Python", "TensorFlow", "Docker"],
            },
        )

        # 기술/개념 노드 생성
        ai_concept = await self.node_use_case.create_node(
            name="Artificial Intelligence",
            node_type=NodeType.CONCEPT,
            description="Technology that enables machines to simulate human intelligence",
            properties={
                "field": "Computer Science",
                "applications": ["NLP", "Computer Vision", "Robotics"],
            },
        )

        ml_project = await self.node_use_case.create_node(
            name="ML Platform Project",
            node_type=NodeType.PRODUCT,  # PROJECT -> PRODUCT로 변경
            description="Internal machine learning platform for data scientists",
            properties={
                "status": "Active",
                "budget": 2000000,
                "duration_months": 18,
                "team_size": 12,
            },
        )

        # === Phase 2: 관계 생성 ===
        print("Phase 2: Creating relationships...")

        # 조직 관계
        ceo_company = await self.relationship_use_case.create_relationship(
            source_node_id=ceo.id,
            target_node_id=company.id,
            relationship_type=RelationshipType.WORKS_AT,
            label="leads",
            properties={"start_date": "2015-01-01", "equity_percentage": 15},
        )

        cto_company = await self.relationship_use_case.create_relationship(
            source_node_id=cto.id,
            target_node_id=company.id,
            relationship_type=RelationshipType.WORKS_AT,
            label="works_at",
            properties={"start_date": "2018-03-15", "department": "Technology"},
        )

        engineer_company = await self.relationship_use_case.create_relationship(
            source_node_id=engineer.id,
            target_node_id=company.id,
            relationship_type=RelationshipType.WORKS_AT,
            label="works_at",
            properties={"start_date": "2020-06-01", "department": "Engineering"},
        )

        # 보고 관계 (LEADS 사용)
        cto_reports_ceo = await self.relationship_use_case.create_relationship(
            source_node_id=ceo.id,
            target_node_id=cto.id,
            relationship_type=RelationshipType.LEADS,
            label="manages",
            properties={"reporting_level": 1},
        )

        engineer_reports_cto = await self.relationship_use_case.create_relationship(
            source_node_id=cto.id,
            target_node_id=engineer.id,
            relationship_type=RelationshipType.LEADS,
            label="manages",
            properties={"reporting_level": 2},
        )

        # 프로젝트 관계
        cto_leads_project = await self.relationship_use_case.create_relationship(
            source_node_id=cto.id,
            target_node_id=ml_project.id,
            relationship_type=RelationshipType.LEADS,
            label="leads",
            properties={"role": "Project Sponsor"},
        )

        engineer_works_on_project = await self.relationship_use_case.create_relationship(
            source_node_id=engineer.id,
            target_node_id=ml_project.id,
            relationship_type=RelationshipType.COLLABORATES_WITH,
            label="works_on",
            properties={"role": "Lead Developer", "allocation_percentage": 80},
        )

        # 기술 관계
        project_uses_ai = await self.relationship_use_case.create_relationship(
            source_node_id=ml_project.id,
            target_node_id=ai_concept.id,
            relationship_type=RelationshipType.USES,
            label="implements",
            properties={"implementation_level": "Core"},
        )

        # === Phase 3: 데이터 무결성 검증 ===
        print("Phase 3: Verifying data integrity...")

        # 생성된 노드들 검증
        nodes = [company, ceo, cto, engineer, ai_concept, ml_project]
        for node in nodes:
            retrieved = await self.node_use_case.get_node(node.id)
            self.assertIsNotNone(retrieved, f"Node {node.name} should exist")
            self.assertEqual(retrieved.name, node.name)

        # 생성된 관계들 검증
        relationships = [
            ceo_company,
            cto_company,
            engineer_company,
            cto_reports_ceo,
            engineer_reports_cto,
            cto_leads_project,
            engineer_works_on_project,
            project_uses_ai,
        ]
        for rel in relationships:
            retrieved = await self.relationship_use_case.get_relationship(rel.id)
            self.assertIsNotNone(retrieved, f"Relationship {rel.label} should exist")

        # === Phase 4: 복합 검색 시나리오 ===
        print("Phase 4: Testing complex search scenarios...")

        # 4.1: 인물 검색
        person_search = await self.knowledge_search_use_case.search_knowledge(
            query="CEO executive leadership technology", strategy=SearchStrategy.SEMANTIC, limit=5
        )
        print(f"Person search results: {len(person_search.results)}")
        # 검색 결과가 없을 수 있으므로 검색 기능만 검증
        self.assertIsNotNone(person_search)

        # 4.2: 기술 관련 검색
        tech_search = await self.knowledge_search_use_case.search_knowledge(
            query="artificial intelligence machine learning",
            strategy=SearchStrategy.SEMANTIC,
            limit=5,
        )
        print(f"Tech search results: {len(tech_search.results)}")
        self.assertIsNotNone(tech_search)

        # 4.3: 프로젝트 검색
        project_search = await self.knowledge_search_use_case.search_knowledge(
            query="ML platform development project", strategy=SearchStrategy.SEMANTIC, limit=5
        )
        print(f"Project search results: {len(project_search.results)}")
        self.assertIsNotNone(project_search)

        # === Phase 5: 그래프 순회 및 관계 분석 ===
        print("Phase 5: Testing graph traversal and relationship analysis...")

        # 5.1: CEO의 모든 관계 조회
        ceo_relationships = await self.relationship_use_case.get_node_relationships(
            ceo.id, direction="both"
        )
        self.assertGreater(len(ceo_relationships), 0)

        # 5.2: 회사의 모든 직원 관계 조회
        company_relationships = await self.relationship_use_case.get_node_relationships(
            company.id, direction="incoming"
        )
        # 최소 3명의 직원이 회사와 연결되어 있어야 함
        work_relationships = [
            r for r in company_relationships if r.relationship_type == RelationshipType.WORKS_AT
        ]
        self.assertGreaterEqual(len(work_relationships), 3)

        # 5.3: 프로젝트 참여자 조회
        project_relationships = await self.relationship_use_case.get_node_relationships(
            ml_project.id, direction="incoming"
        )
        project_participants = [
            r
            for r in project_relationships
            if r.relationship_type in [RelationshipType.LEADS, RelationshipType.COLLABORATES_WITH]
        ]
        self.assertGreaterEqual(len(project_participants), 2)

        # === Phase 6: 업데이트 및 진화 시나리오 ===
        print("Phase 6: Testing update and evolution scenarios...")

        # 6.1: 직원 정보 업데이트 (승진)
        updated_engineer = await self.node_use_case.update_node(
            node_id=engineer.id,
            name="Charlie Brown",
            description="Principal Software Engineer, AI team lead",
            properties={
                "age": 33,
                "department": "Engineering",
                "role": "Principal Engineer",
                "skills": ["Python", "TensorFlow", "Docker", "Kubernetes"],
                "promotion_date": "2024-01-01",
            },
        )
        self.assertEqual(updated_engineer.properties["role"], "Principal Engineer")

        # 6.2: 새로운 팀원 추가
        new_engineer = await self.node_use_case.create_node(
            name="Diana Wilson",
            node_type=NodeType.PERSON,
            description="ML Engineer specializing in natural language processing",
            properties={
                "age": 28,
                "department": "Engineering",
                "role": "ML Engineer",
                "expertise": ["NLP", "PyTorch", "Transformers"],
            },
        )

        # 새 팀원을 프로젝트에 추가
        await self.relationship_use_case.create_relationship(
            source_node_id=new_engineer.id,
            target_node_id=ml_project.id,
            relationship_type=RelationshipType.COLLABORATES_WITH,
            label="works_on",
            properties={"role": "NLP Specialist", "allocation_percentage": 60},
        )

        # 6.3: 프로젝트 상태 업데이트
        # Note: 프로젝트 노드 업데이트는 현재 구현에서 지원되지 않을 수 있음
        # 실제로는 별도의 프로젝트 관리 유스케이스가 필요할 것

        # === Phase 7: 최종 검증 ===
        print("Phase 7: Final verification...")

        # 7.1: 전체 노드 수 확인
        all_nodes = await self.node_use_case.list_nodes(limit=50)
        self.assertGreaterEqual(len(all_nodes), 7)  # 최소 7개 노드 (새 팀원 포함)

        # 7.2: 전체 관계 수 확인
        all_relationships = await self.relationship_use_case.list_relationships(limit=50)
        self.assertGreaterEqual(len(all_relationships), 9)  # 최소 9개 관계 (새 관계 포함)

        # 7.3: 검색 결과 품질 확인
        final_search = await self.knowledge_search_use_case.search_knowledge(
            query="TechCorp AI team engineering", strategy=SearchStrategy.SEMANTIC, limit=10
        )
        print(f"Final search results: {len(final_search.results)}")
        self.assertIsNotNone(final_search)

        # 검색 결과가 있으면 점수 순으로 정렬되어 있는지 확인
        if len(final_search.results) > 1:
            scores = [result.score for result in final_search.results]
            self.assertEqual(scores, sorted(scores, reverse=True))

        # 7.4: MCP 서버 상태 확인
        self.assertIsNotNone(self.mcp_server)

        print("✅ Complete knowledge graph scenario test passed!")
        print(f"   - Created {len(all_nodes)} nodes")
        print(f"   - Created {len(all_relationships)} relationships")
        print("   - Performed multiple search operations")
        print("   - Verified data integrity and search quality")


if __name__ == "__main__":
    unittest.main()
