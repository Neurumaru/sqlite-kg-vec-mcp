"""
지식 그래프 탐색을 위한 Streamlit 웹 애플리케이션.
"""

import argparse
from typing import Any

import streamlit as st

from src.adapters.hnsw.search import VectorSearch
from src.adapters.sqlite3.connection import DatabaseConnection
from src.adapters.sqlite3.exceptions import SQLiteConnectionException
from src.adapters.sqlite3.graph.entities import EntityManager
from src.adapters.sqlite3.graph.relationships import RelationshipManager
from src.adapters.sqlite3.schema import SchemaManager


class KnowledgeGraphStreamlitApp:
    """지식 그래프 탐색을 위한 Streamlit 애플리케이션."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.setup_page_config()
        self.init_database()

    def setup_page_config(self):
        """Streamlit 페이지 설정을 구성합니다."""
        st.set_page_config(page_title="Knowledge Graph Explorer", page_icon="🧠", layout="wide")

    def init_database(self):
        """데이터베이스 연결 및 관리자를 초기화합니다."""
        try:
            self.db_connection = DatabaseConnection(self.db_path)
            connection = self.db_connection.connect()
            self.schema_manager = SchemaManager(self.db_path)
            self.entity_manager = EntityManager(connection)
            self.relationship_manager = RelationshipManager(connection)
            self.vector_search = VectorSearch(
                connection=connection,
                index_dir="vector_indexes",
            )
        except SQLiteConnectionException as exception:
            st.error(f"데이터베이스 연결 실패: {exception.message}")
            st.stop()
        except (OSError, PermissionError) as exception:
            st.error(f"데이터베이스 파일 접근 실패: {exception}")
            st.stop()
        except Exception as exception:
            st.error(f"데이터베이스 초기화 중 예상치 못한 오류: {exception}")
            st.stop()

    def run(self):
        """Streamlit 애플리케이션을 실행합니다."""
        st.title("🧠 Knowledge Graph Explorer")

        # 사이드바 메뉴
        menu = st.sidebar.selectbox(
            "메뉴 선택", ["📊 Dashboard", "🔍 Search", "🏷️ Entities", "🔗 Relationships"]
        )

        if menu == "📊 Dashboard":
            self.render_dashboard()
        elif menu == "🔍 Search":
            self.render_search()
        elif menu == "🏷️ Entities":
            self.render_entities()
        elif menu == "🔗 Relationships":
            self.render_relationships()

    def render_dashboard(self):
        """통계가 포함된 대시보드를 렌더링합니다."""
        st.header("📊 Dashboard")

        try:
            # 기본 통계 가져오기
            entity_count = self.get_entity_count()
            relationship_count = self.get_relationship_count()

            # 메트릭 표시
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Entities", entity_count)

            with col2:
                st.metric("Total Relationships", relationship_count)

            with col3:
                density = relationship_count / max(entity_count, 1)
                st.metric("Graph Density", f"{density:.2f}")

            # 엔티티 유형 분포
            st.subheader("Entity Types Distribution")
            entity_types = self.get_entity_types_distribution()
            if entity_types:
                st.bar_chart(entity_types)
            else:
                st.info("데이터베이스에서 엔티티를 찾을 수 없습니다.")

        except SQLiteConnectionException as exception:
            st.error(f"데이터베이스 연결 오류: {exception.message}")
        except Exception as exception:
            st.error(f"대시보드 로딩 중 예상치 못한 오류: {exception}")

    def render_search(self):
        """검색 인터페이스를 렌더링합니다."""
        st.header("🔍 Search")

        # 검색 입력
        search_query = st.text_input("검색어를 입력하세요:")

        if search_query:
            try:
                # 텍스트 기반 검색
                st.subheader("Text Search Results")
                text_results = self.search_entities_by_text(search_query)

                if text_results:
                    for result in text_results:
                        with st.expander(f"Entity: {result.get('name', 'Unnamed')}"):
                            st.json(result)
                else:
                    st.info("검색 결과가 없습니다.")

                # 벡터 기반 시맨틱 검색 (사용 가능한 경우)
                st.subheader("Semantic Search Results")
                if st.button("벡터 검색 실행"):
                    semantic_results = self.search_entities_semantic(search_query)
                    if semantic_results:
                        for result in semantic_results:
                            with st.expander(
                                f"Entity: {result.get('name', 'Unnamed')} (Score: "
                                f"{result.get('score', 0):.3f})"
                            ):
                                st.json(result)
                    else:
                        st.info("시맨틱 검색 결과가 없습니다.")

            except SQLiteConnectionException as exception:
                st.error(f"데이터베이스 연결 오류: {exception.message}")
            except ValueError as exception:
                st.error(f"검색 입력값 오류: {exception}")
            except Exception as exception:
                st.error(f"검색 중 예상치 못한 오류: {exception}")

    def render_entities(self):
        """엔티티 관리 인터페이스를 렌더링합니다."""
        st.header("🏷️ Entities")

        tab1, tab2 = st.tabs(["Entity List", "Create Entity"])

        with tab1:
            try:
                entities = self.get_all_entities()

                if entities:
                    st.subheader(f"Total Entities: {len(entities)}")

                    # 엔티티 필터
                    entity_types = list({e.get("type", "Unknown") for e in entities})
                    selected_types = st.multiselect(
                        "Entity Types Filter", entity_types, default=entity_types
                    )

                    # 필터링된 엔티티
                    filtered_entities = [
                        e for e in entities if e.get("type", "Unknown") in selected_types
                    ]

                    for entity in filtered_entities:
                        with st.expander(
                            f"{entity.get('name', 'Unnamed')} ({entity.get('type', 'Unknown')})"
                        ):
                            st.json(entity)

                            # 삭제 버튼
                            if st.button(
                                f"Delete {entity.get('id', '')}",
                                key=f"delete_entity_{entity.get('id', '')}",
                            ):
                                entity_id = entity.get("id")
                                if entity_id is not None:
                                    try:
                                        self.delete_entity(int(entity_id))
                                        st.success("Entity deleted!")
                                        st.rerun()
                                    except ValueError:
                                        st.error("Invalid entity ID format")
                                    except Exception as exception:
                                        st.error(f"Entity 삭제 실패: {exception}")
                                else:
                                    st.error("Entity ID가 없습니다")
                else:
                    st.info("엔티티를 찾을 수 없습니다.")

            except SQLiteConnectionException as exception:
                st.error(f"데이터베이스 연결 오류: {exception.message}")
            except Exception as exception:
                st.error(f"엔티티 로딩 중 예상치 못한 오류: {exception}")

        with tab2:
            st.subheader("Create New Entity")

            with st.form("create_entity_form"):
                name = st.text_input("Entity Name")
                entity_type = st.text_input("Entity Type")
                description = st.text_area("Description")

                submitted = st.form_submit_button("Create Entity")

                if submitted:
                    if not name or not name.strip():
                        st.error("Entity name은 필수입니다")
                    elif not entity_type or not entity_type.strip():
                        st.error("Entity type은 필수입니다")
                    else:
                        try:
                            properties = {"description": description} if description else {}
                            self.create_entity(name.strip(), entity_type.strip(), properties)
                            st.success(f"Entity '{name}' created successfully!")
                        except SQLiteConnectionException as exception:
                            st.error(f"데이터베이스 연결 오류: {exception.message}")
                        except ValueError as exception:
                            st.error(f"입력값 오류: {exception}")
                        except Exception as exception:
                            st.error(f"엔티티 생성 실패: {exception}")

    def render_relationships(self):
        """관계 관리 인터페이스를 렌더링합니다."""
        st.header("🔗 Relationships")

        tab1, tab2 = st.tabs(["Relationship List", "Create Relationship"])

        with tab1:
            try:
                relationships = self.get_all_relationships()

                if relationships:
                    st.subheader(f"Total Relationships: {len(relationships)}")

                    for rel in relationships:
                        with st.expander(
                            f"{rel.get('source_name', 'Unknown')} -> "
                            f"{rel.get('target_name', 'Unknown')} ("
                            f"{rel.get('type', 'Unknown')})"
                        ):
                            st.json(rel)

                            # 삭제 버튼
                            if st.button(
                                f"Delete {rel.get('id', '')}",
                                key=f"delete_rel_{rel.get('id', '')}",
                            ):
                                rel_id = rel.get("id")
                                if rel_id is not None:
                                    try:
                                        self.delete_relationship(int(rel_id))
                                        st.success("Relationship deleted!")
                                        st.rerun()
                                    except ValueError:
                                        st.error("Invalid relationship ID format")
                                    except Exception as exception:
                                        st.error(f"Relationship 삭제 실패: {exception}")
                                else:
                                    st.error("Relationship ID가 없습니다")
                else:
                    st.info("관계를 찾을 수 없습니다.")

            except SQLiteConnectionException as exception:
                st.error(f"데이터베이스 연결 오류: {exception.message}")
            except Exception as exception:
                st.error(f"관계 로딩 중 예상치 못한 오류: {exception}")

        with tab2:
            st.subheader("Create New Relationship")

            try:
                entities = self.get_all_entities()
                entity_options = {
                    f"{e.get('name', 'Unnamed')} ({e.get('id', '')})": e.get("id", "")
                    for e in entities
                }

                if len(entity_options) >= 2:
                    with st.form("create_relationship_form"):
                        source_entity = st.selectbox("Source Entity", list(entity_options.keys()))
                        target_entity = st.selectbox("Target Entity", list(entity_options.keys()))
                        relationship_type = st.text_input("Relationship Type")
                        description = st.text_area("Description")

                        submitted = st.form_submit_button("Create Relationship")

                        if submitted:
                            if not source_entity or not target_entity:
                                st.error("Source와 Target entity를 모두 선택해주세요")
                            elif not relationship_type or not relationship_type.strip():
                                st.error("Relationship type은 필수입니다")
                            elif source_entity == target_entity:
                                st.error("Source와 Target entity는 달라야 합니다")
                            else:
                                try:
                                    source_id = entity_options[source_entity]
                                    target_id = entity_options[target_entity]
                                    properties = {"description": description} if description else {}

                                    self.create_relationship(
                                        source_id,
                                        target_id,
                                        relationship_type.strip(),
                                        properties,
                                    )
                                    st.success("Relationship created successfully!")
                                except KeyError as exception:
                                    st.error(f"선택된 entity를 찾을 수 없습니다: {exception}")
                                except SQLiteConnectionException as exception:
                                    st.error(f"데이터베이스 연결 오류: {exception.message}")
                                except ValueError as exception:
                                    st.error(f"입력값 오류: {exception}")
                                except Exception as exception:
                                    st.error(f"관계 생성 실패: {exception}")
                else:
                    st.info("최소 2개의 엔티티가 필요합니다.")

            except SQLiteConnectionException as exception:
                st.error(f"데이터베이스 연결 오류: {exception.message}")
            except Exception as exception:
                st.error(f"관계 생성 폼 로딩 중 예상치 못한 오류: {exception}")

    # 헬퍼 메서드
    def get_entity_count(self) -> int:
        """총 엔티티 수를 가져옵니다."""
        if not self.db_connection.connection:
            raise SQLiteConnectionException(
                db_path=self.db_path, message="데이터베이스 연결이 설정되지 않았습니다"
            )
        cursor = self.db_connection.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM entities")
        result = cursor.fetchone()
        return result[0] if result else 0

    def get_relationship_count(self) -> int:
        """총 관계 수를 가져옵니다."""
        if not self.db_connection.connection:
            raise SQLiteConnectionException(
                db_path=self.db_path, message="데이터베이스 연결이 설정되지 않았습니다"
            )
        cursor = self.db_connection.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM relationships")
        result = cursor.fetchone()
        return result[0] if result else 0

    def get_entity_types_distribution(self) -> dict[str, int]:
        """엔티티 유형의 분포를 가져옵니다."""
        if not self.db_connection.connection:
            raise SQLiteConnectionException(
                db_path=self.db_path, message="데이터베이스 연결이 설정되지 않았습니다"
            )
        cursor = self.db_connection.connection.cursor()
        cursor.execute("SELECT type, COUNT(*) as count FROM entities GROUP BY type")
        return {row[0] or "Unknown": row[1] for row in cursor.fetchall()}

    def get_all_entities(self) -> list[dict[str, Any]]:
        """모든 엔티티를 가져옵니다."""
        if not self.db_connection.connection:
            raise SQLiteConnectionException(
                db_path=self.db_path, message="데이터베이스 연결이 설정되지 않았습니다"
            )
        cursor = self.db_connection.connection.cursor()
        cursor.execute("SELECT id, name, type, properties FROM entities")
        entities = []
        for row in cursor.fetchall():
            entities.append({"id": row[0], "name": row[1], "type": row[2], "properties": row[3]})
        return entities

    def get_all_relationships(self) -> list[dict[str, Any]]:
        """엔티티 이름과 함께 모든 관계를 가져옵니다."""
        if not self.db_connection.connection:
            raise SQLiteConnectionException(
                db_path=self.db_path, message="데이터베이스 연결이 설정되지 않았습니다"
            )
        cursor = self.db_connection.connection.cursor()
        cursor.execute(
            """
            SELECT r.id, r.source_id, r.target_id, r.type, r.properties,
                   e1.name as source_name, e2.name as target_name
            FROM relationships r
            LEFT JOIN entities e1 ON r.source_id = e1.id
            LEFT JOIN entities e2 ON r.target_id = e2.id
        """
        )
        relationships = []
        for row in cursor.fetchall():
            relationships.append(
                {
                    "id": row[0],
                    "source_id": row[1],
                    "target_id": row[2],
                    "type": row[3],
                    "properties": row[4],
                    "source_name": row[5],
                    "target_name": row[6],
                }
            )
        return relationships

    def search_entities_by_text(self, query: str) -> list[dict[str, Any]]:
        """텍스트로 엔티티를 검색합니다."""
        if not query or not query.strip():
            raise ValueError("검색어는 비워 둘 수 없습니다")

        if not self.db_connection.connection:
            raise SQLiteConnectionException(
                db_path=self.db_path, message="데이터베이스 연결이 설정되지 않았습니다"
            )

        query = query.strip()
        cursor = self.db_connection.connection.cursor()
        cursor.execute(
            """
            SELECT id, name, type, properties
            FROM entities
            WHERE name LIKE ? OR type LIKE ? OR properties LIKE ?
        """,
            (f"%{query}%", f"%{query}%", f"%{query}%"),
        )

        results = []
        for row in cursor.fetchall():
            results.append({"id": row[0], "name": row[1], "type": row[2], "properties": row[3]})
        return results

    def search_entities_semantic(self, query: str) -> list[dict[str, Any]]:
        """시맨틱 유사성을 사용하여 엔티티를 검색합니다."""
        try:
            # 여기서는 벡터 검색 기능을 사용합니다.
            # 지금은 자리 표시자로 빈 목록을 반환합니다.
            return []
        except Exception as exception:
            st.error(f"시맨틱 검색 중 예상치 못한 오류: {exception}")
            return []

    def create_entity(self, name: str, entity_type: str, properties: dict[str, Any]):
        """새 엔티티를 생성합니다."""
        if not name or not name.strip():
            raise ValueError("엔티티 이름은 비워 둘 수 없습니다")
        if not entity_type or not entity_type.strip():
            raise ValueError("엔티티 유형은 비워 둘 수 없습니다")

        entity = self.entity_manager.create_entity(name.strip(), entity_type.strip(), properties)
        return entity

    def create_relationship(
        self, source_id: str, target_id: str, rel_type: str, properties: dict[str, Any]
    ):
        """새 관계를 생성합니다."""
        if not source_id or not target_id:
            raise ValueError("소스 및 대상 ID는 비워 둘 수 없습니다")
        if not rel_type or not rel_type.strip():
            raise ValueError("관계 유형은 비워 둘 수 없습니다")
        if source_id == target_id:
            raise ValueError("소스와 대상 엔티티는 달라야 합니다")

        # 관계 관리자를 위해 문자열 ID를 정수로 변환합니다.
        try:
            source_id_int = int(source_id)
            target_id_int = int(target_id)
        except ValueError as exception:
            raise ValueError("엔티티 ID는 유효한 정수여야 합니다") from exception

        relationship = self.relationship_manager.create_relationship(
            source_id_int, target_id_int, rel_type.strip(), properties
        )
        return relationship

    def delete_entity(self, entity_id: int):
        """엔티티를 삭제합니다."""
        if not isinstance(entity_id, int) or entity_id <= 0:
            raise ValueError("엔티티 ID는 양의 정수여야 합니다")

        self.entity_manager.delete_entity(entity_id)

    def delete_relationship(self, relationship_id: int):
        """관계를 삭제합니다."""
        if not isinstance(relationship_id, int) or relationship_id <= 0:
            raise ValueError("관계 ID는 양의 정수여야 합니다")

        self.relationship_manager.delete_relationship(relationship_id)


def main():
    """Streamlit 앱을 실행하는 메인 함수."""
    parser = argparse.ArgumentParser(description="Knowledge Graph Streamlit App")
    parser.add_argument("--db-path", default="knowledge_graph.db", help="Path to SQLite database")
    args = parser.parse_args()

    app = KnowledgeGraphStreamlitApp(args.db_path)
    app.run()


if __name__ == "__main__":
    main()
