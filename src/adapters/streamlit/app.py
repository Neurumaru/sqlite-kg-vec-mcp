"""
Streamlit web application for Knowledge Graph exploration.
"""
import streamlit as st
import sqlite3
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.adapters.sqlite3.connection import DatabaseConnection
from src.adapters.sqlite3.schema import SchemaManager
from src.adapters.sqlite3.graph.entities import EntityManager
from src.adapters.sqlite3.graph.relationships import RelationshipManager
from src.adapters.hnsw.search import VectorSearch


class KnowledgeGraphStreamlitApp:
    """Streamlit application for Knowledge Graph exploration."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.setup_page_config()
        self.init_database()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Knowledge Graph Explorer",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def init_database(self):
        """Initialize database connection and managers."""
        try:
            self.db_connection = DatabaseConnection(self.db_path)
            self.schema_manager = SchemaManager(self.db_path)
            self.entity_manager = EntityManager(self.db_connection.get_connection())
            self.relationship_manager = RelationshipManager(self.db_connection.get_connection())
            self.vector_search = VectorSearch(
                connection=self.db_connection.get_connection(),
                vector_index_dir="vector_indexes"
            )
        except Exception as e:
            st.error(f"데이터베이스 초기화 실패: {e}")
            st.stop()
    
    def run(self):
        """Run the Streamlit application."""
        st.title("🧠 Knowledge Graph Explorer")
        
        # Sidebar menu
        menu = st.sidebar.selectbox(
            "메뉴 선택",
            ["📊 Dashboard", "🔍 Search", "🏷️ Entities", "🔗 Relationships"]
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
        """Render dashboard with statistics."""
        st.header("📊 Dashboard")
        
        try:
            # Get basic statistics
            entity_count = self.get_entity_count()
            relationship_count = self.get_relationship_count()
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Entities", entity_count)
            
            with col2:
                st.metric("Total Relationships", relationship_count)
            
            with col3:
                density = relationship_count / max(entity_count, 1)
                st.metric("Graph Density", f"{density:.2f}")
            
            # Entity types distribution
            st.subheader("Entity Types Distribution")
            entity_types = self.get_entity_types_distribution()
            if entity_types:
                st.bar_chart(entity_types)
            else:
                st.info("No entities found in the database.")
                
        except Exception as e:
            st.error(f"대시보드 로딩 중 오류 발생: {e}")
    
    def render_search(self):
        """Render search interface."""
        st.header("🔍 Search")
        
        # Search input
        search_query = st.text_input("검색어를 입력하세요:")
        
        if search_query:
            try:
                # Text-based search
                st.subheader("Text Search Results")
                text_results = self.search_entities_by_text(search_query)
                
                if text_results:
                    for result in text_results:
                        with st.expander(f"Entity: {result.get('name', 'Unnamed')}"):
                            st.json(result)
                else:
                    st.info("검색 결과가 없습니다.")
                
                # Vector-based semantic search (if available)
                st.subheader("Semantic Search Results")
                if st.button("벡터 검색 실행"):
                    semantic_results = self.search_entities_semantic(search_query)
                    if semantic_results:
                        for result in semantic_results:
                            with st.expander(f"Entity: {result.get('name', 'Unnamed')} (Score: {result.get('score', 0):.3f})"):
                                st.json(result)
                    else:
                        st.info("시맨틱 검색 결과가 없습니다.")
                        
            except Exception as e:
                st.error(f"검색 중 오류 발생: {e}")
    
    def render_entities(self):
        """Render entity management interface."""
        st.header("🏷️ Entities")
        
        tab1, tab2 = st.tabs(["Entity List", "Create Entity"])
        
        with tab1:
            try:
                entities = self.get_all_entities()
                
                if entities:
                    st.subheader(f"Total Entities: {len(entities)}")
                    
                    # Entity filter
                    entity_types = list(set(e.get('type', 'Unknown') for e in entities))
                    selected_types = st.multiselect("Entity Types Filter", entity_types, default=entity_types)
                    
                    # Filtered entities
                    filtered_entities = [e for e in entities if e.get('type', 'Unknown') in selected_types]
                    
                    for entity in filtered_entities:
                        with st.expander(f"{entity.get('name', 'Unnamed')} ({entity.get('type', 'Unknown')})"):
                            st.json(entity)
                            
                            # Delete button
                            if st.button(f"Delete {entity.get('id', '')}", key=f"delete_entity_{entity.get('id', '')}"):
                                self.delete_entity(entity.get('id'))
                                st.success("Entity deleted!")
                                st.rerun()
                else:
                    st.info("No entities found.")
                    
            except Exception as e:
                st.error(f"엔티티 로딩 중 오류 발생: {e}")
        
        with tab2:
            st.subheader("Create New Entity")
            
            with st.form("create_entity_form"):
                name = st.text_input("Entity Name")
                entity_type = st.text_input("Entity Type")
                description = st.text_area("Description")
                
                submitted = st.form_submit_button("Create Entity")
                
                if submitted and name and entity_type:
                    try:
                        properties = {"description": description} if description else {}
                        self.create_entity(name, entity_type, properties)
                        st.success(f"Entity '{name}' created successfully!")
                    except Exception as e:
                        st.error(f"엔티티 생성 실패: {e}")
    
    def render_relationships(self):
        """Render relationship management interface."""
        st.header("🔗 Relationships")
        
        tab1, tab2 = st.tabs(["Relationship List", "Create Relationship"])
        
        with tab1:
            try:
                relationships = self.get_all_relationships()
                
                if relationships:
                    st.subheader(f"Total Relationships: {len(relationships)}")
                    
                    for rel in relationships:
                        with st.expander(f"{rel.get('source_name', 'Unknown')} -> {rel.get('target_name', 'Unknown')} ({rel.get('type', 'Unknown')})"):
                            st.json(rel)
                            
                            # Delete button
                            if st.button(f"Delete {rel.get('id', '')}", key=f"delete_rel_{rel.get('id', '')}"):
                                self.delete_relationship(rel.get('id'))
                                st.success("Relationship deleted!")
                                st.rerun()
                else:
                    st.info("No relationships found.")
                    
            except Exception as e:
                st.error(f"관계 로딩 중 오류 발생: {e}")
        
        with tab2:
            st.subheader("Create New Relationship")
            
            try:
                entities = self.get_all_entities()
                entity_options = {f"{e.get('name', 'Unnamed')} ({e.get('id', '')})": e.get('id', '') for e in entities}
                
                if len(entity_options) >= 2:
                    with st.form("create_relationship_form"):
                        source_entity = st.selectbox("Source Entity", list(entity_options.keys()))
                        target_entity = st.selectbox("Target Entity", list(entity_options.keys()))
                        relationship_type = st.text_input("Relationship Type")
                        description = st.text_area("Description")
                        
                        submitted = st.form_submit_button("Create Relationship")
                        
                        if submitted and source_entity and target_entity and relationship_type:
                            if source_entity != target_entity:
                                try:
                                    source_id = entity_options[source_entity]
                                    target_id = entity_options[target_entity]
                                    properties = {"description": description} if description else {}
                                    
                                    self.create_relationship(source_id, target_id, relationship_type, properties)
                                    st.success("Relationship created successfully!")
                                except Exception as e:
                                    st.error(f"관계 생성 실패: {e}")
                            else:
                                st.error("Source and target entities must be different.")
                else:
                    st.info("최소 2개의 엔티티가 필요합니다.")
                    
            except Exception as e:
                st.error(f"관계 생성 폼 로딩 중 오류 발생: {e}")
    
    # Helper methods
    def get_entity_count(self) -> int:
        """Get total number of entities."""
        cursor = self.db_connection.get_connection().cursor()
        cursor.execute("SELECT COUNT(*) FROM entities")
        return cursor.fetchone()[0]
    
    def get_relationship_count(self) -> int:
        """Get total number of relationships."""
        cursor = self.db_connection.get_connection().cursor()
        cursor.execute("SELECT COUNT(*) FROM relationships")
        return cursor.fetchone()[0]
    
    def get_entity_types_distribution(self) -> Dict[str, int]:
        """Get distribution of entity types."""
        cursor = self.db_connection.get_connection().cursor()
        cursor.execute("SELECT type, COUNT(*) as count FROM entities GROUP BY type")
        return {row[0] or 'Unknown': row[1] for row in cursor.fetchall()}
    
    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Get all entities."""
        cursor = self.db_connection.get_connection().cursor()
        cursor.execute("SELECT id, name, type, properties FROM entities")
        entities = []
        for row in cursor.fetchall():
            entities.append({
                'id': row[0],
                'name': row[1],
                'type': row[2],
                'properties': row[3]
            })
        return entities
    
    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """Get all relationships with entity names."""
        cursor = self.db_connection.get_connection().cursor()
        cursor.execute("""
            SELECT r.id, r.source_id, r.target_id, r.type, r.properties,
                   e1.name as source_name, e2.name as target_name
            FROM relationships r
            LEFT JOIN entities e1 ON r.source_id = e1.id
            LEFT JOIN entities e2 ON r.target_id = e2.id
        """)
        relationships = []
        for row in cursor.fetchall():
            relationships.append({
                'id': row[0],
                'source_id': row[1],
                'target_id': row[2],
                'type': row[3],
                'properties': row[4],
                'source_name': row[5],
                'target_name': row[6]
            })
        return relationships
    
    def search_entities_by_text(self, query: str) -> List[Dict[str, Any]]:
        """Search entities by text."""
        cursor = self.db_connection.get_connection().cursor()
        cursor.execute("""
            SELECT id, name, type, properties
            FROM entities
            WHERE name LIKE ? OR type LIKE ? OR properties LIKE ?
        """, (f'%{query}%', f'%{query}%', f'%{query}%'))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'name': row[1],
                'type': row[2],
                'properties': row[3]
            })
        return results
    
    def search_entities_semantic(self, query: str) -> List[Dict[str, Any]]:
        """Search entities using semantic similarity."""
        try:
            # This would use the vector search functionality
            # For now, return empty list as placeholder
            return []
        except Exception:
            return []
    
    def create_entity(self, name: str, entity_type: str, properties: Dict[str, Any]):
        """Create a new entity."""
        entity = self.entity_manager.create_entity(name, entity_type, properties)
        return entity
    
    def create_relationship(self, source_id: str, target_id: str, rel_type: str, properties: Dict[str, Any]):
        """Create a new relationship."""
        relationship = self.relationship_manager.create_relationship(
            source_id, target_id, rel_type, properties
        )
        return relationship
    
    def delete_entity(self, entity_id: str):
        """Delete an entity."""
        self.entity_manager.delete_entity(entity_id)
    
    def delete_relationship(self, relationship_id: str):
        """Delete a relationship."""
        self.relationship_manager.delete_relationship(relationship_id)


def main():
    """Main function to run the Streamlit app."""
    parser = argparse.ArgumentParser(description="Knowledge Graph Streamlit App")
    parser.add_argument("--db-path", default="knowledge_graph.db", help="Path to SQLite database")
    args = parser.parse_args()
    
    app = KnowledgeGraphStreamlitApp(args.db_path)
    app.run()


if __name__ == "__main__":
    main()