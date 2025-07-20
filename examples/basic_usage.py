"""
Basic usage example for the SQLite KG Vec MCP library.
"""
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlite_kg_vec_mcp import KnowledgeGraph


def main():
    """
    Demonstrate basic usage of the knowledge graph.
    """
    # Create a knowledge graph with a temporary database file
    db_path = "example.db"

    # Remove the database file if it already exists to start fresh
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database file: {db_path}")

    kg = KnowledgeGraph(db_path)
    
    print("Creating nodes and relationships...")
    
    # Create some person nodes
    alice = kg.create_node(
        type="Person",
        name="Alice",
        properties={"age": 30, "occupation": "Data Scientist"}
    )
    
    bob = kg.create_node(
        type="Person",
        name="Bob",
        properties={"age": 35, "occupation": "Software Engineer"}
    )
    
    charlie = kg.create_node(
        type="Person",
        name="Charlie",
        properties={"age": 28, "occupation": "Project Manager"}
    )
    
    # Create some company nodes
    tech_corp = kg.create_node(
        type="Company",
        name="TechCorp",
        properties={"founded": 2010, "industry": "Technology"}
    )
    
    data_inc = kg.create_node(
        type="Company",
        name="DataInc",
        properties={"founded": 2015, "industry": "Data Analytics"}
    )
    
    # Create relationships
    kg.create_edge(
        source_id=alice.id,
        target_id=data_inc.id,
        relation_type="WORKS_FOR",
        properties={"since": 2020, "position": "Senior Data Scientist"}
    )
    
    kg.create_edge(
        source_id=bob.id,
        target_id=tech_corp.id,
        relation_type="WORKS_FOR",
        properties={"since": 2018, "position": "Lead Developer"}
    )
    
    kg.create_edge(
        source_id=charlie.id,
        target_id=tech_corp.id,
        relation_type="WORKS_FOR",
        properties={"since": 2021, "position": "Product Manager"}
    )
    
    kg.create_edge(
        source_id=alice.id,
        target_id=bob.id,
        relation_type="KNOWS",
        properties={"since": 2019, "relationship": "Professional"}
    )
    
    kg.create_edge(
        source_id=bob.id,
        target_id=charlie.id,
        relation_type="KNOWS",
        properties={"since": 2020, "relationship": "Colleague"}
    )
    
    # Find people
    print("\nFinding all people:")
    people, count = kg.find_nodes(type="Person")
    for person in people:
        print(f"- {person.name} ({person.properties.get('occupation')})")
    
    # Find companies
    print("\nFinding all companies:")
    companies, count = kg.find_nodes(type="Company")
    for company in companies:
        print(f"- {company.name} ({company.properties.get('industry')})")
    
    # Get Alice's relationships
    print(f"\nRelationships for {alice.name}:")
    relationships, _ = kg.find_edges(source_id=alice.id, include_entities=True)
    for rel in relationships:
        if rel.target and rel.target.name:
            print(f"- {alice.name} {rel.relation_type} {rel.target.name}")
            if rel.properties:
                print(f"  Properties: {rel.properties}")
    
    # Get who works for TechCorp
    print(f"\nEmployees of {tech_corp.name}:")
    employees = kg.get_neighbors(
        tech_corp.id, 
        direction="incoming", 
        relation_types=["WORKS_FOR"]
    )
    for employee, relationship in employees:
        position = relationship.properties.get("position", "Employee")
        since = relationship.properties.get("since", "N/A")
        print(f"- {employee.name} ({position}) since {since}")
    
    # Find path between Alice and Charlie
    print(f"\nPath from {alice.name} to {charlie.name}:")
    paths = kg.find_paths(alice.id, charlie.id, max_depth=3)
    for path in paths:
        path_str = " -> ".join([node.entity.name for node in path])
        print(f"- {path_str}")
    
    # Update a node
    bob = kg.update_node(
        bob.id,
        properties={"age": 36, "skills": ["Python", "JavaScript", "Docker"]}
    )
    print(f"\nUpdated {bob.name}'s properties:")
    print(f"- Age: {bob.properties.get('age')}")
    print(f"- Skills: {', '.join(bob.properties.get('skills', []))}")
    
    # Clean up
    kg.close()
    print(f"\nExample complete. Database saved to {db_path}")


if __name__ == "__main__":
    main()