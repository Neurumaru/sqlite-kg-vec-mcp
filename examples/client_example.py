"""
Example client for the Knowledge Graph MCP server.
"""
import sys
import argparse
from pathlib import Path

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp import Client


def main():
    """Run the Knowledge Graph MCP client example."""
    parser = argparse.ArgumentParser(description="Run the Knowledge Graph MCP client example")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host where the server is running")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port where the server is running")
    
    args = parser.parse_args()
    
    # Create client
    client_url = f"http://{args.host}:{args.port}"
    print(f"Connecting to Knowledge Graph MCP server at {client_url}")
    client = Client(client_url)
    
    try:
        # Create nodes
        print("\nCreating nodes...")
        person = client.create_node(
            type="Person",
            name="John Doe",
            properties={"age": 30, "occupation": "Software Engineer"}
        )
        print(f"Created person: {person}")
        
        company = client.create_node(
            type="Company",
            name="TechCorp",
            properties={"founded": 2010, "industry": "Technology"}
        )
        print(f"Created company: {company}")
        
        # Create relationship
        print("\nCreating relationship...")
        relationship = client.create_edge(
            source_id=person["node_id"],
            target_id=company["node_id"],
            relation_type="WORKS_AT",
            properties={"since": 2020, "position": "Senior Engineer"}
        )
        print(f"Created relationship: {relationship}")
        
        # Get node
        print("\nRetrieving person node...")
        retrieved_person = client.get_node(id=person["node_id"])
        print(f"Retrieved person: {retrieved_person}")
        
        # Update node
        print("\nUpdating person node...")
        updated_person = client.update_node(
            id=person["node_id"],
            properties={"age": 31, "occupation": "Senior Software Engineer"}
        )
        print(f"Updated person: {updated_person}")
        
        # Get neighbors
        print("\nGetting neighbors of person node...")
        neighbors = client.get_neighbors(node_id=person["node_id"])
        print(f"Person has {len(neighbors['neighbors'])} neighbors:")
        for neighbor in neighbors["neighbors"]:
            print(f"  - {neighbor['node']['name']} ({neighbor['edge']['relation_type']})")
        
        # Find nodes by type
        print("\nFinding all Person nodes...")
        person_nodes = client.find_nodes(type="Person")
        print(f"Found {person_nodes['total_count']} Person nodes")
        
        # Clean up (optional)
        print("\nDeleting nodes and relationships...")
        client.delete_edge(id=relationship["edge_id"])
        client.delete_node(id=person["node_id"])
        client.delete_node(id=company["node_id"])
        print("Deleted all created nodes and relationships")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()