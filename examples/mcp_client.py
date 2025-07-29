"""
Example MCP client for the SQLite KG Vec MCP library.
"""

import argparse
import asyncio
import os
import sys

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# TODO: Fix fastmcp import - package may not be available
# from fastmcp import MCPClient  # noqa: E402


# Temporary replacement with stub class
class MCPClient:
    def __init__(self, url):
        self.url = url
        print("경고: FastMCP Client는 현재 사용할 수 없습니다. Mock 클라이언트를 사용합니다.")
        print("실제 FastMCP 클라이언트를 사용하려면 'fastmcp' 패키지를 설치하세요.")

    async def connect(self):
        print(f"Mock connection to {self.url}")

    async def request(self, method, params=None):
        # Mock responses based on method
        if method == "create_node":
            return {"node_id": f"mock_node_{hash(str(params))}", **params}
        if method == "create_edge":
            return {"edge_id": f"mock_edge_{hash(str(params))}", **params}
        if method == "find_nodes":
            return {"nodes": [], "total_count": 0}
        if method == "get_neighbors":
            return {"neighbors": []}
        if method == "search_by_text":
            return {"results": []}
        return {"success": True}

    async def close(self):
        print("Mock client connection closed")


async def create_sample_data(client):
    """Create sample data in the knowledge graph."""
    print("Creating sample data...")

    # Create person nodes
    alice = await client.request(
        "create_node",
        {
            "type": "Person",
            "name": "Alice",
            "properties": {"age": 30, "occupation": "Data Scientist"},
        },
    )

    bob = await client.request(
        "create_node",
        {
            "type": "Person",
            "name": "Bob",
            "properties": {"age": 35, "occupation": "Software Engineer"},
        },
    )

    charlie = await client.request(
        "create_node",
        {
            "type": "Person",
            "name": "Charlie",
            "properties": {"age": 28, "occupation": "Project Manager"},
        },
    )

    # Create company nodes
    tech_corp = await client.request(
        "create_node",
        {
            "type": "Company",
            "name": "TechCorp",
            "properties": {"founded": 2010, "industry": "Technology"},
        },
    )

    data_inc = await client.request(
        "create_node",
        {
            "type": "Company",
            "name": "DataInc",
            "properties": {"founded": 2015, "industry": "Data Analytics"},
        },
    )

    # Create relationships
    await client.request(
        "create_edge",
        {
            "source_id": alice["node_id"],
            "target_id": data_inc["node_id"],
            "relation_type": "WORKS_FOR",
            "properties": {"since": 2020, "position": "Senior Data Scientist"},
        },
    )

    await client.request(
        "create_edge",
        {
            "source_id": bob["node_id"],
            "target_id": tech_corp["node_id"],
            "relation_type": "WORKS_FOR",
            "properties": {"since": 2018, "position": "Lead Developer"},
        },
    )

    await client.request(
        "create_edge",
        {
            "source_id": charlie["node_id"],
            "target_id": tech_corp["node_id"],
            "relation_type": "WORKS_FOR",
            "properties": {"since": 2021, "position": "Product Manager"},
        },
    )

    await client.request(
        "create_edge",
        {
            "source_id": alice["node_id"],
            "target_id": bob["node_id"],
            "relation_type": "KNOWS",
            "properties": {"since": 2019, "relationship": "Professional"},
        },
    )

    await client.request(
        "create_edge",
        {
            "source_id": bob["node_id"],
            "target_id": charlie["node_id"],
            "relation_type": "KNOWS",
            "properties": {"since": 2020, "relationship": "Colleague"},
        },
    )

    print("Sample data created.")

    return {
        "alice": alice,
        "bob": bob,
        "charlie": charlie,
        "tech_corp": tech_corp,
        "data_inc": data_inc,
    }


async def query_example(client, entities):
    """Run example queries on the knowledge graph."""
    print("\nRunning example queries...")

    # Find all people
    people = await client.request("find_nodes", {"type": "Person", "limit": 10})

    print("\nPeople in the graph:")
    for person in people["nodes"]:
        print(f"- {person['name']} ({person['properties'].get('occupation', 'N/A')})")

    # Find Alice's connections
    alice_id = entities["alice"]["node_id"]
    neighbors = await client.request(
        "get_neighbors", {"node_id": alice_id, "direction": "both", "limit": 10}
    )

    print("\nConnections for Alice:")
    for neighbor in neighbors["neighbors"]:
        node = neighbor["node"]
        edge = neighbor["edge"]
        direction = neighbor["direction"]

        if direction == "outgoing":
            print(f"- Alice {edge['relation_type']} {node['name']}")
        else:
            print(f"- {node['name']} {edge['relation_type']} Alice")

        if "properties" in edge and edge["properties"]:
            props = ", ".join([f"{k}: {v}" for k, v in edge["properties"].items()])
            print(f"  ({props})")

    # Find employees of TechCorp
    tech_corp_id = entities["tech_corp"]["node_id"]
    employees = await client.request(
        "get_neighbors",
        {
            "node_id": tech_corp_id,
            "direction": "incoming",
            "relation_types": ["WORKS_FOR"],
            "limit": 10,
        },
    )

    print("\nEmployees of TechCorp:")
    for neighbor in employees["neighbors"]:
        node = neighbor["node"]
        edge = neighbor["edge"]
        position = edge.get("properties", {}).get("position", "Employee")
        since = edge.get("properties", {}).get("since", "N/A")
        print(f"- {node['name']} ({position}) since {since}")

    # Search by text
    text_results = await client.request(
        "search_by_text",
        {"query": "data science technology", "limit": 5, "entity_types": ["node"]},
    )

    print("\nText search results for 'data science technology':")
    for result in text_results.get("results", []):
        entity = result.get("entity", {})
        score = result.get("distance", 0)
        print(f"- {entity.get('name', 'Unknown')} (Score: {score:.4f})")
        print(f"  Type: {entity.get('type', 'Unknown')}")
        if "properties" in entity and entity["properties"]:
            print(f"  Properties: {entity['properties']}")


async def main():
    """
    Run an MCP client example.
    """
    parser = argparse.ArgumentParser(description="Run a Knowledge Graph MCP client example")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--create-data", action="store_true", help="Create sample data")

    args = parser.parse_args()

    # MCP server URL
    server_url = f"ws://{args.host}:{args.port}"
    print(f"Connecting to MCP server at {server_url}")

    # Connect to the server
    client = MCPClient(server_url)
    await client.connect()

    try:
        entities = {}

        # Create sample data if requested
        if args.create_data:
            entities = await create_sample_data(client)
        else:
            # Find existing entities for examples
            people = await client.request("find_nodes", {"type": "Person", "limit": 3})
            companies = await client.request("find_nodes", {"type": "Company", "limit": 2})

            if people.get("nodes") and len(people["nodes"]) >= 3:
                entities["alice"] = {"node_id": people["nodes"][0]["id"]}
                entities["bob"] = {"node_id": people["nodes"][1]["id"]}
                entities["charlie"] = {"node_id": people["nodes"][2]["id"]}

            if companies.get("nodes") and len(companies["nodes"]) >= 2:
                entities["tech_corp"] = {"node_id": companies["nodes"][0]["id"]}
                entities["data_inc"] = {"node_id": companies["nodes"][1]["id"]}

        # Run example queries
        if entities:
            await query_example(client, entities)
        else:
            print("No entities found. Try running with --create-data to create sample data.")

    finally:
        # Close the connection
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
