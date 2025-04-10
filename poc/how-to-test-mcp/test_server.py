import pytest
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["poc/mcp-basic/server.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)

# Optional: create a sampling callback
async def handle_sampling_message(
    _message: types.CreateMessageRequestParams,
) -> types.CreateMessageResult:
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(
            type="text",
            text="Hello, world! from model",
        ),
        model="gpt-3.5-turbo",
        stopReason="endTurn",
    )

@pytest.mark.asyncio
async def test_echo():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write, sampling_callback=handle_sampling_message
        ) as session:
            # Initialize the connection
            await session.initialize()
            result = await session.call_tool("echo", arguments={"message": "Hello, world!"})
            assert result.content[0].text == "Hello, world!"

@pytest.mark.asyncio
async def test_ping():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write, sampling_callback=handle_sampling_message
        ) as session:
            # Initialize the connection
            await session.initialize()
            result = await session.call_tool("ping")
            assert result.content[0].text == "pong"

@pytest.mark.asyncio
async def test_add_numbers():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write, sampling_callback=handle_sampling_message
        ) as session:
            # Initialize the connection
            await session.initialize()
            result = await session.call_tool("add_numbers", arguments={"a": 1, "b": 2})
            assert float(result.content[0].text) == 3.0