"""Transport layer tests (Task 31).

Tests all three transport implementations and the MCPClient.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _make_server():
    """Create a test MCPServer with a simple tool."""
    from pirag.mcp.protocol import MCPServer
    from pirag.mcp.registry import ToolRegistry, ToolSpec

    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="echo",
        description="Echo back the input",
        capabilities=["test"],
        fn=lambda msg="hello": {"echo": msg},
        schema={"msg": "str"},
    ))
    return MCPServer(registry=registry)


# ---- Test 1: In-process serialization ----
def test_in_process_serialization():
    from pirag.mcp.transport import InProcessTransport

    server = _make_server()
    transport = InProcessTransport(server)

    # Non-serializable types should be handled by default=str
    response = transport.send({
        "jsonrpc": "2.0", "id": 1,
        "method": "initialize",
        "params": {},
    })
    assert "result" in response
    assert response["jsonrpc"] == "2.0"


# ---- Test 2: In-process tool call ----
def test_in_process_tool_call():
    from pirag.mcp.transport import InProcessTransport

    server = _make_server()
    transport = InProcessTransport(server)

    response = transport.send({
        "jsonrpc": "2.0", "id": 2,
        "method": "tools/call",
        "params": {"name": "echo", "arguments": {"msg": "world"}},
    })
    assert "result" in response
    content = response["result"]["content"]
    assert len(content) > 0
    parsed = json.loads(content[0]["text"])
    assert parsed["echo"] == "world"


# ---- Test 3: Stdio transport format ----
def test_stdio_transport_format():
    from pirag.mcp.transport import StdioTransport

    expected_response = {"jsonrpc": "2.0", "id": 1, "result": {"status": "ok"}}
    fake_stdin = io.StringIO(json.dumps(expected_response) + "\n")
    fake_stdout = io.StringIO()

    transport = StdioTransport(proc_stdin=fake_stdin, proc_stdout=fake_stdout)
    result = transport.send({
        "jsonrpc": "2.0", "id": 1,
        "method": "initialize",
        "params": {},
    })

    # Check sent message is newline-delimited JSON
    sent = fake_stdout.getvalue()
    assert sent.endswith("\n")
    lines = sent.strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["jsonrpc"] == "2.0"
    assert parsed["method"] == "initialize"

    # Check received response
    assert result["result"]["status"] == "ok"


# ---- Test 4: MCPClient initialize ----
def test_mcp_client_initialize():
    from pirag.mcp.transport import InProcessTransport, MCPClient

    server = _make_server()
    client = MCPClient(InProcessTransport(server))

    result = client.initialize()
    assert "protocolVersion" in result
    assert "capabilities" in result

    client.close()


# ---- Test 5: MCPClient tool call ----
def test_mcp_client_tool_call():
    from pirag.mcp.transport import InProcessTransport, MCPClient

    server = _make_server()
    client = MCPClient(InProcessTransport(server))

    result = client.call_tool("echo", {"msg": "test123"})
    assert result is not None
    assert result["echo"] == "test123"

    client.close()


# ---- Test 6: MCPClient read resource ----
def test_mcp_client_read_resource():
    from pirag.mcp.protocol import MCPResource
    from pirag.mcp.transport import InProcessTransport, MCPClient

    server = _make_server()
    server.register_resource(MCPResource(
        uri="test://data",
        name="test data",
        description="test",
        read_fn=lambda: {"value": 42},
    ))

    client = MCPClient(InProcessTransport(server))
    result = client.read_resource("test://data")
    assert result is not None
    assert result["value"] == 42

    client.close()


# ---- Test 7: MCPClient get prompt ----
def test_mcp_client_get_prompt():
    from pirag.mcp.protocol import MCPPrompt
    from pirag.mcp.transport import InProcessTransport, MCPClient

    server = _make_server()
    server.register_prompt(MCPPrompt(
        name="test_prompt",
        description="test",
        arguments=[{"name": "topic", "description": "topic"}],
        template_fn=lambda topic="default": f"Tell me about {topic}",
    ))

    client = MCPClient(InProcessTransport(server))
    result = client.get_prompt("test_prompt", {"topic": "spinach"})
    assert "spinach" in result

    client.close()
