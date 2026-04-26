"""MCP transport abstraction.

Separates the protocol layer (JSON-RPC messages) from the transport
mechanism (how messages are delivered). Three transport implementations:

1. ``InProcessTransport`` — Direct in-process dispatch with a real
   JSON serialize / deserialize round-trip on the way in and out
   (so a serialization bug surfaces in tests, not on first remote
   deployment). This is the canonical transport used by both the
   simulator and the FastAPI ``/mcp`` endpoint, via ``MCPClient``.
2. ``StdioTransport`` — Newline-delimited JSON-RPC over stdin/stdout.
   Honors the standard MCP local-client pattern (one server process
   per agent, connected via pipes). Run a stdio server with
   ``python -m pirag.mcp.serve``; pair it with ``StdioTransport`` on
   the client side.
3. ``HTTPTransport`` (formerly ``SSETransport``) — Synchronous JSON-RPC
   over HTTP POST. The "SSE" name was inherited from a draft of the
   MCP spec and was inaccurate: this transport never reads a
   long-lived event stream, it does request/response. ``SSETransport``
   is retained as a deprecated alias for backward compatibility.

All three implement the same ``send(message)`` -> response interface.
"""
from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MCPTransport(ABC):
    """Abstract transport for MCP message delivery."""

    @abstractmethod
    def send(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC message and return the response."""

    @abstractmethod
    def close(self) -> None:
        """Clean up transport resources."""


class InProcessTransport(MCPTransport):
    """Direct in-process message passing (for simulation).

    Messages are serialized to JSON and deserialized to ensure protocol
    compliance even in-process. This catches serialization bugs that
    would only surface in networked deployment.
    """

    def __init__(self, server: Any) -> None:
        self._server = server

    def send(self, message: Dict[str, Any]) -> Dict[str, Any]:
        wire = json.dumps(message, default=str)
        parsed = json.loads(wire)

        from .protocol import MCPMessage
        msg = MCPMessage(
            jsonrpc=parsed.get("jsonrpc", "2.0"),
            id=parsed.get("id"),
            method=parsed.get("method"),
            params=parsed.get("params", {}),
        )
        response = self._server.handle_message(msg)

        resp_dict: Dict[str, Any] = {
            "jsonrpc": response.jsonrpc,
            "id": response.id,
        }
        if response.result is not None:
            resp_dict["result"] = response.result
        if response.error is not None:
            resp_dict["error"] = response.error

        return json.loads(json.dumps(resp_dict, default=str))

    def close(self) -> None:
        pass


class StdioTransport(MCPTransport):
    """JSON-RPC over stdin/stdout (standard MCP local client pattern).

    Writes newline-delimited JSON-RPC messages to stdout, reads responses
    from stdin. Used when agents run as separate processes connected via
    pipes.
    """

    def __init__(
        self,
        proc_stdin: Any = None,
        proc_stdout: Any = None,
    ) -> None:
        self._stdin = proc_stdin or sys.stdin
        self._stdout = proc_stdout or sys.stdout

    def send(self, message: Dict[str, Any]) -> Dict[str, Any]:
        wire = json.dumps(message, default=str) + "\n"
        self._stdout.write(wire)
        self._stdout.flush()

        response_line = self._stdin.readline()
        if not response_line:
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": -32000, "message": "No response from server"},
            }
        return json.loads(response_line)

    def close(self) -> None:
        pass


class HTTPTransport(MCPTransport):
    """Synchronous JSON-RPC over HTTP POST.

    Posts JSON-RPC messages to an HTTP endpoint and reads the response
    body. Uses only stdlib to avoid external dependencies. NOT
    Server-Sent Events: there is no long-lived event stream; each
    ``send`` is a single request/response. The previous name
    ``SSETransport`` was misleading and is retained below as a
    deprecated alias.
    """

    def __init__(self, endpoint_url: str, timeout: int = 30) -> None:
        self._url = endpoint_url
        self._timeout = timeout

    def send(self, message: Dict[str, Any]) -> Dict[str, Any]:
        import urllib.request

        body = json.dumps(message, default=str).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": -32000, "message": str(e)},
            }

    def close(self) -> None:
        pass


# Backward-compat alias. Deprecated; use HTTPTransport.
SSETransport = HTTPTransport


class MCPClient:
    """MCP client that communicates via a pluggable transport.

    Agents use this client to invoke MCP operations. The client does not
    know whether it is talking to a local in-process server or a remote
    network endpoint.
    """

    def __init__(self, transport: MCPTransport) -> None:
        self._transport = transport
        self._id_counter = 0
        self._initialized = False

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def initialize(self) -> Dict[str, Any]:
        """Perform MCP capability negotiation handshake."""
        response = self._transport.send({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "agribrain-agent", "version": "1.0.0"},
            },
        })
        self._initialized = True
        return response.get("result", {})

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools."""
        response = self._transport.send({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {},
        })
        return response.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Invoke an MCP tool by name."""
        response = self._transport.send({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        })
        if "error" in response and response["error"]:
            return None
        content = response.get("result", {}).get("content", [])
        if content and content[0].get("type") == "text":
            try:
                return json.loads(content[0]["text"])
            except (json.JSONDecodeError, KeyError):
                return content[0].get("text")
        return response.get("result")

    def list_resources(self) -> List[Dict[str, Any]]:
        """List available MCP resources."""
        response = self._transport.send({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "resources/list",
            "params": {},
        })
        return response.get("result", {}).get("resources", [])

    def read_resource(self, uri: str) -> Any:
        """Read a value from an MCP resource."""
        response = self._transport.send({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "resources/read",
            "params": {"uri": uri},
        })
        contents = response.get("result", {}).get("contents", [])
        if contents and contents[0].get("text"):
            try:
                return json.loads(contents[0]["text"])
            except (json.JSONDecodeError, KeyError):
                return contents[0].get("text")
        return None

    def list_prompts(self) -> List[Dict[str, Any]]:
        """List available MCP prompts."""
        response = self._transport.send({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "prompts/list",
            "params": {},
        })
        return response.get("result", {}).get("prompts", [])

    def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Expand an MCP prompt template."""
        response = self._transport.send({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "prompts/get",
            "params": {"name": name, "arguments": arguments or {}},
        })
        messages = response.get("result", {}).get("messages", [])
        if messages:
            content = messages[0].get("content", {})
            if isinstance(content, dict):
                return content.get("text", "")
            return str(content)
        return ""

    def close(self) -> None:
        """Close the transport."""
        self._transport.close()
