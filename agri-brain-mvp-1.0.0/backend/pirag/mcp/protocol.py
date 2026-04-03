"""MCP protocol layer: JSON-RPC 2.0, capability negotiation, three primitives.

Implements the Model Context Protocol server following the JSON-RPC 2.0
specification. Supports all three MCP primitives:

- **Tools**: callable functions with schema-based invocation
- **Resources**: URI-addressable live state endpoints
- **Prompts**: parameterized query templates

Reference: https://modelcontextprotocol.io/specification/2025-11-25
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .registry import ToolRegistry, get_default_registry


class MCPCapability(Enum):
    """Capabilities a server can advertise during initialization."""
    TOOLS = "tools"
    RESOURCES = "resources"
    PROMPTS = "prompts"


@dataclass
class MCPMessage:
    """JSON-RPC 2.0 message envelope for MCP communication."""
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    method: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class MCPResource:
    """An MCP resource: a URI-addressable, read-only data endpoint.

    Parameters
    ----------
    uri : resource URI, e.g. ``"agribrain://telemetry/temperature"``.
    name : human-readable name.
    description : what this resource exposes.
    mime_type : content type of the returned data.
    read_fn : callable returning the current value (JSON-serializable).
    """
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"
    read_fn: Optional[Callable[[], Any]] = None


@dataclass
class MCPPrompt:
    """An MCP prompt: a parameterized query template.

    Parameters
    ----------
    name : prompt identifier, e.g. ``"regulatory_compliance_check"``.
    description : what the prompt produces.
    arguments : list of ``{"name": str, "description": str, "required": bool}``.
    template_fn : callable(**arguments) → str producing the query string.
    """
    name: str
    description: str
    arguments: List[Dict[str, str]] = field(default_factory=list)
    template_fn: Optional[Callable[..., str]] = None


# Standard JSON-RPC error codes
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603


class MCPServer:
    """Full MCP server: tools (via registry), resources, prompts, JSON-RPC routing.

    Parameters
    ----------
    server_name : human-readable server name for capability negotiation.
    registry : tool registry to use; defaults to the global singleton.
    """

    PROTOCOL_VERSION = "2024-11-05"
    SERVER_VERSION = "1.0.0"

    def __init__(
        self,
        server_name: str = "agribrain-mcp",
        registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.server_name = server_name
        self._registry = registry or get_default_registry()
        self._resources: Dict[str, MCPResource] = {}
        self._prompts: Dict[str, MCPPrompt] = {}

        self._method_handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "prompts/list": self._handle_prompts_list,
            "prompts/get": self._handle_prompts_get,
        }

    # -----------------------------------------------------------------
    # Registration helpers
    # -----------------------------------------------------------------

    def register_resource(self, resource: MCPResource) -> None:
        """Register an MCP resource."""
        self._resources[resource.uri] = resource

    def register_prompt(self, prompt: MCPPrompt) -> None:
        """Register an MCP prompt template."""
        self._prompts[prompt.name] = prompt

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    # -----------------------------------------------------------------
    # Main dispatch
    # -----------------------------------------------------------------

    def handle_message(self, msg: MCPMessage) -> MCPMessage:
        """Route a JSON-RPC message to the appropriate handler."""
        if msg.method is None:
            return MCPMessage(
                id=msg.id,
                error={"code": _INVALID_REQUEST, "message": "Missing method"},
            )

        handler = self._method_handlers.get(msg.method)
        if handler is None:
            return MCPMessage(
                id=msg.id,
                error={"code": _METHOD_NOT_FOUND, "message": f"Unknown method: {msg.method}"},
            )

        try:
            return handler(msg)
        except Exception as exc:
            return MCPMessage(
                id=msg.id,
                error={"code": _INTERNAL_ERROR, "message": str(exc)},
            )

    # -----------------------------------------------------------------
    # Handlers
    # -----------------------------------------------------------------

    def _handle_initialize(self, msg: MCPMessage) -> MCPMessage:
        """Return protocol version, server info, and capabilities dict."""
        params = msg.params or {}
        client_caps = params.get("capabilities", {})
        capabilities: Dict[str, Any] = {}
        if self._registry.list_tools():
            capabilities["tools"] = {}
        if self._resources:
            capabilities["resources"] = {}
        if self._prompts:
            capabilities["prompts"] = {}
        capabilities["experimental"] = {
            "qosNegotiation": True,
            "reliabilityHints": True,
            "clientCapabilitiesEcho": client_caps,
        }

        return MCPMessage(
            id=msg.id,
            result={
                "protocolVersion": self.PROTOCOL_VERSION,
                "serverInfo": {
                    "name": self.server_name,
                    "version": self.SERVER_VERSION,
                },
                "capabilities": capabilities,
                "extensions": {
                    "qosPolicy": {
                        "supportedLatencyTiers": ["low", "medium", "high"],
                        "supportedReliabilityTiers": ["best_effort", "standard", "high"],
                        "supportedCostTiers": ["low", "medium", "high"],
                    }
                },
            },
        )

    def _handle_tools_list(self, msg: MCPMessage) -> MCPMessage:
        tools = []
        for spec in self._registry._tools.values():
            properties: Dict[str, Any] = {}
            for param_name, param_def in spec.schema.items():
                if isinstance(param_def, dict):
                    # New format: {"type": ..., "description": ...}
                    properties[param_name] = param_def
                else:
                    # Legacy format: bare type string
                    properties[param_name] = {"type": param_def}
            input_schema: Dict[str, Any] = {
                "type": "object",
                "properties": properties,
            }
            tools.append({
                "name": spec.name,
                "description": spec.description,
                "inputSchema": input_schema,
                "x-qos": {
                    "latency_tier": spec.latency_tier,
                    "reliability_tier": spec.reliability_tier,
                    "cost_tier": spec.cost_tier,
                    "role_affinity": spec.role_affinity,
                },
            })
        return MCPMessage(id=msg.id, result={"tools": tools})

    def _handle_tools_call(self, msg: MCPMessage) -> MCPMessage:
        name = msg.params.get("name", "")
        arguments = msg.params.get("arguments", {})

        spec = self._registry.get(name)
        if spec is None:
            return MCPMessage(
                id=msg.id,
                error={"code": _METHOD_NOT_FOUND, "message": f"Unknown tool: {name}"},
            )

        try:
            result = self._registry.invoke(name, **arguments)
            return MCPMessage(
                id=msg.id,
                result={
                    "content": [{"type": "text", "text": json.dumps(result, default=str)}],
                },
            )
        except Exception as exc:
            return MCPMessage(
                id=msg.id,
                result={
                    "content": [{"type": "text", "text": json.dumps({"error": str(exc)})}],
                    "isError": True,
                },
            )

    def _handle_resources_list(self, msg: MCPMessage) -> MCPMessage:
        resources = [
            {
                "uri": r.uri,
                "name": r.name,
                "description": r.description,
                "mimeType": r.mime_type,
            }
            for r in self._resources.values()
        ]
        return MCPMessage(id=msg.id, result={"resources": resources})

    def _handle_resources_read(self, msg: MCPMessage) -> MCPMessage:
        uri = msg.params.get("uri", "")
        resource = self._resources.get(uri)
        if resource is None:
            return MCPMessage(
                id=msg.id,
                error={"code": _INVALID_PARAMS, "message": f"Unknown resource: {uri}"},
            )
        value = resource.read_fn() if resource.read_fn else None
        return MCPMessage(
            id=msg.id,
            result={
                "contents": [{
                    "uri": uri,
                    "mimeType": resource.mime_type,
                    "text": json.dumps(value, default=str),
                }],
            },
        )

    def _handle_prompts_list(self, msg: MCPMessage) -> MCPMessage:
        prompts = [
            {
                "name": p.name,
                "description": p.description,
                "arguments": p.arguments,
            }
            for p in self._prompts.values()
        ]
        return MCPMessage(id=msg.id, result={"prompts": prompts})

    def _handle_prompts_get(self, msg: MCPMessage) -> MCPMessage:
        name = msg.params.get("name", "")
        prompt = self._prompts.get(name)
        if prompt is None:
            return MCPMessage(
                id=msg.id,
                error={"code": _INVALID_PARAMS, "message": f"Unknown prompt: {name}"},
            )
        arguments = msg.params.get("arguments", {})
        text = prompt.template_fn(**arguments) if prompt.template_fn else ""
        return MCPMessage(
            id=msg.id,
            result={
                "description": prompt.description,
                "messages": [{
                    "role": "user",
                    "content": {"type": "text", "text": text},
                }],
            },
        )
