"""MCP protocol layer: JSON-RPC 2.0, capability negotiation, three primitives.

Implements the Model Context Protocol server following the JSON-RPC 2.0
specification. The advertised ``protocolVersion`` is ``2024-11-05``;
the matching reference document is the MCP specification of that
date. Supports all three MCP primitives:

- **Tools**: callable functions with schema-based invocation
- **Resources**: URI-addressable live state endpoints
- **Prompts**: parameterized query templates

Reference: https://modelcontextprotocol.io/specification/2024-11-05
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
        """Route a JSON-RPC message to the appropriate handler.

        Spec compliance (JSON-RPC 2.0 + MCP 2024-11-05):

        - ``jsonrpc`` field is validated to equal ``"2.0"``; anything
          else returns INVALID_REQUEST.
        - ``method`` is required.
        - Notifications (``id`` is None or omitted) are dispatched but
          generate no response. Per JSON-RPC 2.0 §4.1, servers MUST
          NOT respond to notifications. Callers must check
          ``response is None`` before reading.
        - Batch dispatch is provided by :py:meth:`handle_batch` for
          callers that need to send multiple requests in one envelope.
        """
        if msg.jsonrpc != "2.0":
            return MCPMessage(
                id=msg.id,
                error={"code": _INVALID_REQUEST,
                        "message": f"Invalid jsonrpc version: {msg.jsonrpc!r}"},
            )

        if msg.method is None:
            return MCPMessage(
                id=msg.id,
                error={"code": _INVALID_REQUEST, "message": "Missing method"},
            )

        is_notification = msg.id is None

        handler = self._method_handlers.get(msg.method)
        if handler is None:
            if is_notification:
                # Notifications get no response, even on error, per spec.
                return None  # type: ignore[return-value]
            return MCPMessage(
                id=msg.id,
                error={"code": _METHOD_NOT_FOUND,
                        "message": f"Unknown method: {msg.method}"},
            )

        try:
            response = handler(msg)
        except Exception as exc:
            if is_notification:
                return None  # type: ignore[return-value]
            return MCPMessage(
                id=msg.id,
                error={"code": _INTERNAL_ERROR, "message": str(exc)},
            )

        if is_notification:
            return None  # type: ignore[return-value]
        return response

    def handle_batch(self, messages):
        """Dispatch a batch of MCPMessage requests, per JSON-RPC 2.0 §6.

        Returns:
        - A single ``MCPMessage`` carrying INVALID_REQUEST when the
          input batch itself is malformed (empty list).
        - ``None`` when the input is non-empty but every member is a
          notification — per spec the server MUST NOT return an empty
          array; the transport must serialize nothing at all.
        - A list of response messages otherwise, in input order, with
          notifications skipped.
        """
        if not messages:
            return MCPMessage(
                id=None,
                error={"code": _INVALID_REQUEST, "message": "Empty batch"},
            )
        out = []
        for m in messages:
            r = self.handle_message(m)
            if r is not None:
                out.append(r)
        if not out:
            # Spec §6: empty Response array MUST NOT be returned;
            # the transport should serialize nothing.
            return None
        return out

    # -----------------------------------------------------------------
    # Handlers
    # -----------------------------------------------------------------

    # Protocol versions this server can negotiate. The advertised
    # canonical version is PROTOCOL_VERSION; we additionally accept the
    # immediately previous draft for backward compatibility.
    SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05",)

    def _handle_initialize(self, msg: MCPMessage) -> MCPMessage:
        """Return protocol version, server info, and capabilities dict.

        Validates the client's requested ``protocolVersion`` against
        ``SUPPORTED_PROTOCOL_VERSIONS``. If the client did not request a
        version, we return ``PROTOCOL_VERSION`` and let the client
        decide whether to proceed. If the client requested a version we
        don't support, we return ``INVALID_PARAMS`` with the
        intersection so the client can pick an acceptable one.
        """
        params = msg.params or {}
        client_caps = params.get("capabilities", {})
        client_version = params.get("protocolVersion")

        if client_version is not None and client_version not in self.SUPPORTED_PROTOCOL_VERSIONS:
            return MCPMessage(
                id=msg.id,
                error={
                    "code": _INVALID_PARAMS,
                    "message": (
                        f"Unsupported protocolVersion {client_version!r}; "
                        f"server supports {list(self.SUPPORTED_PROTOCOL_VERSIONS)}"
                    ),
                    "data": {
                        "supportedVersions": list(self.SUPPORTED_PROTOCOL_VERSIONS),
                    },
                },
            )

        # Negotiated version: when the client requests a version we do
        # support, echo *that* version (not always PROTOCOL_VERSION).
        # Once SUPPORTED_PROTOCOL_VERSIONS gains a second entry the
        # negotiation will be honest; today the loop is a no-op for the
        # single supported version but keeps the contract correct.
        negotiated_version = client_version if client_version else self.PROTOCOL_VERSION

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
                "protocolVersion": negotiated_version,
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
        # Use the public list_tools() API rather than reaching into
        # the private _tools dict, so a future registry refactor
        # doesn't silently break the protocol.
        for spec_dict in self._registry.list_tools():
            # list_tools returns dicts; reconstruct the spec lookup
            # so we keep the existing schema/QoS shaping logic.
            spec = self._registry.get(spec_dict["name"])
            if spec is None:
                continue
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

    # Backward-compatible argument aliases for common tool parameter names
    _ARG_ALIASES = {
        "calculator": {"expression": "expr"},
        "pirag_query": {"q": "query"},
    }

    def _handle_tools_call(self, msg: MCPMessage) -> MCPMessage:
        name = msg.params.get("name", "")
        arguments = dict(msg.params.get("arguments", {}))

        spec = self._registry.get(name)
        if spec is None:
            return MCPMessage(
                id=msg.id,
                error={"code": _METHOD_NOT_FOUND, "message": f"Unknown tool: {name}"},
            )

        # Apply argument aliases so callers using common alternative names work
        aliases = self._ARG_ALIASES.get(name, {})
        for alt, canonical in aliases.items():
            if alt in arguments and canonical not in arguments:
                arguments[canonical] = arguments.pop(alt)

        # Public-facing transport: enforce policy.yaml rate_limits before
        # dispatch. The simulator's in-process registry calls bypass this
        # bucket (source="registry"); only requests that arrive over the
        # MCP JSON-RPC envelope consume a token. See pirag.mcp.rate_limiter.
        try:
            from .rate_limiter import get_rate_limiter, RateLimitExceeded
            get_rate_limiter().check(name, source="transport")
        except RateLimitExceeded as exc:
            return MCPMessage(
                id=msg.id,
                result={
                    "content": [{"type": "text", "text": json.dumps({
                        "error": str(exc),
                        "code": "rate_limit_exceeded",
                    })}],
                    "isError": True,
                },
            )

        try:
            result = self._registry.invoke(name, **arguments)
            # If the tool returned a structured error envelope (e.g.
            # ``{"_status": "error", ...}`` from chain_query when the
            # FastAPI state isn't populated under the simulator
            # subprocess), surface it as ``isError`` so consumers and
            # the ProtocolRecorder count it as an error rather than a
            # successful payload.
            is_error = (
                isinstance(result, dict)
                and (result.get("_status") == "error" or result.get("error"))
            )
            response_result: Dict[str, Any] = {
                "content": [{"type": "text", "text": json.dumps(result, default=str)}],
            }
            if is_error:
                response_result["isError"] = True
            return MCPMessage(id=msg.id, result=response_result)
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
