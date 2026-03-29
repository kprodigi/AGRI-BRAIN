"""MCP-compliant REST server with JSON-RPC 2.0 endpoint.

Provides both the new ``POST /mcp`` endpoint (JSON-RPC 2.0, standard MCP)
and legacy ``GET /tools`` + ``POST /call`` endpoints for backward
compatibility.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .tools import calculator, units, simulator, policy_oracle
from .tools import compliance, slca_lookup, chain_query


router = APIRouter()

# Legacy tool registry (kept for backward-compatible /tools and /call endpoints)
TOOLS = {
    "calculator": {"fn": calculator.calculate, "schema": {"expr": "str"}},
    "convert_units": {"fn": units.convert, "schema": {"value": "float", "from_unit": "str", "to_unit": "str"}},
    "simulate": {"fn": simulator.simulate, "schema": {"endpoint": "str", "payload": "dict"}},
    "policy_check": {"fn": policy_oracle.check_access, "schema": {"user_id": "str", "tool_name": "str"}},
    "check_compliance": {"fn": compliance.check_compliance, "schema": {"temperature": "float", "humidity": "float", "product_type": "str"}},
    "slca_lookup": {"fn": slca_lookup.lookup_slca_weights, "schema": {"product_type": "str"}},
    "chain_query": {"fn": chain_query.query_recent_decisions, "schema": {"n": "int"}},
}


class CallReq(BaseModel):
    name: str
    args: Dict[str, Any]


class MCPRequest(BaseModel):
    """JSON-RPC 2.0 request envelope."""
    jsonrpc: str = "2.0"
    id: int | None = None
    method: str | None = None
    params: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# MCP server singleton (lazy init)
# ---------------------------------------------------------------------------
_MCP_SERVER = None


def _get_mcp_server():
    """Lazy-initialize the MCPServer with registry, resources, and prompts."""
    global _MCP_SERVER
    if _MCP_SERVER is not None:
        return _MCP_SERVER

    try:
        from .registry import get_default_registry
        from .protocol import MCPServer
        from .prompts import register_prompts

        registry = get_default_registry()
        server = MCPServer(registry=registry)
        register_prompts(server)

        _MCP_SERVER = server
    except Exception:
        _MCP_SERVER = None

    return _MCP_SERVER


# ---------------------------------------------------------------------------
# New MCP-compliant endpoint (JSON-RPC 2.0)
# ---------------------------------------------------------------------------
@router.post("/mcp")
def mcp_endpoint(req: MCPRequest) -> Dict[str, Any]:
    """JSON-RPC 2.0 MCP endpoint. Routes to MCPServer.handle_message()."""
    server = _get_mcp_server()
    if server is None:
        return {
            "jsonrpc": "2.0",
            "id": req.id,
            "error": {"code": -32603, "message": "MCP server initialization failed"},
        }

    from .protocol import MCPMessage
    msg = MCPMessage(
        jsonrpc=req.jsonrpc,
        id=req.id,
        method=req.method,
        params=req.params,
    )
    response = server.handle_message(msg)

    result: Dict[str, Any] = {"jsonrpc": response.jsonrpc, "id": response.id}
    if response.result is not None:
        result["result"] = response.result
    if response.error is not None:
        result["error"] = response.error
    return result


# ---------------------------------------------------------------------------
# Convenience endpoints for resources and prompts
# ---------------------------------------------------------------------------
@router.get("/mcp/resources")
def list_resources() -> Dict[str, Any]:
    """List available MCP resources."""
    server = _get_mcp_server()
    if server is None:
        return {"resources": []}
    from .protocol import MCPMessage
    resp = server.handle_message(MCPMessage(id=0, method="resources/list"))
    return resp.result or {"resources": []}


@router.get("/mcp/prompts")
def list_prompts() -> Dict[str, Any]:
    """List available MCP prompts."""
    server = _get_mcp_server()
    if server is None:
        return {"prompts": []}
    from .protocol import MCPMessage
    resp = server.handle_message(MCPMessage(id=0, method="prompts/list"))
    return resp.result or {"prompts": []}


# ---------------------------------------------------------------------------
# Legacy endpoints (backward-compatible, deprecated)
# ---------------------------------------------------------------------------
@router.get("/tools")
def list_tools():
    """Legacy: list available tools."""
    return {"tools": [{"name": k, "schema": v["schema"]} for k, v in TOOLS.items()]}


@router.post("/call")
def call_tool(req: CallReq):
    """Legacy: invoke a tool by name."""
    if req.name not in TOOLS:
        raise HTTPException(404, f"Unknown tool: {req.name}")
    try:
        fn = TOOLS[req.name]["fn"]
        out = fn(**req.args)
        return {"ok": True, "result": out}
    except Exception as e:
        raise HTTPException(400, f"Tool error: {e}")
