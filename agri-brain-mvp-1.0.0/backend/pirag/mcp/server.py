"""MCP-compliant REST server with JSON-RPC 2.0 endpoint.

Provides both the new ``POST /mcp`` endpoint (JSON-RPC 2.0, standard MCP)
and legacy ``GET /tools`` + ``POST /call`` endpoints for backward
compatibility.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Header, Request
from pydantic import BaseModel
from src.security import enforce_api_key

from .tools import calculator, units, simulator, policy_oracle
from .tools import compliance, slca_lookup, chain_query


router = APIRouter()

# Legacy tool registry (kept for backward-compatible /tools and /call endpoints).
# The full 12-tool registry is available via the JSON-RPC endpoint (POST /mcp).
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

        # Register live state resources
        try:
            from .resources import register_agent_resources

            def _state_fn():
                """Return current telemetry (fallback to defaults if backend state unavailable)."""
                try:
                    from ...src.routers.state import get_current_state
                    state = get_current_state()
                    return {
                        "temp": state.get("temperature", 4.0),
                        "rh": state.get("humidity", 92.0),
                        "inv": state.get("inventory", 10000),
                        "rho": state.get("spoilage_risk", 0.0),
                        "y_hat": state.get("demand", 15.0),
                        "tau": state.get("tau", 0.0),
                    }
                except Exception:
                    return {"temp": 4.0, "rh": 92.0, "inv": 10000, "rho": 0.0, "y_hat": 15.0, "tau": 0.0}

            register_agent_resources(server, _state_fn)
        except ImportError:
            pass

        _MCP_SERVER = server
    except Exception:
        _MCP_SERVER = None

    return _MCP_SERVER


# ---------------------------------------------------------------------------
# New MCP-compliant endpoint (JSON-RPC 2.0)
# ---------------------------------------------------------------------------
@router.post("/mcp")
def mcp_endpoint(
    req: MCPRequest,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> Dict[str, Any]:
    """JSON-RPC 2.0 MCP endpoint. Routes to MCPServer.handle_message()."""
    enforce_api_key(request, x_api_key)
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
def list_resources(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> Dict[str, Any]:
    """List available MCP resources."""
    enforce_api_key(request, x_api_key)
    server = _get_mcp_server()
    if server is None:
        return {"resources": []}
    from .protocol import MCPMessage
    resp = server.handle_message(MCPMessage(id=0, method="resources/list"))
    return resp.result or {"resources": []}


@router.get("/mcp/prompts")
def list_prompts(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> Dict[str, Any]:
    """List available MCP prompts."""
    enforce_api_key(request, x_api_key)
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
def list_tools(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
):
    """Legacy: list available tools."""
    enforce_api_key(request, x_api_key)
    return {"tools": [{"name": k, "schema": v["schema"]} for k, v in TOOLS.items()]}


@router.post("/call")
def call_tool(
    req: CallReq,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
):
    """Legacy: invoke a tool by name."""
    enforce_api_key(request, x_api_key)
    if req.name not in TOOLS:
        raise HTTPException(404, f"Unknown tool: {req.name}")
    try:
        fn = TOOLS[req.name]["fn"]
        out = fn(**req.args)
        return {"ok": True, "result": out}
    except Exception as e:
        raise HTTPException(400, f"Tool error: {e}")
