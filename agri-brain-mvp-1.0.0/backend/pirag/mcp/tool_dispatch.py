"""Role-specific MCP tool dispatch with multi-step composition.

Each agent role has an ordered workflow of conditional tool invocations.
Later steps in a workflow can consume results from earlier steps, enabling
multi-step composition. The dispatcher also accepts a shared context store
so agents can reuse results published by upstream agents.

When an ``mcp_server`` is provided, tool invocations are routed through
the MCP JSON-RPC protocol layer so that the ProtocolRecorder captures
every interaction as genuine protocol traffic.

Trigger functions receive ``(obs, prior_results, shared_context)`` and
return True when the tool should be invoked. Argument functions receive
the same triple and return a kwargs dict for the tool.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..mcp.registry import ToolRegistry


# Type aliases for readability
TriggerFn = Callable[[Any, Dict[str, Any], Any], bool]
ArgsFn = Callable[[Any, Dict[str, Any], Any], Dict[str, Any]]
WorkflowStep = Tuple[str, TriggerFn, ArgsFn]


def _always(obs: Any, prior: Dict[str, Any], shared: Any) -> bool:
    return True


# ---------------------------------------------------------------------------
# Argument builders
# ---------------------------------------------------------------------------

def _compliance_args(obs: Any, prior: Dict[str, Any], shared: Any) -> Dict[str, Any]:
    return {"temperature": obs.temp, "humidity": obs.rh, "product_type": "spinach"}


def _slca_args(obs: Any, prior: Dict[str, Any], shared: Any) -> Dict[str, Any]:
    return {"product_type": "spinach"}


def _chain_query_args(obs: Any, prior: Dict[str, Any], shared: Any) -> Dict[str, Any]:
    return {"n": 5}


def _spoilage_forecast_args(obs: Any, prior: Dict[str, Any], shared: Any) -> Dict[str, Any]:
    return {
        "current_rho": obs.rho,
        "temperature": obs.temp,
        "humidity": obs.rh,
        "hours_ahead": 6,
    }


def _footprint_args(obs: Any, prior: Dict[str, Any], shared: Any) -> Dict[str, Any]:
    steps = getattr(obs, "_steps_completed", 1)
    return {"steps_completed": steps}


def _policy_oracle_args(obs: Any, prior: Dict[str, Any], shared: Any) -> Dict[str, Any]:
    return {"user_id": "system", "tool_name": "surplus_management"}


def _calculator_surplus_args(obs: Any, prior: Dict[str, Any], shared: Any) -> Dict[str, Any]:
    inv = obs.inv
    baseline = 12000.0
    expr = f"{inv} - {baseline}"
    return {"expr": expr}


# ---------------------------------------------------------------------------
# Trigger functions
# ---------------------------------------------------------------------------

def _farm_slca_trigger(obs: Any, prior: Dict[str, Any], shared: Any) -> bool:
    compliance = prior.get("check_compliance", {})
    return not compliance.get("compliant", True) or obs.rho > 0.20


def _farm_spoilage_trigger(obs: Any, prior: Dict[str, Any], shared: Any) -> bool:
    compliance = prior.get("check_compliance", {})
    violations = compliance.get("violations", [])
    return any(v.get("severity") == "critical" for v in violations)


def _processor_compliance_trigger(obs: Any, prior: Dict[str, Any], shared: Any) -> bool:
    if shared is not None:
        upstream = shared.get_upstream_compliance(obs.hour)
        if upstream is not None:
            prior["check_compliance"] = upstream
            return False
    return True


def _processor_policy_trigger(obs: Any, prior: Dict[str, Any], shared: Any) -> bool:
    return obs.surplus_ratio > 0.3


def _processor_chain_trigger(obs: Any, prior: Dict[str, Any], shared: Any) -> bool:
    return obs.surplus_ratio > 0.5


def _processor_calculator_trigger(obs: Any, prior: Dict[str, Any], shared: Any) -> bool:
    return obs.surplus_ratio > 0.5 and prior.get("policy_oracle") is not False


def _distributor_slca_trigger(obs: Any, prior: Dict[str, Any], shared: Any) -> bool:
    return obs.rho > 0.30


def _distributor_spoilage_trigger(obs: Any, prior: Dict[str, Any], shared: Any) -> bool:
    return obs.rho > 0.35


def _distributor_recovery_trigger(obs: Any, prior: Dict[str, Any], shared: Any) -> bool:
    return obs.rho > 0.40


def _distributor_calculator_trigger(obs: Any, prior: Dict[str, Any], shared: Any) -> bool:
    return obs.rho > 0.45


def _cooperative_spoilage_trigger(obs: Any, prior: Dict[str, Any], shared: Any) -> bool:
    return obs.tau > 0.5


# ---------------------------------------------------------------------------
# Role workflows
# ---------------------------------------------------------------------------

FARM_WORKFLOW: List[WorkflowStep] = [
    ("check_compliance", _always, _compliance_args),
    ("slca_lookup", _farm_slca_trigger, _slca_args),
    ("spoilage_forecast", _farm_spoilage_trigger, _spoilage_forecast_args),
]

PROCESSOR_WORKFLOW: List[WorkflowStep] = [
    ("check_compliance", _processor_compliance_trigger, _compliance_args),
    ("policy_oracle", _processor_policy_trigger, _policy_oracle_args),
    ("chain_query", _processor_chain_trigger, _chain_query_args),
    ("calculator", _processor_calculator_trigger, _calculator_surplus_args),
]

COOPERATIVE_WORKFLOW: List[WorkflowStep] = [
    ("slca_lookup", _always, _slca_args),
    ("chain_query", _always, _chain_query_args),
    ("spoilage_forecast", _cooperative_spoilage_trigger, _spoilage_forecast_args),
    ("footprint_query", _always, _footprint_args),
]

DISTRIBUTOR_WORKFLOW: List[WorkflowStep] = [
    ("check_compliance", _always, _compliance_args),
    ("slca_lookup", _distributor_slca_trigger, _slca_args),
    ("spoilage_forecast", _distributor_spoilage_trigger, _spoilage_forecast_args),
    ("recovery_capacity_check", _distributor_recovery_trigger, lambda obs, p, s: {}),
    ("calculator", _distributor_calculator_trigger, _calculator_surplus_args),
]

RECOVERY_WORKFLOW: List[WorkflowStep] = [
    ("chain_query", _always, _chain_query_args),
    ("slca_lookup", _always, _slca_args),
    ("footprint_query", _always, _footprint_args),
]

ROLE_WORKFLOWS: Dict[str, List[WorkflowStep]] = {
    "farm": FARM_WORKFLOW,
    "processor": PROCESSOR_WORKFLOW,
    "cooperative": COOPERATIVE_WORKFLOW,
    "distributor": DISTRIBUTOR_WORKFLOW,
    "recovery": RECOVERY_WORKFLOW,
}


def _invoke_via_protocol(
    server: Any,
    tool_name: str,
    kwargs: Dict[str, Any],
) -> Any:
    """Invoke a tool through the MCP JSON-RPC protocol layer.

    This ensures the ProtocolRecorder captures the interaction as a
    genuine ``tools/call`` request/response pair.
    """
    from .protocol import MCPMessage

    msg = MCPMessage(
        id=0,
        method="tools/call",
        params={"name": tool_name, "arguments": kwargs},
    )
    resp = server.handle_message(msg)

    if resp.error:
        return None

    # Extract result from MCP response envelope
    result = resp.result
    if isinstance(result, dict):
        content = result.get("content", [])
        if content and isinstance(content, list) and content[0].get("type") == "text":
            try:
                return json.loads(content[0]["text"])
            except (json.JSONDecodeError, KeyError, TypeError):
                return content[0].get("text")
    return result


def dispatch_tools(
    role: str,
    obs: Any,
    registry: ToolRegistry,
    shared_context: Any = None,
    mcp_server: Any = None,
) -> Dict[str, Any]:
    """Execute the tool workflow for a given role sequentially.

    Parameters
    ----------
    role : agent role name.
    obs : current Observation.
    registry : MCP tool registry.
    shared_context : optional shared context store.
    mcp_server : optional MCPServer for protocol-level routing.
        When provided, tool calls go through the JSON-RPC protocol
        layer so the ProtocolRecorder captures genuine interactions.

    Returns a dict mapping tool names to their results, plus metadata keys:
    ``_tools_invoked``, ``_tools_failed``, ``_tools_skipped``.
    """
    workflow = ROLE_WORKFLOWS.get(role, [])
    results: Dict[str, Any] = {}
    invoked: List[str] = []
    failed: List[str] = []
    skipped: List[str] = []

    for tool_name, trigger_fn, args_fn in workflow:
        try:
            should_invoke = trigger_fn(obs, results, shared_context)
        except Exception:
            should_invoke = False

        if not should_invoke:
            skipped.append(tool_name)
            continue

        spec = registry.get(tool_name)
        if spec is None:
            skipped.append(tool_name)
            continue

        try:
            kwargs = args_fn(obs, results, shared_context)
            if mcp_server is not None:
                result = _invoke_via_protocol(mcp_server, tool_name, kwargs)
            else:
                result = registry.invoke(tool_name, **kwargs)
            results[tool_name] = result
            invoked.append(tool_name)
        except Exception:
            failed.append(tool_name)
            results[tool_name] = None

    results["_tools_invoked"] = invoked
    results["_tools_failed"] = failed
    results["_tools_skipped"] = skipped
    return results
