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


def _yield_query_args(obs: Any, prior: Dict[str, Any], shared: Any) -> Dict[str, Any]:
    """Path B: pull pre-computed uncertainty from obs.raw when the simulator
    has already run Holt's linear this step. Falls back to inv_history when
    no cached value is present (e.g., FastAPI /decide path).
    """
    raw = getattr(obs, "raw", {}) or {}
    cached_unc = raw.get("supply_uncertainty")
    if cached_unc is not None:
        return {
            "cached_uncertainty": float(cached_unc),
            "cached_forecast":    list(raw.get("supply_hat", [])) if isinstance(raw.get("supply_hat"), (list, tuple)) else [],
            "cached_std":         float(raw.get("supply_std", 0.0)),
        }
    history = list(raw.get("inv_history", []))
    if not history:
        history = [float(getattr(obs, "inv", 0.0))]
    return {"inventory_history": history, "horizon": 6}


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
    # policy_oracle now returns {"allowed": bool, "reason": str, "tool": str};
    # absence (oracle not invoked yet) is permissive (default True), explicit
    # `allowed: False` from the oracle blocks the calculator step.
    oracle = prior.get("policy_oracle") or {}
    return obs.surplus_ratio > 0.5 and bool(oracle.get("allowed", True))


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
    ("yield_query", _always, _yield_query_args),  # Path B
]

COOPERATIVE_WORKFLOW: List[WorkflowStep] = [
    ("slca_lookup", _always, _slca_args),
    ("chain_query", _always, _chain_query_args),
    ("spoilage_forecast", _cooperative_spoilage_trigger, _spoilage_forecast_args),
    ("footprint_query", _always, _footprint_args),
    ("yield_query", _always, _yield_query_args),  # Path B
]

DISTRIBUTOR_WORKFLOW: List[WorkflowStep] = [
    ("check_compliance", _always, _compliance_args),
    ("slca_lookup", _distributor_slca_trigger, _slca_args),
    ("spoilage_forecast", _distributor_spoilage_trigger, _spoilage_forecast_args),
    ("recovery_capacity_check", _distributor_recovery_trigger, lambda obs, p, s: {}),  # KEEP - late-bound
    ("calculator", _distributor_calculator_trigger, _calculator_surplus_args),
    ("yield_query", _always, _yield_query_args),  # Path B
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


_dispatch_id_lock = __import__("threading").Lock()
_dispatch_id_counter = 0


def _next_dispatch_id() -> int:
    """Monotonic id for dispatcher-issued tools/call messages.

    Replaces the previous hardcoded ``id=0`` so recorded traces have a
    real correlation key per request. Thread-safe in the unlikely
    event two coordinator threads race.
    """
    global _dispatch_id_counter
    with _dispatch_id_lock:
        _dispatch_id_counter += 1
        return _dispatch_id_counter


def reset_dispatch_id_counter() -> None:
    """Reset the monotonic dispatch-id counter to 0.

    Called by ``AgentCoordinator.reset`` so per-episode protocol
    traces use comparable id ranges (otherwise the counter grows
    unboundedly across the simulator's 5-scenario × 20-mode loop and
    reviewers comparing two scenario runs see disjoint id ranges).
    """
    global _dispatch_id_counter
    with _dispatch_id_lock:
        _dispatch_id_counter = 0


class _ToolProtocolError(RuntimeError):
    """Raised by ``_invoke_via_protocol`` when the protocol-routed tool
    call returned a JSON-RPC error or a structured ``isError: True``
    response. Caught by ``dispatch_tools`` so the failure populates
    ``_tools_failed`` instead of being silently masked as a successful
    payload (the previous behaviour after the 2026-04 audit fix that
    introduced ``isError`` surfacing).
    """


def _invoke_via_protocol(
    server: Any,
    tool_name: str,
    kwargs: Dict[str, Any],
) -> Any:
    """Invoke a tool through the MCP JSON-RPC protocol layer.

    Sends a real ``tools/call`` message through ``server.handle_message``
    with a monotonic id. Recorded by ProtocolRecorder as an in-process
    dispatch entry (see ``protocol_recorder.py`` docstring for the
    distinction between dispatch traces and wire traffic). Raises
    ``_ToolProtocolError`` on JSON-RPC error or ``result.isError = True``
    so the caller can record the failure rather than masking it.
    """
    from .protocol import MCPMessage

    msg = MCPMessage(
        id=_next_dispatch_id(),
        method="tools/call",
        params={"name": tool_name, "arguments": kwargs},
    )
    resp = server.handle_message(msg)

    if resp is None:
        # Notification path — should not happen for tools/call, but
        # defend the contract.
        raise _ToolProtocolError(f"{tool_name}: no response (notification path)")

    if resp.error:
        raise _ToolProtocolError(f"{tool_name}: jsonrpc error {resp.error}")

    # Extract result from MCP response envelope
    result = resp.result
    parsed: Any = result
    if isinstance(result, dict):
        content = result.get("content", [])
        if content and isinstance(content, list) and content[0].get("type") == "text":
            try:
                parsed = json.loads(content[0]["text"])
            except (json.JSONDecodeError, KeyError, TypeError):
                parsed = content[0].get("text")
        # MCP 2024-11-05: tool failures arrive as result.isError = True
        # with the error payload in content.text. Surface as failure so
        # _tools_failed in dispatch_tools is honest.
        if result.get("isError"):
            raise _ToolProtocolError(f"{tool_name}: tool reported isError; payload={parsed!r}")
    return parsed


def dispatch_tools(
    role: str,
    obs: Any,
    registry: ToolRegistry,
    shared_context: Any = None,
    mcp_server: Any = None,
    dispatch_config: Optional[Dict[str, Any]] = None,
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
    cfg = dispatch_config or {}
    workflow = _select_workflow(role, registry, cfg)
    results: Dict[str, Any] = {}
    invoked: List[str] = []
    failed: List[str] = []
    skipped: List[str] = []

    def _execute(tool_name: str, trigger_fn, args_fn) -> None:
        """Run a single workflow step: trigger -> args -> invoke."""
        try:
            should_invoke = trigger_fn(obs, results, shared_context)
        except Exception:
            should_invoke = False

        if not should_invoke:
            skipped.append(tool_name)
            return

        spec = registry.get(tool_name)
        if spec is None:
            skipped.append(tool_name)
            return

        try:
            kwargs = args_fn(obs, results, shared_context)
            result = _invoke_with_reliability(
                registry=registry,
                mcp_server=mcp_server,
                tool_name=tool_name,
                kwargs=kwargs,
                cfg=cfg,
            )
            results[tool_name] = result
            invoked.append(tool_name)
        except Exception:
            failed.append(tool_name)
            results[tool_name] = None

    # Pass 1 — static role workflow.
    for step in workflow:
        _execute(*step)

    # Pass 2 — observe-then-decide loop (real ReAct closed loop).
    # If the static workflow surfaced a critical compliance violation
    # but the agent has not yet looked up forward spoilage risk, invoke
    # spoilage_forecast as a follow-up. Then if the forecast says the
    # produce will be at high risk in the next 6 hours, re-run
    # compliance with a tightened temperature target so the next-step
    # decision sees the escalated state. This is the only place in the
    # dispatcher where one tool's *result* drives whether another tool
    # runs and with what arguments — i.e. an actual "perceive ->
    # think -> act -> observe" iteration.
    react_iterations = 0
    react_max_iter = int(cfg.get("react_max_iter", 2))
    while react_iterations < react_max_iter:
        react_iterations += 1
        compliance = results.get("check_compliance") or {}
        is_critical = (
            isinstance(compliance, dict)
            and not compliance.get("compliant", True)
            and any(
                v.get("severity") == "critical"
                for v in compliance.get("violations", []) or []
            )
        )
        if not is_critical:
            break
        # Followup A: ensure spoilage_forecast was run with the actual
        # observed conditions; if not, run it now.
        spec_sf = registry.get("spoilage_forecast")
        if spec_sf is not None and "spoilage_forecast" not in invoked:
            try:
                sf = _invoke_with_reliability(
                    registry=registry,
                    mcp_server=mcp_server,
                    tool_name="spoilage_forecast",
                    kwargs={
                        "current_rho": float(getattr(obs, "rho", 0.0)),
                        "temperature": float(getattr(obs, "temp", 8.0)),
                        "humidity": float(getattr(obs, "rh", 90.0)),
                        "hours_ahead": 6,
                    },
                    cfg=cfg,
                )
                results["spoilage_forecast"] = sf
                invoked.append("spoilage_forecast")
            except Exception:
                failed.append("spoilage_forecast")
                results["spoilage_forecast"] = None
        # Observe: did spoilage_forecast confirm the threat?
        sf = results.get("spoilage_forecast") or {}
        forecast_high = isinstance(sf, dict) and sf.get("urgency") in {"high", "critical"}
        # Followup B: if both compliance is critical AND forecast is
        # high, escalate by re-running compliance with the tightened
        # FDA target threshold (4 C, the leafy-greens tighter floor).
        # This produces a fresh `check_compliance` result tagged
        # `_react_iteration` so downstream consumers can audit which
        # call actually drove the decision.
        if forecast_high and "check_compliance_react" not in results:
            spec_cc = registry.get("check_compliance")
            if spec_cc is not None:
                try:
                    cc2 = _invoke_with_reliability(
                        registry=registry,
                        mcp_server=mcp_server,
                        tool_name="check_compliance",
                        kwargs={
                            "temperature": float(getattr(obs, "temp", 8.0)),
                            "humidity": float(getattr(obs, "rh", 90.0)),
                            "product_type": "spinach_tightened",
                        },
                        cfg=cfg,
                    )
                    if isinstance(cc2, dict):
                        cc2["_react_iteration"] = react_iterations
                    results["check_compliance_react"] = cc2
                    invoked.append("check_compliance_react")
                except Exception:
                    failed.append("check_compliance_react")
                    results["check_compliance_react"] = None
            # Loop terminates here: we have the escalated compliance
            # result; further iterations would not produce new
            # information given the static argument schema.
            break
        # No new tool to call; exit the loop.
        break

    results["_tools_invoked"] = invoked
    results["_tools_failed"] = failed
    results["_tools_skipped"] = skipped
    results["_dispatch_profile"] = cfg.get("qos_profile", "legacy")
    results["_react_iterations"] = react_iterations
    return results


def _select_workflow(role: str, registry: ToolRegistry, cfg: Dict[str, Any]) -> List[WorkflowStep]:
    workflow = ROLE_WORKFLOWS.get(role, [])
    if not cfg.get("enable_qos_routing", False):
        return workflow
    # QoS profile currently reorders by latency/cost tiers while preserving role workflow membership.
    scored: List[Tuple[int, WorkflowStep]] = []
    preferred_qos = cfg.get("role_preferred_qos", "standard")
    for step in workflow:
        tool_name = step[0]
        spec = registry.get(tool_name)
        if spec is None:
            scored.append((999, step))
            continue
        latency_rank = {"low": 0, "medium": 1, "high": 2}.get(spec.latency_tier, 1)
        cost_rank = {"low": 0, "medium": 1, "high": 2}.get(spec.cost_tier, 1)
        rel_rank = {"high": 0, "standard": 1, "best_effort": 2}.get(spec.reliability_tier, 1)
        score = latency_rank * 3 + cost_rank * 2 + rel_rank
        if preferred_qos == "low_latency":
            score = latency_rank * 4 + cost_rank + rel_rank
        elif preferred_qos == "low_cost":
            score = cost_rank * 4 + latency_rank + rel_rank
        elif preferred_qos == "high_reliability":
            score = rel_rank * 4 + latency_rank + cost_rank
        scored.append((score, step))
    scored.sort(key=lambda x: x[0])
    return [s for _, s in scored]


_CB = None


def _invoke_with_reliability(
    registry: ToolRegistry,
    mcp_server: Any,
    tool_name: str,
    kwargs: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Any:
    if not cfg.get("enable_reliability", False):
        if mcp_server is not None:
            return _invoke_via_protocol(mcp_server, tool_name, kwargs)
        return registry.invoke(tool_name, **kwargs)

    from .reliability import CircuitBreaker, invoke_with_retry

    global _CB
    if _CB is None:
        _CB = CircuitBreaker()

    if not _CB.allow(tool_name):
        return None

    def _do_call() -> Any:
        if mcp_server is not None:
            return _invoke_via_protocol(mcp_server, tool_name, kwargs)
        return registry.invoke(tool_name, **kwargs)

    try:
        out = invoke_with_retry(_do_call, retries=cfg.get("retries", 1))
        _CB.on_success(tool_name)
        return out
    except Exception:
        _CB.on_failure(tool_name)
        raise
