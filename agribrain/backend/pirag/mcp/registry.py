"""MCP tool registry with capability-based discovery and episode-level caching.

Provides the central tool registry for AGRI-BRAIN's MCP implementation.
Agents query capabilities (e.g., ["regulatory", "temperature"]) and the
registry returns matching tools, enabling dynamic discovery rather than
hardcoded imports.

Caching is opt-in per tool via ``cacheable`` and ``cache_key_params``.
Episode-scoped: call ``clear_cache()`` between episodes.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


_log = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """Specification for an MCP-registered tool.

    Parameters
    ----------
    name : unique tool identifier.
    description : human-readable purpose.
    capabilities : capability tags used for discovery queries.
    fn : callable implementing the tool.
    schema : parameter name → type string mapping.
    cacheable : whether results can be cached within an episode.
    cache_key_params : parameter names forming the cache key.
    """
    name: str
    description: str
    capabilities: List[str]
    fn: Callable[..., Any]
    schema: Dict[str, Any]  # param_name -> {"type": ..., "description": ...} or legacy "type_str"
    cacheable: bool = False
    cache_key_params: List[str] = field(default_factory=list)
    latency_tier: str = "medium"
    reliability_tier: str = "standard"
    cost_tier: str = "medium"
    role_affinity: List[str] = field(default_factory=list)


class ToolRegistry:
    """Registry of MCP tools with capability-based discovery and caching."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self._cache: Dict[str, Any] = {}
        # Per-instance record of optional tools that failed to register
        # (import errors, missing env, etc.). Previously a module-global
        # — that leaked failures across registries built in the same
        # process (notably the test suite, where a fresh ``ToolRegistry``
        # would inherit stale failures from the singleton build). Each
        # instance now owns its own dict, so a freshly constructed
        # registry starts clean.
        self._registration_failures: Dict[str, str] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register a tool specification.

        Permissive by default (preserves the legacy overwrite
        semantics that ``register_*_capabilities`` helpers rely on).
        Duplicate registrations with a *different underlying callable*
        log at WARN with the previous and replacement tool ids so silent
        shadowing is auditable. Re-registrations of the *same callable*
        (which is what happens when benchmark code recreates a
        coordinator/registry per window — common in
        ``run_external_validity.py`` and the stress suite) are silent;
        the dispatch behaviour is unchanged so a warning would just be
        log noise. Callers that want strict deduplication can use
        :meth:`replace` (explicit overwrite) or :meth:`register_strict`
        (raise on duplicate).
        """
        existing = self._tools.get(spec.name)
        if existing is not None and existing is not spec:
            # Compare the underlying functions, not the ToolSpec objects.
            # When a benchmark loop reconstructs the coordinator each
            # iteration, it builds new ToolSpec wrappers around the same
            # capability function; ``existing.fn is spec.fn`` is the
            # right "this is genuinely the same registration" check.
            if existing.fn is not spec.fn:
                _log.warning(
                    "MCP registry: overwriting tool %r (previous fn=%s, new fn=%s)",
                    spec.name,
                    getattr(existing.fn, "__qualname__", existing.fn),
                    getattr(spec.fn, "__qualname__", spec.fn),
                )
        self._tools[spec.name] = spec

    def register_strict(self, spec: ToolSpec) -> None:
        """Register a tool, raising ``ValueError`` on duplicates."""
        existing = self._tools.get(spec.name)
        if existing is not None and existing is not spec:
            raise ValueError(
                f"Tool {spec.name!r} already registered "
                "(use register() / replace() to overwrite intentionally)"
            )
        self._tools[spec.name] = spec

    def replace(self, spec: ToolSpec) -> None:
        """Overwrite an existing registration without raising."""
        self._tools[spec.name] = spec

    def unregister(self, tool_name: str) -> bool:
        """Remove a tool from the registry. Returns True if removed."""
        return self._tools.pop(tool_name, None) is not None

    def discover(self, capabilities: List[str]) -> List[ToolSpec]:
        """Return tools matching ANY requested capability."""
        cap_set = set(capabilities)
        return [
            spec for spec in self._tools.values()
            if cap_set & set(spec.capabilities)
        ]

    def invoke(self, tool_name: str, **kwargs: Any) -> Any:
        """Invoke a tool by name. Uses cache if applicable.

        Raises
        ------
        KeyError
            If ``tool_name`` is not registered.
        """
        spec = self._tools[tool_name]

        if spec.cacheable and spec.cache_key_params:
            key_data = {p: kwargs.get(p) for p in spec.cache_key_params}
            # SHA-256 instead of MD5 — same cache-key role, but Bandit /
            # security scanners flag MD5 by default and it tends to
            # raise questions. Cost difference is irrelevant at
            # registry-cache frequency.
            cache_key = f"{tool_name}:{hashlib.sha256(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()}"
            if cache_key in self._cache:
                return self._cache[cache_key]
            result = spec.fn(**kwargs)
            self._cache[cache_key] = result
            return result

        return spec.fn(**kwargs)

    def get(self, tool_name: str) -> Optional[ToolSpec]:
        """Return a tool spec by name, or None if not found."""
        return self._tools.get(tool_name)

    def clear_cache(self) -> None:
        """Clear all cached tool results (call between episodes)."""
        self._cache.clear()

    def registration_failures(self) -> Dict[str, str]:
        """Return a defensive copy of optional-tool registration failures.

        Keys are tool names that failed to register; values are the
        stringified exception (or a human-readable reason such as
        ``"SIM_API_BASE not configured"``). The dict is a copy, so
        mutating it does not affect the registry's internal state.
        """
        return dict(self._registration_failures)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return serializable list of all registered tools."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "capabilities": s.capabilities,
                "schema": s.schema,
                "qos": {
                    "latency_tier": s.latency_tier,
                    "reliability_tier": s.reliability_tier,
                    "cost_tier": s.cost_tier,
                    "role_affinity": s.role_affinity,
                },
            }
            for s in self._tools.values()
        ]


_DEFAULT_REGISTRY: Optional[ToolRegistry] = None


def get_default_registry() -> ToolRegistry:
    """Lazy singleton with all built-in tools registered."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is not None:
        return _DEFAULT_REGISTRY

    registry = ToolRegistry()

    from .tools.compliance import check_compliance
    from .tools.slca_lookup import lookup_slca_weights
    from .tools.chain_query import query_recent_decisions
    from .tools.policy_oracle import check_access
    from .tools.calculator import calculate
    from .tools.units import convert
    # `simulate` is registered conditionally below — it depends on
    # SIM_API_BASE being set, which is empty under the simulator
    # subprocess. Importing it eagerly is fine; gating the registration
    # so the simulator-mode registry doesn't expose a tool that always
    # returns _status: error is the honest move.
    from .tools.simulator import simulate

    registry.register(ToolSpec(
        name="check_compliance",
        description="Check FDA temperature and humidity compliance for produce",
        capabilities=["regulatory", "temperature", "quality"],
        fn=check_compliance,
        schema={
            "temperature": {"type": "number", "description": "Current product temperature in degrees Celsius"},
            "humidity": {"type": "number", "description": "Relative humidity as a percentage (0-100)"},
            "product_type": {"type": "string", "description": "Produce type (e.g., spinach, lettuce, romaine)"},
        },
        cacheable=False,
    ))
    registry.register(ToolSpec(
        name="slca_lookup",
        description="Look up SLCA weights and base scores by product type",
        capabilities=["social", "scoring", "routing"],
        fn=lookup_slca_weights,
        schema={
            "product_type": {"type": "string", "description": "Produce type for SLCA scoring lookup"},
        },
        cacheable=True,
        cache_key_params=["product_type"],
    ))
    registry.register(ToolSpec(
        name="chain_query",
        description="Query recent routing decisions from blockchain audit trail",
        capabilities=["blockchain", "audit", "history"],
        fn=query_recent_decisions,
        schema={
            "n": {"type": "integer", "description": "Number of recent decisions to retrieve (default: 10)"},
        },
        cacheable=False,
    ))
    registry.register(ToolSpec(
        name="policy_oracle",
        description="Check governance policy access for a user and tool",
        capabilities=["governance", "access", "policy"],
        fn=check_access,
        schema={
            "user_id": {"type": "string", "description": "Agent or user identifier requesting access"},
            "tool_name": {"type": "string", "description": "Name of the tool to check access for"},
        },
        cacheable=True,
        cache_key_params=["user_id", "tool_name"],
    ))
    registry.register(ToolSpec(
        name="calculator",
        description="Evaluate a safe arithmetic expression",
        capabilities=["math", "estimation"],
        fn=calculate,
        schema={
            "expr": {"type": "string", "description": "Safe arithmetic expression to evaluate (e.g., '2.5 * 3.14')"},
        },
        cacheable=False,
    ))
    registry.register(ToolSpec(
        name="convert_units",
        description="Convert a numeric value between measurement units",
        capabilities=["units", "conversion"],
        fn=convert,
        schema={
            "value": {"type": "number", "description": "Numeric value to convert"},
            "from_unit": {"type": "string", "description": "Source unit (e.g., kg, lb, celsius, fahrenheit)"},
            "to_unit": {"type": "string", "description": "Target unit to convert into"},
        },
        cacheable=True,
        cache_key_params=["value", "from_unit", "to_unit"],
    ))
    # Conditional registration: `simulate` is only useful when the
    # internal simulation API base is configured (SIM_API_BASE). Under
    # the simulator subprocess the base is empty, so registering the
    # tool there means every protocol-routed call returns
    # `_status: error` — which then inflates the recorder's
    # tool_iserror_responses count. The honest behaviour is to skip
    # registration unless the base is set; FastAPI process-style runs
    # will register it because they have SIM_API_BASE configured.
    try:
        from src.settings import SETTINGS as _SETTINGS
        _sim_base = (getattr(_SETTINGS, "sim_api_base", "") or "").strip()
    except Exception:
        _sim_base = ""
    if _sim_base:
        registry.register(ToolSpec(
            name="simulate",
            description="Run a forward simulation via the simulation API",
            capabilities=["simulation", "forecast"],
            fn=simulate,
            schema={
                "endpoint": {"type": "string", "description": "Simulation API endpoint path"},
                "payload": {"type": "object", "description": "JSON payload with simulation parameters"},
            },
            cacheable=False,
        ))
    else:
        _log.info(
            "MCP tool 'simulate' not registered: SIM_API_BASE is empty "
            "(simulator subprocess); set SIM_API_BASE in the runtime env "
            "to enable forward-simulation MCP calls."
        )
        registry._registration_failures["simulate"] = "SIM_API_BASE not configured"

    # New tools (Tasks 3-4) — registered if importable. Failures are
    # logged at WARN and recorded on the registry instance's
    # ``_registration_failures`` dict so the operator can detect partial
    # registration via the ``mcp_registration_status()`` helper or the
    # ``mcp.registry.status`` MCP resource.
    try:
        from .tools.spoilage_forecast import forecast_spoilage
        registry.register(ToolSpec(
            name="spoilage_forecast",
            description="Integrate Arrhenius-Baranyi ODE forward for spoilage prediction",
            capabilities=["spoilage", "forecast", "physics"],
            fn=forecast_spoilage,
            schema={
                "current_rho": {"type": "number", "description": "Current spoilage risk (0.0 = fresh, 1.0 = fully spoiled)"},
                "temperature": {"type": "number", "description": "Current product temperature in degrees Celsius"},
                "humidity": {"type": "number", "description": "Relative humidity as a percentage (0-100)"},
                "hours_ahead": {"type": "integer", "description": "Hours to forecast ahead (default: 6)"},
            },
            cacheable=False,
        ))
    except Exception as exc:  # 2026-04: catch ALL registration failures, not only ImportError
        _log.warning("MCP tool 'spoilage_forecast' not registered: %s", exc)
        registry._registration_failures["spoilage_forecast"] = str(exc)

    try:
        from .tools.footprint_query import query_footprint
        registry.register(ToolSpec(
            name="footprint_query",
            description="Return cumulative and per-step energy and water footprint",
            capabilities=["footprint", "green_ai", "monitoring"],
            fn=query_footprint,
            schema={
                "steps_completed": {"type": "integer", "description": "Number of simulation steps completed"},
                "energy_per_step_j": {"type": "number", "description": "Energy consumption per step in joules"},
                "water_per_step_l": {"type": "number", "description": "Water consumption per step in litres"},
            },
            cacheable=False,
        ))
    except Exception as exc:  # 2026-04: catch ALL registration failures, not only ImportError
        _log.warning("MCP tool 'footprint_query' not registered: %s", exc)
        registry._registration_failures["footprint_query"] = str(exc)

    # piRAG, explanation, and context feature tools
    try:
        from .tools.pirag_query import pirag_query
        registry.register(ToolSpec(
            name="pirag_query",
            description="Query the piRAG knowledge base with physics-informed retrieval",
            capabilities=["retrieval", "knowledge", "pirag", "regulatory"],
            fn=pirag_query,
            schema={
                "query": {"type": "string", "description": "Natural language search query for the knowledge base"},
                "k": {"type": "integer", "description": "Number of documents to retrieve (default: 4, max: 10)"},
                "role": {"type": "string", "description": "Agent role for query context (farm, processor, distributor, recovery)"},
                "temperature": {"type": "number", "description": "Current temperature in Celsius for physics-informed expansion"},
                "rho": {"type": "number", "description": "Current spoilage risk (0-1) for physics-informed expansion"},
                "humidity": {"type": "number", "description": "Relative humidity for physics-informed reranking"},
            },
            cacheable=False,
        ))
    except Exception as exc:  # 2026-04: catch ALL registration failures, not only ImportError
        _log.warning("MCP tool 'pirag_query' not registered: %s", exc)
        registry._registration_failures["pirag_query"] = str(exc)

    try:
        from .tools.explain_tool import explain
        registry.register(ToolSpec(
            name="explain",
            description="Generate a causal explanation for a routing decision with provenance",
            capabilities=["explanation", "explainability", "audit"],
            fn=explain,
            schema={
                "action": {"type": "string", "description": "Routing action (cold_chain, local_redistribute, recovery)"},
                "role": {"type": "string", "description": "Agent role making the decision"},
                "hour": {"type": "number", "description": "Simulation hour of the decision"},
                "rho": {"type": "number", "description": "Current spoilage risk (0-1)"},
                "temperature": {"type": "number", "description": "Current product temperature in Celsius"},
                "scenario": {"type": "string", "description": "Scenario (baseline, heatwave, cyber_outage, overproduction, adaptive_pricing)"},
            },
            cacheable=False,
        ))
    except Exception as exc:  # 2026-04: catch ALL registration failures, not only ImportError
        _log.warning("MCP tool 'explain' not registered: %s", exc)
        registry._registration_failures["explain"] = str(exc)

    try:
        from .tools.context_features import read_context_features
        registry.register(ToolSpec(
            name="context_features",
            description="Read the current MCP/piRAG context feature vector and logit modifier",
            capabilities=["context", "monitoring", "transparency"],
            fn=read_context_features,
            schema={},
            cacheable=False,
        ))
    except Exception as exc:  # 2026-04: catch ALL registration failures, not only ImportError
        _log.warning("MCP tool 'context_features' not registered: %s", exc)
        registry._registration_failures["context_features"] = str(exc)

    # Holt's linear yield/supply forecast tool. Canonical entry point for
    # supply forecasting; the simulator and REST endpoint both route
    # through this so production and benchmark paths share one code path.
    try:
        from .tools.yield_query import query_yield
        registry.register(ToolSpec(
            name="yield_query",
            description="Holt's linear yield/supply forecast returning point forecast, residual std, and normalised CV uncertainty",
            capabilities=["supply", "forecast", "uncertainty"],
            fn=query_yield,
            schema={
                "inventory_history":  {"type": "array",   "description": "Recent inventory_units observations for forecasting"},
                "horizon":            {"type": "integer", "description": "Forecast horizon in 15-min steps (default: 1)"},
                "cached_uncertainty": {"type": "number",  "description": "Pre-computed CV in [0,1] (short-circuits Holt's linear when present)"},
                "cached_forecast":    {"type": "array",   "description": "Pre-computed point forecast (paired with cached_uncertainty)"},
                "cached_std":         {"type": "number",  "description": "Pre-computed std (paired with cached_uncertainty)"},
            },
            cacheable=False,
        ))
    except Exception as exc:  # 2026-04: catch ALL registration failures, not only ImportError
        _log.warning("MCP tool 'yield_query' not registered: %s", exc)
        registry._registration_failures["yield_query"] = str(exc)

    # Symmetric demand forecast tool. Mirrors yield_query so supply and
    # demand share the same MCP contract and both can short-circuit from
    # a simulator-cached payload on obs.raw.
    try:
        from .tools.demand_query import query_demand
        registry.register(ToolSpec(
            name="demand_query",
            description="Demand forecast (LSTM or Holt's linear) returning point forecast, residual std, and normalised CV uncertainty",
            capabilities=["demand", "forecast", "uncertainty"],
            fn=query_demand,
            schema={
                "demand_history":     {"type": "array",   "description": "Recent demand_units observations for forecasting"},
                "horizon":            {"type": "integer", "description": "Forecast horizon in 15-min steps (default: 1)"},
                "method":             {"type": "string",  "description": "Forecast method: 'lstm' (default) or 'holt_winters'"},
                "cached_uncertainty": {"type": "number",  "description": "Pre-computed CV in [0,1] (short-circuits the forecaster when present)"},
                "cached_forecast":    {"type": "array",   "description": "Pre-computed point forecast (paired with cached_uncertainty)"},
                "cached_std":         {"type": "number",  "description": "Pre-computed std (paired with cached_uncertainty)"},
            },
            cacheable=False,
        ))
    except Exception as exc:  # 2026-04: catch ALL registration failures, not only ImportError
        _log.warning("MCP tool 'demand_query' not registered: %s", exc)
        registry._registration_failures["demand_query"] = str(exc)

    _DEFAULT_REGISTRY = registry
    return registry


def mcp_registration_status() -> Dict[str, Any]:
    """Return the registration outcome of every optional MCP tool.

    Provides a structured view of which optional tools registered and
    which failed with their import error so partial registrations are
    visible. Used by the MCP `mcp.registry.status` resource.
    """
    reg = get_default_registry()
    registered = sorted(reg._tools.keys())
    failures = reg.registration_failures()
    return {
        "registered": registered,
        "registered_count": len(registered),
        "failed": failures,
        "failed_count": len(failures),
    }
