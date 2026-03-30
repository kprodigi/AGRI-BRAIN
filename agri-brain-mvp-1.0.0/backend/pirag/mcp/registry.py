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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


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
    schema: Dict[str, str]
    cacheable: bool = False
    cache_key_params: List[str] = field(default_factory=list)


class ToolRegistry:
    """Registry of MCP tools with capability-based discovery and caching."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self._cache: Dict[str, Any] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register a tool specification."""
        self._tools[spec.name] = spec

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
            cache_key = f"{tool_name}:{hashlib.md5(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()}"
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

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return serializable list of all registered tools."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "capabilities": s.capabilities,
                "schema": s.schema,
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
    from .tools.simulator import simulate

    registry.register(ToolSpec(
        name="check_compliance",
        description="Check FDA temperature and humidity compliance for produce",
        capabilities=["regulatory", "temperature", "quality"],
        fn=check_compliance,
        schema={"temperature": "float", "humidity": "float", "product_type": "str"},
        cacheable=False,
    ))
    registry.register(ToolSpec(
        name="slca_lookup",
        description="Look up SLCA weights and base scores by product type",
        capabilities=["social", "scoring", "routing"],
        fn=lookup_slca_weights,
        schema={"product_type": "str"},
        cacheable=True,
        cache_key_params=["product_type"],
    ))
    registry.register(ToolSpec(
        name="chain_query",
        description="Query recent routing decisions from blockchain audit trail",
        capabilities=["blockchain", "audit", "history"],
        fn=query_recent_decisions,
        schema={"n": "int"},
        cacheable=False,
    ))
    registry.register(ToolSpec(
        name="policy_oracle",
        description="Check governance policy access for a user and tool",
        capabilities=["governance", "access", "policy"],
        fn=check_access,
        schema={"user_id": "str", "tool_name": "str"},
        cacheable=True,
        cache_key_params=["user_id", "tool_name"],
    ))
    registry.register(ToolSpec(
        name="calculator",
        description="Evaluate a safe arithmetic expression",
        capabilities=["math", "estimation"],
        fn=calculate,
        schema={"expr": "str"},
        cacheable=False,
    ))
    registry.register(ToolSpec(
        name="convert_units",
        description="Convert a numeric value between measurement units",
        capabilities=["units", "conversion"],
        fn=convert,
        schema={"value": "float", "from_unit": "str", "to_unit": "str"},
        cacheable=True,
        cache_key_params=["value", "from_unit", "to_unit"],
    ))
    registry.register(ToolSpec(
        name="simulate",
        description="Run a forward simulation via the simulation API",
        capabilities=["simulation", "forecast"],
        fn=simulate,
        schema={"endpoint": "str", "payload": "dict"},
        cacheable=False,
    ))

    # New tools (Tasks 3–4) — registered if importable
    try:
        from .tools.spoilage_forecast import forecast_spoilage
        registry.register(ToolSpec(
            name="spoilage_forecast",
            description="Integrate Arrhenius-Baranyi ODE forward for spoilage prediction",
            capabilities=["spoilage", "forecast", "physics"],
            fn=forecast_spoilage,
            schema={"current_rho": "float", "temperature": "float", "humidity": "float", "hours_ahead": "int"},
            cacheable=False,
        ))
    except ImportError:
        pass

    try:
        from .tools.footprint_query import query_footprint
        registry.register(ToolSpec(
            name="footprint_query",
            description="Return cumulative and per-step energy and water footprint",
            capabilities=["footprint", "green_ai", "monitoring"],
            fn=query_footprint,
            schema={"steps_completed": "int", "energy_per_step_j": "float", "water_per_step_l": "float"},
            cacheable=False,
        ))
    except ImportError:
        pass

    # piRAG, explanation, and context feature tools
    try:
        from .tools.pirag_query import pirag_query
        registry.register(ToolSpec(
            name="pirag_query",
            description="Query the piRAG knowledge base with physics-informed retrieval",
            capabilities=["retrieval", "knowledge", "pirag", "regulatory"],
            fn=pirag_query,
            schema={"query": "str", "k": "int", "role": "str", "temperature": "float",
                     "rho": "float", "humidity": "float"},
            cacheable=False,
        ))
    except ImportError:
        pass

    try:
        from .tools.explain_tool import explain
        registry.register(ToolSpec(
            name="explain",
            description="Generate a causal explanation for a routing decision with provenance",
            capabilities=["explanation", "explainability", "audit"],
            fn=explain,
            schema={"action": "str", "role": "str", "hour": "float", "rho": "float",
                     "temperature": "float", "scenario": "str"},
            cacheable=False,
        ))
    except ImportError:
        pass

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
    except ImportError:
        pass

    _DEFAULT_REGISTRY = registry
    return registry
