"""MCP Resources: URI-addressable live state endpoints.

Exposes real-time supply chain telemetry, quality metrics, forecasts,
and policy parameters as MCP resources. Each resource has a URI following
the ``agribrain://`` scheme and a ``read_fn`` that returns the current
value from the agent's observable state.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from .protocol import MCPResource, MCPServer


def register_agent_resources(
    server: MCPServer,
    agent_state_fn: Callable[[], Dict[str, Any]],
) -> None:
    """Register MCP resources for an agent's observable state.

    Parameters
    ----------
    server : MCP server to register resources on.
    agent_state_fn : callable returning a dict with keys:
        temp, rh, inv, rho, freshness, y_hat, tau, policy_params,
        cumulative_footprint.  Missing keys yield None.
    """
    _RESOURCE_DEFS = [
        ("agribrain://telemetry/temperature",
         "Temperature",
         "Current ambient temperature in Celsius",
         lambda: agent_state_fn().get("temp")),

        ("agribrain://telemetry/humidity",
         "Humidity",
         "Current relative humidity in percent",
         lambda: agent_state_fn().get("rh")),

        ("agribrain://inventory/level",
         "Inventory Level",
         "Current inventory in units",
         lambda: agent_state_fn().get("inv")),

        ("agribrain://quality/spoilage_risk",
         "Spoilage Risk",
         "Current spoilage risk rho in [0, 1]",
         lambda: agent_state_fn().get("rho")),

        ("agribrain://quality/freshness",
         "Freshness",
         "Current freshness score (1 - rho)",
         lambda: agent_state_fn().get("freshness", 1.0 - (agent_state_fn().get("rho") or 0.0))),

        ("agribrain://forecast/demand",
         "Demand Forecast",
         "Current demand forecast (units per step)",
         lambda: agent_state_fn().get("y_hat")),

        ("agribrain://regime/volatility",
         "Volatility Regime",
         "Volatility indicator tau (1.0 = anomaly, 0.0 = normal)",
         lambda: agent_state_fn().get("tau")),

        ("agribrain://policy/parameters",
         "Policy Parameters",
         "Current policy configuration parameters",
         lambda: agent_state_fn().get("policy_params", {})),

        ("agribrain://footprint/cumulative",
         "Cumulative Footprint",
         "Cumulative green AI footprint (energy, water, CO2)",
         lambda: agent_state_fn().get("cumulative_footprint", {})),
    ]

    for uri, name, description, read_fn in _RESOURCE_DEFS:
        server.register_resource(MCPResource(
            uri=uri,
            name=name,
            description=description,
            read_fn=read_fn,
        ))

    # Context feature and explanation resources
    try:
        from .tools.context_features import get_context_cache
        for uri, name, desc, key in [
            ("agribrain://context/features",
             "Context Feature Vector",
             "Current 5D MCP/piRAG context feature vector",
             "features"),
            ("agribrain://context/modifier",
             "Context Logit Modifier",
             "Current 3D logit adjustment from MCP/piRAG context",
             "modifier"),
            ("agribrain://context/explanation",
             "Latest Decision Explanation",
             "Most recent causal decision explanation with provenance",
             "explanation"),
        ]:
            _key = key  # capture for closure
            server.register_resource(MCPResource(
                uri=uri, name=name, description=desc,
                read_fn=lambda _k=_key: get_context_cache().get(_k, {}),
            ))
    except ImportError:
        pass
