"""Agent-as-MCP-Server: each agent exposes unique capabilities as MCP tools.

Other agents can discover and invoke these capabilities via the MCP
registry, enabling agent-to-agent coordination through the protocol
layer rather than direct method calls.
"""
from __future__ import annotations

from typing import Any, Dict

from .registry import ToolSpec
from .protocol import MCPServer


def register_farm_capabilities(server: MCPServer, farm_agent: Any) -> None:
    """Register farm agent's freshness assessment as an MCP tool."""
    def farm_freshness_assessment() -> Dict[str, Any]:
        state = farm_agent.state
        steps = state.get("steps_handled", 0)
        at_risk = state.get("at_risk_count", 0)
        at_risk_fraction = at_risk / max(steps, 1)

        if at_risk_fraction > 0.50:
            recommendation = "immediate_diversion"
        elif at_risk_fraction > 0.25:
            recommendation = "accelerated_processing"
        else:
            recommendation = "continue_cold_chain"

        return {
            "steps_handled": steps,
            "at_risk_count": at_risk,
            "at_risk_fraction": round(at_risk_fraction, 3),
            "recommendation": recommendation,
        }

    server.registry.register(ToolSpec(
        name="farm_freshness_assessment",
        description="Assess current freshness risk from farm agent state",
        capabilities=["freshness", "quality", "farm"],
        fn=farm_freshness_assessment,
        schema={},
    ))


def register_recovery_capabilities(server: MCPServer, recovery_agent: Any) -> None:
    """Register recovery agent's capacity check as an MCP tool."""
    def recovery_capacity_check() -> Dict[str, Any]:
        remaining = (
            recovery_agent.MAX_CAPACITY_BROADCASTS
            - recovery_agent._capacity_broadcasts
        )

        if remaining > 60:
            preferred_pathway = "composting"
        elif remaining > 30:
            preferred_pathway = "anaerobic_digestion"
        else:
            preferred_pathway = "energy_recovery"

        return {
            "remaining_broadcasts": remaining,
            "preferred_pathway": preferred_pathway,
            "steps_handled": recovery_agent.state.get("steps_handled", 0),
        }

    server.registry.register(ToolSpec(
        name="recovery_capacity_check",
        description="Check recovery agent remaining capacity and preferred pathway",
        capabilities=["recovery", "capacity", "waste"],
        fn=recovery_capacity_check,
        schema={},
    ))


def register_cooperative_capabilities(server: MCPServer, cooperative_agent: Any) -> None:
    """Register cooperative agent's coordination status as an MCP tool."""
    def cooperative_coordination_status() -> Dict[str, Any]:
        remaining = (
            cooperative_agent.MAX_COORDINATION_BROADCASTS
            - cooperative_agent._coordination_broadcasts
        )
        return {
            "broadcasts_remaining": remaining,
            "steps_handled": cooperative_agent.state.get("steps_handled", 0),
        }

    server.registry.register(ToolSpec(
        name="cooperative_coordination_status",
        description="Check cooperative agent coordination broadcast capacity",
        capabilities=["coordination", "cooperative"],
        fn=cooperative_coordination_status,
        schema={},
    ))


def register_processor_capabilities(server: MCPServer, processor_agent: Any) -> None:
    """Register processor agent's throughput status as an MCP tool."""
    def processor_throughput_status() -> Dict[str, Any]:
        state = processor_agent.state
        steps = state.get("steps_handled", 0)
        waste = state.get("cumulative_waste", 0.0)

        if steps > 0 and waste / steps > 0.10:
            surplus_assessment = "high_waste"
        elif steps > 0 and waste / steps > 0.05:
            surplus_assessment = "moderate_waste"
        else:
            surplus_assessment = "normal"

        return {
            "steps_handled": steps,
            "cumulative_waste": round(waste, 4),
            "surplus_assessment": surplus_assessment,
        }

    server.registry.register(ToolSpec(
        name="processor_throughput_status",
        description="Check processor agent throughput and waste status",
        capabilities=["processing", "throughput"],
        fn=processor_throughput_status,
        schema={},
    ))


def register_distributor_capabilities(server: MCPServer, distributor_agent: Any) -> None:
    """Register distributor agent's route feasibility as an MCP tool."""
    def distributor_route_feasibility() -> Dict[str, Any]:
        state = distributor_agent.state
        reroute_count = state.get("routed_count", 0)
        at_risk = state.get("at_risk_count", 0)
        steps = state.get("steps_handled", 0)
        at_risk_fraction = at_risk / max(steps, 1)

        return {
            "reroute_count": reroute_count,
            "at_risk_fraction": round(at_risk_fraction, 3),
            "steps_handled": steps,
        }

    server.registry.register(ToolSpec(
        name="distributor_route_feasibility",
        description="Check distributor route feasibility and reroute history",
        capabilities=["routing", "distribution"],
        fn=distributor_route_feasibility,
        schema={},
    ))


def register_all_agent_capabilities(server: MCPServer, agents: dict) -> None:
    """Register all agent capabilities on the MCP server.

    Parameters
    ----------
    server : MCP server instance.
    agents : dict mapping role names to agent instances.
    """
    _REGISTRARS = {
        "farm": register_farm_capabilities,
        "recovery": register_recovery_capabilities,
        "cooperative": register_cooperative_capabilities,
        "processor": register_processor_capabilities,
        "distributor": register_distributor_capabilities,
    }
    for role, agent in agents.items():
        registrar = _REGISTRARS.get(role)
        if registrar is not None:
            registrar(server, agent)
