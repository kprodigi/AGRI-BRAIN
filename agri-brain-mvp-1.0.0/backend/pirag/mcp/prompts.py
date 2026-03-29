"""MCP Prompt Templates for piRAG query construction.

Exposes parameterized query templates as MCP prompts. Each prompt
generates a role-relevant piRAG query string that can be used for
knowledge base retrieval.
"""
from __future__ import annotations

from .protocol import MCPPrompt, MCPServer


def _regulatory_compliance_template(
    product_type: str = "spinach",
    temperature: str = "4.0",
    humidity: str = "90.0",
) -> str:
    """Generate a regulatory compliance query."""
    return (
        f"FDA cold chain compliance requirements for {product_type} "
        f"at {temperature} degrees Celsius and {humidity} percent relative humidity. "
        f"Include FSMA Produce Safety Rule thresholds, traceability requirements, "
        f"and corrective action procedures for temperature excursions."
    )


def _waste_hierarchy_template(
    spoilage_risk: str = "0.30",
    product_type: str = "spinach",
    hours_remaining: str = "12",
) -> str:
    """Generate a waste hierarchy assessment query."""
    return (
        f"Food waste hierarchy assessment for {product_type} with spoilage risk {spoilage_risk} "
        f"and {hours_remaining} hours remaining shelf life. "
        f"Evaluate redistribution to food banks, animal feed diversion, composting, "
        f"and anaerobic digestion pathways per EU Waste Framework Directive 2008/98/EC."
    )


def _emergency_rerouting_template(
    scenario: str = "heatwave",
    current_action: str = "cold_chain",
    urgency: str = "high",
) -> str:
    """Generate an emergency rerouting query."""
    return (
        f"Emergency rerouting standard operating procedure under {scenario} conditions. "
        f"Current routing action is {current_action} with {urgency} urgency. "
        f"Include notification chain requirements, transport time adjustments, "
        f"and fallback procedures for degraded connectivity."
    )


def _slca_routing_template(
    action: str = "local_redistribute",
    surplus_ratio: str = "0.5",
    product_type: str = "spinach",
) -> str:
    """Generate an SLCA routing guidance query."""
    return (
        f"SLCA scoring methodology for {action} routing of {product_type} "
        f"with surplus ratio {surplus_ratio}. "
        f"Evaluate labour fairness, community resilience, price transparency, "
        f"and carbon footprint impact of the proposed routing decision."
    )


def _governance_policy_template(
    decision_type: str = "rerouting",
    agent_role: str = "cooperative",
) -> str:
    """Generate a governance policy lookup query."""
    return (
        f"Cooperative governance policy for {decision_type} decisions "
        f"by {agent_role} agent. Include quorum thresholds, voting periods, "
        f"SLCA reward and slashing criteria, and parameter bounds for "
        f"autonomous decision-making authority."
    )


def register_prompts(server: MCPServer) -> None:
    """Register all piRAG prompt templates on the MCP server."""
    server.register_prompt(MCPPrompt(
        name="regulatory_compliance_check",
        description="Generate a regulatory compliance query for FDA cold chain requirements",
        arguments=[
            {"name": "product_type", "description": "Produce type (e.g. spinach)", "required": "false"},
            {"name": "temperature", "description": "Current temperature in Celsius", "required": "false"},
            {"name": "humidity", "description": "Current relative humidity in percent", "required": "false"},
        ],
        template_fn=_regulatory_compliance_template,
    ))

    server.register_prompt(MCPPrompt(
        name="waste_hierarchy_assessment",
        description="Generate a waste hierarchy assessment query for food diversion pathways",
        arguments=[
            {"name": "spoilage_risk", "description": "Current spoilage risk (0-1)", "required": "false"},
            {"name": "product_type", "description": "Produce type", "required": "false"},
            {"name": "hours_remaining", "description": "Remaining shelf life hours", "required": "false"},
        ],
        template_fn=_waste_hierarchy_template,
    ))

    server.register_prompt(MCPPrompt(
        name="emergency_rerouting",
        description="Generate an emergency rerouting SOP query",
        arguments=[
            {"name": "scenario", "description": "Current scenario (heatwave, cyber_outage, etc.)", "required": "false"},
            {"name": "current_action", "description": "Current routing action", "required": "false"},
            {"name": "urgency", "description": "Urgency level (low, medium, high, critical)", "required": "false"},
        ],
        template_fn=_emergency_rerouting_template,
    ))

    server.register_prompt(MCPPrompt(
        name="slca_routing_guidance",
        description="Generate an SLCA scoring methodology query for routing decisions",
        arguments=[
            {"name": "action", "description": "Proposed routing action", "required": "false"},
            {"name": "surplus_ratio", "description": "Current surplus ratio", "required": "false"},
            {"name": "product_type", "description": "Produce type", "required": "false"},
        ],
        template_fn=_slca_routing_template,
    ))

    server.register_prompt(MCPPrompt(
        name="governance_policy_lookup",
        description="Generate a cooperative governance policy query",
        arguments=[
            {"name": "decision_type", "description": "Type of decision requiring governance", "required": "false"},
            {"name": "agent_role", "description": "Role of the requesting agent", "required": "false"},
        ],
        template_fn=_governance_policy_template,
    ))
