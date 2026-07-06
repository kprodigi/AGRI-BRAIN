"""MCP Prompt Templates for piRAG query construction.

Exposes parameterized query templates as MCP prompts. Each prompt
generates a role-relevant piRAG query string that can be used for
knowledge base retrieval.  When a non-baseline ``scenario`` is passed,
scenario-specific search terms are appended so that BM25 retrieval
surfaces the corresponding KB documents (e.g. heatwave_contingency_plan
for heatwave, cyber_outage_contingency for cyber_outage).
"""
from __future__ import annotations

from typing import Dict

from .protocol import MCPPrompt, MCPServer


# Scenario-specific terms that match content in KB documents.
# Each value contains terms that appear in the corresponding document(s)
# so BM25/TF-IDF retrieval can discriminate between scenarios.
SCENARIO_SEARCH_TERMS: Dict[str, str] = {
    "heatwave": (
        "heatwave contingency ambient temperature exceedance thermal stress "
        "cooling capacity reduced transport distance"
    ),
    "cyber_outage": (
        "communications outage fallback cached routing manual override "
        "offline decision-making reconnection synchronization"
    ),
    "overproduction": (
        "demand volatility surplus redistribution excess inventory "
        "food bank diversion Bollinger band threshold"
    ),
    "adaptive_pricing": (
        "price transparency cooperative pricing demand surge demand trough "
        "volatile market pricing audit"
    ),
    "baseline": "",
}


def _scenario_suffix(scenario: str) -> str:
    """Return scenario-specific search terms to append to a query."""
    terms = SCENARIO_SEARCH_TERMS.get(scenario, "")
    return f" {terms}" if terms else ""


def _regulatory_compliance_template(
    product_type: str = "spinach",
    temperature: str = "4.0",
    humidity: str = "90.0",
    scenario: str = "baseline",
) -> str:
    """Generate a regulatory compliance query."""
    base = (
        f"FDA cold chain compliance requirements for {product_type} "
        f"at {temperature} degrees Celsius and {humidity} percent relative humidity. "
        f"Include FSMA Produce Safety Rule thresholds, traceability requirements, "
        f"corrective action procedures including temperature excursion severity classification, "
        f"and IoT sensor calibration standards for continuous monitoring."
    )
    return base + _scenario_suffix(scenario)


def _waste_hierarchy_template(
    spoilage_risk: str = "0.30",
    product_type: str = "spinach",
    hours_remaining: str = "12",
    scenario: str = "baseline",
) -> str:
    """Generate a waste hierarchy assessment query."""
    base = (
        f"Food waste hierarchy assessment for {product_type} with spoilage risk {spoilage_risk} "
        f"and {hours_remaining} hours remaining shelf life. "
        f"Evaluate redistribution to food banks, animal feed diversion, composting, "
        f"and anaerobic digestion pathways per EU Waste Framework Directive 2008/98/EC."
    )
    return base + _scenario_suffix(scenario)


def _emergency_rerouting_template(
    scenario: str = "heatwave",
    current_action: str = "cold_chain",
    urgency: str = "high",
) -> str:
    """Generate an emergency rerouting query."""
    base = (
        f"Emergency rerouting standard operating procedure under {scenario} conditions. "
        f"Current routing action is {current_action} with {urgency} urgency. "
        f"Include notification chain requirements, transport time adjustments, "
        f"fallback procedures for degraded connectivity, "
        f"and carbon accounting for refrigerated transport emission factors."
    )
    return base + _scenario_suffix(scenario)


def _slca_routing_template(
    action: str = "local_redistribute",
    surplus_ratio: str = "0.5",
    product_type: str = "spinach",
    scenario: str = "baseline",
) -> str:
    """Generate an SLCA routing guidance query."""
    base = (
        f"SLCA scoring methodology for {action} routing of {product_type} "
        f"with surplus ratio {surplus_ratio}. "
        f"Evaluate labour fairness including shift duration standards, "
        f"community resilience, price transparency, "
        f"and carbon footprint impact of the proposed routing decision. "
        f"Include energy consumption reporting and green AI efficiency metrics."
    )
    return base + _scenario_suffix(scenario)


def _governance_policy_template(
    decision_type: str = "rerouting",
    agent_role: str = "cooperative",
    scenario: str = "baseline",
) -> str:
    """Generate a governance policy lookup query."""
    base = (
        f"Cooperative governance policy for {decision_type} decisions "
        f"by {agent_role} agent. Include quorum thresholds, voting periods, "
        f"SLCA reward and slashing criteria, parameter bounds for "
        f"autonomous decision-making authority, "
        f"and blockchain audit trail requirements including immutable decision hash."
    )
    return base + _scenario_suffix(scenario)


def register_prompts(server: MCPServer) -> None:
    """Register all piRAG prompt templates on the MCP server."""
    _scenario_arg = {
        "name": "scenario",
        "description": "Current scenario (baseline, heatwave, cyber_outage, overproduction, adaptive_pricing)",
        "required": False,
    }

    server.register_prompt(MCPPrompt(
        name="regulatory_compliance_check",
        description="Generate a regulatory compliance query for FDA cold chain requirements",
        arguments=[
            {"name": "product_type", "description": "Produce type (e.g. spinach)", "required": False},
            {"name": "temperature", "description": "Current temperature in Celsius", "required": False},
            {"name": "humidity", "description": "Current relative humidity in percent", "required": False},
            _scenario_arg,
        ],
        template_fn=_regulatory_compliance_template,
    ))

    server.register_prompt(MCPPrompt(
        name="waste_hierarchy_assessment",
        description="Generate a waste hierarchy assessment query for food diversion pathways",
        arguments=[
            {"name": "spoilage_risk", "description": "Current spoilage risk (0-1)", "required": False},
            {"name": "product_type", "description": "Produce type", "required": False},
            {"name": "hours_remaining", "description": "Remaining shelf life hours", "required": False},
            _scenario_arg,
        ],
        template_fn=_waste_hierarchy_template,
    ))

    server.register_prompt(MCPPrompt(
        name="emergency_rerouting",
        description="Generate an emergency rerouting SOP query",
        arguments=[
            {"name": "scenario", "description": "Current scenario (heatwave, cyber_outage, etc.)", "required": False},
            {"name": "current_action", "description": "Current routing action", "required": False},
            {"name": "urgency", "description": "Urgency level (low, medium, high, critical)", "required": False},
        ],
        template_fn=_emergency_rerouting_template,
    ))

    server.register_prompt(MCPPrompt(
        name="slca_routing_guidance",
        description="Generate an SLCA scoring methodology query for routing decisions",
        arguments=[
            {"name": "action", "description": "Proposed routing action", "required": False},
            {"name": "surplus_ratio", "description": "Current surplus ratio", "required": False},
            {"name": "product_type", "description": "Produce type", "required": False},
            _scenario_arg,
        ],
        template_fn=_slca_routing_template,
    ))

    server.register_prompt(MCPPrompt(
        name="governance_policy_lookup",
        description="Generate a cooperative governance policy query",
        arguments=[
            {"name": "decision_type", "description": "Type of decision requiring governance", "required": False},
            {"name": "agent_role", "description": "Role of the requesting agent", "required": False},
            _scenario_arg,
        ],
        template_fn=_governance_policy_template,
    ))
