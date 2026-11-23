"""MCP Prompt Templates for piR query construction.

Exposes parameterized query templates as MCP prompts. Each prompt
generates a role-relevant piR query string that can be used for
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
        "synthetic heatwave perturbation hour 24 temperature humidity "
        "mechanistic spoilage thermal stress"
    ),
    "cyber_outage": (
        "synthetic cyber outage hour 24 demand multiplier temperature excursion "
        "MCP channel unavailable institutional retrieval remains"
    ),
    "overproduction": (
        "synthetic overproduction hour 12 inventory multiplier surplus ratio "
        "temperature excursion Bollinger volatility"
    ),
    "adaptive_pricing": (
        "synthetic adaptive pricing demand sinusoid keyed noise volatility "
        "price transparency proxy not measured market outcome"
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
    """Generate a source-labelled operating-envelope guidance query."""
    base = (
        f"Source-scope and synthetic operating-envelope context relevant to {product_type} "
        f"at {temperature} degrees Celsius and {humidity} percent relative humidity. "
        f"Return the simulator's author-declared 8 degree Celsius spinach ceiling, "
        f"85-to-95-percent humidity envelope, and mechanistic-model limitations. "
        f"Do not infer current law, food safety, traceability certification, or legal compliance."
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
        f"Synthetic food-waste-hierarchy proxy for {product_type} with modeled spoilage risk {spoilage_risk} "
        f"and {hours_remaining} modeled hours remaining. "
        f"Return the executable continuous food-bank, animal-feed, and composting heuristic equations "
        f"and state that they do not determine safety, eligibility, or product disposition."
    )
    return base + _scenario_suffix(scenario)


def _emergency_rerouting_template(
    scenario: str = "heatwave",
    current_action: str = "cold_chain",
    urgency: str = "high",
) -> str:
    """Generate an emergency rerouting query."""
    base = (
        f"Synthetic scenario-aware routing context under {scenario} conditions. "
        f"Current routing action is {current_action} with {urgency} urgency. "
        f"Return the exact scenario perturbation, channel availability, fixed route distances, "
        f"and modeled transport-emissions equation. Do not infer notifications, operational "
        f"fallback procedures, safety eligibility, or mandatory rerouting."
    )
    return base + _scenario_suffix(scenario)


def _slca_routing_template(
    action: str = "local_redistribute",
    surplus_ratio: str = "0.5",
    product_type: str = "spinach",
    scenario: str = "baseline",
) -> str:
    """Generate a social-performance-proxy routing query."""
    base = (
        f"Author-declared social-performance-proxy methodology for {action} routing of {product_type} "
        f"with surplus ratio {surplus_ratio}. "
        f"Identify the declared labour, community, transparency, and modeled-emissions proxy inputs. "
        f"Do not describe the numerical proxy as a measured labour, equity, community, or life-cycle effect. "
        f"Include separately labelled computational-footprint assumptions."
    )
    return base + _scenario_suffix(scenario)


def _governance_policy_template(
    decision_type: str = "rerouting",
    agent_role: str = "cooperative",
    scenario: str = "baseline",
) -> str:
    """Generate a coordinator-mediated cooperative-overlay query."""
    base = (
        f"Coordinator-mediated cooperative-overlay assumptions for {decision_type} decisions "
        f"involving the {agent_role} role. Identify the declared overlay window and context blend, "
        f"and distinguish off-chain calculation-trace hashes from optional external anchoring. "
        f"State that the benchmark has no institutional quorum, stakeholder voting, deployed "
        f"governance, independent organizational control, or immutable publication evidence."
    )
    return base + _scenario_suffix(scenario)


def register_prompts(server: MCPServer) -> None:
    """Register all piR prompt templates on the MCP server."""
    _scenario_arg = {
        "name": "scenario",
        "description": "Current scenario (baseline, heatwave, cyber_outage, overproduction, adaptive_pricing)",
        "required": False,
    }

    server.register_prompt(MCPPrompt(
        name="regulatory_compliance_check",
        description="Generate a source-labelled operating-envelope and traceability guidance query",
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
        description="Generate a declared social-performance-proxy methodology query",
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
        description="Generate a coordinator-mediated cooperative-overlay query",
        arguments=[
            {"name": "decision_type", "description": "Type of decision requiring governance", "required": False},
            {"name": "agent_role", "description": "Role of the requesting agent", "required": False},
            _scenario_arg,
        ],
        template_fn=_governance_policy_template,
    ))
