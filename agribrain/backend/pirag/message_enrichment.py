"""piRAG-enriched inter-agent messages.

Attaches piRAG guidance and MCP status to outgoing inter-agent messages,
enriching coordination signals with context-aware intelligence. The
enrichment is non-mutating: a new message is returned with an expanded
payload.
"""
from __future__ import annotations

from typing import Any, Dict

from src.agents.message import InterAgentMessage, MessageType


def enrich_message(
    msg: InterAgentMessage,
    rag_context: Dict[str, Any],
    mcp_results: Dict[str, Any],
) -> InterAgentMessage:
    """Attach piRAG guidance and MCP status to an outgoing message.

    Enrichment rules by message type:
    - SPOILAGE_ALERT → regulatory or SOP guidance
    - SURPLUS_ALERT → SOP or waste hierarchy guidance
    - REROUTE_REQUEST → emergency SOP guidance
    - COORDINATION_UPDATE → SLCA or governance guidance
    - CAPACITY_UPDATE → waste hierarchy guidance

    Also adds compliance_status and urgency_level from MCP results.

    Parameters
    ----------
    msg : original inter-agent message.
    rag_context : piRAG retrieval results.
    mcp_results : MCP tool dispatch results.

    Returns
    -------
    New InterAgentMessage with enriched payload.
    """
    enriched_payload = dict(msg.payload)

    # Select guidance based on message type
    guidance = ""
    if msg.msg_type == MessageType.SPOILAGE_ALERT:
        guidance = (
            rag_context.get("regulatory_guidance", "")
            or rag_context.get("sop_guidance", "")
        )
    elif msg.msg_type == MessageType.SURPLUS_ALERT:
        guidance = (
            rag_context.get("sop_guidance", "")
            or rag_context.get("waste_hierarchy_guidance", "")
        )
    elif msg.msg_type == MessageType.REROUTE_REQUEST:
        guidance = rag_context.get("sop_guidance", "")
    elif msg.msg_type == MessageType.COORDINATION_UPDATE:
        guidance = (
            rag_context.get("slca_guidance", "")
            or rag_context.get("governance_guidance", "")
        )
    elif msg.msg_type == MessageType.CAPACITY_UPDATE:
        guidance = rag_context.get("waste_hierarchy_guidance", "")

    if guidance:
        enriched_payload["pirag_guidance"] = guidance[:200]

    # Add MCP compliance status
    compliance = mcp_results.get("check_compliance")
    if isinstance(compliance, dict):
        enriched_payload["compliance_status"] = compliance.get("compliant", True)

    # Add urgency level from spoilage forecast
    forecast = mcp_results.get("spoilage_forecast")
    if isinstance(forecast, dict):
        enriched_payload["urgency_level"] = forecast.get("urgency", "low")

    return InterAgentMessage(
        sender=msg.sender,
        recipient=msg.recipient,
        msg_type=msg.msg_type,
        payload=enriched_payload,
        hour=msg.hour,
    )
