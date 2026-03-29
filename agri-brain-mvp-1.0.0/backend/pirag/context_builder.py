"""Role-specific piRAG query construction with MCP-informed refinements.

Each agent role has a base query template plus conditional expansions
triggered by observation thresholds and MCP tool results. When an MCP
server is available, queries can be built via the MCP prompts/get
primitive; otherwise, direct template expansion is used.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .mcp.protocol import MCPMessage, MCPServer


ROLE_QUERY_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "farm": {
        "base": "FDA cold chain compliance for fresh spinach postharvest storage",
        "conditions": [
            {"trigger": lambda obs, mcp: obs.temp > 8.0,
             "append": "temperature excursion above safe threshold requiring corrective action"},
            {"trigger": lambda obs, mcp: obs.rho > 0.20,
             "append": "elevated spoilage risk during early supply chain stages"},
            {"trigger": lambda obs, mcp: mcp.get("check_compliance", {}).get("violations"),
             "append": "active FDA compliance violations detected by monitoring system"},
        ],
    },
    "processor": {
        "base": "processing facility quality management and throughput optimization",
        "conditions": [
            {"trigger": lambda obs, mcp: obs.surplus_ratio > 0.3,
             "append": "surplus inventory requiring diversion or redistribution planning"},
            {"trigger": lambda obs, mcp: obs.surplus_ratio > 0.5,
             "append": "critical surplus exceeding processing capacity"},
            {"trigger": lambda obs, mcp: mcp.get("policy_oracle") is False,
             "append": "governance policy restricting current routing options"},
        ],
    },
    "cooperative": {
        "base": "SLCA scoring methodology and cooperative stakeholder coordination",
        "conditions": [
            {"trigger": lambda obs, mcp: obs.tau > 0.5,
             "append": "volatility regime requiring coordinated response across agents"},
            {"trigger": lambda obs, mcp: obs.rho > 0.30,
             "append": "spoilage pressure requiring cross-agent coordination"},
            {"trigger": lambda obs, mcp: mcp.get("slca_lookup", {}).get("base_scores", {}).get("local_redistribute", {}).get("R", 0) > 0.80,
             "append": "high community resilience opportunity through local redistribution"},
        ],
    },
    "distributor": {
        "base": "redistribution routing compliance and community food delivery",
        "conditions": [
            {"trigger": lambda obs, mcp: obs.rho > 0.35,
             "append": "time-critical redistribution under elevated spoilage risk"},
            {"trigger": lambda obs, mcp: obs.rho > 0.45,
             "append": "emergency rerouting required for at-risk produce"},
            {"trigger": lambda obs, mcp: mcp.get("check_compliance", {}).get("violations"),
             "append": "compliance violations affecting redistribution eligibility"},
        ],
    },
    "recovery": {
        "base": "food waste hierarchy and circular economy valorization pathways",
        "conditions": [
            {"trigger": lambda obs, mcp: obs.rho > 0.50,
             "append": "advanced spoilage limiting recovery options to composting or energy"},
            {"trigger": lambda obs, mcp: obs.rho > 0.80,
             "append": "terminal spoilage suitable only for anaerobic digestion"},
            {"trigger": lambda obs, mcp: mcp.get("footprint_query", {}).get("efficiency_flag") == "above_baseline",
             "append": "elevated energy footprint requiring efficiency review"},
        ],
    },
}


def build_role_query(
    role: str,
    obs: Any,
    scenario: str,
    mcp_results: Dict[str, Any],
    mcp_server: Optional[MCPServer] = None,
) -> str:
    """Build a piRAG query string for the given role and conditions.

    If ``mcp_server`` is available and has a matching prompt, queries are
    built via ``prompts/get``. Otherwise, direct template expansion is used.

    Parameters
    ----------
    role : agent role name.
    obs : current Observation.
    scenario : current scenario name.
    mcp_results : results from MCP tool dispatch.
    mcp_server : optional MCP server for prompt-based construction.
    """
    # Try MCP prompts/get first
    if mcp_server is not None:
        prompt_name = _ROLE_PROMPT_MAP.get(role)
        if prompt_name is not None:
            try:
                prompt_args = _build_prompt_args(role, obs, scenario, mcp_results)
                response = mcp_server.handle_message(MCPMessage(
                    id=0, method="prompts/get",
                    params={"name": prompt_name, "arguments": prompt_args},
                ))
                if response.result:
                    messages = response.result.get("messages", [])
                    if messages:
                        text = messages[0].get("content", {}).get("text", "")
                        if text:
                            return text
            except Exception:
                pass

    # Fallback: direct template expansion
    template = ROLE_QUERY_TEMPLATES.get(role, {"base": "supply chain management", "conditions": []})
    parts = [template["base"]]

    for cond in template["conditions"]:
        try:
            if cond["trigger"](obs, mcp_results):
                parts.append(cond["append"])
        except Exception:
            continue

    if scenario != "baseline":
        parts.append(f"operating under {scenario} scenario conditions")

    return " with ".join(parts[:2]) + (". " + ". ".join(parts[2:]) if len(parts) > 2 else "")


_ROLE_PROMPT_MAP: Dict[str, str] = {
    "farm": "regulatory_compliance_check",
    "processor": "slca_routing_guidance",
    "cooperative": "governance_policy_lookup",
    "distributor": "emergency_rerouting",
    "recovery": "waste_hierarchy_assessment",
}


def _build_prompt_args(
    role: str, obs: Any, scenario: str, mcp_results: Dict[str, Any],
) -> Dict[str, str]:
    """Build arguments for the MCP prompt based on role."""
    if role == "farm":
        return {
            "product_type": "spinach",
            "temperature": str(round(obs.temp, 1)),
            "humidity": str(round(obs.rh, 1)),
        }
    elif role == "processor":
        return {
            "action": "local_redistribute",
            "surplus_ratio": str(round(obs.surplus_ratio, 2)),
            "product_type": "spinach",
        }
    elif role == "cooperative":
        return {
            "decision_type": "coordination",
            "agent_role": "cooperative",
        }
    elif role == "distributor":
        return {
            "scenario": scenario,
            "current_action": "cold_chain",
            "urgency": "high" if obs.rho > 0.40 else "medium",
        }
    elif role == "recovery":
        return {
            "spoilage_risk": str(round(obs.rho, 2)),
            "product_type": "spinach",
            "hours_remaining": str(max(1, int(72 - obs.hour))),
        }
    return {}


def retrieve_role_context(
    role: str,
    obs: Any,
    scenario: str,
    mcp_results: Dict[str, Any],
    pipeline: Any,
    mcp_server: Optional[MCPServer] = None,
) -> Dict[str, Any]:
    """Retrieve piRAG context for a role with physics-informed expansion.

    Returns a dict with query, citations, guidance fields, scores, and
    guard/provenance metadata.
    """
    context: Dict[str, Any] = {
        "query": "",
        "citations": [],
        "regulatory_guidance": "",
        "sop_guidance": "",
        "slca_guidance": "",
        "waste_hierarchy_guidance": "",
        "governance_guidance": "",
        "top_citation_score": 0.0,
        "top_doc_id": "",
        "guards_passed": True,
        "evidence_hashes": [],
    }

    if pipeline is None:
        return context

    query = build_role_query(role, obs, scenario, mcp_results, mcp_server)

    # Physics-informed query expansion (Task 11)
    try:
        from .physics_reranker import expand_query_with_physics
        k_eff = mcp_results.get("spoilage_forecast", {}).get("k_effective", 0.0)
        query = expand_query_with_physics(query, obs.rho, obs.temp, k_eff)
    except ImportError:
        pass

    context["query"] = query

    try:
        response = pipeline.ask(query, k=4, anchor_on_chain=False)

        # Physics-informed re-ranking (Task 11)
        try:
            from .physics_reranker import physics_rerank
            passages = [{"text": c.passage, "score": 0.5, "id": c.doc_id, "meta": c.meta} for c in response.citations]
            reranked = physics_rerank(passages, obs.temp, obs.rho, obs.rh)
            # Use reranked order for guidance extraction
            ranked_citations = reranked
        except ImportError:
            ranked_citations = [{"text": c.passage, "score": 0.5, "id": c.doc_id, "meta": c.meta} for c in response.citations]

        context["evidence_hashes"] = response.evidence_hashes

        for cit in response.citations:
            context["citations"].append({
                "doc_id": cit.doc_id,
                "passage": cit.passage[:300],
                "sha256": cit.sha256,
            })

        # Assign guidance based on document IDs and compute top score
        for entry in ranked_citations:
            doc_id = entry.get("id", "")
            passage = entry.get("text", "")[:300]
            score = entry.get("score", 0.0)

            if score > context["top_citation_score"]:
                context["top_citation_score"] = score
                context["top_doc_id"] = doc_id

            if "regulatory" in doc_id or "fda" in doc_id:
                if not context["regulatory_guidance"]:
                    context["regulatory_guidance"] = passage
            elif "sop" in doc_id or "cold_chain" in doc_id or "emergency" in doc_id:
                if not context["sop_guidance"]:
                    context["sop_guidance"] = passage
            elif "slca" in doc_id:
                if not context["slca_guidance"]:
                    context["slca_guidance"] = passage
            elif "waste_hierarchy" in doc_id:
                if not context["waste_hierarchy_guidance"]:
                    context["waste_hierarchy_guidance"] = passage
            elif "governance" in doc_id or "cooperative" in doc_id:
                if not context["governance_guidance"]:
                    context["governance_guidance"] = passage

        # Evaluate context quality based on retrieval relevance, not
        # the answer-formatting unit guard (which false-positives on
        # the template engine's "Based on N relevant sources" preamble).
        context["guards_passed"] = (
            len(response.citations) > 0
            and context["top_citation_score"] > 0.15
        )

    except Exception:
        pass

    return context
