"""Role-specific piRAG query construction with MCP-informed refinements.

Each agent role has a base query template plus conditional expansions
triggered by observation thresholds and MCP tool results. When an MCP
server is available, queries can be built via the MCP prompts/get
primitive; otherwise, direct template expansion is used.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .guards.retrieval_guard import retrieval_quality_ok
from .mcp.protocol import MCPMessage, MCPServer

_log = logging.getLogger(__name__)


ROLE_QUERY_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "farm": {
        "base": ("FDA cold chain compliance for fresh spinach postharvest storage "
                 "including temperature excursion severity classification "
                 "and IoT sensor calibration standards for continuous monitoring"),
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
        "base": ("processing facility quality management and throughput optimization "
                 "including energy consumption reporting and green AI efficiency metrics "
                 "with cooperative governance quorum requirements"),
        "conditions": [
            {"trigger": lambda obs, mcp: obs.surplus_ratio > 0.3,
             "append": "surplus inventory requiring diversion or redistribution planning"},
            {"trigger": lambda obs, mcp: obs.surplus_ratio > 0.5,
             "append": "critical surplus exceeding processing capacity"},
            {"trigger": lambda obs, mcp: not (mcp.get("policy_oracle", {}) or {}).get("allowed", True),
             "append": "governance policy restricting current routing options"},
        ],
    },
    "cooperative": {
        "base": ("SLCA scoring methodology and cooperative stakeholder coordination "
                 "with blockchain audit trail requirements including immutable decision hash "
                 "and labor fairness shift duration standards"),
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
        "base": ("redistribution routing compliance and community food delivery "
                 "with carbon accounting for refrigerated transport emission factors"),
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
        "base": ("food waste hierarchy and circular economy valorization pathways "
                 "including animal feed diversion standards and composting bioenergy requirements"),
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
            except Exception as _exc:
                _log.debug("MCP prompt fetch for role %s skipped: %s", role, _exc)

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
        try:
            from .mcp.prompts import SCENARIO_SEARCH_TERMS
            scenario_terms = SCENARIO_SEARCH_TERMS.get(scenario, "")
        except ImportError:
            scenario_terms = ""
        parts.append(f"operating under {scenario} scenario conditions {scenario_terms}".strip())

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
    """Build arguments for the MCP prompt based on role.

    Every role now receives the ``scenario`` parameter so that prompt
    templates can append scenario-specific search terms for discriminative
    piRAG retrieval.
    """
    base: Dict[str, str] = {"scenario": scenario}

    if role == "farm":
        base.update({
            "product_type": "spinach",
            "temperature": str(round(obs.temp, 1)),
            "humidity": str(round(obs.rh, 1)),
        })
    elif role == "processor":
        base.update({
            "action": "local_redistribute",
            "surplus_ratio": str(round(obs.surplus_ratio, 2)),
            "product_type": "spinach",
        })
    elif role == "cooperative":
        base.update({
            "decision_type": "coordination",
            "agent_role": "cooperative",
        })
    elif role == "distributor":
        base.update({
            "current_action": "cold_chain",
            "urgency": "high" if obs.rho > 0.40 else "medium",
        })
    elif role == "recovery":
        base.update({
            "spoilage_risk": str(round(obs.rho, 2)),
            "product_type": "spinach",
            "hours_remaining": str(max(1, int(72 - obs.hour))),
        })
    return base


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
    # Fault-injection paths may pass sparse/None MCP payloads; normalize once.
    mcp_results = mcp_results or {}
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
        "retrieval_metrics": {},
        "counterfactual": {},
        "physics_consistency_score": 1.0,
    }

    if pipeline is None:
        return context

    query = build_role_query(role, obs, scenario, mcp_results, mcp_server)

    # Physics-informed query expansion (Task 11)
    try:
        from .physics_reranker import expand_query_with_physics
        spoilage_forecast = mcp_results.get("spoilage_forecast") or {}
        if not isinstance(spoilage_forecast, dict):
            spoilage_forecast = {}
        k_eff = spoilage_forecast.get("k_effective", 0.0)
        query = expand_query_with_physics(query, obs.rho, obs.temp, k_eff)
    except ImportError:
        pass

    context["query"] = query

    try:
        response = pipeline.ask(query, k=4, anchor_on_chain=False)

        # Lexical + Arrhenius re-ranking. We import the canonical name so
        # production exercises the renamed function rather than the
        # deprecated `physics_rerank` alias.
        try:
            from .physics_reranker import lexical_arrhenius_rerank

            # Surface the Arrhenius rate to the reranker so the
            # thermodynamic component actually fires (was previously
            # always 0 because k_eff defaulted to 0).
            try:
                from .mcp.tools.spoilage_forecast import forecast_spoilage as _fs
                _sf = _fs(obs.rho, obs.temp, obs.rh, hours_ahead=1)
                _k_eff_for_rerank = float(_sf.get("k_effective", 0.0) or 0.0)
            except Exception:
                _k_eff_for_rerank = 0.0

            passages = [{"text": c.passage, "score": float(getattr(c, "score", 0.0)), "id": c.doc_id, "meta": c.meta} for c in response.citations]
            reranked = lexical_arrhenius_rerank(
                passages, obs.temp, obs.rho, obs.rh, k_eff=_k_eff_for_rerank,
            )
            # Use reranked order for guidance extraction.
            ranked_citations = reranked
            if reranked:
                # 2026-04 honesty fix: aggregate the *Arrhenius
                # consistency factor* alone — not the lexical bonus —
                # so the gate at `context_to_logits.compute_context_modifier`
                # operates on the only true thermodynamic signal in
                # the rerank pipeline.
                context["physics_consistency_score"] = float(
                    sum(float(p.get("arrhenius_consistency", 1.0)) for p in reranked)
                    / len(reranked)
                )
                # Keep the legacy mean-bonus aggregate available for
                # back-compat consumers (was the meaning of this field
                # before the audit), but with a renamed key.
                context["lexical_bonus_mean"] = float(
                    sum(float(p.get("lexical_bonus", 0.0)) for p in reranked) / len(reranked)
                )
        except ImportError:
            ranked_citations = [{"text": c.passage,
                                 "score": float(getattr(c, "score", 0.0)),
                                 "id": c.doc_id,
                                 "meta": c.meta}
                                for c in response.citations]

        context["evidence_hashes"] = response.evidence_hashes

        for cit in response.citations:
            context["citations"].append({
                "doc_id": cit.doc_id,
                "passage": cit.passage[:300],
                "sha256": cit.sha256,
            })

        # 2026-04 fix: honour the rerank order — top is rank 1, not the
        # max-by-score-over-iteration which is sensitive to dict order
        # on RRF ties.
        if ranked_citations:
            top_entry = ranked_citations[0]
            context["top_citation_score"] = float(top_entry.get("score", 0.0))
            context["top_doc_id"] = top_entry.get("id", "")

        # Assign guidance based on document IDs.
        for entry in ranked_citations:
            doc_id = entry.get("id", "")
            passage = entry.get("text", "")[:300]

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

        # Evaluate context quality via the named retrieval-quality guard
        # (third of the paper Section 3.7 guard triple, alongside the
        # unit and feasibility guards). Using the answer-formatting unit
        # guard here would false-positive on the template engine's
        # "Based on N relevant sources" preamble, so the retrieval-level
        # check is intentionally distinct.
        context["guards_passed"] = retrieval_quality_ok(
            response.citations, context["top_citation_score"]
        )

        # Retrieval quality diagnostics for research reporting.
        # Faithfulness@k and evidence_coverage compare the answer-proxy
        # text against the *full* retrieved passage, not the truncated
        # 300-char preview that lands in ``context["citations"]``. The
        # 2026-04 audit caught the truncation bug (it under-reported
        # faithfulness when the matching span lived past char 300);
        # the fix here uses ``response.citations`` directly.
        try:
            from .eval.metrics import (
                faithfulness_at_k,
                evidence_coverage,
            )
            metric_citations = [
                {"excerpt": c.passage, "id": c.doc_id}
                for c in response.citations
            ]
            answer_proxy = (
                context.get("regulatory_guidance")
                or context.get("sop_guidance")
                or context.get("slca_guidance")
                or ""
            )
            context["retrieval_metrics"] = {
                "faithfulness_at_3": faithfulness_at_k(answer_proxy, metric_citations, k=3),
                "evidence_coverage": evidence_coverage(answer_proxy, metric_citations),
                "n_citations": len(metric_citations),
            }
        except Exception:
            context["retrieval_metrics"] = {}

        # Optional counterfactual retrieval: remove the top doc and compare.
        policy_flags = {}
        if hasattr(obs, "raw") and isinstance(obs.raw, dict):
            policy_flags = obs.raw.get("policy_flags", {})
        if policy_flags.get("enable_pirag_counterfactual_eval", False):
            try:
                cf_query = query + " counterfactual alternative guidance"
                cf_resp = pipeline.ask(cf_query, k=4, anchor_on_chain=False)
                cf_top = cf_resp.citations[0].doc_id if cf_resp.citations else ""
                context["counterfactual"] = {
                    "query": cf_query,
                    "top_doc_id": cf_top,
                    "top_doc_changed": bool(cf_top and cf_top != context.get("top_doc_id", "")),
                    "n_citations": len(cf_resp.citations),
                }
            except Exception as _exc:
                _log.debug("counterfactual retrieval skipped: %s", _exc)
                context["counterfactual"] = {}

    except Exception as _exc:
        _log.debug("retrieve_role_context fell through to fallback: %s", _exc)

    # Extract actionable keywords from guidance passages
    try:
        from .keyword_extractor import extract_keywords_by_type
        keywords: Dict[str, Any] = {}
        for field in ["regulatory_guidance", "sop_guidance", "slca_guidance",
                      "waste_hierarchy_guidance", "governance_guidance"]:
            text = context.get(field, "")
            if text:
                kw_type = field.replace("_guidance", "")
                keywords[kw_type] = extract_keywords_by_type(text)
        context["keywords"] = keywords
    except ImportError:
        context["keywords"] = {}

    return context
