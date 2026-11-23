"""Role-specific piR query construction with MCP-informed refinements.

Each agent role has a base query template plus conditional expansions
triggered by observation thresholds and MCP tool results. When an MCP
server is available, queries can be built via the MCP prompts/get
primitive; otherwise, direct template expansion is used.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional

from .guards.feasibility_guard import within_ranges
from .guards.retrieval_guard import (
    MIN_TOP_CITATION_SCORE,
    retrieval_quality_ok,
)
from .guards.unit_guard import units_consistent
from .mcp.protocol import MCPMessage, MCPServer
from .strict_validation import handle_unexpected_failure

# Default feasibility-guard constraints applied to retrieval excerpts
# in the routing context pipeline. These are intentionally permissive
# because the retrieval text is documentary, not a numeric simulator
# answer; any number that lands strictly inside a sensible engineering
# range passes. See ``guards/feasibility_guard.within_ranges`` for the
# parser. The constraints are exposed as a module-level constant so
# tests can monkeypatch them.
DEFAULT_GUARD_CONSTRAINTS: Dict[str, Any] = {
    "min": -1.0e9,
    "max": 1.0e9,
}

_log = logging.getLogger(__name__)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_sha256(value: Any) -> Optional[str]:
    """Hash JSON-native metadata without inventing a string coercion."""

    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def _base_citation_record(citation: Any, rank: int) -> Dict[str, Any]:
    """Retain every score already computed by the base retriever."""

    score = float(getattr(citation, "score", 0.0))
    fused_value = getattr(citation, "fused_score", None)
    # Backward compatibility for callers constructing Citation(score=...) with
    # the newly added fused_score field left at its default zero.
    fused_score = float(
        score if fused_value is None or (fused_value == 0.0 and score != 0.0)
        else fused_value
    )
    passage = str(citation.passage)
    content_sha256 = str(
        getattr(citation, "sha256", "") or _text_sha256(passage)
    )
    doc_id = str(citation.doc_id)
    metadata = getattr(citation, "meta", {}) or {}
    document_sha256 = _json_sha256({
        "doc_id": doc_id,
        "content_sha256": content_sha256,
    })
    retrieval_rank = int(getattr(citation, "retrieval_rank", 0) or rank)
    return {
        "text": passage,
        "id": doc_id,
        "meta": metadata,
        "base_rank": retrieval_rank,
        "raw_score": fused_score,
        "score": fused_score,
        "fused_score": fused_score,
        "rerank_score": fused_score,
        "sparse_rank": getattr(citation, "sparse_rank", None),
        "raw_sparse_score": float(
            getattr(citation, "sparse_score", 0.0)
        ),
        "sparse_rrf": float(getattr(citation, "sparse_rrf", 0.0)),
        "dense_rank": getattr(citation, "dense_rank", None),
        "raw_dense_score": float(
            getattr(citation, "dense_score", 0.0)
        ),
        "dense_rrf": float(getattr(citation, "dense_rrf", 0.0)),
        "fusion": str(getattr(citation, "fusion", "")),
        "content_sha256": content_sha256,
        "document_sha256": document_sha256,
        "metadata_sha256": _json_sha256(metadata),
    }


def _public_ranked_citation(
    entry: Dict[str, Any], final_rank: int,
) -> Dict[str, Any]:
    """Expose one ranked hit without duplicating its passage text."""

    passage = str(entry.get("text", ""))
    return {
        "rank": int(final_rank),
        "base_rank": int(entry.get("base_rank", final_rank)),
        "doc_id": str(entry.get("id", "")),
        "raw_score": float(entry.get("raw_score", 0.0)),
        "score": float(entry.get("score", 0.0)),
        "fused_score": float(entry.get("fused_score", 0.0)),
        "rerank_score": float(
            entry.get("rerank_score", entry.get("score", 0.0))
        ),
        "raw_sparse_score": float(entry.get("raw_sparse_score", 0.0)),
        "sparse_rank": entry.get("sparse_rank"),
        "sparse_rrf": float(entry.get("sparse_rrf", 0.0)),
        "raw_dense_score": float(entry.get("raw_dense_score", 0.0)),
        "dense_rank": entry.get("dense_rank"),
        "dense_rrf": float(entry.get("dense_rrf", 0.0)),
        "fusion": str(entry.get("fusion", "")),
        "lexical_score": entry.get("lexical_score"),
        "lexical_bonus": entry.get("lexical_bonus"),
        "arrhenius_score": entry.get("arrhenius_score"),
        "arrhenius_consistency": entry.get("arrhenius_consistency"),
        "physics_bonus": entry.get("physics_bonus"),
        "physics_consistency": entry.get("physics_consistency"),
        "content_sha256": entry.get("content_sha256"),
        "document_sha256": entry.get("document_sha256"),
        "metadata_sha256": entry.get("metadata_sha256"),
        "passage_preview_sha256": _text_sha256(passage[:300]),
        "passage_preview_character_count": len(passage[:300]),
        "passage_character_count": len(passage),
    }


ROLE_QUERY_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "farm": {
        "base": ("source-scoped leafy-greens context and the declared synthetic "
                 "spinach operating envelope, mechanistic spoilage-risk equation, "
                 "and telemetry-input limitations"),
        "conditions": [
            {"trigger": lambda obs, mcp: obs.temp > 8.0,
             "append": "temperature above the author-declared 8 degree Celsius benchmark envelope"},
            {"trigger": lambda obs, mcp: obs.rho > 0.20,
             "append": "elevated modeled spoilage risk; do not infer food safety"},
            {"trigger": lambda obs, mcp: (mcp.get("check_compliance") or {}).get("violations"),
             "append": "declared operating-envelope violations detected by monitoring"},
        ],
    },
    "processor": {
        "base": ("synthetic processing-state and surplus context, including declared "
                 "computational-footprint estimates and their measurement boundaries"),
        "conditions": [
            {"trigger": lambda obs, mcp: obs.surplus_ratio > 0.3,
             "append": "surplus ratio above the declared policy-oracle query trigger"},
            {"trigger": lambda obs, mcp: obs.surplus_ratio > 0.5,
             "append": "surplus ratio above the declared chain-query and calculator trigger"},
            {"trigger": lambda obs, mcp: not (mcp.get("policy_oracle", {}) or {}).get("allowed", True),
             "append": "author-declared governance-policy oracle result excludes the queried option"},
        ],
    },
    "cooperative": {
        "base": ("author-declared social-performance proxy and coordinator-mediated "
                 "peer-message context, with off-chain calculation-trace integrity "
                 "and optional external anchoring limitations"),
        "conditions": [
            {"trigger": lambda obs, mcp: obs.tau > 0.5,
             "append": "synthetic volatility flag active during the cooperative overlay"},
            {"trigger": lambda obs, mcp: obs.rho > 0.30,
             "append": "modeled spoilage risk above the declared recovery-logit knee"},
            {"trigger": lambda obs, mcp: (mcp.get("slca_lookup") or {}).get("base_scores", {}).get("local_redistribute", {}).get("R", 0) >= 0.78,
             "append": "local redistribution has the declared R prior 0.78; it is not a measured community outcome"},
        ],
    },
    "distributor": {
        "base": ("local-redistribution proxy scope and synthetic operating-envelope "
                 "context with the exact modeled transport-emissions equation"),
        "conditions": [
            {"trigger": lambda obs, mcp: obs.rho > 0.35,
             "append": "modeled risk above the declared spoilage-forecast tool trigger"},
            {"trigger": lambda obs, mcp: obs.rho > 0.45,
             "append": "modeled risk above the declared calculator-query trigger; no eligibility inference"},
            {"trigger": lambda obs, mcp: (mcp.get("check_compliance") or {}).get("violations"),
             "append": "author-declared operating-envelope flag; it does not determine redistribution eligibility"},
        ],
    },
    "recovery": {
        "base": ("aggregate recovery and circular-economy proxy equations, including "
                 "the limitations of food-bank, animal-feed, and composting heuristics"),
        "conditions": [
            {"trigger": lambda obs, mcp: obs.rho > 0.30,
             "append": "modeled risk above the declared recovery-logit knee; retrieve continuous proxy scores, not disposition rules"},
            {"trigger": lambda obs, mcp: (mcp.get("footprint_query") or {}).get("efficiency_flag") == "above_baseline",
             "append": "declared computational-footprint proxy flag; not hardware telemetry"},
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
    """Build a piR query string for the given role and conditions.

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
                handle_unexpected_failure(
                    f"MCP prompt retrieval for role {role}", _exc, _log,
                )

    # Fallback: direct template expansion
    template = ROLE_QUERY_TEMPLATES.get(role, {"base": "supply chain management", "conditions": []})
    parts = [template["base"]]

    for cond in template["conditions"]:
        try:
            if cond["trigger"](obs, mcp_results):
                parts.append(cond["append"])
        except Exception as _exc:
            handle_unexpected_failure(
                f"role-query condition evaluation for role {role}", _exc, _log,
            )
            continue

    if scenario != "baseline":
        try:
            from .mcp.prompts import SCENARIO_SEARCH_TERMS
            scenario_terms = SCENARIO_SEARCH_TERMS.get(scenario, "")
        except ImportError as _exc:
            handle_unexpected_failure(
                "scenario retrieval-term lookup", _exc, _log,
            )
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
    piR retrieval.
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
    retrieval_kind: str = "pirag",
) -> Dict[str, Any]:
    """Retrieve either piR or the declared standard-RAG comparator.

    Returns a dict with query, citations, guidance fields, scores, and
    guard/provenance metadata. ``standard`` retains the same corpus, base
    pipeline, citation guards, and downstream mapping but removes the three
    piR additions declared by the protocol: state-conditioned query
    expansion, lexical/Arrhenius reranking, and (downstream) temporal
    continuity weighting.
    """
    if retrieval_kind not in ("pirag", "standard"):
        raise ValueError(
            "retrieval_kind must be 'pirag' or 'standard', got "
            f"{retrieval_kind!r}"
        )
    # Fault-injection paths may pass sparse/None MCP payloads; normalize once.
    mcp_results = mcp_results or {}
    context: Dict[str, Any] = {
        "retrieval_kind": retrieval_kind,
        "query": "",
        "citations": [],
        # Compatibility citations above remain in source-retrieval order.
        # The new record below is the exact policy-used final top-k order with
        # score decomposition and hashes but no duplicated passage text.
        "ranked_citations": [],
        "ranked_evidence_hashes": [],
        "query_transform_metadata": {},
        "ranking_transform_metadata": {},
        "pipeline_retrieval_metadata": {},
        "pipeline_guard_decisions": {},
        "regulatory_guidance": "",
        "sop_guidance": "",
        "slca_guidance": "",
        "waste_hierarchy_guidance": "",
        "governance_guidance": "",
        # Raw reciprocal-rank-fusion strength of the document selected after
        # reranking.  ``top_citation_score`` remains a compatibility alias for
        # this raw value because existing trace consumers use that key.
        "top_fused_score": 0.0,
        "top_citation_score": 0.0,
        # Adjusted score used only to order passages after the declared
        # lexical + Arrhenius rerank.  It is not a calibrated confidence.
        "top_rerank_score": 0.0,
        "top_doc_id": "",
        # Fail closed: an unavailable retrieval pipeline cannot authorize a
        # retrieved-evidence modifier. Separately computed MCP features are
        # handled on their own channel and remain available downstream.
        "guards_passed": False,
        "guard_decisions": {},
        "evidence_hashes": [],
        "retrieval_metrics": {},
        "counterfactual": {},
        "alternative_query_retrieval": {},
        "physics_consistency_score": 1.0,
    }

    if pipeline is None:
        return context

    query = build_role_query(role, obs, scenario, mcp_results, mcp_server)
    base_query = query
    query_expansion_attempted = False
    query_expansion_executed = False
    query_expansion_k_eff = None

    # Physics-informed expansion belongs only to piR.  The standard-RAG
    # comparator sends the same role query directly to the same base pipeline.
    if retrieval_kind == "pirag":
        query_expansion_attempted = True
        try:
            from .physics_reranker import expand_query_with_physics
            spoilage_forecast = mcp_results.get("spoilage_forecast") or {}
            if not isinstance(spoilage_forecast, dict):
                spoilage_forecast = {}
            k_eff = spoilage_forecast.get("k_effective", 0.0)
            query = expand_query_with_physics(query, obs.rho, obs.temp, k_eff)
            query_expansion_executed = True
            try:
                query_expansion_k_eff = float(k_eff)
            except (TypeError, ValueError):
                query_expansion_k_eff = None
        except ImportError as _exc:
            handle_unexpected_failure(
                "physics-informed query expansion", _exc, _log,
            )

    context["query"] = query
    context["query_transform_metadata"] = {
        "role": role,
        "scenario": scenario,
        "retrieval_kind": retrieval_kind,
        "base_query_source": "build_role_query",
        "base_query": base_query,
        "base_query_sha256": _text_sha256(base_query),
        "final_query": query,
        "final_query_sha256": _text_sha256(query),
        "physics_expansion_attempted": query_expansion_attempted,
        "physics_expansion_executed": query_expansion_executed,
        "physics_expansion_changed_query": query != base_query,
        "physics_expansion_inputs": {
            "rho": float(obs.rho),
            "temperature": float(obs.temp),
            "k_effective": query_expansion_k_eff,
        } if query_expansion_attempted else None,
        "requested_k": 4,
        "anchor_on_chain": False,
    }

    try:
        response = pipeline.ask(query, k=4, anchor_on_chain=False)
        context["pipeline_retrieval_metadata"] = dict(
            getattr(response, "retrieval_metadata", {}) or {}
        )
        context["pipeline_guard_decisions"] = {
            "guards_passed": getattr(response, "guards_passed", None),
            "breakdown": dict(
                getattr(response, "guard_breakdown", {}) or {}
            ),
        }

        # Base-pipeline order is the standard-RAG order. piR alone adds the
        # lexical/Arrhenius reranker; both arms retain raw fused scores for the
        # same author-declared RRF-floor gate.
        base_ranked_citations = [
            _base_citation_record(citation, rank)
            for rank, citation in enumerate(response.citations, start=1)
        ]
        ranked_citations = [dict(item) for item in base_ranked_citations]
        rerank_attempted = retrieval_kind == "pirag"
        rerank_executed = False
        rerank_k_eff = None
        if retrieval_kind == "pirag":
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
                    _sf = _fs(
                        obs.rho, obs.temp, obs.rh, hours_ahead=1,
                        age_hours=float(getattr(obs, "hour", 0.0)),
                    )
                    _k_eff_for_rerank = float(
                        _sf.get("k_effective", 0.0) or 0.0
                    )
                except Exception as _exc:
                    handle_unexpected_failure(
                        "spoilage-rate calculation for lexical-Arrhenius reranking",
                        _exc,
                        _log,
                    )
                    _k_eff_for_rerank = 0.0
                rerank_k_eff = _k_eff_for_rerank

                passages = [dict(item) for item in base_ranked_citations]
                reranked = lexical_arrhenius_rerank(
                    passages, obs.temp, obs.rho, obs.rh,
                    k_eff=_k_eff_for_rerank,
                )
                # Use reranked order for guidance extraction.
                ranked_citations = reranked
                rerank_executed = True
                if reranked:
                    # Aggregate the Arrhenius consistency factor alone so the
                    # downstream physics gate operates on a thermodynamic
                    # signal rather than a lexical bonus.
                    context["physics_consistency_score"] = float(
                        sum(
                            float(p.get("arrhenius_consistency", 1.0))
                            for p in reranked
                        ) / len(reranked)
                    )
                    context["lexical_bonus_mean"] = float(
                        sum(
                            float(p.get("lexical_bonus", 0.0))
                            for p in reranked
                        ) / len(reranked)
                    )
            except ImportError as _exc:
                handle_unexpected_failure(
                    "lexical-Arrhenius reranker import", _exc, _log,
                )

        context["ranking_transform_metadata"] = {
            "base_order_source": "PiRResponse.citations",
            "final_order_source": (
                "lexical_arrhenius_rerank"
                if rerank_executed else "PiRResponse.citations"
            ),
            "rerank_attempted": rerank_attempted,
            "rerank_executed": rerank_executed,
            "rerank_inputs": {
                "temperature": float(obs.temp),
                "rho": float(obs.rho),
                "humidity": float(obs.rh),
                "k_effective": rerank_k_eff,
            } if rerank_attempted else None,
            "base_count": len(base_ranked_citations),
            "final_count": len(ranked_citations),
        }

        context["evidence_hashes"] = response.evidence_hashes

        for cit in response.citations:
            context["citations"].append({
                "doc_id": cit.doc_id,
                "passage": cit.passage[:300],
                "sha256": cit.sha256,
            })

        base_by_id = {
            str(entry.get("id", "")): entry
            for entry in base_ranked_citations
        }
        public_ranked_citations = []
        for final_rank, entry in enumerate(ranked_citations, start=1):
            # Preserve base-retrieval fields even if a custom reranker returns
            # only its adjusted fields. Live lexical_arrhenius_rerank already
            # carries them through via dict expansion.
            merged = dict(base_by_id.get(str(entry.get("id", "")), {}))
            merged.update(entry)
            public_ranked_citations.append(
                _public_ranked_citation(merged, final_rank)
            )
        context["ranked_citations"] = public_ranked_citations
        context["ranked_evidence_hashes"] = [
            citation["content_sha256"]
            for citation in public_ranked_citations
            if citation.get("content_sha256")
        ]

        # Honor the rerank order, but keep the score scales separate.  The
        # selected document's raw RRF strength drives psi_2, psi_3, and the
        # author-declared RRF-floor gate; the adjusted rerank score only establishes
        # ordering.  Mixing these values makes an additive lexical bonus look
        # like calibrated confidence and invalidates the declared RRF-floor gate.
        top_entry = None
        if ranked_citations:
            top_entry = ranked_citations[0]
            top_fused_score = float(
                top_entry.get("fused_score", top_entry.get("score", 0.0))
            )
            top_rerank_score = float(
                top_entry.get("rerank_score", top_entry.get("score", 0.0))
            )
            context["top_fused_score"] = top_fused_score
            context["top_citation_score"] = top_fused_score
            context["top_rerank_score"] = top_rerank_score
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

        # Aggregate the three Section 3.7 gates (dimensional analysis,
        # feasibility, and the author-declared RRF floor) into a single
        # ``guards_passed``
        # flag. Any guard returning False causes
        # ``context_to_logits.compute_context_modifier`` to zero only the
        # piR-derived term. Separately computed MCP operating-envelope,
        # modeled-forecast, and history features remain active. This does not prove that the
        # guards detect every bad input or that guarded performance cannot
        # degrade. The per-guard outcomes are surfaced so an operator (or the
        # explainability panel) can see which guard tripped.
        retrieval_ok = retrieval_quality_ok(
            ranked_citations, context["top_fused_score"]
        )

        # Run unit + feasibility guards over the top-ranked passage
        # (the canonical evidence the policy is being asked to rely on).
        # When no usable citation exists the unit/feasibility checks are
        # vacuously True; the retrieval guard will already have failed
        # and the aggregate stays False.
        top_passage = str(top_entry.get("text", "")) if top_entry else ""
        unit_ok = units_consistent(top_passage) if top_passage else True
        feasibility_ok = within_ranges(top_passage, DEFAULT_GUARD_CONSTRAINTS) if top_passage else True

        context["guard_breakdown"] = {
            "retrieval": bool(retrieval_ok),
            "unit": bool(unit_ok),
            "feasibility": bool(feasibility_ok),
        }
        context["guards_passed"] = bool(retrieval_ok and unit_ok and feasibility_ok)
        context["guard_decisions"] = {
            "retrieval_rrf_floor": {
                "passed": bool(retrieval_ok),
                "citation_count": len(ranked_citations),
                "evaluated_doc_id": context.get("top_doc_id", ""),
                "top_fused_score": float(context["top_fused_score"]),
                "minimum_exclusive": float(MIN_TOP_CITATION_SCORE),
            },
            "unit": {
                "passed": bool(unit_ok),
                "evaluated_doc_id": context.get("top_doc_id", ""),
            },
            "feasibility": {
                "passed": bool(feasibility_ok),
                "evaluated_doc_id": context.get("top_doc_id", ""),
                "constraints": dict(DEFAULT_GUARD_CONSTRAINTS),
            },
            "aggregate": {
                "passed": bool(context["guards_passed"]),
                "rule": "retrieval_rrf_floor AND unit AND feasibility",
            },
        }

        # Descriptive lexical-overlap diagnostics only. They are excluded from
        # retrieval-quality claims and do not replace independent judgments.
        # The metrics compare an answer-proxy copied from retrieval against
        # text against the *full* retrieved passage, not the truncated
        # 300-char preview that lands in ``context["citations"]``. The
        # 2026-04 cleanup caught the truncation bug (it under-reported
        # faithfulness when the matching span lived past char 300);
        # the fix here uses ``response.citations`` directly.
        try:
            from .eval.metrics import (
                citation_token_overlap_at_k,
                sentence_token_overlap_coverage,
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
            citation_overlap = citation_token_overlap_at_k(
                answer_proxy, metric_citations, k=3
            )
            sentence_overlap = sentence_token_overlap_coverage(
                answer_proxy, metric_citations
            )
            context["retrieval_metrics"] = {
                "interpretation": "descriptive lexical overlap only; not retrieval-quality evidence",
                "citation_token_overlap_at_3": citation_overlap,
                "sentence_token_overlap_coverage": sentence_overlap,
                "faithfulness_at_3": citation_overlap,  # legacy alias
                "evidence_coverage": sentence_overlap,  # legacy alias
                "n_citations": len(metric_citations),
            }
        except Exception as _exc:
            handle_unexpected_failure(
                "retrieval-metric calculation", _exc, _log,
            )
            context["retrieval_metrics"] = {}

        # Optional alternative-query retrieval diagnostic. The legacy flag and
        # ``counterfactual`` output key are retained for compatibility, but
        # this calculation appends an alternative-guidance phrase and does not
        # exclude the original top document. It is not a counterfactual.
        policy_flags = {}
        if hasattr(obs, "raw") and isinstance(obs.raw, dict):
            policy_flags = obs.raw.get("policy_flags", {})
        if policy_flags.get("enable_pirag_counterfactual_eval", False):
            try:
                cf_query = query + " alternative guidance"
                cf_resp = pipeline.ask(cf_query, k=4, anchor_on_chain=False)
                cf_top = cf_resp.citations[0].doc_id if cf_resp.citations else ""
                diagnostic = {
                    "kind": "alternative_query_retrieval",
                    "query": cf_query,
                    "top_doc_id": cf_top,
                    "top_doc_changed": bool(cf_top and cf_top != context.get("top_doc_id", "")),
                    "n_citations": len(cf_resp.citations),
                }
                context["alternative_query_retrieval"] = diagnostic
                context["counterfactual"] = diagnostic  # legacy alias
            except Exception as _exc:
                handle_unexpected_failure(
                    "optional alternative-query retrieval", _exc, _log,
                )
                context["alternative_query_retrieval"] = {}
                context["counterfactual"] = {}

    except Exception as _exc:
        handle_unexpected_failure(
            f"role-context retrieval for role {role}", _exc, _log,
        )

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
    except ImportError as _exc:
        handle_unexpected_failure(
            "retrieval keyword extraction", _exc, _log,
        )
        context["keywords"] = {}

    return context
