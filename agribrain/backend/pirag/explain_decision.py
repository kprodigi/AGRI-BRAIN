"""Feature-attribution explanation engine for routing decisions.

Generates human-readable explanations with:

1. **Dominant-feature attribution** — picks ``argmax(|THETA_CONTEXT[a] *
   psi|)`` and reports the matching context feature as the rationale.
   Earlier wording called this a "causal chain"; that label was
   inaccurate because the system has no structural causal model and
   no intervention semantics. The current explanation is a linear
   feature-attribution readout, which is what the code actually
   computes.
2. **Ablation delta** — what the action probability would be if the
   MCP/piRAG context modifier were zeroed (same RNG seed, same
   environment). Earlier wording called this a "counterfactual"; in
   the Pearlian sense it is not (no twin-network, no abduction). The
   correct framing is a leave-one-out / ablation delta, which is
   what the code computes.
3. **Source citations** — inline ``[KB:]`` references to the
   retrieved knowledge-base document IDs. The ``[KB:]`` tag is shared
   across multi-field explanations because only the top-ranked doc
   is currently surfaced; reviewers should not over-interpret a
   single ``[KB:]`` per paragraph as a distinct citation per claim.
4. **Provenance** — SHA-256 evidence hashes plus a Merkle root over
   the explanation payload. Optional on-chain anchoring is governed
   by ``CHAIN_REQUIRE_PRIVKEY`` and ``CHAIN_SUBMIT``; the default
   path produces an off-chain root with no verifying reader.

Output schema retains the legacy field name ``causal_chain`` for
backward compatibility with the frontend/explainability panel; the
new alias ``attribution_chain`` carries the same content.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .provenance.hasher import hash_artifact
from .context_to_logits import THETA_CONTEXT

_log = logging.getLogger(__name__)


_FEATURE_NAMES = [
    "compliance severity", "spoilage forecast urgency",
    "retrieval confidence", "regulatory pressure", "recovery saturation",
]

_ACTION_LABELS = {
    "cold_chain": "cold chain (long-haul)",
    "local_redistribute": "local redistribution",
    "recovery": "recovery/composting",
}

_ACTION_INDEX = {"cold_chain": 0, "local_redistribute": 1, "recovery": 2}


def explain_decision(
    action: str,
    role: str,
    hour: float,
    obs: Any,
    mcp_results: Dict[str, Any],
    rag_context: Dict[str, Any],
    slca_score: float,
    carbon_kg: float,
    waste: float,
    context_features: Optional[np.ndarray] = None,
    logit_adjustment: Optional[np.ndarray] = None,
    action_probs: Optional[np.ndarray] = None,
    counterfactual_action: Optional[str] = None,
    counterfactual_probs: Optional[np.ndarray] = None,
    governance_override: bool = False,
    keywords: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a causal explanation for a routing decision.

    Parameters
    ----------
    action : selected action name.
    role : active agent role.
    hour : simulation hour.
    obs : current Observation.
    mcp_results : results from MCP tool dispatch.
    rag_context : results from piRAG retrieval.
    slca_score : composite SLCA score.
    carbon_kg : carbon emissions.
    waste : waste rate.
    context_features : 5D context feature vector (psi).
    logit_adjustment : 3D logit modifier (THETA_CONTEXT @ psi).
    action_probs : probability vector WITH context.
    counterfactual_action : action WITHOUT context.
    counterfactual_probs : probability vector WITHOUT context.
    governance_override : whether governance override was triggered.
    keywords : extracted keywords per guidance type.

    Returns
    -------
    Dict with summary, full_explanation, causal_chain, counterfactual,
    keywords, evidence_hashes, provenance data.
    """
    action_label = _ACTION_LABELS.get(action, action)
    action_idx = _ACTION_INDEX.get(action, 1)

    # --- Evidence hashes ---
    tools_invoked = mcp_results.get("_tools_invoked", [])
    mcp_hashes: List[str] = []

    compliance = mcp_results.get("check_compliance")
    if isinstance(compliance, dict):
        mcp_hashes.append(hash_artifact({"tool": "check_compliance", "result": compliance}))

    forecast = mcp_results.get("spoilage_forecast")
    if isinstance(forecast, dict):
        mcp_hashes.append(hash_artifact({"tool": "spoilage_forecast", "result": forecast}))

    slca_data = mcp_results.get("slca_lookup")
    if isinstance(slca_data, dict):
        mcp_hashes.append(hash_artifact({"tool": "slca_lookup", "result": slca_data}))

    pirag_hashes = rag_context.get("evidence_hashes", [])
    all_hashes = pirag_hashes + mcp_hashes

    # Merkle root
    merkle_root = ""
    if all_hashes:
        try:
            from .provenance.merkle import merkle_root as _mr
            merkle_root = _mr(all_hashes)
        except Exception as _exc:
            _log.debug("merkle root for explanation skipped: %s", _exc)

    # --- Physical basis ---
    physical_basis = (
        f"rho={obs.rho:.3f}, T={obs.temp:.1f}C, RH={obs.rh:.0f}%, "
        f"inventory={obs.inv:.0f}, surplus={getattr(obs, 'surplus_ratio', 0):.2f}"
    )

    # --- Paragraph 1: Causal chain ---
    para1 = _build_causal_paragraph(
        action, action_label, role, hour, obs,
        mcp_results, rag_context, context_features,
        governance_override,
    )

    # --- Paragraph 2: Context features and logit shift ---
    para2 = ""
    contributions = None
    dominant_idx = 0
    if context_features is not None and logit_adjustment is not None:
        contributions = THETA_CONTEXT[action_idx] * context_features
        dominant_idx = int(np.argmax(np.abs(contributions)))
        dominant_name = _FEATURE_NAMES[dominant_idx]

        prob_str = ""
        if action_probs is not None:
            prob_str = f", making {action_label} {action_probs[action_idx]*100:.1f}% probable"

        para2 = (
            f"These signals activated the {dominant_name} "
            f"(psi={context_features[dominant_idx]:.1f}) context feature, "
            f"shifting the cold chain logit by {logit_adjustment[0]:+.2f} "
            f"and the redistribution logit by {logit_adjustment[1]:+.2f}"
            f"{prob_str}."
        )

    # --- Paragraph 3: Ablation comparison (formerly labelled
    # "counterfactual"; the framing is honest now: same policy, same
    # phi(s), same RNG seed, with psi := 0). ---
    para3 = ""
    if counterfactual_probs is not None and action_probs is not None:
        delta_lr = (action_probs[1] - counterfactual_probs[1]) * 100
        para3 = (
            f"WITHOUT MCP/piRAG context, cold chain probability would have been "
            f"{counterfactual_probs[0]*100:.1f}% and redistribution "
            f"{counterfactual_probs[1]*100:.1f}%. "
            f"Context injection shifted {abs(delta_lr):.1f} percentage points "
            f"{'toward' if delta_lr > 0 else 'away from'} redistribution"
        )
        if counterfactual_action and counterfactual_action != action:
            cf_label = _ACTION_LABELS.get(counterfactual_action, counterfactual_action)
            para3 += f", changing the selected action from {cf_label} to {action_label}"
        para3 += "."

    # --- Paragraph 4: Source citations with keywords ---
    para4 = _build_citation_paragraph(rag_context, keywords)

    # --- Paragraph 5: Provenance ---
    n_mcp = len(mcp_hashes)
    n_pirag = len(pirag_hashes)
    para5 = (
        f"Provenance: {len(all_hashes)} evidence items "
        f"({n_mcp} MCP tool outputs + {n_pirag} piRAG citations)"
    )
    if merkle_root:
        para5 += f", Merkle root: {merkle_root[:12]}..."
    para5 += "."

    # --- Summary (one-line) ---
    summary = (
        f"{role} agent selected {action} at hour {hour:.1f}. "
        f"Spoilage risk: {obs.rho:.3f}. SLCA: {slca_score:.3f}. "
    )
    if isinstance(compliance, dict) and not compliance.get("compliant"):
        summary += "Compliance violations detected. "
    if isinstance(forecast, dict):
        summary += f"Spoilage forecast: {forecast.get('urgency', '')}. "

    # --- Full explanation ---
    paragraphs = [p for p in [para1, para2, para3, para4, para5] if p]
    full_explanation = "\n\n".join(paragraphs)

    # --- Feature-attribution structured data (legacy field name kept) ---
    attribution_chain: Dict[str, Any] = {}
    if contributions is not None:
        sorted_indices = sorted(range(5), key=lambda i: abs(contributions[i]), reverse=True)
        attribution_chain = {
            "primary_feature": _FEATURE_NAMES[sorted_indices[0]],
            "primary_contribution": float(contributions[sorted_indices[0]]),
            "secondary_feature": _FEATURE_NAMES[sorted_indices[1]] if len(sorted_indices) > 1 else None,
            "all_contributions": dict(zip(_FEATURE_NAMES, contributions.tolist())),
            # Legacy aliases for callers that still read ``primary_cause``.
            "primary_cause": _FEATURE_NAMES[sorted_indices[0]],
            "secondary_cause": _FEATURE_NAMES[sorted_indices[1]] if len(sorted_indices) > 1 else None,
        }

    # --- Ablation-delta structured data (formerly "counterfactual") ---
    # Honestly labelled: this is what the same policy, with the same
    # state vector phi(s) and the same RNG seed, would have selected if
    # the MCP/piRAG context modifier (Delta_z = THETA_CONTEXT @ psi) had
    # been zero. It is *not* a Pearl-style counterfactual: there is no
    # twin-network and no abduction step. It is a leave-one-out ablation
    # of the context layer.
    ablation_delta: Dict[str, Any] = {
        "kind": "ablation_psi_zero",
        "description": (
            "Action and probabilities the same policy would have produced "
            "with psi := 0 (i.e. with the MCP/piRAG context modifier "
            "disabled). Same RNG seed, same phi(s). This is an ablation "
            "delta, not a Pearl-style counterfactual."
        ),
        "action_without_context": counterfactual_action,
        "probs_without_context": counterfactual_probs.tolist() if counterfactual_probs is not None else None,
        "probs_with_context": action_probs.tolist() if action_probs is not None else None,
        "action_changed": (counterfactual_action != action) if counterfactual_action else False,
        "probability_shift": (action_probs - counterfactual_probs).tolist()
            if action_probs is not None and counterfactual_probs is not None else None,
    }

    return {
        "summary": summary.strip(),
        "physical_basis": physical_basis,
        "mcp_evidence": _build_mcp_evidence_str(mcp_results),
        "regulatory_context": (rag_context.get("regulatory_guidance", "") or "")[:200],
        "social_performance": f"SLCA: {slca_score:.3f}, carbon: {carbon_kg:.2f} kg, waste: {waste:.4f}",
        "full_explanation": full_explanation,
        "evidence_hashes": all_hashes,
        "tools_invoked": tools_invoked,
        "citations": rag_context.get("citations", []),
        "guards_passed": rag_context.get("guards_passed", True),
        "provenance_ready": bool(merkle_root),
        "merkle_root": merkle_root,
        # New honest field names + legacy aliases for backward compat.
        "attribution_chain": attribution_chain,
        "ablation_delta": ablation_delta,
        "causal_chain": attribution_chain,
        "counterfactual": ablation_delta,
        "keywords": keywords or {},
        "governance_override": governance_override,
    }


def _build_causal_paragraph(
    action: str, action_label: str, role: str, hour: float, obs: Any,
    mcp_results: Dict, rag_context: Dict, context_features: Optional[np.ndarray],
    governance_override: bool,
) -> str:
    """Build the BECAUSE paragraph identifying dominant causes."""
    if governance_override:
        return (
            f"The {role} agent MANDATED {action_label} at hour {hour:.1f} "
            f"via governance override BECAUSE simultaneous critical compliance violation "
            f"AND high spoilage forecast triggered the MCP governance constraint. "
            f"This override bypasses probabilistic routing and deterministically "
            f"selects the safest routing pathway."
        )

    action_idx = _ACTION_INDEX.get(action, 1)
    causes: List[str] = []

    if context_features is not None:
        contributions = THETA_CONTEXT[action_idx] * context_features
        sorted_idx = sorted(range(5), key=lambda i: abs(contributions[i]), reverse=True)

        for rank, idx in enumerate(sorted_idx[:2]):
            if abs(contributions[idx]) < 0.05:
                break
            if rank == 1 and abs(contributions[idx]) < 0.3 * abs(contributions[sorted_idx[0]]):
                break
            causes.append(_build_cause_phrase(idx, context_features, mcp_results, rag_context, obs))

    if not causes:
        return (
            f"The {role} agent routed produce to {action_label} at hour {hour:.1f} "
            f"based on the combined policy and context signals."
        )

    cause_text = causes[0]
    if len(causes) > 1:
        cause_text += f" AND {causes[1]}"

    return (
        f"The {role} agent routed produce to {action_label} at hour {hour:.1f} "
        f"BECAUSE {cause_text}."
    )


def _build_cause_phrase(
    feature_idx: int,
    context_features: np.ndarray,
    mcp_results: Dict,
    rag_context: Dict,
    obs: Any,
) -> str:
    """Build a causal phrase for a specific context feature."""
    if feature_idx == 0:
        violations = mcp_results.get("check_compliance", {}).get("violations", [])
        v = violations[0] if violations else {}
        severity = v.get("severity", "unknown")
        limit = v.get("limit", obs.temp)
        delta = obs.temp - limit if isinstance(limit, (int, float)) else 0
        return (
            f"the MCP compliance check detected a {severity} temperature violation "
            f"({obs.temp:.1f}C exceeding the {limit}C limit by {delta:.1f}C)"
        )
    elif feature_idx == 1:
        fc = mcp_results.get("spoilage_forecast", {})
        return (
            f"the spoilage forecast predicted quality will decline to "
            f"rho={fc.get('forecast_rho', '?')} within {fc.get('hours_ahead', '?')} hours "
            f"(urgency: {fc.get('urgency', '?')})"
        )
    elif feature_idx == 2:
        return (
            f"piRAG retrieved high-confidence guidance from "
            f"{rag_context.get('top_doc_id', 'the knowledge base')} "
            f"(relevance: {rag_context.get('top_citation_score', 0):.2f})"
        )
    elif feature_idx == 3:
        return (
            f"piRAG retrieved regulatory guidance from "
            f"{rag_context.get('top_doc_id', 'the knowledge base')} "
            f"(relevance: {rag_context.get('top_citation_score', 0):.2f}) "
            f"indicating regulatory pressure for shorter transport routes"
        )
    elif feature_idx == 4:
        return (
            "recent decisions are heavily skewed toward recovery, "
            "requiring rebalancing toward forward supply chain routes"
        )
    return f"context feature {feature_idx} was active"


def _build_citation_paragraph(
    rag_context: Dict, keywords: Optional[Dict],
) -> str:
    """Build the source citation paragraph with inline keywords."""
    sources: List[str] = []

    guidance_fields = [
        ("regulatory_guidance", "regulatory"),
        ("sop_guidance", "sop"),
        ("waste_hierarchy_guidance", "waste_hierarchy"),
        ("governance_guidance", "governance"),
        ("slca_guidance", "slca"),
    ]

    for field, kw_type in guidance_fields:
        text = rag_context.get(field, "")
        if not text:
            continue
        doc = rag_context.get("top_doc_id", "unknown")
        score = rag_context.get("top_citation_score", 0)

        kw_str = ""
        if keywords:
            kw_data = keywords.get(kw_type, {})
            if isinstance(kw_data, dict):
                all_kw = (
                    kw_data.get("thresholds", [])
                    + kw_data.get("required_actions", [])
                    + kw_data.get("regulations", [])
                )
                if all_kw:
                    kw_str = f" (key: {', '.join(all_kw[:3])})"

        sources.append(f"[KB: {doc}, relevance={score:.2f}]{kw_str}")

    if not sources:
        return ""
    return "This decision is supported by " + " and ".join(sources[:3]) + "."


def _build_mcp_evidence_str(mcp_results: Dict) -> str:
    """Build a summary string of MCP tool evidence."""
    parts: List[str] = []
    compliance = mcp_results.get("check_compliance")
    if isinstance(compliance, dict):
        status = "compliant" if compliance.get("compliant") else "non-compliant"
        n_viol = len(compliance.get("violations", []))
        parts.append(f"Compliance: {status} ({n_viol} violations)")

    forecast = mcp_results.get("spoilage_forecast")
    if isinstance(forecast, dict):
        parts.append(
            f"Spoilage forecast: rho={forecast.get('forecast_rho', '?')} "
            f"({forecast.get('urgency', '?')})"
        )

    slca_data = mcp_results.get("slca_lookup")
    if isinstance(slca_data, dict):
        parts.append(f"SLCA data: product={slca_data.get('product_type', '?')}")

    return "; ".join(parts) if parts else "No MCP tools invoked"
