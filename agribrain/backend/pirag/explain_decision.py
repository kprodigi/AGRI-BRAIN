"""Feature-attribution explanation engine for routing decisions.

Generates human-readable explanations with:

1. **Dominant-component attribution** — uses the recorded feature allocation
   of the final context modifier after retrieval-only temporal/physics scaling,
   clipping, and cooperative blending. A separate residual represents
   any declared fixed cooperative adjustment.
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
   is currently surfaced; a single ``[KB:]`` per paragraph should
   not be over-interpreted as a distinct citation per claim.
4. **Local evidence commitment** — SHA-256 evidence hashes plus a Merkle root
   over selected MCP outputs and retrieval passages. This function never
   submits the root on-chain and emits no Merkle inclusion paths.

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
    "operating-envelope exceedance", "modeled-spoilage forecast signal",
    "retrieval-score signal", "retrieved-policy signal",
    "modeled recovery-capacity signal",
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
    ablation_action: Optional[str] = None,
    ablation_probs: Optional[np.ndarray] = None,
    governance_override: bool = False,
    keywords: Optional[Dict[str, Any]] = None,
    *,
    effective_context_theta: Optional[np.ndarray] = None,
    chosen_action_context_contributions: Optional[np.ndarray] = None,
    chosen_action_context_residual: Optional[float] = None,
    context_attribution_scope: Optional[str] = None,
    context_integration_trace: Optional[Dict[str, Any]] = None,
    counterfactual_action: Optional[str] = None,
    counterfactual_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Generate a policy-trace explanation for a routing decision.

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
    ablation_action : action the same policy would have selected with
        ``psi := 0`` (i.e. the MCP/piRAG context modifier zeroed). This
        is an ablation, not a Pearl-style counterfactual; see
        ``ablation_delta.kind`` in the returned dict for the explicit
        framing. The deprecated alias ``counterfactual_action`` is
        accepted for backward compat.
    ablation_probs : probability vector under the same psi := 0
        ablation. Deprecated alias: ``counterfactual_probs``.
    governance_override : legacy compatibility field indicating whether the
        declared probability-gap rule activated.
    keywords : extracted keywords per guidance type.
    effective_context_theta : optional 3x5 matrix snapshotted before the
        decision's learner update. When omitted, the declared fixed matrix is
        used for backward-compatible hypothetical calls.
    chosen_action_context_contributions : optional length-5 allocation of the
        chosen action's final context modifier across features. It includes
        retrieval-only temporal/physics scaling, clipping, and cooperative
        blending.
    chosen_action_context_residual : non-feature remainder, such as a
        declared cooperative fixed adjustment. Contributions plus this residual must equal the
        chosen-action component of ``logit_adjustment``.
    context_attribution_scope : ``primary_context``, ``cooperative_blend``, or
        the legacy key ``cooperative_veto`` for the recorded cooperative
        operating-envelope adjustment.
    context_integration_trace : JSON-native forward trace that separates MCP
        and piRAG components, gates, clipping, and the learner Jacobian.

    Returns
    -------
    Dict with summary, full_explanation, attribution_chain, ablation_delta,
    keywords, evidence_hashes, provenance data. ``causal_chain`` and
    ``counterfactual`` are kept as legacy aliases for the same content.
    """
    # Back-compat: accept the legacy ``counterfactual_*`` kwargs but
    # internally use the honest ``ablation_*`` names. If both are
    # supplied the explicit ablation_* parameter wins; this matches
    # how callers were migrated in 2026-04.
    if ablation_action is None and counterfactual_action is not None:
        ablation_action = counterfactual_action
    if ablation_probs is None and counterfactual_probs is not None:
        ablation_probs = counterfactual_probs
    action_label = _ACTION_LABELS.get(action, action)
    action_idx = _ACTION_INDEX.get(action, 1)

    contributions = None
    contribution_residual = float(chosen_action_context_residual or 0.0)
    attribution_basis = None
    if chosen_action_context_contributions is not None:
        contributions = np.asarray(
            chosen_action_context_contributions, dtype=float,
        ).reshape(-1)
        if contributions.shape != (5,):
            raise ValueError("chosen_action_context_contributions must have length 5")
        attribution_basis = (
            "recorded_final_modifier_feature_allocation_plus_explicit_residual"
        )
    elif context_features is not None:
        theta = np.asarray(
            effective_context_theta
            if effective_context_theta is not None else THETA_CONTEXT,
            dtype=float,
        )
        if theta.shape != (3, 5):
            raise ValueError("effective_context_theta must have shape (3, 5)")
        raw_contributions = theta[action_idx] * np.asarray(context_features, dtype=float)
        contributions = raw_contributions.copy()
        if logit_adjustment is not None:
            target = float(np.asarray(logit_adjustment, dtype=float)[action_idx])
            raw_sum = float(raw_contributions.sum())
            if abs(raw_sum) > 1e-15:
                contributions *= target / raw_sum
            contribution_residual = target - float(contributions.sum())
        attribution_basis = "proportional_allocation_from_effective_theta"

    if contributions is not None and logit_adjustment is not None:
        target = float(np.asarray(logit_adjustment, dtype=float)[action_idx])
        reconstructed = float(contributions.sum()) + contribution_residual
        if not np.isclose(reconstructed, target, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "chosen-action feature allocation and residual do not "
                "reconstruct the final context modifier"
            )

    # --- Evidence hashes ---
    tools_invoked = mcp_results.get("_tools_invoked", [])
    mcp_hashes: List[str] = []
    mcp_evidence_hashes: Dict[str, str] = {}

    compliance = mcp_results.get("check_compliance")
    if isinstance(compliance, dict):
        artifact_hash = hash_artifact(
            {"tool": "check_compliance", "result": compliance}
        )
        mcp_hashes.append(artifact_hash)
        mcp_evidence_hashes["check_compliance"] = artifact_hash

    forecast = mcp_results.get("spoilage_forecast")
    if isinstance(forecast, dict):
        artifact_hash = hash_artifact(
            {"tool": "spoilage_forecast", "result": forecast}
        )
        mcp_hashes.append(artifact_hash)
        mcp_evidence_hashes["spoilage_forecast"] = artifact_hash

    slca_data = mcp_results.get("slca_lookup")
    if isinstance(slca_data, dict):
        artifact_hash = hash_artifact(
            {"tool": "slca_lookup", "result": slca_data}
        )
        mcp_hashes.append(artifact_hash)
        mcp_evidence_hashes["slca_lookup"] = artifact_hash

    pirag_hashes = list(rag_context.get("evidence_hashes", []))
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

    # --- Paragraph 1: dominant-feature policy trace ---
    para1 = _build_policy_trace_paragraph(
        action, action_label, role, hour, obs,
        mcp_results, rag_context, context_features,
        governance_override, action_probs, contributions, contribution_residual,
        context_attribution_scope,
    )

    # --- Paragraph 2: Context features and logit shift ---
    para2 = ""
    dominant_idx = 0
    if (context_features is not None and logit_adjustment is not None
            and contributions is not None):
        dominant_idx = int(np.argmax(np.abs(contributions)))
        dominant_name = (
            "the non-feature cooperative/rule adjustment"
            if abs(contribution_residual) > abs(contributions[dominant_idx])
            else _FEATURE_NAMES[dominant_idx]
        )

        prob_str = ""
        if action_probs is not None:
            prob_str = f", making {action_label} {action_probs[action_idx]*100:.1f}% probable"

        para2 = (
            f"The final context-modifier allocation was largest for "
            f"{dominant_name}, "
            f"shifting the cold chain logit by {logit_adjustment[0]:+.2f} "
            f"and the redistribution logit by {logit_adjustment[1]:+.2f}"
            f"{prob_str}."
        )

    # --- Paragraph 3: Ablation comparison (psi := 0).
    # Honest framing: same policy, same phi(s), same RNG seed, with the
    # MCP/piRAG context modifier zeroed. Not a Pearl-style counterfactual.
    # The comparison is explicitly labeled as a calculation-layer ablation so
    # it is not mistaken for a Pearlian intervention.
    para3 = ""
    if ablation_probs is not None and action_probs is not None:
        delta_lr = (action_probs[1] - ablation_probs[1]) * 100
        para3 = (
            f"Ablation (psi := 0): with the MCP/piRAG modifier zeroed, the "
            f"calculated cold-chain probability is {ablation_probs[0]*100:.1f}% and redistribution "
            f"{ablation_probs[1]*100:.1f}% under the same policy, phi(s) and RNG seed. "
            f"Context injection shifted {abs(delta_lr):.1f} percentage points "
            f"{'toward' if delta_lr > 0 else 'away from'} redistribution"
        )
        if ablation_action and ablation_action != action:
            cf_label = _ACTION_LABELS.get(ablation_action, ablation_action)
            para3 += f", changing the selected action from {cf_label} to {action_label}"
        para3 += "."

    # --- Paragraph 4: Source citations with keywords ---
    para4 = _build_citation_paragraph(rag_context, keywords)

    # --- Paragraph 5: Provenance ---
    n_mcp = len(mcp_hashes)
    n_pirag = len(pirag_hashes)
    para5 = (
        f"Local evidence commitment: {len(all_hashes)} exposed leaf hashes "
        f"({n_mcp} MCP tool outputs + {n_pirag} piRAG citations)"
    )
    if merkle_root:
        para5 += f", local Merkle root: {merkle_root[:12]}..."
    para5 += ". No Merkle inclusion paths or on-chain root anchor are claimed."

    # --- Summary (one-line) ---
    summary = (
        f"{role} agent selected {action} at hour {hour:.1f}. "
        f"Spoilage risk: {obs.rho:.3f}. Social-performance proxy: {slca_score:.3f}. "
    )
    if isinstance(compliance, dict) and not compliance.get("compliant"):
        summary += "Readings outside the declared benchmark envelope were detected. "
    if isinstance(forecast, dict):
        summary += f"Spoilage forecast: {forecast.get('urgency', '')}. "

    # --- Full explanation ---
    paragraphs = [p for p in [para1, para2, para3, para4, para5] if p]
    full_explanation = "\n\n".join(paragraphs)

    # --- Feature-attribution structured data (legacy field name kept) ---
    attribution_chain: Dict[str, Any] = {}
    if contributions is not None:
        sorted_indices = sorted(range(5), key=lambda i: abs(contributions[i]), reverse=True)
        residual_dominates = (
            abs(contribution_residual) > abs(contributions[sorted_indices[0]])
        )
        primary_feature = (
            "non-feature cooperative/rule adjustment"
            if residual_dominates else _FEATURE_NAMES[sorted_indices[0]]
        )
        primary_contribution = (
            contribution_residual
            if residual_dominates else float(contributions[sorted_indices[0]])
        )
        attribution_chain = {
            "primary_feature": primary_feature,
            "primary_contribution": float(primary_contribution),
            "secondary_feature": _FEATURE_NAMES[sorted_indices[1]] if len(sorted_indices) > 1 else None,
            "all_contributions": dict(zip(_FEATURE_NAMES, contributions.tolist())),
            "nonfeature_residual": float(contribution_residual),
            "scope": context_attribution_scope,
            "reconstructed_chosen_action_modifier": float(
                contributions.sum() + contribution_residual
            ),
            "basis": attribution_basis,
            # Legacy aliases for callers that still read ``primary_cause``.
            "primary_cause": primary_feature,
            "secondary_cause": _FEATURE_NAMES[sorted_indices[1]] if len(sorted_indices) > 1 else None,
        }

    # --- Ablation-delta structured data (formerly "counterfactual") ---
    # Honestly labelled: this is what the same policy, with the same
    # state vector phi(s) and the same RNG seed, would have selected if
    # the final MCP/piRAG context modifier (derived from THETA_CONTEXT @ psi)
    # had
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
        "action_without_context": ablation_action,
        "probs_without_context": ablation_probs.tolist() if ablation_probs is not None else None,
        "probs_with_context": action_probs.tolist() if action_probs is not None else None,
        "action_changed": (ablation_action != action) if ablation_action else False,
        "probability_shift": (action_probs - ablation_probs).tolist()
            if action_probs is not None and ablation_probs is not None else None,
    }

    return {
        "summary": summary.strip(),
        "physical_basis": physical_basis,
        "mcp_evidence": _build_mcp_evidence_str(mcp_results),
        "regulatory_context": (rag_context.get("regulatory_guidance", "") or "")[:200],
        "social_performance": f"social proxy: {slca_score:.3f}, carbon: {carbon_kg:.2f} kg, waste: {waste:.4f}",
        "full_explanation": full_explanation,
        # The full, ordered leaf list is returned so callers can recompute the
        # local commitment.  Earlier API code exposed only the first five
        # retrieval hashes while returning a root that also covered selected
        # MCP outputs, which made the public trail look complete when it was
        # not.  These are commitment leaves, not Merkle inclusion paths and
        # not evidence of an on-chain anchor.
        "evidence_hashes": all_hashes,
        "retrieval_evidence_hashes": pirag_hashes,
        "mcp_evidence_hashes": mcp_evidence_hashes,
        "evidence_hash_count": len(all_hashes),
        "evidence_hashes_complete": True,
        "commitment_type": "local_merkle_root",
        "merkle_inclusion_paths_exposed": False,
        "merkle_root_anchored_on_chain": False,
        "tools_invoked": tools_invoked,
        "citations": rag_context.get("citations", []),
        # Preserve the three-state contract: True/False are evaluated
        # outcomes, while None means the guard pipeline was not evaluated.
        "guards_passed": rag_context.get("guards_passed"),
        "provenance_ready": bool(merkle_root),
        "merkle_root": merkle_root,
        # New honest field names + legacy aliases for backward compat.
        "attribution_chain": attribution_chain,
        "ablation_delta": ablation_delta,
        "causal_chain": attribution_chain,
        "counterfactual": ablation_delta,
        "keywords": keywords or {},
        "probability_gap_override": governance_override,
        "probability_gap_rule": _governance_predicate_record(
            action_probs, governance_override,
        ),
        # Legacy schema aliases retained for existing clients.
        "governance_override": governance_override,
        "governance_predicate": _governance_predicate_record(
            action_probs, governance_override,
        ),
        "context_integration": context_integration_trace or {},
    }


def _build_policy_trace_paragraph(
    action: str, action_label: str, role: str, hour: float, obs: Any,
    mcp_results: Dict, rag_context: Dict, context_features: Optional[np.ndarray],
    governance_override: bool,
    action_probs: Optional[np.ndarray] = None,
    contributions: Optional[np.ndarray] = None,
    contribution_residual: float = 0.0,
    attribution_scope: Optional[str] = None,
) -> str:
    """Build a calculation-trace paragraph for dominant contributions."""
    if governance_override:
        predicate = _governance_predicate_record(action_probs, True)
        p_cold = predicate.get("p_cold_chain")
        gap = predicate.get("p_local_minus_cold")
        ceiling = predicate["p_cold_chain_ceiling"]
        minimum_gap = predicate["p_local_minus_cold_minimum"]
        observed = (
            f" (recorded p(cold)={p_cold:.6f}, "
            f"p(local)-p(cold)={gap:.6f})"
            if p_cold is not None and gap is not None else ""
        )
        return (
            f"The simulated {role} policy selected {action_label} at hour {hour:.1f} "
            f"after the author-declared probability-gap rule activated because "
            f"the recorded policy probabilities met p(cold) < {ceiling:.3f} and "
            f"p(local)-p(cold) > {minimum_gap:.2f}{observed}. "
            f"Operating-envelope and spoilage quantities can influence the logits, "
            f"but they are not separate predicates of this rule."
        )

    contributions_text: List[str] = []

    if context_features is not None and contributions is not None:
        sorted_idx = sorted(range(5), key=lambda i: abs(contributions[i]), reverse=True)

        if abs(contribution_residual) > abs(contributions[sorted_idx[0]]):
            scope = _attribution_scope_label(attribution_scope)
            return (
                f"The {role} agent routed produce to {action_label} at hour {hour:.1f} "
                f"because the {scope} non-feature adjustment was the largest "
                f"component of the recorded context modifier."
            )

        if attribution_scope in {"cooperative_blend", "cooperative_veto"}:
            dominant = _FEATURE_NAMES[sorted_idx[0]]
            scope = _attribution_scope_label(attribution_scope)
            return (
                f"The {role} agent routed produce to {action_label} at hour {hour:.1f} "
                f"because the {scope} allocation was largest for {dominant}."
            )

        for rank, idx in enumerate(sorted_idx[:2]):
            if abs(contributions[idx]) < 0.05:
                break
            if rank == 1 and abs(contributions[idx]) < 0.3 * abs(contributions[sorted_idx[0]]):
                break
            contributions_text.append(
                _build_contribution_phrase(
                    idx, context_features, mcp_results, rag_context, obs
                )
            )

    if not contributions_text:
        return (
            f"The {role} agent routed produce to {action_label} at hour {hour:.1f} "
            f"based on the combined policy and context signals."
        )

    contribution_text = contributions_text[0]
    if len(contributions_text) > 1:
        contribution_text += f" AND {contributions_text[1]}"

    return (
        f"The {role} agent routed produce to {action_label} at hour {hour:.1f} "
        f"because {contribution_text}."
    )


def _attribution_scope_label(scope: Optional[str]) -> str:
    """Return cautious display text while preserving legacy schema keys."""

    return {
        "primary_context": "primary-context",
        "cooperative_blend": "cooperative weighted-composition",
        "cooperative_veto": "cooperative operating-envelope",
    }.get(scope or "", (scope or "context").replace("_", " "))


def _governance_predicate_record(
    action_probs: Optional[np.ndarray], governance_override: bool,
) -> Dict[str, Any]:
    """Return the exact probability-space rule and recorded operands."""

    from src.models.action_selection import (
        GOVERNANCE_CC_PROB_CEILING,
        GOVERNANCE_LOCAL_ADVANTAGE_MIN,
    )

    record: Dict[str, Any] = {
        "rule": "p_cold_chain < ceiling AND p_local_minus_cold > minimum_gap",
        "p_cold_chain_ceiling": float(GOVERNANCE_CC_PROB_CEILING),
        "p_local_minus_cold_minimum": float(GOVERNANCE_LOCAL_ADVANTAGE_MIN),
        "triggered": bool(governance_override),
    }
    if action_probs is None:
        record.update({
            "p_cold_chain": None,
            "p_local_minus_cold": None,
            "predicate_recomputed": None,
        })
        return record

    probs = np.asarray(action_probs, dtype=float)
    if probs.shape != (3,):
        raise ValueError("action_probs must have length 3")
    p_cold = float(probs[0])
    gap = float(probs[1] - probs[0])
    record.update({
        "p_cold_chain": p_cold,
        "p_local_minus_cold": gap,
        "predicate_recomputed": bool(
            p_cold < GOVERNANCE_CC_PROB_CEILING
            and gap > GOVERNANCE_LOCAL_ADVANTAGE_MIN
        ),
    })
    return record


def _build_contribution_phrase(
    feature_idx: int,
    context_features: np.ndarray,
    mcp_results: Dict,
    rag_context: Dict,
    obs: Any,
) -> str:
    """Build a contribution phrase for a specific context feature."""
    if feature_idx == 0:
        violations = mcp_results.get("check_compliance", {}).get("violations", [])
        v = violations[0] if violations else {}
        severity = v.get("severity", "unknown")
        parameter = v.get("parameter", "unspecified")
        value = v.get("value")
        limit = v.get("limit")
        if isinstance(value, (int, float)) and isinstance(limit, (int, float)):
            if parameter == "temperature":
                detail = (
                    f"temperature reading {value:.1f}C above the declared maximum "
                    f"{limit:.1f}C by {value - limit:.1f}C"
                )
            elif parameter == "humidity_low":
                detail = (
                    f"relative-humidity reading {value:.1f}% below the declared minimum "
                    f"{limit:.1f}% by {limit - value:.1f} percentage points"
                )
            elif parameter == "humidity_high":
                detail = (
                    f"relative-humidity reading {value:.1f}% above the declared maximum "
                    f"{limit:.1f}% by {value - limit:.1f} percentage points"
                )
            else:
                detail = f"{parameter} reading {value} outside the declared value {limit}"
        else:
            detail = str(v.get("message") or "an unspecified envelope rule activation")
        return (
            f"the MCP synthetic operating-envelope check recorded a {severity} "
            f"excursion ({detail})"
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
            f"piRAG returned a normalized fused-rank signal from "
            f"{rag_context.get('top_doc_id', 'the knowledge base')} "
            f"(normalized policy input: {float(context_features[2]):.2f})"
        )
    elif feature_idx == 3:
        return (
            f"piRAG retrieved a source-labelled guidance note from "
            f"{rag_context.get('top_doc_id', 'the knowledge base')} "
            f"(raw fused-rank strength: {rag_context.get('top_citation_score', 0):.4f})"
        )
    elif feature_idx == 4:
        return (
            "the recent-decision recovery fraction contributed the declared "
            "rebalancing term toward forward supply-chain routes"
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

        sources.append(f"[KB: {doc}, raw fused-rank strength={score:.4f}]{kw_str}")

    if not sources:
        return ""
    return "Retrieved calculation inputs included " + " and ".join(sources[:3]) + "."


def _build_mcp_evidence_str(mcp_results: Dict) -> str:
    """Build a summary string of MCP tool evidence."""
    parts: List[str] = []
    compliance = mcp_results.get("check_compliance")
    if isinstance(compliance, dict):
        status = "within envelope" if compliance.get("compliant") else "outside envelope"
        n_viol = len(compliance.get("violations", []))
        parts.append(f"Synthetic benchmark operating envelope: {status} ({n_viol} excursions)")

    forecast = mcp_results.get("spoilage_forecast")
    if isinstance(forecast, dict):
        parts.append(
            f"Spoilage forecast: rho={forecast.get('forecast_rho', '?')} "
            f"({forecast.get('urgency', '?')})"
        )

    slca_data = mcp_results.get("slca_lookup")
    if isinstance(slca_data, dict):
        parts.append(
            f"Declared social-performance priors: product={slca_data.get('product_type', '?')}"
        )

    return "; ".join(parts) if parts else "No MCP tools invoked"
