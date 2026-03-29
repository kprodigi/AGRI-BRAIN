"""Confidence-weighted, guard-gated, learnable context modifier.

Converts MCP tool results and piRAG retrieval context into a logit
modifier vector of shape (3,) — one element per routing action
[cold_chain, local_redistribute, recovery].

Three design principles:
1. **Confidence-weighted**: modifier magnitude scales with retrieval score.
2. **Guard-gated**: returns zeros when piRAG guards fail.
3. **Temporally modulated**: stronger during regime transitions (low
   continuity), weaker during stable periods (high continuity).

Set ``CONTEXT_MODIFIER_SCALE = 0.0`` to disable for ablation studies.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .temporal_context import TemporalContextWindow


CONTEXT_MODIFIER_SCALE: float = 1.0
"""Global scale factor. 0.0 disables context injection for ablation."""

# Per-element clamp bounds
_MODIFIER_CLAMP = 0.30

# Action indices: 0=cold_chain, 1=local_redistribute, 2=recovery
MODIFIER_RULES: List[Dict[str, Any]] = [
    {
        "name": "compliance_critical",
        "condition": lambda mcp, rag, obs: (
            any(v.get("severity") == "critical"
                for v in mcp.get("check_compliance", {}).get("violations", []))
        ),
        "base_modifier": np.array([-0.25, +0.15, +0.10]),
        "confidence_source": "mcp",
        "weight": 1.0,
    },
    {
        "name": "compliance_warning",
        "condition": lambda mcp, rag, obs: (
            any(v.get("severity") == "warning"
                for v in mcp.get("check_compliance", {}).get("violations", []))
            and not any(v.get("severity") == "critical"
                        for v in mcp.get("check_compliance", {}).get("violations", []))
        ),
        "base_modifier": np.array([-0.10, +0.05, +0.05]),
        "confidence_source": "mcp",
        "weight": 1.0,
    },
    {
        "name": "slca_redistribution_preference",
        "condition": lambda mcp, rag, obs: (
            mcp.get("slca_lookup", {}).get("base_scores", {}).get("local_redistribute", {}).get("R", 0) > 0.80
        ),
        "base_modifier": np.array([-0.05, +0.10, -0.05]),
        "confidence_source": "mcp",
        "weight": 1.0,
    },
    {
        "name": "recovery_saturation",
        "condition": lambda mcp, rag, obs: (
            mcp.get("recovery_capacity_check", {}).get("remaining_broadcasts", 80) < 20
            if isinstance(mcp.get("recovery_capacity_check"), dict) else False
        ),
        "base_modifier": np.array([+0.05, +0.05, -0.10]),
        "confidence_source": "mcp",
        "weight": 1.0,
    },
    {
        "name": "spoilage_forecast_critical",
        "condition": lambda mcp, rag, obs: (
            mcp.get("spoilage_forecast", {}).get("urgency") in ("critical", "high")
        ),
        "base_modifier": np.array([-0.15, +0.10, +0.05]),
        "confidence_source": "mcp",
        "weight": 1.0,
    },
    {
        "name": "regulatory_pressure",
        "condition": lambda mcp, rag, obs: bool(rag.get("regulatory_guidance")),
        "base_modifier": np.array([-0.05, +0.05, 0.00]),
        "confidence_source": "retrieval",
        "weight": 1.0,
    },
    {
        "name": "waste_hierarchy_guidance",
        "condition": lambda mcp, rag, obs: bool(rag.get("waste_hierarchy_guidance")),
        "base_modifier": np.array([-0.05, +0.05, +0.05]),
        "confidence_source": "retrieval",
        "weight": 1.0,
    },
    {
        "name": "emergency_sop_active",
        "condition": lambda mcp, rag, obs: bool(rag.get("sop_guidance")) and obs.rho > 0.35,
        "base_modifier": np.array([-0.10, +0.10, 0.00]),
        "confidence_source": "retrieval",
        "weight": 1.0,
    },
]


def compute_context_modifier(
    mcp_results: Dict[str, Any],
    rag_context: Dict[str, Any],
    obs: Any,
    temporal_window: Optional[TemporalContextWindow] = None,
    rule_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the context modifier vector for softmax logits.

    Parameters
    ----------
    mcp_results : results from MCP tool dispatch.
    rag_context : results from piRAG retrieval.
    obs : current Observation.
    temporal_window : optional temporal context for modulation.
    rule_weights : optional per-rule weights from online learner.

    Returns
    -------
    Modifier vector of shape (3,), clamped to [-0.30, +0.30] per element.
    Returns zeros when guards fail or inputs are empty.
    """
    modifier = np.zeros(3, dtype=np.float64)

    # Guard gate: if piRAG guards failed, return zeros
    if not rag_context.get("guards_passed", True):
        return modifier

    # If both inputs are empty/missing, return zeros
    if not mcp_results and not rag_context:
        return modifier

    # Retrieval confidence (from piRAG top citation score)
    retrieval_confidence = min(rag_context.get("top_citation_score", 0.0), 1.0)

    # MCP confidence is binary: 1.0 if tools returned results, 0.0 otherwise
    mcp_confidence = 1.0 if any(
        k for k in mcp_results
        if not k.startswith("_") and mcp_results[k] is not None
    ) else 0.0

    rules_fired: List[int] = []

    for idx, rule in enumerate(MODIFIER_RULES):
        try:
            if not rule["condition"](mcp_results, rag_context, obs):
                continue
        except Exception:
            continue

        rules_fired.append(idx)

        # Select confidence source
        if rule["confidence_source"] == "mcp":
            confidence = mcp_confidence
        else:
            confidence = retrieval_confidence

        # Apply rule weight (from online learner)
        weight = 1.0
        if rule_weights is not None and idx < len(rule_weights):
            weight = float(rule_weights[idx])
        else:
            weight = rule.get("weight", 1.0)

        modifier += rule["base_modifier"] * confidence * weight

    # Temporal modulation: low continuity → stronger, high continuity → weaker
    if temporal_window is not None:
        continuity = temporal_window.context_continuity_score(obs.hour)
        # Scale: continuity near 1.0 → 0.7, near 0.0 → 1.3
        temporal_scale = 1.3 - 0.6 * continuity
        modifier *= temporal_scale

    # Apply global scale
    modifier *= CONTEXT_MODIFIER_SCALE

    # Clamp per element
    modifier = np.clip(modifier, -_MODIFIER_CLAMP, _MODIFIER_CLAMP)

    return modifier
