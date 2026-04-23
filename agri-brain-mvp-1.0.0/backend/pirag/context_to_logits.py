"""Confidence-weighted, guard-gated, learnable context modifier.

Converts MCP tool results and piRAG retrieval context into a logit modifier
vector of shape (3,), one element per routing action
``[cold_chain, local_redistribute, recovery]``.

Three-layer context integration:

1. **Context feature vector**: MCP and piRAG outputs become structured
   institutional / coordination features psi(context) in R^5 with weight
   matrix THETA_CONTEXT in R^(3x5).
2. **Guard-gated**: returns zeros when the piRAG retrieval-quality guard
   fails (routing-path gate; not the three-guard aggregate used by /ask).
3. **Temporally modulated**: stronger during regime transitions (low
   continuity), weaker during stable periods (high continuity).

Set ``CONTEXT_MODIFIER_SCALE = 0.0`` to disable for ablation studies.

Supply and demand forecast information (point estimates and
uncertainties) no longer enters the context vector. Both signals are
now symmetric state features in ``phi(s)`` at indices 6-8; see
``backend.src.models.action_selection.build_feature_vector``. The
``yield_query`` MCP tool continues to produce the supply forecast, which
is consumed through the state vector (not through psi).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_log = logging.getLogger(__name__)

from .temporal_context import TemporalContextWindow


CONTEXT_MODIFIER_SCALE: float = 1.0
"""Global scale factor. 0.0 disables context injection for ablation."""

# Per-element clamp bounds (widened from +/-0.30 to +/-1.0 for meaningful impact)
_MODIFIER_CLAMP = 1.0


URGENCY_MAP: Dict[str, float] = {
    "low": 0.1,
    "medium": 0.4,
    "high": 0.7,
    "critical": 1.0,
}


# Context weight matrix (3 actions x 5 context features)
# Sign-justified alongside THETA in the paper.
THETA_CONTEXT: np.ndarray = np.array([
    # psi_0 compl  psi_1 fcst   psi_2 conf   psi_3 reg    psi_4 rec_sat
    [ -0.80,       -0.60,       -0.15,       -0.30,       +0.25],   # ColdChain
    [ +0.50,       +0.40,       +0.20,       +0.25,       +0.10],   # LocalRedistribute
    [ +0.30,       +0.20,       -0.05,       +0.05,       -0.35],   # Recovery
], dtype=np.float64)
"""Context weight matrix mapping 5 institutional context features to 3
action logit adjustments.

Sign justifications:

- Compliance severity (psi_0):    violations disfavor cold chain (-0.80),
                                  favor redistribution (+0.50).
- Forecast urgency (psi_1):       high predicted spoilage disfavors cold
                                  chain (-0.60).
- Retrieval confidence (psi_2):   high-confidence retrieval slightly shifts
                                  toward redistribution (+0.20).
- Regulatory pressure (psi_3):    regulatory docs shift away from cold
                                  chain (-0.30).
- Recovery saturation (psi_4):    heavy recent recovery disfavors further
                                  recovery (-0.35).
"""


# Feature masks for ablation modes (5-element; supply and demand
# forecast signals now live in the state vector phi, not here).
_MCP_FEATURE_MASK   = np.array([1.0, 1.0, 0.0, 0.0, 1.0])  # psi_0, psi_1, psi_4
_PIRAG_FEATURE_MASK = np.array([0.0, 0.0, 1.0, 1.0, 0.0])  # psi_2, psi_3


def extract_context_features(
    mcp_results: Dict[str, Any],
    rag_context: Dict[str, Any],
    obs: Any,
) -> np.ndarray:
    """Extract a 5D context feature vector from MCP and piRAG outputs.

    Returns np.ndarray of shape (5,) with values in [0, 1].

    Features:
        psi_0: Compliance severity (0.0 compliant, 0.5 warning, 1.0 critical)
        psi_1: Forecast urgency (mapped from spoilage_forecast urgency level)
        psi_2: Retrieval confidence (normalized top citation score)
        psi_3: Regulatory pressure (1.0 if top doc is regulatory with score > 0.4)
        psi_4: Recovery saturation (fraction of recent decisions that were recovery)
    """
    psi = np.zeros(5, dtype=np.float64)

    if mcp_results is None:
        mcp_results = {}
    if rag_context is None:
        rag_context = {}

    compliance = mcp_results.get("check_compliance") or {}
    if not compliance.get("compliant", True):
        violations = compliance.get("violations", [])
        if any(v.get("severity") == "critical" for v in violations):
            psi[0] = 1.0
        elif violations:
            psi[0] = 0.5

    forecast = mcp_results.get("spoilage_forecast") or {}
    urgency = forecast.get("urgency", "")
    psi[1] = URGENCY_MAP.get(urgency, 0.0)

    top_score = rag_context.get("top_citation_score", 0.0)
    psi[2] = min(top_score / 0.8, 1.0)

    top_doc = rag_context.get("top_doc_id", "")
    top_doc_score = rag_context.get("top_citation_score", 0.0)
    if top_doc_score > 0.4 and any(kw in top_doc.lower() for kw in ("regulatory", "fda", "emergency")):
        psi[3] = 1.0

    chain = mcp_results.get("chain_query", [])
    if isinstance(chain, list) and len(chain) >= 3:
        recovery_frac = sum(1 for d in chain if d.get("action") == "recovery") / len(chain)
        psi[4] = recovery_frac

    return psi


def compute_context_modifier(
    mcp_results: Dict[str, Any],
    rag_context: Dict[str, Any],
    obs: Any,
    temporal_window: Optional[TemporalContextWindow] = None,
    rule_weights: Optional[np.ndarray] = None,
    theta_override: Optional[np.ndarray] = None,
    slca_amp_override: Optional[float] = None,
    temporal_params_override: Optional[Tuple[float, float]] = None,
    context_mode: str = "full",
) -> np.ndarray:
    """Compute context logit adjustment via THETA_CONTEXT @ psi(context).

    ``context_mode`` accepts:
        - "full"       : all 5 features active.
        - "mcp_only"   : only MCP-derived features (psi_0, psi_1, psi_4).
        - "pirag_only" : only piRAG-derived features (psi_2, psi_3).

    Returns
    -------
    Modifier vector of shape (3,), clamped to [-1.0, +1.0] per element.
    Returns zeros when guards fail or inputs are empty.
    """
    if CONTEXT_MODIFIER_SCALE == 0.0:
        return np.zeros(3)

    # Guard gate: routing-path retrieval-quality flag (set by
    # context_builder.retrieve_role_context). Not the three-guard aggregate.
    if not rag_context.get("guards_passed", True):
        return np.zeros(3)

    physics_score = float(rag_context.get("physics_consistency_score", 1.0))
    physics_gate_enabled = False
    if hasattr(obs, "raw") and isinstance(obs.raw, dict):
        flags = obs.raw.get("policy_flags", {})
        physics_gate_enabled = bool(flags.get("enable_physics_consistency_gate", False))

    if physics_gate_enabled and physics_score < 0.03:
        return np.zeros(3)

    if not mcp_results and not rag_context:
        return np.zeros(3)

    psi = extract_context_features(mcp_results, rag_context, obs)

    if context_mode == "mcp_only":
        psi = psi * _MCP_FEATURE_MASK
    elif context_mode == "pirag_only":
        psi = psi * _PIRAG_FEATURE_MASK

    theta = theta_override if theta_override is not None else THETA_CONTEXT

    if theta_override is None and rule_weights is not None and len(rule_weights) >= 5:
        feature_weights = np.array(rule_weights[:5], dtype=np.float64)
        psi = psi * feature_weights

    modifier = theta @ psi

    if temporal_window is not None:
        try:
            t_base, t_scale = temporal_params_override or (1.3, 0.6)
            continuity = temporal_window.context_continuity_score(
                getattr(obs, "hour", 0.0)
            )
            temporal_mod = t_base - t_scale * continuity
            modifier *= temporal_mod
        except Exception as _exc:
            _log.debug("temporal continuity modulation skipped: %s", _exc)

    modifier *= CONTEXT_MODIFIER_SCALE
    if physics_gate_enabled:
        modifier *= max(0.0, min(1.0, physics_score / 0.15))
    modifier = np.clip(modifier, -_MODIFIER_CLAMP, _MODIFIER_CLAMP)

    return modifier


# Feature names for the learner / evaluator (5 entries, matches psi shape).
MODIFIER_RULES: List[Dict[str, Any]] = [
    {"name": "compliance_severity",  "feature_idx": 0},
    {"name": "forecast_urgency",     "feature_idx": 1},
    {"name": "retrieval_confidence", "feature_idx": 2},
    {"name": "regulatory_pressure",  "feature_idx": 3},
    {"name": "recovery_saturation",  "feature_idx": 4},
]
