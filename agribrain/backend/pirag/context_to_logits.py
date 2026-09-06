"""Guard-gated, learnable context modifier using normalized retrieval rank strength.

Converts MCP tool results and piR retrieval context into a logit modifier
vector of shape (3,), one element per routing action
``[cold_chain, local_redistribute, recovery]``.

Three-layer context integration:

1. **Context feature vector**: MCP and piR outputs become structured
   institutional / coordination features psi(context) in R^5 with weight
   matrix THETA_CONTEXT in R^(3x5).
2. **Channel-separated guards**: the author-declared RRF floor applies only
   to the piR-derived columns. The optional hard physics-consistency gate
   also affects only piR when explicitly enabled, but it is disabled in the
   locked confirmatory run. A failed retrieval must not suppress separately
   computed MCP operating-envelope, modeled-forecast, or history features.
3. **Retrieval-only temporal modulation**: piR evidence is stronger during
   regime transitions (low continuity) and weaker during stable periods (high
   continuity); MCP evidence is not a function of retrieval persistence.

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

from .temporal_context import TemporalContextWindow
from .strict_validation import handle_unexpected_failure

_log = logging.getLogger(__name__)


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


# Context weight matrix (3 actions x 5 context features). Signs and magnitudes
# are author-specified design priors, not estimates from field observations.
THETA_CONTEXT: np.ndarray = np.array([
    # psi_0 envelope psi_1 forecast psi_2 rank psi_3 guidance psi_4 rec_sat
    [ -0.40,       -0.30,       -0.10,       -0.15,       +0.12],   # ColdChain
    [ +0.30,       +0.25,       +0.15,       +0.18,       +0.08],   # LocalRedistribute
    [ +0.15,       +0.10,       -0.05,       +0.05,       -0.20],   # Recovery
], dtype=np.float64)
"""Context weight matrix mapping 5 institutional context features to 3
action logit adjustments.

Declared sign rationale:

- Operating-envelope severity (psi_0): excursions disfavor cold chain (-0.40),
                                  favor redistribution (+0.30).
- Forecast urgency (psi_1):       high predicted spoilage disfavors cold
                                  chain (-0.30).
- Retrieval-score signal (psi_2): a larger normalized fused-rank score shifts
                                  toward redistribution (+0.15); it is not a
                                  calibrated confidence.
- Source-labelled guidance flag (psi_3): a retrieved guidance-note filename
                                  pattern shifts away from cold chain (-0.15).
- Recovery saturation (psi_4):    heavy recent recovery disfavors further
                                  recovery (-0.20).

The separate 100-point structural sensitivity varies the context-prior scale.
The volatility-regime term belongs to the base policy rather than this context
matrix and is the action-specific vector
``b_tau = [0.25, 0.05, -0.25]`` multiplied by the binary regime flag; its
three coordinates are swept independently. Because the same synthetic scenario
family was used during development, none of these weights may be presented as
independently calibrated or externally validated.
"""


# Feature partitions for the separated MCP / piR formulation.  Retaining one
# learned 3x5 matrix is algebraically equivalent to two matrices whose columns
# are ``MCP_FEATURE_INDICES`` and ``PIR_FEATURE_INDICES`` respectively.
MCP_FEATURE_INDICES: Tuple[int, ...] = (0, 1, 4)
PIR_FEATURE_INDICES: Tuple[int, ...] = (2, 3)

# Feature masks for ablation modes (5-element; supply and demand forecast
# signals now live in the state vector phi, not here).
_MCP_FEATURE_MASK = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
_PIR_FEATURE_MASK = np.array([0.0, 0.0, 1.0, 1.0, 0.0])


def apply_context_mode_feature_mask(
    psi: np.ndarray,
    context_mode: str = "full",
) -> np.ndarray:
    """Return the feature vector actually presented to ``THETA_CONTEXT``.

    The coordinator records this effective vector in the decision ledger so
    explanations and learner updates refer to the same inputs as routing.  A
    copy is always returned; callers may therefore retain the unmasked feature
    extraction separately when needed for diagnostics.
    """
    effective = np.asarray(psi, dtype=np.float64).copy()
    if effective.shape != (5,):
        raise ValueError(f"context feature vector must have shape (5,), got {effective.shape}")
    if context_mode == "mcp_only":
        effective *= _MCP_FEATURE_MASK
    elif context_mode == "pirag_only":
        effective *= _PIR_FEATURE_MASK
    return effective


def extract_context_features(
    mcp_results: Dict[str, Any],
    rag_context: Dict[str, Any],
    obs: Any,
) -> np.ndarray:
    """Extract a 5D context feature vector from MCP and piR outputs.

    Returns np.ndarray of shape (5,) with values in [0, 1].

    Features:
        psi_0: Operating-envelope severity (0.0 within, 0.5 warning, 1.0 critical)
        psi_1: Forecast urgency (mapped from spoilage_forecast urgency level)
        psi_2: Normalized fused-rank retrieval-score signal (not confidence)
        psi_3: Source-labelled guidance flag (1.0 if a declared guidance-document
               filename pattern clears the author-declared RRF floor)
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

    # psi_2: Normalized fused-rank score rescaled for the RRF retriever.
    # The hybrid retriever now returns Reciprocal Rank Fusion scores
    # bounded by 1/(K+1) per list (~0.0164 for K=60). The previous
    # divisor of 0.8 was calibrated for the deprecated min-max merge
    # whose top score saturated around 1.0; with RRF that divisor would
    # cap psi_2 at ~0.02, killing the feature. The new normalisation
    # uses the maximum theoretical RRF score (both lists rank the doc
    # at position 1 -> 2/(K+1)) as the [0,1] ceiling so a top hit on
    # both retrievers yields psi_2 ≈ 1.0, matching the pre-RRF semantics.
    # Use the raw RRF strength, never the lexical/Arrhenius rerank score.
    # ``top_citation_score`` is the legacy alias retained for callers that
    # predate the explicit split.
    top_score = rag_context.get(
        "top_fused_score", rag_context.get("top_citation_score", 0.0)
    )
    try:
        from .pyrag.hybrid_retriever import HybridRetriever as _HR
        _rrf_k = float(_HR.RRF_K)
    except Exception as _exc:
        handle_unexpected_failure(
            "RRF normalization-constant lookup", _exc, _log,
        )
        _rrf_k = 60.0
    _rrf_top_max = 2.0 / (_rrf_k + 1.0)
    psi[2] = float(min(top_score / max(_rrf_top_max, 1e-9), 1.0))

    # psi_3: Source-labelled guidance flag. Old code used
    # `top_doc_score > 0.4`
    # scaled for [0,1] min-max scores. With RRF max ≈ 2/(K+1) ≈
    # 0.0328, the 0.4 threshold was unreachable. The new threshold is
    # the retrieval-guard floor itself: the doc must clear the guard
    # (already enforced upstream when the modifier is computed) AND
    # match a declared guidance-document filename pattern. Using the floor directly
    # rather than a multiple keeps the gate consistent with the
    # declared floor (RRF top scores live in a narrow band so a
    # multiplicative buffer would push the threshold outside the
    # achievable range).
    top_doc = rag_context.get("top_doc_id", "")
    top_doc_score = rag_context.get(
        "top_fused_score", rag_context.get("top_citation_score", 0.0)
    )
    try:
        from .guards.retrieval_guard import MIN_TOP_CITATION_SCORE as _RG_MIN
    except Exception as _exc:
        handle_unexpected_failure(
            "retrieval-guard threshold lookup", _exc, _log,
        )
        _RG_MIN = 0.0246
    if top_doc_score > _RG_MIN and any(
        kw in top_doc.lower() for kw in ("regulatory", "fda", "emergency")
    ):
        psi[3] = 1.0

    chain = mcp_results.get("chain_query", {})
    # chain_query returns a structured dict {_status, records}; older code paths
    # may still pass a bare list, so accept both shapes here.
    chain_records = chain.get("records", []) if isinstance(chain, dict) else chain
    if isinstance(chain_records, list) and len(chain_records) >= 3:
        recovery_frac = sum(1 for d in chain_records if d.get("action") == "recovery") / len(chain_records)
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
    retrieval_kind: str = "pirag",
    trace_out: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Compute the channel-separated context logit adjustment.

    The production mapping is

    ``clip(S * (Theta_MCP @ psi_MCP
                 + g_retrieval * g_physics * tau *
                   Theta_RAG @ psi_RAG), -1, +1)``.

    The RRF-floor, physics-consistency, and temporal-continuity terms
    therefore regulate only retrieved evidence.  They never suppress the MCP
    columns.  ``trace_out['modifier_theta_jacobian']`` records the exact
    derivative of the returned modifier with respect to the effective 3x5
    context matrix, including the chosen zero subgradient for clipped rows.

    ``context_mode`` accepts:
        - "full"       : all 5 features active.
        - "mcp_only"   : only MCP-derived features (psi_0, psi_1, psi_4).
        - "pirag_only" : only piR-derived features (psi_2, psi_3).

    ``retrieval_kind='standard'`` retains the author-declared RRF-floor gate but
    removes piR's physics-consistency and temporal-continuity multipliers.

    Returns
    -------
    Modifier vector of shape (3,), clamped to [-1.0, +1.0] per element.
    Failed retrieval guards zero only the piR component.  The whole vector is
    zero only when the global scale is zero or both channels are empty.
    """
    if retrieval_kind not in ("pirag", "standard"):
        raise ValueError(
            "retrieval_kind must be 'pirag' or 'standard', got "
            f"{retrieval_kind!r}"
        )
    declared_retrieval_kind = rag_context.get("retrieval_kind")
    if (declared_retrieval_kind is not None
            and declared_retrieval_kind != retrieval_kind):
        raise ValueError(
            "retrieval kind mismatch between modifier argument "
            f"({retrieval_kind!r}) and retrieved context "
            f"({declared_retrieval_kind!r})"
        )

    theta = np.asarray(
        theta_override if theta_override is not None else THETA_CONTEXT,
        dtype=np.float64,
    )
    if theta.shape != (3, 5):
        raise ValueError(f"context matrix must have shape (3, 5), got {theta.shape}")

    temporal_gate_requested = bool(
        retrieval_kind == "pirag"
        and temporal_window is not None
        and context_mode != "mcp_only"
    )
    temporal_gate_applied = False
    temporal_continuity_score: float | None = None
    temporal_base_value: float | None = None
    temporal_decay_value: float | None = None

    if trace_out is not None:
        trace_out.clear()
        trace_out.update({
            "context_mode": context_mode,
            "retrieval_kind": retrieval_kind,
            # Retained audit fields pin the retired over-steering path off.
            "over_steer": False,
            "clip_applied": True,
            "effective_theta": theta.copy(),
            "raw_psi": None,
            "effective_psi": np.zeros(5, dtype=np.float64),
            "linear_feature_contributions": np.zeros((3, 5), dtype=np.float64),
            "channel_scaled_feature_contributions": np.zeros((3, 5), dtype=np.float64),
            "feature_contributions": np.zeros((3, 5), dtype=np.float64),
            "nonfeature_residual": np.zeros(3, dtype=np.float64),
            "mcp_preclip_component": np.zeros(3, dtype=np.float64),
            "pirag_preclip_component": np.zeros(3, dtype=np.float64),
            "retrieval_gate": 0.0,
            "retrieval_blocked_reason": None,
            "temporal_scale": 1.0,
            "temporal_gate_requested": temporal_gate_requested,
            "temporal_gate_applied": False,
            "temporal_continuity_score": None,
            "temporal_base": None,
            "temporal_decay": None,
            "physics_scale": 1.0,
            "rag_total_scale": 0.0,
            "global_scale": float(CONTEXT_MODIFIER_SCALE),
            "preclip_modifier": np.zeros(3, dtype=np.float64),
            "clip_derivative": np.ones(3, dtype=np.float64),
            "modifier_theta_jacobian": np.zeros((3, 5), dtype=np.float64),
            "final_modifier": np.zeros(3, dtype=np.float64),
            "blocked_reason": None,
        })

    def _zero(reason: str) -> np.ndarray:
        if trace_out is not None:
            trace_out["blocked_reason"] = reason
        return np.zeros(3, dtype=np.float64)

    if CONTEXT_MODIFIER_SCALE == 0.0:
        return _zero("global_scale_zero")

    if not mcp_results and not rag_context:
        return _zero("empty_context")

    psi = extract_context_features(mcp_results, rag_context, obs)
    raw_psi = psi.copy()

    # Single-channel ablation modes (mcp_only / pirag_only). The
    # structural gating in coordinator._compute_step_context already produces
    # channel-level differentiation by
    # skipping dispatch_tools or retrieve_role_context entirely, so
    # the feature mask below is the second line of defence rather
    # than the primary differentiator. The mask zeroes out the
    # psi-dimensions that originate from the *gated-out* channel so
    # any residual signal that leaked through (e.g. via the
    # cooperative overlay or shared_context cache) does not
    # contaminate the ablation. An earlier post-audit fix added a
    # small mode-specific logit bias on top of the mask; that bias
    # has been retired (it was an author-knob that engineered the
    # very ablation difference being evaluated) now that the structural
    # gates themselves define the channel contrast. Any empirical separation
    # must be established by the fresh, commit-bound benchmark run.
    psi = apply_context_mode_feature_mask(psi, context_mode)

    if theta_override is None and rule_weights is not None and len(rule_weights) >= 5:
        feature_weights = np.array(rule_weights[:5], dtype=np.float64)
        psi = psi * feature_weights

    # The aggregate retrieval guard combines the RRF floor, dimensional
    # consistency, and feasibility. It is fail-closed for retrieved evidence:
    # a missing flag cannot authorize piR.
    retrieval_gate = float(bool(rag_context.get("guards_passed", False)))
    if retrieval_gate:
        retrieval_blocked_reason = None
    elif "guards_passed" not in rag_context:
        retrieval_blocked_reason = "retrieval_guard_missing"
    else:
        failed = sorted(
            name for name, passed in
            (rag_context.get("guard_breakdown", {}) or {}).items()
            if not bool(passed)
        )
        retrieval_blocked_reason = (
            "retrieval_guard:" + ",".join(failed)
            if failed else "retrieval_guard"
        )

    # Temporal continuity is a property of piR retrieval persistence.  Do not
    # calculate or apply it for an MCP-only arm.
    temporal_mod = 1.0
    if temporal_gate_requested:
        try:
            t_base, t_scale = temporal_params_override or (1.3, 0.6)
            continuity = temporal_window.context_continuity_score(
                getattr(obs, "hour", 0.0)
            )
            temporal_base_value = float(t_base)
            temporal_decay_value = float(t_scale)
            temporal_continuity_score = float(continuity)
            if not all(np.isfinite(value) for value in (
                temporal_base_value,
                temporal_decay_value,
                temporal_continuity_score,
            )) or temporal_base_value < 0.0 or temporal_decay_value < 0.0:
                raise ValueError(
                    "temporal gate base/decay must be finite and non-negative"
                )
            if not 0.0 <= temporal_continuity_score <= 1.0:
                raise ValueError(
                    "temporal continuity score must lie in [0, 1]"
                )
            temporal_mod = (
                temporal_base_value
                - temporal_decay_value * temporal_continuity_score
            )
            if not np.isfinite(temporal_mod) or temporal_mod < 0.0:
                raise ValueError(
                    "temporal gate scale must be finite and non-negative"
                )
            temporal_gate_applied = True
        except Exception as _exc:
            temporal_mod = 1.0
            temporal_gate_applied = False
            temporal_continuity_score = None
            temporal_base_value = None
            temporal_decay_value = None
            handle_unexpected_failure(
                "temporal continuity modulation", _exc, _log,
            )

    # Physics consistency describes retrieved passages, not MCP tool results.
    # Only when the optional gate is enabled does the declared hard threshold
    # zero the retrieved term or apply the bounded soft scale. The locked
    # confirmatory run leaves this gate disabled; the MCP term remains unchanged.
    physics_score = float(rag_context.get("physics_consistency_score", 1.0))
    physics_gate_enabled = False
    if hasattr(obs, "raw") and isinstance(obs.raw, dict):
        flags = obs.raw.get("policy_flags", {})
        physics_gate_enabled = bool(flags.get("enable_physics_consistency_gate", False))

    physics_scale = 1.0
    if retrieval_kind == "pirag" and physics_gate_enabled:
        if physics_score < 0.03:
            physics_scale = 0.0
            if retrieval_blocked_reason is None:
                retrieval_blocked_reason = "physics_gate"
        else:
            physics_scale = max(0.0, min(1.0, physics_score / 0.15))

    rag_total_scale = retrieval_gate * physics_scale * temporal_mod

    # Keep feature-resolved terms through each transform.  ``linear`` is the
    # ungated Theta*psi allocation.  ``preclip_contributions`` is the actual
    # channel-separated forward mapping before the single total clip.
    linear_feature_contributions = theta * psi[np.newaxis, :]
    feature_scales = np.ones(5, dtype=np.float64)
    feature_scales[list(PIR_FEATURE_INDICES)] = rag_total_scale
    jacobian_features = (
        float(CONTEXT_MODIFIER_SCALE) * feature_scales * psi
    )
    preclip_contributions = (
        float(CONTEXT_MODIFIER_SCALE)
        * linear_feature_contributions
        * feature_scales[np.newaxis, :]
    )
    mcp_preclip_component = preclip_contributions[
        :, list(MCP_FEATURE_INDICES)
    ].sum(axis=1)
    pirag_preclip_component = preclip_contributions[
        :, list(PIR_FEATURE_INDICES)
    ].sum(axis=1)
    preclip_modifier = mcp_preclip_component + pirag_preclip_component
    modifier = np.clip(preclip_modifier, -_MODIFIER_CLAMP, _MODIFIER_CLAMP)
    # The clip is nondifferentiable at the boundary. Use the conservative zero
    # subgradient for |u| >= cap, and the exact unit derivative inside.
    clip_derivative = (
        np.abs(preclip_modifier) < _MODIFIER_CLAMP
    ).astype(np.float64)

    modifier_theta_jacobian = (
        clip_derivative[:, np.newaxis] * jacobian_features[np.newaxis, :]
    )

    # After clipping, allocate each action's clipped value proportionally over
    # the pre-clip terms.  This is an attribution that reconstructs the forward
    # modifier; it is intentionally distinct from the derivative above.
    allocated = preclip_contributions.copy()
    residual = np.zeros(3, dtype=np.float64)
    for action_idx in range(3):
        preclip = float(preclip_modifier[action_idx])
        final = float(modifier[action_idx])
        if abs(preclip) > 1e-15:
            allocated[action_idx] *= final / preclip
        elif abs(final) > 1e-15:
            residual[action_idx] = final

    if trace_out is not None:
        trace_out.update({
            "raw_psi": raw_psi,
            "effective_psi": psi.copy(),
            "linear_feature_contributions": linear_feature_contributions,
            "channel_scaled_feature_contributions": preclip_contributions,
            "mcp_preclip_component": mcp_preclip_component,
            "pirag_preclip_component": pirag_preclip_component,
            "retrieval_gate": float(retrieval_gate),
            "retrieval_blocked_reason": retrieval_blocked_reason,
            "temporal_scale": float(temporal_mod),
            "temporal_gate_requested": temporal_gate_requested,
            "temporal_gate_applied": temporal_gate_applied,
            "temporal_continuity_score": temporal_continuity_score,
            "temporal_base": temporal_base_value,
            "temporal_decay": temporal_decay_value,
            "global_scale": float(CONTEXT_MODIFIER_SCALE),
            "physics_scale": float(physics_scale),
            "rag_total_scale": float(rag_total_scale),
            "preclip_modifier": preclip_modifier,
            "clip_derivative": clip_derivative,
            "modifier_theta_jacobian": modifier_theta_jacobian,
            "feature_contributions": allocated,
            "nonfeature_residual": residual,
            "final_modifier": modifier.copy(),
        })

    return modifier


# Compatibility feature keys for the learner/evaluator (5 entries, matches psi
# shape). Two legacy names are retained to avoid breaking stored traces; their
# display labels are the non-probabilistic descriptions below.
MODIFIER_RULES: List[Dict[str, Any]] = [
    {"name": "compliance_severity",  "display_name": "operating_envelope_severity", "feature_idx": 0},
    {"name": "forecast_urgency",     "display_name": "forecast_urgency", "feature_idx": 1},
    {"name": "retrieval_confidence", "display_name": "normalized_fused_rank_strength", "feature_idx": 2},
    {"name": "regulatory_pressure",  "display_name": "source_labelled_guidance_flag", "feature_idx": 3},
    {"name": "recovery_saturation",  "display_name": "recovery_saturation", "feature_idx": 4},
]
