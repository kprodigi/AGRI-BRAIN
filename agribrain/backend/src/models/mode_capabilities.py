"""Single source of truth for benchmark-mode behavior.

The simulator, coordinator, and online learners previously maintained partly
overlapping mode-name sets.  That made it possible for a mode to be labelled
"learned" while its learner was never constructed, or to use a learnable logit
term without routing gradients to that term.  This module records those
capabilities once and exposes small derived sets for compatibility.

``context_kind`` describes the external tool/retrieval modifier only.  The
``no_context`` arm deliberately keeps dormant context infrastructure for
structural parity, but bypasses tool/retrieval injection and peer-message
influence at decision time. This revised definition is not compatible with
the historical d3286ae no_context results, which retained peer messages.

New AGRI-BRAIN runs use standard RAG. The archived d3286ae ``agribrain`` key
used piRAG; its ``agribrain_standard_rag`` key is the standard-RAG comparator.
That explicit key remains an executable compatibility alias for the revised
default, not a distinct ablation. Other retrieval-enabled diagnostics use
standard RAG too, preserving their one-factor differences from the default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ContextKind = Literal["full", "mcp_only", "pirag_only"]
RetrievalKind = Literal["pirag", "standard"]
FixedRule = Literal[
    "always_redistribute", "always_recovery", "risk_threshold",
    "capacity_informed", "greedy_informed",
]
GuardKind = Literal["none", "eligibility", "reservation"]
ContextInterface = Literal["bounded", "feature"]
ContextWeightInit = Literal["prior", "zero"]


@dataclass(frozen=True)
class ModeCapabilities:
    """Declared behavioral capabilities of one operating mode."""

    context_kind: ContextKind | None
    context_infrastructure: bool
    context_matrix_learning: bool
    policy_delta_learning: bool
    reward_shaping_learning: bool
    agribrain_logits: bool
    spoilage_residual: bool = True
    peer_messages: bool = True
    retrieval_kind: RetrievalKind = "standard"
    sign_constrained_learning: bool = True
    frozen_learners: bool = False
    fixed_rule: FixedRule | None = None
    # The five fields below take effect under the revised context design only.
    # ``guard``: applied before sampling. ``eligibility`` removes a route when
    # the product is outside the acceptance band stated in the retrieved
    # acceptance document, or when the destination reports no room at all.
    # ``reservation`` also requires reported room for the dispatch. A check
    # whose information the mode never requests is not applied: a mode
    # without tools has no capacity check, a mode without retrieval has no
    # eligibility check.
    guard: GuardKind = "none"
    # ``context_adjustment=False`` keeps tools and retrieval running for the
    # guard but applies no score adjustment.
    context_adjustment: bool = True
    # ``feature`` removes the output bound, the sign constraint and the change
    # bound, i.e. the context enters as ordinary input features.
    context_interface: ContextInterface = "bounded"
    context_weight_init: ContextWeightInit = "prior"
    # A constant vector adapted over the same episodes replaces the context.
    constant_adjustment: bool = False

    @property
    def learned(self) -> bool:
        """Whether at least one policy parameter is adapted online."""

        return bool(
            self.context_matrix_learning
            or self.policy_delta_learning
            or self.reward_shaping_learning
        )

    @property
    def episode_count(self) -> int:
        """Three adaptation episodes plus one frozen evaluation episode."""

        return 4 if self.learned else 1

    @property
    def adaptation_episode_count(self) -> int:
        """Number of within-block episodes allowed to update learners."""

        return 3 if self.learned else 0


def _caps(
    *,
    context_kind: ContextKind | None = None,
    context_infrastructure: bool | None = None,
    context_matrix_learning: bool = False,
    policy_delta_learning: bool = False,
    reward_shaping_learning: bool = False,
    agribrain_logits: bool = False,
    spoilage_residual: bool = True,
    peer_messages: bool = True,
    retrieval_kind: RetrievalKind = "standard",
    sign_constrained_learning: bool = True,
    frozen_learners: bool = False,
    fixed_rule: FixedRule | None = None,
    guard: GuardKind = "none",
    context_adjustment: bool = True,
    context_interface: ContextInterface = "bounded",
    context_weight_init: ContextWeightInit = "prior",
    constant_adjustment: bool = False,
) -> ModeCapabilities:
    if context_infrastructure is None:
        context_infrastructure = context_kind is not None
    return ModeCapabilities(
        context_kind=context_kind,
        context_infrastructure=bool(context_infrastructure),
        context_matrix_learning=bool(context_matrix_learning),
        policy_delta_learning=bool(policy_delta_learning),
        reward_shaping_learning=bool(reward_shaping_learning),
        agribrain_logits=bool(agribrain_logits),
        spoilage_residual=bool(spoilage_residual),
        peer_messages=bool(peer_messages),
        retrieval_kind=retrieval_kind,
        sign_constrained_learning=bool(sign_constrained_learning),
        frozen_learners=bool(frozen_learners),
        fixed_rule=fixed_rule,
        guard=guard,
        context_adjustment=bool(context_adjustment),
        context_interface=context_interface,
        context_weight_init=context_weight_init,
        constant_adjustment=bool(constant_adjustment),
    )


# The insertion order is the canonical public mode order used by the policy.
MODE_CAPABILITIES: dict[str, ModeCapabilities] = {
    "static": _caps(),
    # Hybrid RL uses the common base-policy correction but no external context
    # or social-proxy logit vectors.
    "hybrid_rl": _caps(policy_delta_learning=True),
    # One-factor PINN ablation: identical to AGRI-BRAIN in context, policy,
    # social-proxy shaping and peer communication; only the frozen synthetic
    # residual is removed. Common-random-number streams remain paired.
    "no_pinn": _caps(
        context_kind="full", context_matrix_learning=True,
        policy_delta_learning=True, reward_shaping_learning=True,
        agribrain_logits=True, spoilage_residual=False, guard="reservation",
    ),
    "no_slca": _caps(
        context_kind="full", context_matrix_learning=True,
        policy_delta_learning=True, agribrain_logits=True, guard="reservation",
    ),
    # Keep the infrastructure available for symmetry/diagnostics, while the
    # missing context_kind makes the external modifier structurally absent.
    "no_context": _caps(
        context_infrastructure=True, policy_delta_learning=True,
        reward_shaping_learning=True, agribrain_logits=True, peer_messages=False,
    ),
    "mcp_only": _caps(
        context_kind="mcp_only", context_matrix_learning=True,
        policy_delta_learning=True, reward_shaping_learning=True,
        agribrain_logits=True, guard="reservation",
    ),
    "pirag_only": _caps(
        context_kind="pirag_only", context_matrix_learning=True,
        policy_delta_learning=True, reward_shaping_learning=True,
        agribrain_logits=True, guard="reservation",
    ),
    "agribrain": _caps(
        context_kind="full", context_matrix_learning=True,
        policy_delta_learning=True, reward_shaping_learning=True,
        agribrain_logits=True, guard="reservation",
    ),
    # Compatibility alias: identical to the revised standard-RAG default.
    # Keep the old key so archived standard-RAG evidence remains identifiable.
    "agribrain_standard_rag": _caps(
        context_kind="full", context_matrix_learning=True,
        policy_delta_learning=True, reward_shaping_learning=True,
        agribrain_logits=True, retrieval_kind="standard", guard="reservation",
    ),
    # Secondary one-factor learned ablations share the revised retrieval kind,
    # priors, update budget, and final evaluation stream of the full system.
    "agribrain_no_peer": _caps(
        context_kind="full", context_matrix_learning=True,
        policy_delta_learning=True, reward_shaping_learning=True,
        agribrain_logits=True, peer_messages=False, guard="reservation",
    ),
    "agribrain_sign_unconstrained": _caps(
        context_kind="full", context_matrix_learning=True,
        policy_delta_learning=True, reward_shaping_learning=True,
        agribrain_logits=True, sign_constrained_learning=False, guard="reservation",
    ),
    # Comparison arms of the revised design. Each differs from a named
    # reference in the stated respect only.
    # Score adjustment without the guard and without peers.
    "adjustment_only": _caps(
        context_kind="full", context_matrix_learning=True,
        policy_delta_learning=True, reward_shaping_learning=True,
        agribrain_logits=True, peer_messages=False,
    ),
    # Eligibility guard without the score adjustment and without peers.
    "eligibility_only": _caps(
        context_kind="full", policy_delta_learning=True,
        reward_shaping_learning=True, agribrain_logits=True,
        peer_messages=False, guard="eligibility", context_adjustment=False,
    ),
    # Reservation guard without the score adjustment and without peers.
    "reservation_only": _caps(
        context_kind="full", policy_delta_learning=True,
        reward_shaping_learning=True, agribrain_logits=True,
        peer_messages=False, guard="reservation", context_adjustment=False,
    ),
    # Score adjustment with the eligibility guard, without and with peers.
    # They differ from ``agribrain_no_peer`` and ``agribrain`` only in the
    # guard: capacity is left to the score adjustment.
    "agribrain_eligibility_no_peer": _caps(
        context_kind="full", context_matrix_learning=True,
        policy_delta_learning=True, reward_shaping_learning=True,
        agribrain_logits=True, peer_messages=False, guard="eligibility",
    ),
    "agribrain_eligibility": _caps(
        context_kind="full", context_matrix_learning=True,
        policy_delta_learning=True, reward_shaping_learning=True,
        agribrain_logits=True, guard="eligibility",
    ),
    # Full system without the guard.
    "agribrain_no_guard": _caps(
        context_kind="full", context_matrix_learning=True,
        policy_delta_learning=True, reward_shaping_learning=True,
        agribrain_logits=True,
    ),
    # Full system with the context weights held at their prescribed values.
    "agribrain_fixed_context": _caps(
        context_kind="full", policy_delta_learning=True,
        reward_shaping_learning=True, agribrain_logits=True, guard="reservation",
    ),
    # The three modes below are compared with ``agribrain_eligibility_no_peer``
    # and differ from it in the stated respect only.
    # The same context as input features of an unconstrained linear term.
    "feature_inputs_prior": _caps(
        context_kind="full", context_matrix_learning=True,
        policy_delta_learning=True, reward_shaping_learning=True,
        agribrain_logits=True, peer_messages=False, guard="eligibility",
        context_interface="feature",
    ),
    "feature_inputs_zero": _caps(
        context_kind="full", context_matrix_learning=True,
        policy_delta_learning=True, reward_shaping_learning=True,
        agribrain_logits=True, peer_messages=False, guard="eligibility",
        context_interface="feature", context_weight_init="zero",
    ),
    # A learned constant adjustment in place of the context.
    "constant_adjustment": _caps(
        context_kind="full", context_matrix_learning=True,
        policy_delta_learning=True, reward_shaping_learning=True,
        agribrain_logits=True, peer_messages=False, guard="eligibility",
        context_weight_init="zero", constant_adjustment=True,
    ),
    # Rule references. Like ``static`` they are fixed decision rules: no
    # learning, no context modifier, no peer influence and no random draw.
    # ``risk_threshold`` uses the estimated risk and the published acceptance
    # criteria only. ``capacity_informed`` is a priority rule with reservation
    # that also uses the capacity reports. ``greedy_informed`` chooses the
    # route with the highest predicted step value and allows partial
    # acceptance; it knows the outcome equations. The rules read the same
    # criteria and reports as the guard of the learned modes.
    "rule_always_redistribute": _caps(
        fixed_rule="always_redistribute", peer_messages=False,
    ),
    "rule_always_recovery": _caps(
        fixed_rule="always_recovery", peer_messages=False,
    ),
    "rule_risk_threshold": _caps(
        fixed_rule="risk_threshold", peer_messages=False,
    ),
    "rule_capacity_informed": _caps(
        fixed_rule="capacity_informed", peer_messages=False,
    ),
    "rule_greedy_informed": _caps(
        fixed_rule="greedy_informed", peer_messages=False,
    ),
}


PRIMARY_MODES: tuple[str, ...] = (
    "static", "hybrid_rl", "no_pinn", "no_slca", "no_context",
    "mcp_only", "pirag_only", "agribrain",
)
SECONDARY_ABLATION_MODES: tuple[str, ...] = (
    "agribrain_standard_rag",
    "agribrain_no_peer",
    "agribrain_sign_unconstrained",
)
PUBLICATION_BENCHMARK_MODES: tuple[str, ...] = (
    *PRIMARY_MODES, *SECONDARY_ABLATION_MODES,
)
REVISION_COMPARISON_MODES: tuple[str, ...] = (
    "adjustment_only",
    "eligibility_only",
    "reservation_only",
    "agribrain_eligibility_no_peer",
    "agribrain_eligibility",
    "agribrain_no_guard",
    "agribrain_fixed_context",
    "feature_inputs_prior",
    "feature_inputs_zero",
    "constant_adjustment",
)
RULE_REFERENCE_MODES: tuple[str, ...] = (
    "rule_always_redistribute",
    "rule_always_recovery",
    "rule_risk_threshold",
    "rule_capacity_informed",
    "rule_greedy_informed",
)

VALID_MODES: tuple[str, ...] = tuple(MODE_CAPABILITIES)
DECISION_OWNER_ROLES: tuple[str, ...] = (
    "farm", "processor", "distributor", "recovery",
)
CONTEXT_INFRASTRUCTURE_MODES = frozenset(
    mode for mode, caps in MODE_CAPABILITIES.items()
    if caps.context_infrastructure
)
CONTEXT_MODE_MAP: dict[str, ContextKind] = {
    mode: caps.context_kind
    for mode, caps in MODE_CAPABILITIES.items()
    if caps.context_kind is not None
}
POLICY_DELTA_LEARNING_MODES = frozenset(
    mode for mode, caps in MODE_CAPABILITIES.items()
    if caps.policy_delta_learning
)
REWARD_SHAPING_LEARNING_MODES = frozenset(
    mode for mode, caps in MODE_CAPABILITIES.items()
    if caps.reward_shaping_learning
)
CONTEXT_MATRIX_LEARNING_MODES = frozenset(
    mode for mode, caps in MODE_CAPABILITIES.items()
    if caps.context_matrix_learning
)
AGRIBRAIN_LOGIT_MODES = frozenset(
    mode for mode, caps in MODE_CAPABILITIES.items()
    if caps.agribrain_logits
)
PINN_RESIDUAL_MODES = frozenset(
    mode for mode, caps in MODE_CAPABILITIES.items()
    if caps.spoilage_residual
)
MULTI_EPISODE_MODES: dict[str, int] = {
    mode: caps.episode_count
    for mode, caps in MODE_CAPABILITIES.items()
    if caps.episode_count > 1
}


def capabilities_for(mode: str) -> ModeCapabilities:
    """Return declared capabilities, failing loudly on an unknown mode."""

    try:
        return MODE_CAPABILITIES[mode]
    except KeyError as exc:
        raise ValueError(
            f"unknown operating mode {mode!r}; expected one of {VALID_MODES!r}"
        ) from exc


__all__ = [
    "ModeCapabilities", "MODE_CAPABILITIES", "VALID_MODES",
    "PRIMARY_MODES", "SECONDARY_ABLATION_MODES", "PUBLICATION_BENCHMARK_MODES",
    "RULE_REFERENCE_MODES", "REVISION_COMPARISON_MODES",
    "DECISION_OWNER_ROLES", "CONTEXT_INFRASTRUCTURE_MODES",
    "CONTEXT_MODE_MAP", "POLICY_DELTA_LEARNING_MODES",
    "REWARD_SHAPING_LEARNING_MODES", "CONTEXT_MATRIX_LEARNING_MODES",
    "AGRIBRAIN_LOGIT_MODES", "PINN_RESIDUAL_MODES", "MULTI_EPISODE_MODES",
    "capabilities_for",
]
