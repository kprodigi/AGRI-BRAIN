"""Three-mode journal distribution. Retained capability values match the evaluated source.
Legacy internal vocabulary is preserved for archived ledger compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ContextKind = Literal["full", "mcp_only", "pirag_only"]
RetrievalKind = Literal["pirag", "standard"]


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
    )


# The insertion order is the canonical public mode order used by the policy.
MODE_CAPABILITIES: dict[str, ModeCapabilities] = {
    "static": _caps(),
    "no_context": _caps(
        context_infrastructure=True, policy_delta_learning=True,
        reward_shaping_learning=True, agribrain_logits=True, peer_messages=False,
    ),
    "agribrain": _caps(
        context_kind="full", context_matrix_learning=True,
        policy_delta_learning=True, reward_shaping_learning=True,
        agribrain_logits=True,
    ),
}

PRIMARY_MODES: tuple[str, ...] = ("static", "no_context", "agribrain")
SECONDARY_ABLATION_MODES: tuple[str, ...] = ()
PUBLICATION_BENCHMARK_MODES = PRIMARY_MODES

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
    "DECISION_OWNER_ROLES", "CONTEXT_INFRASTRUCTURE_MODES",
    "CONTEXT_MODE_MAP", "POLICY_DELTA_LEARNING_MODES",
    "REWARD_SHAPING_LEARNING_MODES", "CONTEXT_MATRIX_LEARNING_MODES",
    "AGRIBRAIN_LOGIT_MODES", "PINN_RESIDUAL_MODES", "MULTI_EPISODE_MODES",
    "capabilities_for",
]
