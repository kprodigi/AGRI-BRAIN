"""
Supply chain resilience metrics: ARI, RLE, and equity.

Adaptive Resilience Index (ARI)
-------------------------------
Weighted composite of three supply chain performance dimensions
(Pettit et al., 2013; Christopher & Peck, 2004):

    ARI = (1 − waste) × SLCA_composite × (1 − ρ)

where:
    (1 − waste)         = operational stability (product not lost)
    SLCA_composite      = social performance (UNEP/SETAC, 2020)
    (1 − ρ)             = freshness quality (product condition)

Each factor is in [0, 1], producing ARI ∈ [0, 1].

Higher ARI indicates a more resilient supply chain: low waste, high
social performance, and fresh product reaching consumers.

Reverse Logistics Efficiency (RLE)
----------------------------------
Two variants are reported.

Binary RLE — the standard fraction-routed metric:

    RLE = recovered / (recovered + unrecovered_waste)

where:
    recovered          = count of at-risk batches routed to
                         local_redistribute or recovery
    unrecovered_waste  = count of at-risk batches that remained in
                         cold chain (no intervention)

The threshold (default 0.10) corresponds to 10 % quality loss — the
point where produce is still marketable but beginning to degrade and
should be considered for rerouting.

The binary RLE saturates at 1.0 for any policy that always reroutes
at-risk batches, even if the chosen reroute is suboptimal (e.g.
sending a mildly degraded batch to compost rather than to local
redistribution). To capture *which* reroute and *when*, we additionally
report a value-weighted graded RLE in the Recovery Value Index /
Effective Recovery Rate family (Govindan, Soleimani & Kannan 2015;
Steeneck & Sarin 2018), reflecting the cascading-use hierarchy of the
EU Waste Framework Directive (2008/98/EC, Article 4):

                    Σ_t ρ_t · e(a_t, ρ_t) · 1[ρ_t > θ]
    RLE_graded  =  ─────────────────────────────────────
                    Σ_t ρ_t · e*(ρ_t)     · 1[ρ_t > θ]

where e(a, ρ) is the action-specific recovery factor at spoilage state
ρ (fraction of original product value salvaged), and e*(ρ) = max_a
e(a, ρ) is the best achievable factor at that state. The metric is
1.0 only when the policy picks the optimal action at every at-risk
step; sending every at-risk batch to ``recovery`` even when
``local_redistribute`` would salvage more value scores well below 1.

Equity (Gini-inspired)
----------------------
Distribution uniformity of per-step SLCA scores across the episode:

    equity = 1 − σ(SLCA_values)

where σ is the standard deviation. Higher equity means more consistent
social performance across all timesteps. Static mode achieves equity = 1.0
trivially (constant SLCA), while adaptive methods have lower equity due
to policy variation — but this is acceptable when mean SLCA is higher.

Based on the Gini coefficient concept (Gini, 1912), adapted for
continuous performance metrics rather than income distribution.

References
----------
    - Pettit, T.J., Croxton, K.L. & Fiksel, J. (2013). Ensuring supply
      chain resilience: Development and implementation of an assessment
      tool. J. Business Logistics, 34(1), 46–76.
    - Christopher, M. & Peck, H. (2004). Building the resilient supply
      chain. Int. J. Logistics Management, 15(2), 1–14.
    - Gini, C. (1912). Variabilità e mutabilità. Reprinted in Memorie
      di metodologica statistica. Libreria Eredi Virgilio Veschi, Rome.
    - UNEP/SETAC (2020). Guidelines for Social Life Cycle Assessment of
      Products and Organizations.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .action_aliases import resolve_action as _resolve_action


# ---------------------------------------------------------------------------
# RLE threshold
# ---------------------------------------------------------------------------
RLE_THRESHOLD: float = 0.10
"""Spoilage risk threshold above which a batch is considered "at-risk".

0.10 corresponds to 10 % quality loss — produce is still marketable but
beginning to degrade. Rerouting at this point maximises recovery value.
"""


# ---------------------------------------------------------------------------
# ARI
# ---------------------------------------------------------------------------

def compute_ari(waste: float, slca_composite: float, rho: float) -> float:
    """Compute the Adaptive Resilience Index for a single timestep.

    ARI = (1 − waste) × SLCA_composite × (1 − ρ)

    Parameters
    ----------
    waste : net waste fraction after intervention, in [0, 1].
    slca_composite : attenuated SLCA composite score, in [0, 1].
    rho : spoilage risk (1 − shelf_left), in [0, 1].

    Returns
    -------
    ARI value in [0, 1].
    """
    return (1.0 - waste) * slca_composite * (1.0 - rho)


# ---------------------------------------------------------------------------
# RLE
# ---------------------------------------------------------------------------

class RLETracker:
    """Stateful tracker for Reverse Logistics Efficiency across an episode.

    Call :meth:`update` at each timestep with the spoilage risk and chosen
    action.  Call :meth:`rle` to get the current efficiency.
    """

    def __init__(self, threshold: float = RLE_THRESHOLD) -> None:
        self.threshold = threshold
        self.at_risk: int = 0
        self.routed: int = 0

    def update(self, rho: float, action: str) -> None:
        """Record one timestep.

        Parameters
        ----------
        rho : spoilage risk at this timestep.
        action : routing action taken (``cold_chain``, ``local_redistribute``,
                 or ``recovery``).
        """
        if rho > self.threshold:
            self.at_risk += 1
            canonical = _resolve_action(action)
            if canonical in ("local_redistribute", "recovery"):
                self.routed += 1

    @property
    def rle(self) -> float:
        """Current RLE = routed / max(at_risk, 1)."""
        return self.routed / max(self.at_risk, 1)


def compute_rle(rho_values: List[float], actions: List[str],
                threshold: float = RLE_THRESHOLD) -> float:
    """Compute RLE over a full episode.

    Parameters
    ----------
    rho_values : per-step spoilage risk values.
    actions : per-step routing action names.
    threshold : spoilage risk threshold for "at-risk".

    Returns
    -------
    RLE in [0, 1].  Returns 0 when no batches are at-risk.
    """
    tracker = RLETracker(threshold=threshold)
    for rho, action in zip(rho_values, actions):
        tracker.update(rho, action)
    return tracker.rle


# ---------------------------------------------------------------------------
# Graded RLE (Recovery Value Index / Effective Recovery Rate family)
# ---------------------------------------------------------------------------
#
# The binary :func:`compute_rle` saturates at 1.0 for any policy that always
# reroutes at-risk batches, regardless of *which* reroute is chosen or *when*.
# That is fine as a coarse "did we intervene?" signal, but undersells the
# action-quality dimension of reverse logistics: redistributing at the right
# moment recovers ~85 % of unit value, whereas late composting recovers only
# ~30 %, and the binary metric treats both as identical "routed" events.
#
# The graded variant below is in the Recovery Value Index / Effective Recovery
# Rate family (Govindan, Soleimani & Kannan 2015; Steeneck & Sarin 2018) and
# also reflects the cascading-use hierarchy in the EU Waste Framework
# Directive (2008/98/EC, Article 4): redistribution > recovery > disposal.
# It penalises (a) late intervention (high ρ when the action could have been
# taken earlier), (b) wrong action choice for the spoilage state, and
# (c) failure to intervene at all.
#
# Recovery factor table — fraction of original product value salvaged when
# action ``a`` is taken at spoilage state ρ. Calibrated from cold-chain
# reverse-logistics literature; the values below are illustrative defaults
# meant for a published sensitivity analysis rather than fixed constants.
# Each list is a sorted sequence of ``(rho_upper, factor)`` pairs; the first
# bucket whose ``rho_upper`` covers the current ρ wins.
RLE_RECOVERY_FACTORS = {
    "local_redistribute": [
        (0.30, 0.85),  # mild at-risk -> high market salvage at discount
        (0.60, 0.50),  # moderate     -> reduced market value
        (1.00, 0.20),  # severe       -> mostly secondary market only
    ],
    "recovery": [
        (1.00, 0.30),  # composting / feed / biofuel: roughly constant low recovery
    ],
    "cold_chain": [
        (1.00, 0.0),   # no intervention -> no recovery beyond residual shelf life
    ],
}


def _recovery_factor(action: str, rho: float) -> float:
    """Recovery factor e(a, rho) — fraction of original value salvaged."""
    canonical = _resolve_action(action)
    factors = RLE_RECOVERY_FACTORS.get(canonical)
    if not factors:
        return 0.0
    for rho_upper, value in factors:
        if rho <= rho_upper:
            return float(value)
    return 0.0


def optimal_recovery_factor(rho: float) -> float:
    """Best achievable recovery factor at this spoilage state.

    e*(rho) = max_a e(a, rho). Used as the per-step normaliser in
    :func:`compute_rle_graded` so the metric reads as "fraction of the
    maximum possible recovery the policy actually achieved".
    """
    return float(max(
        _recovery_factor(action, rho)
        for action in RLE_RECOVERY_FACTORS
    ))


class RLEGradedTracker:
    """Severity-weighted, value-aware reverse-logistics efficiency.

    Tracks two cumulative quantities across an episode:

      numerator   = Σ ρ_t · e(a_t, ρ_t) · 1[ρ_t > threshold]
      denominator = Σ ρ_t · e*(ρ_t)     · 1[ρ_t > threshold]

    The graded RLE is ``numerator / denominator`` ∈ [0, 1]: 1.0 means
    every at-risk batch was routed to the *optimal* action for its
    severity, 0.0 means none were rerouted at all. Unlike the binary
    RLE, a policy that always picks ``recovery`` even when
    ``local_redistribute`` would salvage more value scores well below 1.
    """

    def __init__(self, threshold: float = RLE_THRESHOLD) -> None:
        self.threshold = threshold
        self.numerator: float = 0.0
        self.denominator: float = 0.0

    def update(self, rho: float, action: str) -> None:
        if rho > self.threshold:
            self.numerator += rho * _recovery_factor(action, rho)
            self.denominator += rho * optimal_recovery_factor(rho)

    @property
    def rle_graded(self) -> float:
        if self.denominator <= 0.0:
            return 0.0
        return self.numerator / self.denominator


def compute_rle_graded(rho_values: List[float], actions: List[str],
                       threshold: float = RLE_THRESHOLD) -> float:
    """Compute the severity-weighted, value-aware RLE over a full episode.

    See :class:`RLEGradedTracker` for the formula and rationale.

    Returns
    -------
    RLE_graded in [0, 1]. Returns 0 when no batches are at-risk.
    """
    tracker = RLEGradedTracker(threshold=threshold)
    for rho, action in zip(rho_values, actions):
        tracker.update(rho, action)
    return tracker.rle_graded


# ---------------------------------------------------------------------------
# Equity
# ---------------------------------------------------------------------------

def compute_equity(slca_values: List[float] | np.ndarray) -> float:
    """Quality-weighted temporal consistency of social performance.

    Equity = mean(SLCA) * (1 - std(SLCA))

    The original form ``1 - std(SLCA)`` was a pure uniformity measure and
    therefore gave a high score to any policy that produced constant SLCA,
    including a degenerate static policy whose SLCA is uniformly low. The
    revised definition multiplies uniformity by mean SLCA so that a high
    equity value requires *both* that per-step SLCA be temporally stable
    *and* that the stable level be high. A static cold-chain policy with
    mean SLCA ~0.5 therefore cannot score higher than an integrated
    policy with mean SLCA ~0.85 regardless of how flat its trajectory
    is. This mirrors the standard cooperative-economics practice of
    pairing a consistency term with a quality term rather than reporting
    them independently (Atkinson 1970; UNEP/SETAC SLCA guidelines 2020).

    Parameters
    ----------
    slca_values : per-step attenuated SLCA composite scores, in [0, 1].

    Returns
    -------
    Equity value in [0, 1]. Higher = more uniform AND higher mean SLCA.
    """
    arr = np.asarray(slca_values, dtype=float)
    if arr.size == 0:
        return 0.0
    mean_s = float(np.mean(arr))
    std_s = float(np.std(arr))
    # SLCA is bounded in [0, 1] so std cannot exceed 0.5 in practice;
    # clip defensively so equity stays in [0, 1] for downstream consumers
    # that assume a unit-interval metric.
    uniformity = max(0.0, min(1.0, 1.0 - std_s))
    return max(0.0, min(1.0, mean_s * uniformity))
