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
Fraction of at-risk batches (spoilage risk ρ > threshold) that are
proactively routed to redistribution or recovery:

    RLE = recovered / (recovered + unrecovered_waste)

where:
    recovered          = count of at-risk batches routed to
                         local_redistribute or recovery
    unrecovered_waste  = count of at-risk batches that remained in
                         cold chain (no intervention)

The threshold (default 0.10) corresponds to 10 % quality loss — the
point where produce is still marketable but beginning to degrade and
should be considered for rerouting.

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
# Equity
# ---------------------------------------------------------------------------

def compute_equity(slca_values: List[float] | np.ndarray) -> float:
    """Compute uniformity-based equity from per-step SLCA scores.

    equity = 1 − σ(SLCA_values)

    This is a standard-deviation-based uniformity measure (not a Gini
    coefficient). Higher values indicate more consistent social performance
    across time steps.

    Parameters
    ----------
    slca_values : per-step attenuated SLCA composite scores.

    Returns
    -------
    Equity value in [0, 1].  Higher = more uniform social performance.
    """
    return 1.0 - float(np.std(slca_values))
