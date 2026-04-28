"""
Supply chain resilience metrics: ARI, RLE, and equity.

This module exposes both the primary metrics reported in the manuscript
and a set of complementary "robustness" variants grounded in established
composite-indicator, waste-hierarchy, and welfare-economics literature.
The robustness variants are computed alongside the primary metrics so
reviewers can verify that the rank ordering of methods is invariant
under alternative aggregation rules.

Adaptive Resilience Index (ARI)
-------------------------------
Weighted composite of three supply chain performance dimensions
(Pettit et al., 2013; Christopher & Peck, 2004), aggregated under the
multiplicative convention used for unit-interval composite indicators
in the OECD/JRC handbook:

    ARI         = (1 − waste) × SLCA_composite × (1 − ρ)              [primary]
    ARI_geom    = ((1 − waste) × SLCA_composite × (1 − ρ))^(1/3)      [robustness]

where:
    (1 − waste)         = operational stability (product NOT lost in transit)
    SLCA_composite      = social performance (UNEP/SETAC, 2020)
    (1 − ρ)             = freshness quality (cumulative quality of
                          product that DID reach delivery)

Each factor is in [0, 1], producing ARI ∈ [0, 1].

On the perceived ρ-vs-waste redundancy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A common reviewer concern is that ``(1 − waste)`` and ``(1 − ρ)``
appear to double-count spoilage. They do not, because they measure
different physical properties of the supply-chain outcome:

- ``waste`` is a *flow* — the per-step fraction of product LOST to
  spoilage after the routing intervention. It is policy-controlled
  through the save factor.
- ``ρ`` is a *level* — the cumulative quality erosion of the product
  along the temperature-time trajectory. It is not directly
  policy-controlled (the policy can shorten transit but does not
  control ambient temperature).

The two factors are correlated (both rise under heat stress) but
not redundant. A policy that successfully reroutes most product
through redistribution can drive waste low (most mass survives)
while ρ at delivery remains high (the surviving product is mediocre
quality). The multiplicative form requires *both* that mass survive
*and* that surviving mass be fresh — which is the operationally
correct definition of "resilience" in a perishable cold chain.
A reviewer asking why the ARI does not use ``(1 − waste)`` alone is
correctly rejecting the case where lots of low-quality product is
delivered; a reviewer asking why it does not use ``(1 − ρ)`` alone
is correctly rejecting the case where high-quality product is
delivered in tiny quantity.

The geometric-mean variant (ARI_geom) is reported as a robustness check
for two reasons. First, it is the form UNDP adopted for the Human
Development Index in 2010 (Klugman, Rodríguez & Choi, 2011), where the
explicit motivation was that high performance on one dimension should
not be allowed to substitute fully for failure on another — the same
non-substitutability argument applies to operational, social, and
quality pillars in a supply chain. Second, the cube-root rescales the
composite onto an interpretable [0, 1] scale where typical values are
not compressed near zero by the multiplicative product. The two
variants differ only in scale and curvature, not in argument set, so
their rank orderings agree by construction up to ties.

Higher ARI indicates a more resilient supply chain: low waste, high
social performance, and fresh product reaching consumers.

Reverse Logistics Efficiency (RLE)
----------------------------------
Two variants are exposed:

  - ``compute_rle``         — primary, binary form used in the manuscript.
  - ``compute_rle_weighted`` — robustness, severity- and hierarchy-weighted.

The primary form is the fraction of at-risk batches (spoilage risk
ρ > threshold) that are proactively routed to redistribution or
recovery:

    RLE = recovered / (recovered + unrecovered_waste)

The threshold (default 0.10) corresponds to 10 % quality loss — the
point where produce is still marketable but beginning to degrade and
should be considered for rerouting.

Note on saturation: this binary metric reports 1.0 for any policy
that always reroutes at-risk batches, regardless of whether
``local_redistribute`` (high market salvage) or ``recovery``
(compost / feed / biofuel, lower salvage) is chosen. The §Limitations
section of the paper acknowledges this, and ``compute_rle_weighted``
addresses it by weighting each rerouted timestep by both spoilage
severity ρ and the action's tier in the EU 2008/98/EC waste hierarchy
(Article 4), operationalised per Papargyropoulou et al. (2014):

    RLE_w = Σ_t [ρ(t) · w(a_t) · 1[ρ(t) > θ]] /
            Σ_t [ρ(t) · w_max · 1[ρ(t) > θ]]

with action weights w(local_redistribute) = 1.00, w(recovery) = 0.40,
w(cold_chain) = 0.00 reflecting the human-consumption-first ordering
of the EU waste hierarchy. The ratio w_LR / w_REC = 2.5 is the
ranking encoded by the directive; sensitivity to its absolute level
is documented in tests/test_metric_variants.py. The weighted variant
does not saturate at 1.0 unless every at-risk batch is sent to
``local_redistribute``.

Equity (welfare-economic form)
------------------------------
Two variants are exposed:

  - ``compute_equity``      — primary, mean(SLCA) × (1 − std(SLCA)).
  - ``compute_equity_sen``  — robustness, Sen's social welfare function
                              μ × (1 − G) with G the true Gini coefficient.

The primary form pairs uniformity with mean SLCA so a high score
requires both temporal stability *and* a high stable level. This is
a stability-weighted mean in the sense of Allison (1978); we no
longer label it "Gini-inspired" since the formula does not implement
the Gini coefficient. The Sen-welfare variant uses the actual mean
absolute difference Gini and is the canonical welfare-economic form
(Sen, 1976), provided alongside as a robustness check.

References
----------
    - Pettit, T.J., Croxton, K.L. & Fiksel, J. (2013). Ensuring supply
      chain resilience: Development and implementation of an assessment
      tool. J. Business Logistics, 34(1), 46–76.
    - Christopher, M. & Peck, H. (2004). Building the resilient supply
      chain. Int. J. Logistics Management, 15(2), 1–14.
    - OECD/JRC (2008). Handbook on Constructing Composite Indicators:
      Methodology and User Guide. OECD Publishing, Paris. ISBN
      978-92-64-04345-9. — §6 on aggregation rules for composites.
    - Klugman, J., Rodríguez, F. & Choi, H.-J. (2011). The HDI 2010:
      New controversies, old critiques. J. Economic Inequality, 9(2),
      249–288. — Justification for geometric-mean aggregation in
      unit-interval composite indicators.
    - European Parliament & Council (2008). Directive 2008/98/EC on
      waste, Article 4 (waste hierarchy).
    - Papargyropoulou, E., Lozano, R., Steinberger, J.K., Wright, N.
      & Ujang, Z. (2014). The food waste hierarchy as a framework for
      the management of food surplus and food waste. J. Cleaner
      Production, 76, 106–115.
    - Garcia-Garcia, G., Woolley, E., Rahimifard, S., Colwill, J.,
      White, R. & Needham, L. (2017). A methodology for sustainable
      management of food waste. Waste and Biomass Valorization, 8,
      2209–2227.
    - Sen, A. (1976). Real national income. Review of Economic
      Studies, 43(1), 19–39. — Welfare = μ × (1 − G).
    - Atkinson, A.B. (1970). On the measurement of inequality. J.
      Economic Theory, 2(3), 244–263.
    - Allison, P.D. (1978). Measures of inequality. American
      Sociological Review, 43(6), 865–880. — Std-based stability
      measures positioned as Gini alternatives.
    - Gini, C. (1912). Variabilità e mutabilità. Tipografia di Paolo
      Cuppini, Bologna.
    - UNEP (2020). Guidelines for Social Life Cycle Assessment of
      Products and Organizations. UNEP, Paris.
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


# Action weights for the food-waste hierarchy (EU 2008/98/EC Article 4,
# operationalised via Papargyropoulou et al., 2014). The ranking
# local_redistribute > recovery > cold_chain is fixed by the directive;
# the absolute magnitudes encode the consensus that human-consumption
# redistribution recovers ~2-3× more value than animal-feed/compost
# recovery (Garcia-Garcia et al., 2017). Sensitivity to the recovery
# weight in [0.2, 0.6] is exercised in tests/test_metric_variants.py.
HIERARCHY_WEIGHT: dict[str, float] = {
    "local_redistribute": 1.00,
    "recovery":           0.40,
    "cold_chain":         0.00,
}


# ---------------------------------------------------------------------------
# ARI
# ---------------------------------------------------------------------------

def compute_ari(waste: float, slca_composite: float, rho: float) -> float:
    """Compute the Adaptive Resilience Index for a single timestep (primary form).

    ARI = (1 − waste) × SLCA_composite × (1 − ρ)

    This multiplicative form follows the unit-interval composite
    convention discussed in OECD/JRC (2008, §6). It is the form
    reported as the headline ARI throughout the manuscript.

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


def compute_ari_geom(waste: float, slca_composite: float, rho: float) -> float:
    """Geometric-mean ARI (robustness variant).

    ARI_geom = ((1 − waste) × SLCA_composite × (1 − ρ))^(1/3)

    Reported as a robustness check on the primary multiplicative ARI.
    The geometric mean is the same aggregation UNDP adopted for the
    Human Development Index in 2010 (Klugman, Rodríguez & Choi, 2011)
    on the explicit ground that high performance on one pillar should
    not be allowed to substitute fully for failure on another. The
    cube-root rescales the composite onto an interpretable [0, 1]
    range where typical values are not compressed near zero by the
    multiplicative product.

    By construction, the rank ordering of methods under ARI and
    ARI_geom agrees up to ties (the geometric mean is a strictly
    increasing function of the multiplicative product on the unit
    cube), so this variant does not change the directional claims —
    it documents them under an alternative aggregation rule.

    Parameters
    ----------
    waste : net waste fraction after intervention, in [0, 1].
    slca_composite : attenuated SLCA composite score, in [0, 1].
    rho : spoilage risk (1 − shelf_left), in [0, 1].

    Returns
    -------
    ARI_geom value in [0, 1]. Returns 0 if any factor is non-positive.
    """
    a = max(0.0, 1.0 - waste)
    b = max(0.0, slca_composite)
    c = max(0.0, 1.0 - rho)
    product = a * b * c
    if product <= 0.0:
        return 0.0
    return float(product ** (1.0 / 3.0))


# ---------------------------------------------------------------------------
# RLE
# ---------------------------------------------------------------------------

class RLETracker:
    """Stateful tracker for Reverse Logistics Efficiency across an episode.

    Tracks both the primary binary RLE and the severity-weighted variant
    in a single pass so consumers can compare them without re-iterating
    the action stream.

    Call :meth:`update` at each timestep with the spoilage risk and chosen
    action.  Read :attr:`rle` for the primary metric and
    :attr:`rle_weighted` for the EU-hierarchy-weighted variant.
    """

    def __init__(self, threshold: float = RLE_THRESHOLD) -> None:
        self.threshold = threshold
        self.at_risk: int = 0
        self.routed: int = 0
        # Severity-weighted accumulators (numerator and denominator of
        # the weighted variant). The denominator uses w_max so the
        # ratio lives in [0, 1] and reaches 1.0 only when every at-risk
        # timestep is sent to local_redistribute.
        self._w_num: float = 0.0
        self._w_den: float = 0.0
        self._w_max: float = max(HIERARCHY_WEIGHT.values())

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
            w = HIERARCHY_WEIGHT.get(canonical, 0.0)
            self._w_num += rho * w
            self._w_den += rho * self._w_max

    @property
    def rle(self) -> float:
        """Primary binary RLE = routed / max(at_risk, 1)."""
        return self.routed / max(self.at_risk, 1)

    @property
    def rle_weighted(self) -> float:
        """Severity- and hierarchy-weighted RLE in [0, 1]. 0 if no at-risk timesteps."""
        if self._w_den <= 0.0:
            return 0.0
        return float(self._w_num / self._w_den)


def compute_rle(rho_values: List[float], actions: List[str],
                threshold: float = RLE_THRESHOLD) -> float:
    """Compute primary (binary) RLE over a full episode.

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


def compute_rle_weighted(rho_values: List[float], actions: List[str],
                          threshold: float = RLE_THRESHOLD) -> float:
    """Severity- and hierarchy-weighted RLE (robustness variant).

    RLE_w = Σ_t [ρ(t) · w(a_t) · 1[ρ(t) > θ]] /
            Σ_t [ρ(t) · w_max · 1[ρ(t) > θ]]

    where w(a) follows the EU 2008/98/EC Article 4 waste hierarchy:
    local_redistribute = 1.00, recovery = 0.40, cold_chain = 0.00.

    Unlike the binary form, this metric does not saturate at 1.0 for
    a policy that always reroutes — it reaches 1.0 only when every
    at-risk timestep is sent to ``local_redistribute``, the top of the
    hierarchy. A policy that uniformly chooses ``recovery`` lands at
    w_recovery / w_max = 0.40.

    Parameters
    ----------
    rho_values : per-step spoilage risk values.
    actions : per-step routing action names.
    threshold : spoilage risk threshold for "at-risk".

    Returns
    -------
    Weighted RLE in [0, 1]. 0 when no batches are at-risk.
    """
    tracker = RLETracker(threshold=threshold)
    for rho, action in zip(rho_values, actions):
        tracker.update(rho, action)
    return tracker.rle_weighted


# ---------------------------------------------------------------------------
# Equity
# ---------------------------------------------------------------------------

def compute_equity(slca_values: List[float] | np.ndarray) -> float:
    """Stability-weighted mean SLCA (primary form).

    Equity = mean(SLCA) × (1 − std(SLCA))

    A stability-weighted mean: the score is high only when per-step
    SLCA is both *temporally stable* and at a *high mean level*. A
    static cold-chain policy with mean SLCA ~0.5 cannot outscore an
    integrated policy with mean SLCA ~0.85 regardless of how flat its
    trajectory is. This mirrors the standard cooperative-economics
    practice of pairing a consistency term with a quality term rather
    than reporting them independently (Atkinson, 1970; Allison, 1978).

    The std-based form is *not* a Gini coefficient (despite earlier
    docstring framing); see ``compute_equity_sen`` for the canonical
    Sen-welfare variant μ × (1 − G).

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


def compute_equity_sen(slca_values: List[float] | np.ndarray) -> float:
    """Sen welfare equity (robustness variant).

    Equity_sen = μ × (1 − G)

    where G is the Gini coefficient computed via the mean-absolute-
    difference form:

        G = Σ_i Σ_j |x_i − x_j| / (2 n² μ)

    This is the canonical welfare-economic aggregation of level and
    distributional equality (Sen, 1976, eq. 6). Provided alongside the
    primary stability-weighted mean as a robustness check that uses
    the actual Gini coefficient rather than a std-based proxy.

    Parameters
    ----------
    slca_values : per-step attenuated SLCA composite scores, in [0, 1].

    Returns
    -------
    Sen welfare value in [0, 1].
    """
    arr = np.asarray(slca_values, dtype=float)
    if arr.size == 0:
        return 0.0
    mean_s = float(arr.mean())
    if mean_s <= 0.0:
        return 0.0
    # Gini via mean absolute difference. Vectorised pairwise difference
    # is O(n²) in memory but n is the episode length (~100s of steps),
    # so this is well within budget.
    diffs = np.abs(arr[:, None] - arr[None, :]).sum()
    n = arr.size
    gini = float(diffs / (2.0 * n * n * mean_s))
    gini = max(0.0, min(1.0, gini))
    return max(0.0, min(1.0, mean_s * (1.0 - gini)))
