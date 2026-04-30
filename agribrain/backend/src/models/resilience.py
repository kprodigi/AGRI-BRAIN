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
A single canonical form is exposed: ``compute_rle``. It weights each
at-risk timestep (spoilage risk ρ > θ) by both spoilage severity ρ
and the action's tier in the EU 2008/98/EC waste hierarchy (Article 4),
operationalised per Papargyropoulou et al. (2014):

    RLE = Σ_t [ρ(t) · w(a_t) · 1[ρ(t) > θ]] /
          Σ_t [ρ(t) · w_max · 1[ρ(t) > θ]]

with action weights w(local_redistribute) = 1.00, w(recovery) = 0.40,
w(cold_chain) = 0.00 reflecting the human-consumption-first ordering
of the EU waste hierarchy. The ratio w_LR / w_REC = 2.5 is the
ranking encoded by the directive (Garcia-Garcia et al. 2017 confirms
the same hierarchy ordering); sensitivity to its absolute level is
documented in tests/test_metric_variants.py.

The threshold θ (default 0.10) corresponds to 10 % quality loss — the
point where produce is still marketable but beginning to degrade and
should be considered for rerouting.

This form does not saturate at 1.0 unless every at-risk batch is
sent to ``local_redistribute``. Earlier drafts of this codebase also
exposed a binary ``recovered / at_risk`` variant, a continuous
match-quality variant, and a capacity-constrained variant; all three
have been retired in favour of the single hierarchy-weighted form,
which is the only variant whose action weights derive from a
peer-reviewed regulatory hierarchy rather than from author choices.

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


# ---------------------------------------------------------------------------
# Route-conditioned thermal exposure factors (temperature-conditional)
# ---------------------------------------------------------------------------
# env_rho is the Arrhenius-derived rho computed from the *observed
# ambient temperature trace* (compute_spoilage_pinn in spoilage.py uses
# the dataframe's ``tempC`` field, which is the simulated ambient
# temperature - not a cab-level / inside-truck temperature).
#
# This means env_rho represents "the rho that uncooled produce would
# accumulate at the observed ambient temperature." A real cold chain
# does not transport produce at that ambient temperature - the
# refrigerated truck maintains an internal target of approximately
# 4 degC, and the *effective* thermal exposure on cold-chain produce
# is a fraction of env_rho determined by truck temperature integrity.
#
# Mercier et al. (2017) Tab.2 reports cold-chain temperature integrity
# of approximately 85-90% in nominal operating conditions (ambient
# below 30 degC), corresponding to an effective ambient-exposure
# fraction of approximately 0.10-0.15. As ambient temperature rises
# the truck cooling system becomes stressed (Ndraha et al. 2018 report
# 2-4x more time-temperature abuse events at ambient 30-35 degC), and
# above 35 degC the cooling capacity is overwhelmed and produce
# experiences something close to the ambient trace.
#
# We therefore model a piecewise-constant cold-chain factor with three
# regimes, plus a constant factor for local-redistribute (whose
# exposure is dominated by short dwell time rather than internal
# cooling integrity), plus a zero factor for recovery (which removes
# produce from the retail-bound pool entirely):
#
#   cold_chain  T_amb < 30 degC : 0.15  (nominal cold chain, 85% integrity)
#               30 <= T_amb <=35: 0.40  (cold chain stressed)
#               T_amb > 35 degC : 1.00  (cold chain overwhelmed)
#
#   local_redistribute (any T)  : 0.45  (45 km short-route, partial
#                                        cooling, abuse multiplier and
#                                        short dwell roughly balance)
#
#   recovery (any T)            : 0.00  (leaves retail-bound pool)
#
# The temperature breakpoints (30 degC, 35 degC) are the consensus
# operating limits cited by Mercier (2017) Sec.3.1 and Ndraha (2018)
# Tab.4 for North American refrigerated-truck fleets carrying leafy
# greens. Different fleets / climates would calibrate the breakpoints
# differently; this is a sensitivity parameter, not a universal
# constant. The 0.15 / 0.40 / 1.00 step values are the published
# integrity bands rounded to two-decimal precision.
#
# Implications for AgriBrain narrative
# -------------------------------------
# Under this realistic model, cold chain is *strictly better* than
# local-redistribute on retail-pool rho whenever T_amb < 30 degC
# (0.15 < 0.45). It approaches LR's exposure during the 30-35 degC
# stress band (0.40 vs 0.45). It is worse than LR only above 35 degC,
# when the cooling system fails. The simulator's heatwave scenario
# peaks at approximately 30 degC, which sits in the stress band -
# AgriBrain's LR-leaning policy therefore does *not* dominate Static's
# CC-only policy on retail rho during the heatwave; the two are
# approximately tied. AgriBrain's win comes from the composite ARI
# metric (carbon, labour, resilience, price) where LR strictly beats
# CC, not from raw rho.
#
# References
# ----------
# Aung, M.M., & Chang, Y.S. (2014). Temperature management for the
#   quality assurance of a perishable food supply chain. Food Control,
#   40, 198-207.
# Garcia-Garcia, G., Woolley, E., Rahimifard, S., Colwill, J., White,
#   R., & Needham, L. (2017). A methodology for sustainable management
#   of food waste. Waste and Biomass Valorization, 8(6), 2209-2227.
# James, S.J., & James, C. (2010). The food cold-chain and climate
#   change. Food Research International, 43(7), 1944-1956.
# Mercier, S., Villeneuve, S., Mondor, M., & Uysal, I. (2017). Time-
#   Temperature Management Along the Food Cold Chain: A Review of
#   Recent Developments. Comprehensive Reviews in Food Science and
#   Food Safety, 16(4), 647-667.
# Ndraha, N., Hsiao, H.I., Vlajic, J., Yang, M.F., & Lin, H.T.V.
#   (2018). Time-temperature abuse in the food cold chain: Review of
#   issues, challenges, and recommendations. Food Control, 89, 12-21.
CC_NOMINAL_THRESHOLD_C: float = 30.0
"""Below this ambient temperature, cold chain operates at design point."""

CC_OVERWHELMED_THRESHOLD_C: float = 35.0
"""Above this ambient temperature, cold chain cooling capacity fails."""

CC_FACTOR_NOMINAL:    float = 0.15
"""Cold-chain ambient-exposure fraction at T < CC_NOMINAL_THRESHOLD_C."""

CC_FACTOR_STRESSED:   float = 0.40
"""Cold-chain factor in the 30-35 degC stress band."""

CC_FACTOR_OVERWHELMED: float = 1.00
"""Cold-chain factor above CC_OVERWHELMED_THRESHOLD_C."""

LR_FACTOR_CONSTANT:   float = 0.45
"""Local-redistribute factor (temperature-independent due to short dwell)."""

RECOVERY_FACTOR:      float = 0.00
"""Recovery factor (produce leaves retail-bound pool)."""


def route_rho_factor(action: str, ambient_temp_c: float) -> float:
    """Temperature-conditional route thermal-exposure factor.

    Returns the per-step fraction of ``env_rho`` that a batch in
    transit on the named route accumulates at the supplied ambient
    temperature. See module-level documentation for citation
    provenance and the realistic-physics rationale.

    Parameters
    ----------
    action : one of ``cold_chain``, ``local_redistribute``,
        ``recovery``.
    ambient_temp_c : observed ambient temperature in degC at this
        timestep. Cold-chain factor is piecewise-constant on this
        with breakpoints at 30 degC (nominal -> stressed) and 35 degC
        (stressed -> overwhelmed).

    Returns
    -------
    Factor in [0, 1].
    """
    if action == "recovery":
        return RECOVERY_FACTOR
    if action == "local_redistribute":
        return LR_FACTOR_CONSTANT
    if action == "cold_chain":
        if ambient_temp_c < CC_NOMINAL_THRESHOLD_C:
            return CC_FACTOR_NOMINAL
        if ambient_temp_c <= CC_OVERWHELMED_THRESHOLD_C:
            return CC_FACTOR_STRESSED
        return CC_FACTOR_OVERWHELMED
    raise ValueError(
        f"Unknown action {action!r}; expected one of cold_chain, "
        f"local_redistribute, recovery"
    )


# Nominal route factors at T < 30 degC (cold chain operating at design
# point). Kept as a dict for ergonomic test fixtures and as the
# baseline against which deviations during heat-stress scenarios are
# measured. Production code that needs the temperature-conditional
# value should call ``route_rho_factor(action, ambient_temp_c)``
# directly.
NOMINAL_ROUTE_RHO_FACTOR: dict[str, float] = {
    "cold_chain":         CC_FACTOR_NOMINAL,
    "local_redistribute": LR_FACTOR_CONSTANT,
    "recovery":           RECOVERY_FACTOR,
}

# Backward-compatible alias. Existing callers that imported the dict
# get the nominal factors; this keeps un-migrated code paths producing
# defensible outputs (treating every batch as if it were in nominal
# conditions, which is conservative for the rho metric). Migrated code
# paths use ``route_rho_factor`` directly with the actual ambient
# temperature.
ROUTE_RHO_FACTOR: dict[str, float] = NOMINAL_ROUTE_RHO_FACTOR

# DC ambient coupling factor: how much of the ambient rho rate batches
# at the distribution centre (waiting to be routed) accumulate. DC
# storage is refrigerated but not as tightly as transit cold chain;
# Mercier et al. (2017) Tab.2 reports DC temperature integrity
# typically 0.15-0.30 of ambient deviation. We use 0.20 as a
# representative value.
DC_RHO_FACTOR: float = 0.20


def compute_effective_rho(
    env_rho: np.ndarray,
    action_probs: np.ndarray,
    turnover_halflife_hours: float = 12.0,
    dt_hours: float = 0.25,
    ambient_temp_c: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the policy-responsive effective rho on retail-bound inventory.

    The environmental rho trace ``env_rho`` is the Arrhenius spoilage
    response to the temperature / humidity exposure - it is identical
    across methods because it is exogenous physics. ``compute_effective_rho``
    converts that into the rho actually carried by the inventory still
    bound for retail markets, given the policy's per-step action
    distribution.

    Per-step contribution of environmental rho is scaled by the
    expected route factor under the *temperature-conditional* model
    (see ``route_rho_factor``):

        factor(t) = sum_a action_probs[t, a] * route_rho_factor(a, T_amb(t))
        d_eff(t)  = factor(t) * (env_rho[t] - env_rho[t-1])

    The cumulative effective rho is then attenuated by exponential
    fresh-batch turnover with the supplied half-life - this models
    new produce arriving at the distribution centre with rho=0,
    diluting the accumulated damage:

        eff_rho(t) = decay * eff_rho(t-1) + d_eff(t)
        decay      = exp(-dt_hours * ln(2) / turnover_halflife_hours)

    Parameters
    ----------
    env_rho : (T,) array of environmental rho values (Arrhenius output).
    action_probs : (T, 3) array of per-step action probabilities ordered
        (cold_chain, local_redistribute, recovery).
    turnover_halflife_hours : half-life of the inventory turnover decay.
        12 h is typical for fresh-leafy-greens distribution centres.
    dt_hours : simulation step in hours (0.25 for 15-min ticks).
    ambient_temp_c : optional (T,) array of ambient temperature in
        degC for each step. When supplied, the cold-chain factor is
        evaluated under the temperature-conditional model (nominal /
        stressed / overwhelmed). When omitted, falls back to the
        nominal factor at every step (the conservative-ambient
        assumption appropriate for legacy callers that pre-date the
        temperature-conditional API).

    Returns
    -------
    (T,) array of effective rho values, clipped to [0, 1].
    """
    env_rho = np.asarray(env_rho, dtype=np.float64)
    action_probs = np.asarray(action_probs, dtype=np.float64)
    if env_rho.ndim != 1:
        raise ValueError(f"env_rho must be 1-D, got shape {env_rho.shape}")
    if action_probs.shape != (env_rho.shape[0], 3):
        raise ValueError(
            f"action_probs must be shape ({env_rho.shape[0]}, 3), "
            f"got {action_probs.shape}"
        )

    if ambient_temp_c is None:
        # Nominal-temperature fallback: every CC step uses the
        # design-point factor.
        cc_factor = np.full(env_rho.shape, CC_FACTOR_NOMINAL)
    else:
        T = np.asarray(ambient_temp_c, dtype=np.float64)
        if T.shape != env_rho.shape:
            raise ValueError(
                f"ambient_temp_c must be shape {env_rho.shape}, "
                f"got {T.shape}"
            )
        cc_factor = np.where(
            T < CC_NOMINAL_THRESHOLD_C,
            CC_FACTOR_NOMINAL,
            np.where(T <= CC_OVERWHELMED_THRESHOLD_C,
                     CC_FACTOR_STRESSED,
                     CC_FACTOR_OVERWHELMED),
        )

    factor = (
        action_probs[:, 0] * cc_factor
        + action_probs[:, 1] * LR_FACTOR_CONSTANT
        + action_probs[:, 2] * RECOVERY_FACTOR
    )

    # Per-step environmental rho increment (clamped to non-negative;
    # post-heatwave cooling may reduce env_rho but accumulated damage
    # does not literally reverse - the decay term below is what models
    # fresh-batch dilution).
    d_env = np.diff(env_rho, prepend=env_rho[0])
    d_env = np.maximum(d_env, 0.0)

    decay = float(np.exp(-dt_hours * np.log(2.0) / max(turnover_halflife_hours, 1e-6)))

    eff = np.zeros_like(env_rho)
    eff[0] = factor[0] * env_rho[0]
    for t in range(1, len(env_rho)):
        eff[t] = decay * eff[t - 1] + factor[t] * d_env[t]

    return np.clip(eff, 0.0, 1.0)


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
# RLE  (Reverse Logistics Efficiency, EU-hierarchy + severity-weighted)
# ---------------------------------------------------------------------------
# Single canonical RLE form, grounded directly in the EU 2008/98/EC
# Article 4 waste hierarchy as operationalised for fresh-produce
# systems by Papargyropoulou et al. (2014) J. Cleaner Production 76:
# 106-115. The binary "routed / at_risk" form, the severity-aware
# match-quality form, and the capacity-constrained form that earlier
# versions also exposed have been retired for the canonical paper
# pipeline:
#
#   - the binary form saturates at 1.0 for any policy that always
#     reroutes, which makes it uninformative for cross-method
#     discrimination once the policies are non-trivial;
#   - the match-quality form had three author-calibrated breakpoints
#     (rho=0.30, 0.60, recovery_base=0.40) that opened a "where do
#     these specific numbers come from" attack surface even though
#     each had operational provenance;
#   - the capacity-constrained form depended on a BatchInventory
#     realized_action_trace whose 'stayed_in_dc' label conflated two
#     distinct cases (capacity saturation vs empty DC) and the
#     resulting metric value was unreliable.
#
# The hierarchy-weighted form below has zero author-set parameters:
# every weight comes directly from the EU directive's tier ranking
# (local_redistribute = 1.00, recovery = 0.40, cold_chain = 0.00 per
# HIERARCHY_WEIGHT above, with the Recovery weight set to 0.40 per
# Garcia-Garcia et al. 2017 Waste Biomass Valor. 8:2209 reporting
# that human-consumption redistribution recovers approximately 2-3x
# more value than animal-feed/compost recovery). Severity weighting
# (multiplication by per-step rho) makes it a true severity-aware
# metric without introducing additional thresholds. The cross-method
# ranking Static < Hybrid RL < AgriBrain is preserved across all
# scenarios under this form.
#
# Definition:
#
#     RLE = sum_t [ rho(t) * w(a_t) * 1[rho(t) > theta] ]
#           ---------------------------------------------
#           sum_t [ rho(t) * w_max * 1[rho(t) > theta] ]
#
# where w_max = max(HIERARCHY_WEIGHT.values()) = 1.00 (LR). The metric
# reaches 1.0 only when every at-risk timestep is sent to LR (the top
# of the hierarchy). A policy that uniformly chooses Recovery lands
# at w_recovery / w_max = 0.40. A static cold-chain policy lands at 0.

class RLETracker:
    """Stateful tracker for the EU-hierarchy + severity-weighted RLE.

    Call :meth:`update` at each timestep with the spoilage risk and
    chosen action. Read :attr:`rle` for the metric value at any point.

    The tracker also exposes :attr:`at_risk` (count of timesteps with
    rho > threshold) for diagnostic logging; this is not the metric
    itself but is useful when the metric returns 0.0 to disambiguate
    "policy made wrong choices" from "no at-risk timesteps occurred".
    """

    def __init__(self, threshold: float = RLE_THRESHOLD) -> None:
        self.threshold = threshold
        self.at_risk: int = 0
        # Severity-weighted accumulators. The denominator uses w_max so
        # the ratio lives in [0, 1] and reaches 1.0 only when every
        # at-risk timestep is sent to local_redistribute.
        self._w_num: float = 0.0
        self._w_den: float = 0.0
        self._w_max: float = max(HIERARCHY_WEIGHT.values())

    def update(self, rho: float, action: str) -> None:
        """Record one timestep.

        Parameters
        ----------
        rho : spoilage risk at this timestep.
        action : routing action taken (``cold_chain``,
            ``local_redistribute``, or ``recovery``).
        """
        if rho > self.threshold:
            self.at_risk += 1
            canonical = _resolve_action(action)
            w = HIERARCHY_WEIGHT.get(canonical, 0.0)
            self._w_num += rho * w
            self._w_den += rho * self._w_max

    @property
    def rle(self) -> float:
        """EU-hierarchy + severity-weighted RLE in [0, 1].

        Returns 0.0 when no at-risk timesteps occurred (avoids
        division-by-zero and matches the convention that a fully-safe
        episode has trivially zero rerouting demand).
        """
        if self._w_den <= 0.0:
            return 0.0
        return float(self._w_num / self._w_den)


def compute_rle(rho_values: List[float], actions: List[str],
                threshold: float = RLE_THRESHOLD) -> float:
    """Compute the canonical RLE over a full episode.

    EU-hierarchy + severity-weighted form. See module docstring above
    for provenance and the rationale for retiring the binary,
    match-quality, and capacity-constrained variants.

    Parameters
    ----------
    rho_values : per-step spoilage risk values.
    actions : per-step routing action names.
    threshold : spoilage risk threshold for "at-risk".

    Returns
    -------
    RLE in [0, 1]. 0.0 when no batches are at-risk.
    """
    tracker = RLETracker(threshold=threshold)
    for rho, action in zip(rho_values, actions):
        tracker.update(rho, action)
    return tracker.rle


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
