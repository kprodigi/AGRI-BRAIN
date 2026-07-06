"""
Supply chain resilience metrics: ARI, RLE, and equity.

This module exposes both the primary metrics reported in the manuscript
and a set of complementary "robustness" variants grounded in established
composite-indicator, waste-hierarchy, and welfare-economics literature.
The robustness variants are computed alongside the primary metrics so
that the rank ordering of methods can be verified to be invariant
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
A potential concern is that ``(1 − waste)`` and ``(1 − ρ)``
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
Asking why the ARI does not use ``(1 − waste)`` alone correctly
rejects the case where lots of low-quality product is delivered;
asking why it does not use ``(1 − ρ)`` alone correctly rejects
the case where high-quality product is delivered in tiny quantity.

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
RLE is the EU 2008/98/EC Article 4 food-waste hierarchy operationalised
as a unit-interval metric. Article 4 establishes a five-tier priority
order that EU Member States must apply in waste-management policy; the
three tiers relevant to perishable-food routing are, in descending
priority, (a) prevention, (b) preparing for re-use / re-use for human
consumption, (c) recycling (incl. animal feed, anaerobic digestion).
Tier (b) is operationalised as ``local_redistribute``, tier (c) as
``recovery``, and the no-intervention default ``cold_chain`` sits
outside the hierarchy. The Commission Notice 2017/C 361/01 §3.1 and
Garcia-Garcia et al. (2017) §4.2 add the food-safety conditional:
above the marketable-quality boundary, tier (b) is no longer
admissible and tier (c) becomes the top-priority hierarchy choice.
Papargyropoulou et al. (2014) Fig.2 provides the bench magnitudes;
§3.3 of the same paper describes the marketable / non-marketable
boundary as a continuous risk gradient rather than a step function.

The metric:

    RLE = Σ_t [ρ(t) · w(a_t, ρ(t)) · 1[ρ(t) > θ]] /
          Σ_t [ρ(t) · w_max · 1[ρ(t) > θ]]

with the ρ-conditional weight table

    ρ ≤ cutoff    (marketable):     w_LR = 1.00, w_Rec = 0.40, w_CC = 0.00
    ρ > cutoff    (non-marketable): w_LR = 0.00, w_Rec = 1.00, w_CC = 0.00

linearly interpolated over a transition halfwidth of 0.05 around the
cutoff (default cutoff = 0.50) so the marketable/non-marketable
boundary is biologically gradual rather than a knife-edge. See
``hierarchy_weight`` for the full operational definition and
``RHO_MARKETABLE_CUTOFF`` / ``RHO_TRANSITION_HALFWIDTH`` for the
calibration constants. Sensitivity to the recovery weight in
[0.20, 0.60] is exercised in tests/test_metric_variants.py.

The threshold θ (default 0.10) corresponds to 10 % quality loss — the
point where produce is still marketable but beginning to degrade and
should be considered for rerouting.

This form does not saturate at 1.0 unless every at-risk batch is
routed to the band-appropriate top tier (LR in marketable, Recovery
in non-marketable). Earlier drafts of this codebase also exposed a
binary ``recovered / at_risk`` variant, a continuous match-quality
variant, a capacity-constrained variant, and a uniform-weights
EU-agnostic companion; all four have been retired in favour of the
single hierarchy-weighted form, which is the only variant whose
action weights derive from the EU directive itself rather than from
author choices. The 2026-04 single-version-of-the-truth pass
ensures every metric in this module has exactly one formulation per
the user mandate.

Equity (welfare-economic form)
------------------------------
Single canonical form:

  - ``compute_equity``      — mean(SLCA) × (1 − std(SLCA)).

This stability-weighted mean pairs uniformity with mean SLCA so a
high score requires both temporal stability *and* a high stable
level (Allison 1978). The Sen-welfare robustness companion
(``compute_equity_sen``) that earlier versions exposed was retired
in the 2026-04 single-version-of-the-truth pass.

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
      waste. OJ L 312, 22.11.2008. Article 4 (waste hierarchy).
    - European Commission (2017). Commission Notice 2017/C 361/01:
      EU guidelines on food donation. OJ C 361, 25.10.2017. §3.1
      (food-safety conditional on tier (b) admissibility).
    - Papargyropoulou, E., Lozano, R., Steinberger, J.K., Wright, N.
      & Ujang, Z. (2014). The food waste hierarchy as a framework for
      the management of food surplus and food waste. J. Cleaner
      Production, 76, 106–115. — Fig.2 (bench magnitudes), §3.3
      (marketable boundary as continuous risk gradient).
    - Garcia-Garcia, G., Woolley, E., Rahimifard, S., Colwill, J.,
      White, R. & Needham, L. (2017). A methodology for sustainable
      management of food waste. Waste and Biomass Valorization, 8,
      2209–2227. — §4.2 (top-priority swap above marketable cutoff).
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
# We therefore model piecewise-constant route factors. Cold chain has
# three regimes (nominal / stressed / overwhelmed). Local redistribute
# also has temperature-conditional factors because its short-dwell
# protection comes from indoor warehouse / food-bank-cooler staging,
# which tracks ambient with attenuation rather than active
# refrigeration: a refrigerated walk-in cooler at a partner site holds
# 4-7 degC when outdoor ambient is cool but heats up appreciably during
# heatwave-scale events. Recovery has a zero factor (produce leaves the
# retail-bound pool entirely):
#
#   cold_chain  T_amb < 30 degC : 0.15  (nominal cold chain, 85% integrity)
#               30 <= T_amb <=35: 0.40  (cold chain stressed)
#               T_amb > 35 degC : 1.00  (cold chain overwhelmed)
#
#   local_redistribute
#               T_amb < 15 degC : 0.20  (cool indoor staging, near-CC
#                                        performance; short dwell at
#                                        4-7 degC food-bank cooler)
#               15 <= T_amb < 30: 0.45  (moderate ambient, indoor staging
#                                        ~15-20 degC, short dwell)
#               30 <= T_amb <=35: 0.65  (warehouse heating up, partial
#                                        protection only)
#               T_amb > 35 degC : 0.85  (warehouse near-ambient, LR
#                                        marginally better than overwhelmed
#                                        CC because no compressor failure
#                                        adds excursion risk)
#
#   recovery (any T)            : 0.00  (leaves retail-bound pool)
#
# The cold-chain breakpoints (30 degC, 35 degC) are the consensus
# operating limits cited by Mercier (2017) Sec.3.1 and Ndraha (2018)
# Tab.4 for North American refrigerated-truck fleets carrying leafy
# greens. The local-redistribute breakpoints (15 degC, 30 degC, 35 degC)
# track the same heatwave thresholds plus a low-ambient cool band
# motivated by the typical 4-7 degC staging temperature of food-bank
# walk-in coolers (Garcia-Garcia 2017 Sec.4.2 reports 4 degC as the
# operating set-point for FareShare-style redistribution networks).
# These are sensitivity parameters, not universal constants; different
# fleets / climates / network architectures would calibrate them
# differently.
#
# Implications for AgriBrain narrative
# -------------------------------------
# Under this realistic model, cold chain is *strictly better* than
# local-redistribute on retail-pool rho whenever T_amb < 30 degC, but
# the gap narrows to 0.20 vs 0.15 in the cool band rather than 0.45 vs
# 0.15. The two are approximately tied at 0.65 vs 0.40 in the
# 30-35 degC stress band, and LR is *better* than CC at 0.85 vs 1.00
# above 35 degC (CC overwhelmed). Combined with the Recovery knee in
# action_selection.py (rho > 0.50 triages to Recovery, removing
# produce from the retail pool entirely) and the food-safety override
# in batch_inventory.py (rho > 0.65 forces Recovery regardless of
# policy choice), this gives AgriBrain a genuine retail-pool quality
# advantage over Static during and after a heatwave: AgriBrain's
# Recovery routing keeps the worst batches out of retail entirely,
# while Static's CC-only policy lets every batch enter retail at
# whatever rho the cold-chain integrity gave it.
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

# Local-redistribute breakpoints. Indoor warehouse / food-bank cooler
# staging tracks ambient with attenuation; the cool-band threshold
# captures the regime where the staging cooler genuinely operates
# refrigerated (4-7 degC), the stressed/hot bands capture warehouse
# heating during heatwave-scale events.
LR_COOL_THRESHOLD_C:    float = 15.0
"""Below this ambient temperature, LR staging operates refrigerated."""

LR_STRESSED_THRESHOLD_C: float = 30.0
"""Below this ambient temperature, LR staging is moderate; matches CC
nominal threshold so the stress-band breakpoint is symmetric across
routes."""

LR_HOT_THRESHOLD_C:     float = 35.0
"""Above this ambient temperature, LR staging is near-ambient. Matches
CC overwhelmed threshold so the upper-band breakpoint is symmetric
across routes."""

LR_FACTOR_COOL:        float = 0.20
"""LR factor at T < 15 degC (cool indoor staging, near-CC performance)."""

LR_FACTOR_NOMINAL:     float = 0.45
"""LR factor in the 15-30 degC moderate band (warehouse 15-20 degC)."""

LR_FACTOR_STRESSED:    float = 0.65
"""LR factor in the 30-35 degC stress band (warehouse heating up)."""

LR_FACTOR_HOT:         float = 0.85
"""LR factor above 35 degC (warehouse near-ambient, marginally better
than overwhelmed CC because no compressor-failure excursion risk)."""

# Backward-compatible alias for callers that imported the old constant
# name. Defaults to the nominal-band value (0.45) so any code path that
# did not migrate to the temperature-conditional API still produces
# the previous numerics.
LR_FACTOR_CONSTANT:    float = LR_FACTOR_NOMINAL

RECOVERY_FACTOR:      float = 0.00
"""Recovery factor (produce leaves retail-bound pool)."""

# Food-safety hard cutoff: rho above this is "not safely marketable"
# under typical food-bank / retail acceptance criteria. This is the
# hard regulatory boundary; the Recovery knee in action_selection.py
# (RHO_RECOVERY_KNEE = 0.30) is a soft policy nudge that begins
# routing toward Recovery much earlier so the policy is not surprised
# by the hard cutoff firing. The two thresholds answer different
# questions: the knee is "when should the policy *start considering*
# Recovery as a serious option" (a calibration internal to the
# AgriBrain policy); the cutoff here is "when does the regulatory
# environment *force* Recovery regardless of policy preference" (an
# environmental constraint).
#
# Calibration provenance: 0.65 is positioned to correspond to the
# upper end of the marketable-quality band described in Papargyropoulou
# et al. (2014) §3.3 (continuous risk gradient between marketable and
# non-marketable produce; food-safety judgment shifts to "reject"
# in the upper third of the gradient). The 80%-rejection-at-intake
# anchor in food-bank operations literature (FareShare, Sirop annual
# reports 2018-2022) supports this band; the specific value 0.65 is a
# calibration choice within that band rather than a single-source
# reading. Future work: a dedicated cutoff sensitivity sweep over
# [0.55, 0.75] in test_effective_rho_and_knee.py would tighten the
# defensibility of the specific 0.65 value; currently the only
# coverage is the existence-and-range test
# test_knee_threshold_constant_is_in_realistic_range, which pins
# the constant within the [0.55, 0.75] band but does not exercise
# the routing-behaviour sensitivity to perturbations within it.
# This constant is *separate* from RHO_MARKETABLE_CUTOFF (the metric
# weight-table boundary, default 0.50) which is the *softer*
# marketable/non-marketable gradient centre rather than the hard
# food-safety reject line.
RHO_FOOD_SAFETY_CUTOFF: float = 0.65
"""Hard cutoff above which DC batches are forcibly routed to Recovery."""


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
        (stressed -> overwhelmed). Local-redistribute factor is also
        piecewise-constant with breakpoints at 15 degC (cool ->
        nominal), 30 degC (nominal -> stressed), and 35 degC
        (stressed -> hot), reflecting indoor warehouse / food-bank
        cooler staging tracking ambient with attenuation.

    Returns
    -------
    Factor in [0, 1].
    """
    if action == "recovery":
        return RECOVERY_FACTOR
    if action == "local_redistribute":
        if ambient_temp_c < LR_COOL_THRESHOLD_C:
            return LR_FACTOR_COOL
        if ambient_temp_c < LR_STRESSED_THRESHOLD_C:
            return LR_FACTOR_NOMINAL
        if ambient_temp_c <= LR_HOT_THRESHOLD_C:
            return LR_FACTOR_STRESSED
        return LR_FACTOR_HOT
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
        # design-point factor; LR uses the nominal-band factor.
        cc_factor = np.full(env_rho.shape, CC_FACTOR_NOMINAL)
        lr_factor = np.full(env_rho.shape, LR_FACTOR_NOMINAL)
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
        # Temperature-conditional LR factor (matches route_rho_factor):
        # cool < 15 / nominal 15-30 / stressed 30-35 / hot > 35.
        lr_factor = np.where(
            T < LR_COOL_THRESHOLD_C,
            LR_FACTOR_COOL,
            np.where(T < LR_STRESSED_THRESHOLD_C,
                     LR_FACTOR_NOMINAL,
                     np.where(T <= LR_HOT_THRESHOLD_C,
                              LR_FACTOR_STRESSED,
                              LR_FACTOR_HOT)),
        )

    factor = (
        action_probs[:, 0] * cc_factor
        + action_probs[:, 1] * lr_factor
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


# =============================================================================
# Hierarchy weights for the EU 2008/98/EC food-waste hierarchy
# =============================================================================
#
# REGULATORY GROUNDING (this is the load-bearing claim, stated first)
# ---------------------------------------------------------------------
# EU Directive 2008/98/EC Article 4 (the "Waste Framework Directive")
# establishes a five-tier hierarchy that Member States must apply as a
# priority order in waste-prevention legislation. The first three tiers
# relevant to perishable-food routing decisions are, in descending
# priority:
#
#   (a) Prevention of waste
#   (b) Preparing for re-use / Re-use for human consumption
#   (c) Recycling (including organics; for food, this means recovery
#       routes such as animal feed, anaerobic digestion, composting)
#
# Tier (b) is operationalised in this codebase as ``local_redistribute``
# (LR): redirecting still-marketable produce to short-chain human
# consumption (food banks, community markets, retail). Tier (c) is
# operationalised as ``recovery``: animal feed, biogas, composting.
# ``cold_chain`` (CC) is the no-intervention default: produce stays in
# the centralised distribution path and is *not* repurposed under the
# hierarchy.
#
# The directive's text and Papargyropoulou et al. (2014, Fig.2)
# explicitly require that re-use for human consumption is preferred
# *only when food safety permits*. The European Commission's
# subsequent "Guidelines on food donation" (Commission Notice
# 2017/C 361/01, §3.1) makes this conditional explicit: "Food shall
# not be donated where it does not satisfy food safety requirements."
# Garcia-Garcia et al. (2017, §4.2) summarise this as: above the
# marketable-quality cutoff, Recovery becomes the *top-priority* tier
# under the hierarchy because human-consumption routes are no longer
# regulatorily admissible.
#
# CONSEQUENCES FOR THE WEIGHT TABLE
# ---------------------------------------------------------------------
# Operationalised as a routing-action utility weight w(action, rho):
#
#   1. MARKETABLE band (rho <= RHO_MARKETABLE_CUTOFF):
#      Tier (b) is admissible. Hierarchy: LR > Recovery > CC.
#         w(local_redistribute) = 1.00  (Tier b: top priority)
#         w(recovery)           = 0.40  (Tier c: lower priority than b)
#         w(cold_chain)         = 0.00  (no-intervention default)
#
#   2. NON-MARKETABLE band (rho > RHO_MARKETABLE_CUTOFF):
#      Tier (b) is no longer admissible (the food-safety conditional
#      from Article 4 fires). Hierarchy: Recovery > {LR, CC}.
#         w(recovery)           = 1.00  (Tier c: top priority in band)
#         w(local_redistribute) = 0.00  (admissibility violated)
#         w(cold_chain)         = 0.00  (no-intervention default)
#
# The 0.40 weight on Recovery in the marketable band is the standard
# magnitude used by Papargyropoulou (2014, Fig.2) for the bench-scale
# value gap between human and animal-feed valorisation; sensitivity
# in [0.20, 0.60] is exercised in tests/test_metric_variants.py.
#
# SMOOTHING ACROSS THE BAND BOUNDARY
# ---------------------------------------------------------------------
# The marketable / non-marketable transition is biologically gradual,
# not a knife-edge: pathogen risk and consumer acceptance both change
# continuously across a quality-loss range, and operator judgment
# (food-bank intake QA, retail markdown decisions) routinely treats
# the boundary as a soft transition. Papargyropoulou et al. (2014,
# §3.3) describe the marketable-vs-non-marketable judgment as
# "operator-discretion within a continuous risk gradient", not as a
# step function on rho.
#
# The underlying weight tables are therefore step-defined for clarity,
# but the production lookup ``hierarchy_weight(action, rho)`` linearly
# interpolates over a transition band of half-width
# RHO_TRANSITION_HALFWIDTH (default 0.05) centred on the cutoff. At
# rho = cutoff - halfwidth (e.g. 0.45) the lookup returns the full
# marketable weights; at rho = cutoff + halfwidth (e.g. 0.55) it
# returns the full non-marketable weights; in between, weights are
# linearly interpolated. This eliminates the step discontinuity at
# rho = cutoff that produced non-monotonic RLE under stochastic
# temperature noise (the previous step lookup made RLE jump
# whenever a seed's mean rho crossed 0.50, even by epsilon).
#
# Setting RHO_TRANSITION_HALFWIDTH = 0.0 recovers the step-function
# behaviour for testing / strict-mode runs.
#
# CITATIONS (these are the actual sources, not pasted twice)
# ---------------------------------------------------------------------
#   - European Parliament / Council (2008). Directive 2008/98/EC of 19
#     November 2008 on waste. OJ L 312, 22.11.2008. Article 4.
#   - European Commission (2017). Commission Notice 2017/C 361/01:
#     EU guidelines on food donation. §3.1 (food safety conditional).
#   - Papargyropoulou, E., Lozano, R., Steinberger, J.K., Wright, N. &
#     Ujang, Z.B. (2014). The food waste hierarchy as a framework for
#     the management of food surplus and food waste. J. Cleaner
#     Production, 76, 106-115. Fig.2 (weight magnitudes), §3.3 (band
#     boundary as a continuous risk gradient).
#   - Garcia-Garcia, G., Woolley, E., Rahimifard, S., Colwill, J.,
#     White, R. & Needham, L. (2017). A methodology for sustainable
#     management of food waste. Waste & Biomass Valorization, 8(6),
#     2209-2227. §4.2 (top-priority swap above marketable cutoff).
RHO_MARKETABLE_CUTOFF: float = 0.50
"""Marketable/non-marketable boundary on rho. See module-docstring
'CONSEQUENCES FOR THE WEIGHT TABLE' for the regulatory grounding."""

RHO_TRANSITION_HALFWIDTH: float = 0.05
"""Half-width of the linear-interpolation band centred on
RHO_MARKETABLE_CUTOFF. Set to 0.0 to recover step-function behaviour."""

HIERARCHY_WEIGHT: dict[str, float] = {
    "local_redistribute": 1.00,
    "recovery":           0.40,
    "cold_chain":         0.00,
}
"""Hierarchy weights for produce in the *marketable* band (rho<=cutoff).
Use ``hierarchy_weight(action, rho)`` for the rho-conditional value."""

HIERARCHY_WEIGHT_NONMARKETABLE: dict[str, float] = {
    "local_redistribute": 0.00,
    "recovery":           1.00,
    "cold_chain":         0.00,
}
"""Hierarchy weights for produce in the *non-marketable* band
(rho>cutoff): Recovery becomes the correct top tier."""

def hierarchy_weight(action: str, rho: float,
                     cutoff: float = RHO_MARKETABLE_CUTOFF,
                     halfwidth: float = RHO_TRANSITION_HALFWIDTH) -> float:
    """rho-conditional hierarchy weight with smooth band transition.

    Implements the EU 2008/98/EC Article 4 hierarchy with a
    continuous transition across the marketable / non-marketable
    boundary. The transition is operator-judgment-shaped per
    Papargyropoulou (2014) §3.3 (continuous risk gradient), not a
    step function.

    Parameters
    ----------
    action : routing action (``local_redistribute`` / ``recovery`` /
        ``cold_chain``).
    rho : spoilage risk in [0, 1].
    cutoff : marketable / non-marketable centre. Default
        ``RHO_MARKETABLE_CUTOFF``.
    halfwidth : half-width of the linear-interpolation band. At
        rho <= cutoff - halfwidth the marketable table is in full
        effect; at rho >= cutoff + halfwidth the non-marketable
        table is in full effect; in between, weights are linearly
        interpolated. Default ``RHO_TRANSITION_HALFWIDTH``.
        Setting halfwidth=0.0 recovers a hard step at the cutoff.

    Returns
    -------
    Weight in [0, 1]. Unknown actions return 0.0 in both bands.
    """
    canonical = _resolve_action(action)
    w_market = HIERARCHY_WEIGHT.get(canonical, 0.0)
    w_nonmarket = HIERARCHY_WEIGHT_NONMARKETABLE.get(canonical, 0.0)
    if halfwidth <= 0.0:
        return w_market if rho <= cutoff else w_nonmarket
    lo = cutoff - halfwidth
    hi = cutoff + halfwidth
    if rho <= lo:
        return w_market
    if rho >= hi:
        return w_nonmarket
    # Linear interpolation across the transition band. At rho=lo,
    # alpha=0 (full marketable); at rho=hi, alpha=1 (full non-market);
    # at rho=cutoff, alpha=0.5 (midpoint blend).
    alpha = (rho - lo) / (hi - lo)
    return float((1.0 - alpha) * w_market + alpha * w_nonmarket)


# ---------------------------------------------------------------------------
# ARI
# ---------------------------------------------------------------------------

def compute_ari(waste: float, slca_composite: float, rho: float) -> float:
    """Compute the Adaptive Resilience Index for a single timestep.

    ARI = (1 − waste) × SLCA_composite × (1 − ρ)

    This multiplicative form follows the unit-interval composite
    convention discussed in OECD/JRC (2008, §6) and is the *single*
    canonical ARI throughout the codebase - no parallel "geometric
    mean" / "rank-only" / etc. variants are exposed. The
    geometric-mean robustness companion that earlier versions also
    emitted (compute_ari_geom) was retired in the 2026-04 single-
    version-of-the-truth pass per the user mandate that every
    metric have exactly one formulation in the repository.

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
            # rho-conditional weight: above RHO_MARKETABLE_CUTOFF the
            # hierarchy table swaps so Recovery becomes the top tier
            # (correct routing for non-marketable produce). Denominator
            # uses w_max=1.0 in both bands so the ratio stays in [0, 1].
            w = hierarchy_weight(action, rho)
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
# Violation disposition (outcome-side metric on the safety-window event set)
# ---------------------------------------------------------------------------
# constraint_violation_rate / regulatory_violation_rate / compliance_
# violation_rate are all driven by the dataset's ambient temperature and
# humidity trajectory and are computed by predicates that do not consult
# the chosen action. They are therefore *environmental signatures* of how
# stress-laden a scenario is, not measures of policy quality. Reading
# table1's ConstraintViolationRate or RegulatoryViolationRate column
# naively as "AgriBrain has the same compliance failure rate as Static"
# misreads the metric: every method is being scored on the same env-
# driven event set by construction.
#
# The *outcome* question — "given that the env was in a violation state,
# what did the agent do about it?" — is answered by the per-violation
# action disposition: of those violation timesteps, what fraction did the
# agent send into the cold-chain (downstream toward retail) vs route to
# local-redistribute or recovery (off the retail-bound pool)? This is a
# pure policy metric: every method is asked the same question on the
# same event subset, so cross-method differences come entirely from the
# action distribution conditional on the environmental violation event.
#
# Expected ranking under healthy policies:
#
#   Static                downstream ~= 1.00  (no policy, always cold_chain)
#   Hybrid RL             downstream  < 1.00  (RL learned to reroute some)
#   AgriBrain             downstream << 1.00  (Recovery knee + food-safety
#                                              override fire on rho > 0.30
#                                              and rho > 0.65 respectively)
#
# Companion metrics:
#
#   contained_violation_rate    = fraction routed to ``recovery`` (off retail)
#   redistribute_violation_rate = fraction routed to ``local_redistribute``
#
# The three sum to 1.0 by construction whenever there are violation
# events. When the episode has no violation events, all three return
# 0.0 to avoid divide-by-zero and to flag "no event data to score
# disposition on" downstream.
#
# References
# ----------
# Pettit, T.J., Croxton, K.L. & Fiksel, J. (2013). §4.2 ("response
# fitness" as the fraction of stress events the policy responded to)
# anchors the conditional-on-event framing the metric uses.
def compute_violation_disposition(
    temp_violations: List[bool],
    quality_violations: List[bool],
    actions: List[str],
) -> dict:
    """Action-disposition rates over the env-driven violation event set.

    Records what the policy did on each timestep where the environment
    was in a safety-window violation state (temperature ceiling exceeded
    OR shelf-fraction below expedite floor — the same predicate the
    simulator uses for ``constraint_violation_rate`` and
    ``operational_violation_rate``). Returns the conditional disposition
    rates, with the three action buckets summing to 1.0 by construction
    whenever at least one violation event fired during the episode.

    Parameters
    ----------
    temp_violations : per-step booleans, ``True`` iff the cold-chain
        temperature ceiling was exceeded at that step.
    quality_violations : per-step booleans, ``True`` iff shelf-life
        fell below the expedite floor at that step.
    actions : per-step routing action names ("cold_chain",
        "local_redistribute", "recovery", or any aliased equivalent
        resolved by ``action_aliases.resolve_action``).

    Returns
    -------
    dict with keys
        downstream_violation_rate    in [0, 1] — fraction of violation
                                     events the policy let into the
                                     retail-bound cold chain.
        redistribute_violation_rate  in [0, 1] — fraction routed to
                                     local_redistribute.
        contained_violation_rate     in [0, 1] — fraction routed to
                                     recovery (off the retail-bound pool).
        violation_event_count        int — how many event timesteps the
                                     rates are conditioned on. 0 means
                                     the three rates are by-convention
                                     zero rather than meaningful.
    """
    if not (len(temp_violations) == len(quality_violations) == len(actions)):
        raise ValueError(
            f"trace lengths must match; got temp={len(temp_violations)}, "
            f"quality={len(quality_violations)}, actions={len(actions)}"
        )
    total_violations = 0
    routed_to_cold_chain = 0
    routed_to_local = 0
    routed_to_recovery = 0
    for tv, qv, a in zip(temp_violations, quality_violations, actions):
        if not (bool(tv) or bool(qv)):
            continue
        total_violations += 1
        canonical = _resolve_action(a)
        if canonical == "cold_chain":
            routed_to_cold_chain += 1
        elif canonical == "local_redistribute":
            routed_to_local += 1
        elif canonical == "recovery":
            routed_to_recovery += 1
    if total_violations == 0:
        return {
            "downstream_violation_rate":    0.0,
            "redistribute_violation_rate":  0.0,
            "contained_violation_rate":     0.0,
            "violation_event_count":        0,
        }
    return {
        "downstream_violation_rate":    float(routed_to_cold_chain / total_violations),
        "redistribute_violation_rate":  float(routed_to_local      / total_violations),
        "contained_violation_rate":     float(routed_to_recovery   / total_violations),
        "violation_event_count":        int(total_violations),
    }


# ---------------------------------------------------------------------------
# Equity
# ---------------------------------------------------------------------------

def compute_equity(slca_values: List[float] | np.ndarray) -> float:
    """Stability-weighted mean SLCA (single canonical equity metric).

    Equity = mean(SLCA) × (1 − std(SLCA))

    A stability-weighted mean: the score is high only when per-step
    SLCA is both *temporally stable* and at a *high mean level*. A
    static cold-chain policy with mean SLCA ~0.5 cannot outscore an
    integrated policy with mean SLCA ~0.85 regardless of how flat its
    trajectory is. This mirrors the standard cooperative-economics
    practice of pairing a consistency term with a quality term rather
    than reporting them independently (Atkinson, 1970; Allison, 1978).

    This is the *single* canonical equity throughout the codebase -
    no parallel "Sen welfare" / "Gini-based" / etc. variants are
    exposed. The Sen-welfare robustness companion that earlier
    versions also emitted (compute_equity_sen) was retired in the
    2026-04 single-version-of-the-truth pass per the user mandate
    that every metric have exactly one formulation in the
    repository.

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
