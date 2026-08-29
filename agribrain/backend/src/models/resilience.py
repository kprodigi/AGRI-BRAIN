"""
Synthetic resilience metrics: ARI, severity-weighted RLE, and a temporal social-performance stability proxy.

All three are author-defined simulation metrics; they are not
standardized or externally validated indices. This module exposes one
canonical form of each metric so the code, tables, figures, and manuscript use
the same definitions.

Adaptive Resilience Index (ARI)
-------------------------------
Author-defined multiplicative composite of three unit-interval dimensions:

    ARI = (1 − waste) × social_proxy × (1 − ρ)

where:
    (1 − waste)         = operational stability for the simulated routing
                          opportunity
    social_proxy        = the study's stylized sustainability/social proxy
    (1 − ρ)             = freshness under the common latent environmental
                          temperature-humidity exposure

Each factor is in [0, 1], producing ARI ∈ [0, 1].

On the perceived ρ-vs-waste redundancy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A potential concern is that ``(1 − waste)`` and ``(1 − ρ)``
appear to double-count spoilage. They do not, because they measure
different physical properties of the supply-chain outcome:

- ``waste`` is an action-dependent *flow proxy*: the modelled fraction lost
  at one standardized routing opportunity after the action-specific save
  factor.
- ``ρ`` is an action-independent *state*: cumulative mechanistic quality
  erosion under the latent environmental temperature-humidity trajectory.
  It is identical across paired methods for a given seed, scenario, and
  retained episode.

The two factors are correlated (both rise under heat stress) but
not redundant. An action may reduce the modelled loss fraction at an
opportunity while the common environmental trajectory still implies
substantial accumulated quality erosion. The multiplicative form encodes the
study's declared requirement that operational stability, proxy social
performance, and environmental freshness all be high. It does not claim a
validated product mass balance or a policy-induced change in mechanistic rho.

Higher ARI indicates lower simulated waste, higher proxy social performance,
and higher modelled freshness under this definition.

Severity-weighted reverse-logistics score (RLE)
------------------------------------------------
RLE is an author-defined unit-interval score inspired by the qualitative waste
hierarchy in EU 2008/98/EC Article 4. The directive does not prescribe this
formula, the action mapping, the numerical weights, or cutoffs on this study's
modelled-risk scale. The cutoff, weights, and smoothing below are transparent
synthetic-case assumptions; they do not classify real product safety or legal
eligibility.

The metric:

    RLE = Σ_t [ρ(t) · w(a_t, ρ(t)) · 1[ρ(t) > θ]] /
          Σ_t [ρ(t) · w_max · 1[ρ(t) > θ]]

with the ρ-conditional weight table

    ρ ≤ cutoff    (lower-risk band):  w_LR = 1.00, w_Rec = 0.40, w_CC = 0.00
    ρ > cutoff    (higher-risk band): w_LR = 0.00, w_Rec = 1.00, w_CC = 0.00

linearly interpolated over a transition halfwidth of 0.05 around the
cutoff (default cutoff = 0.50) so the transition between the two
author-defined weight tables is gradual rather than a knife-edge. See
``hierarchy_weight`` for the full operational definition and
``RHO_ACTION_WEIGHT_CUTOFF`` / ``RHO_TRANSITION_HALFWIDTH`` for the
benchmark constants. Sensitivity to the recovery weight in
[0.20, 0.60] is exercised in tests/test_metric_variants.py.

The threshold θ = 0.10 is a declared benchmark trigger for the modelled-risk
scale; it is not a measured marketability threshold.

This form does not saturate at 1.0 unless every at-risk batch receives the
top-weighted action for its band (LR in the lower-risk band, Recovery in the
higher-risk band). Earlier drafts of this codebase also exposed a
binary ``recovered / at_risk`` variant, a continuous match-quality
variant, a capacity-constrained variant, and a uniform-weights
EU-agnostic companion; all four have been retired in favour of the
single hierarchy-weighted form. Its numerical action weights remain author
choices. The 2026-04 single-version-of-the-truth pass
ensures every metric in this module has exactly one formulation per
the user mandate.

Temporal social-performance stability proxy
-------------------------------------------
Single canonical form:

  - ``compute_equity``      — mean(SLCA) × (1 − std(SLCA)).

This author-defined stability-weighted mean pairs temporal uniformity with the
mean proxy score. It is not a demographic or distributional equity measure.

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
      Production, 76, 106–115. — qualitative hierarchy motivation only.
    - Garcia-Garcia, G., Woolley, E., Rahimifard, S., Colwill, J.,
      White, R. & Needham, L. (2017). A methodology for sustainable
      management of food waste. Waste and Biomass Valorization, 8,
      2209–2227. — qualitative food-waste management context only.
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

The 0.10 value is a declared benchmark trigger on the modelled-risk scale, not
an empirical marketability threshold.
"""


# ---------------------------------------------------------------------------
# Exploratory legacy route-conditioned thermal exposure model
# ---------------------------------------------------------------------------
# The confirmatory publication pipeline does not call the functions or
# constants in this section. Its ARI, RLE, reward, and operating-envelope
# outcomes use the common latent environmental rho directly. The API remains
# for backward compatibility and explicitly labelled exploratory tests; its
# route factors, turnover half-life, and disposition cutoff are unvalidated
# synthetic assumptions and must not be used as confirmatory evidence.
# env_rho is the Arrhenius-derived rho computed from the synthetic
# temperature trace (compute_spoilage in spoilage.py uses the dataframe's
# ``tempC`` field). This exploratory helper applies declared route multipliers
# to that common benchmark quantity; it does not simulate vehicle or facility
# temperatures.
#
# Reviews of cold-chain temperature management motivate modelling stronger
# exposure under adverse thermal conditions. They do not supply the numerical
# factors or breakpoints below; those values are author-declared synthetic-case
# assumptions and have not been calibrated to a specific fleet.
#
# We therefore expose piecewise-constant author-declared route factors for
# exploratory sensitivity work. The regime names are benchmark labels, not
# measured truck or facility states. Recovery has a zero factor because that
# action leaves the modeled retail-bound pool:
#
#   cold_chain  T_amb < 30 degC : 0.15  (nominal benchmark band)
#               30 <= T_amb <=35: 0.40  (cold chain stressed)
#               T_amb > 35 degC : 1.00  (cold chain overwhelmed)
#
#   local_redistribute
#               T_amb < 15 degC : 0.20  (cool benchmark band)
#               15 <= T_amb < 30: 0.45  (nominal benchmark band)
#               30 <= T_amb <=35: 0.65  (stressed benchmark band)
#               T_amb > 35 degC : 0.85  (hot benchmark band)
#
#   recovery (any T)            : 0.00  (leaves retail-bound pool)
#
# The cold-chain and redistribution breakpoints are scenario parameters, not
# consensus operating limits. They require fleet- and facility-specific
# calibration before deployment.
#
# Implications within this synthetic model
# -------------------------------------
# Under this stylized model, cold chain is assigned less exposure than
# local-redistribute on retail-pool rho whenever T_amb < 30 degC, but
# the gap narrows to 0.20 vs 0.15 in the cool band rather than 0.45 vs
# 0.15. The two are approximately tied at 0.65 vs 0.40 in the
# 30-35 degC stress band, and LR receives the smaller factor above 35 degC.
# These assigned factors, together with the declared recovery rules, can create
# method differences in modelled retail-pool quality. Results must therefore be
# interpreted as conditional on this synthetic route-exposure model.
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
"""Declared lower breakpoint for the synthetic route-exposure model."""

CC_OVERWHELMED_THRESHOLD_C: float = 35.0
"""Declared upper breakpoint for the synthetic route-exposure model."""

CC_FACTOR_NOMINAL:    float = 0.15
"""Cold-chain ambient-exposure fraction at T < CC_NOMINAL_THRESHOLD_C."""

CC_FACTOR_STRESSED:   float = 0.40
"""Cold-chain factor in the 30-35 degC stress band."""

CC_FACTOR_OVERWHELMED: float = 1.00
"""Cold-chain factor above CC_OVERWHELMED_THRESHOLD_C."""

# Local-redistribute breakpoints for the exploratory synthetic route model.
LR_COOL_THRESHOLD_C:    float = 15.0
"""Lower breakpoint for the exploratory LR route-factor table."""

LR_STRESSED_THRESHOLD_C: float = 30.0
"""Middle breakpoint for the exploratory LR route-factor table."""

LR_HOT_THRESHOLD_C:     float = 35.0
"""Upper breakpoint for the exploratory LR route-factor table."""

LR_FACTOR_COOL:        float = 0.20
"""LR factor in the declared T < 15 degC benchmark band."""

LR_FACTOR_NOMINAL:     float = 0.45
"""LR factor in the declared 15-30 degC benchmark band."""

LR_FACTOR_STRESSED:    float = 0.65
"""LR factor in the declared 30-35 degC benchmark band."""

LR_FACTOR_HOT:         float = 0.85
"""LR factor in the declared above-35 degC benchmark band."""

# Backward-compatible alias for callers that imported the old constant
# name. Defaults to the nominal-band value (0.45) so any code path that
# did not migrate to the temperature-conditional API still produces
# the previous numerics.
LR_FACTOR_CONSTANT:    float = LR_FACTOR_NOMINAL

RECOVERY_FACTOR:      float = 0.00
"""Recovery factor (produce leaves retail-bound pool)."""

# Synthetic disposition cutoff on the modelled-risk scale. It is applied
# uniformly to every mode as a benchmark constraint. No cited regulation or
# field study defines rho or establishes 0.65 as a food-safety boundary; a real
# deployment would replace this rule with product-specific inspection and
# regulatory criteria.
RHO_DISPOSITION_CUTOFF: float = 0.65
"""Declared benchmark cutoff above which batches are routed to recovery."""

# Backward-compatible import name. The value is not a food-safety threshold.
RHO_FOOD_SAFETY_CUTOFF: float = RHO_DISPOSITION_CUTOFF


def route_rho_factor(action: str, ambient_temp_c: float) -> float:
    """Exploratory temperature-conditional route thermal-exposure factor.

    Returns the per-step fraction of ``env_rho`` that a batch in
    transit on the named route accumulates at the supplied ambient
    temperature. See module-level documentation for the declared synthetic
    assumptions and qualitative literature motivation.

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
        (stressed -> hot).

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

# DC ambient-coupling factor for the synthetic case. The literature motivates
# accounting for incomplete temperature control, but 0.20 is not extracted from
# or validated by a specific field dataset.
DC_RHO_FACTOR: float = 0.20


def compute_effective_rho(
    env_rho: np.ndarray,
    action_probs: np.ndarray,
    turnover_halflife_hours: float = 12.0,
    dt_hours: float = 0.25,
    ambient_temp_c: np.ndarray | None = None,
) -> np.ndarray:
    """Compute an exploratory policy-responsive retail-pool rho proxy.

    This helper is excluded from the confirmatory benchmark and publication
    artifacts. It is retained only for backward compatibility and sensitivity
    experiments because its route factors and turnover recurrence have not
    been calibrated to a measured shipment-flow dataset.

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
        The default 12 h is a synthetic-case inventory-turnover assumption.
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
# QUALITATIVE POLICY MOTIVATION AND SYNTHETIC OPERATIONALIZATION
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
# Tier (b) motivates ``local_redistribute`` (LR): redirecting product to a
# short-chain human-consumption route in the synthetic benchmark. Tier (c) is
# operationalised as ``recovery``: animal feed, biogas, composting.
# ``cold_chain`` (CC) is the no-intervention default: produce stays in
# the centralised distribution path and is *not* repurposed under the
# hierarchy.
#
# Food-donation guidance requires food safety for human-consumption routes.
# Neither that guidance nor the cited academic literature defines this model's
# rho variable, a rho cutoff, or the numerical utilities below.
#
# CONSEQUENCES FOR THE WEIGHT TABLE
# ---------------------------------------------------------------------
# Operationalised as a routing-action utility weight w(action, rho):
#
#   1. LOWER-RISK band (rho <= RHO_ACTION_WEIGHT_CUTOFF):
#      The declared ordering is LR > Recovery > CC.
#         w(local_redistribute) = 1.00  (Tier b: top priority)
#         w(recovery)           = 0.40  (Tier c: lower priority than b)
#         w(cold_chain)         = 0.00  (no-intervention default)
#
#   2. HIGHER-RISK band (rho > RHO_ACTION_WEIGHT_CUTOFF):
#      The synthetic score assigns no utility to redistribution in this band.
#      This is not a food-safety or legal determination. Ordering: Recovery >
#      {LR, CC}.
#         w(recovery)           = 1.00  (Tier c: top priority in band)
#         w(local_redistribute) = 0.00  (author-assigned zero in this band)
#         w(cold_chain)         = 0.00  (no-intervention default)
#
# The 0.40 recovery weight is author-specified. Sensitivity in [0.20, 0.60]
# is exercised in tests/test_metric_variants.py.
#
# SMOOTHING ACROSS THE BAND BOUNDARY
# ---------------------------------------------------------------------
# Linear smoothing is an author-specified numerical choice that avoids a
# discontinuity in the synthetic score. It is not a biological or regulatory
# relationship validated on the rho scale.
#
# The underlying weight tables are therefore step-defined for clarity,
# but the production lookup ``hierarchy_weight(action, rho)`` linearly
# interpolates over a transition band of half-width
# RHO_TRANSITION_HALFWIDTH (default 0.05) centred on the cutoff. At
# rho = cutoff - halfwidth (e.g. 0.45) the lookup returns the full
# lower-risk weights; at rho = cutoff + halfwidth (e.g. 0.55) it
# returns the full higher-risk weights; in between, weights are
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
#     Production, 76, 106-115. Qualitative hierarchy motivation only.
#   - Garcia-Garcia, G., Woolley, E., Rahimifard, S., Colwill, J.,
#     White, R. & Needham, L. (2017). A methodology for sustainable
#     management of food waste. Waste & Biomass Valorization, 8(6),
#     2209-2227. Qualitative food-waste management context only.
RHO_ACTION_WEIGHT_CUTOFF: float = 0.50
"""Declared centre of the synthetic RLE action-weight transition."""

# Backward-compatible import name retained for archived analysis scripts. It
# does not classify real product marketability.
RHO_MARKETABLE_CUTOFF: float = RHO_ACTION_WEIGHT_CUTOFF

RHO_TRANSITION_HALFWIDTH: float = 0.05
"""Half-width of the linear-interpolation band centred on
RHO_ACTION_WEIGHT_CUTOFF. Set to 0.0 to recover step-function behaviour."""

HIERARCHY_WEIGHT_LOW_RISK: dict[str, float] = {
    "local_redistribute": 1.00,
    "recovery":           0.40,
    "cold_chain":         0.00,
}
"""Author-specified weights in the lower-risk band (rho<=cutoff).
Use ``hierarchy_weight(action, rho)`` for the rho-conditional value."""

# Backward-compatible name for the lower-risk table.
HIERARCHY_WEIGHT = HIERARCHY_WEIGHT_LOW_RISK

HIERARCHY_WEIGHT_HIGH_RISK: dict[str, float] = {
    "local_redistribute": 0.00,
    "recovery":           1.00,
    "cold_chain":         0.00,
}
"""Author-specified weights in the higher-risk band (rho>cutoff)."""

# Backward-compatible name retained for archived imports. The table is not a
# real marketability classification.
HIERARCHY_WEIGHT_NONMARKETABLE = HIERARCHY_WEIGHT_HIGH_RISK

def hierarchy_weight(action: str, rho: float,
                     cutoff: float = RHO_ACTION_WEIGHT_CUTOFF,
                     halfwidth: float = RHO_TRANSITION_HALFWIDTH) -> float:
    """rho-conditional hierarchy weight with smooth band transition.

    Implements the study's synthetic action weights with a continuous
    transition. EU 2008/98/EC motivates the qualitative hierarchy only; it does
    not prescribe these weights or the rho transition.

    Parameters
    ----------
    action : routing action (``local_redistribute`` / ``recovery`` /
        ``cold_chain``).
    rho : spoilage risk in [0, 1].
    cutoff : centre of the lower-risk / higher-risk action-weight transition.
        Default ``RHO_ACTION_WEIGHT_CUTOFF``.
    halfwidth : half-width of the linear-interpolation band. At
        rho <= cutoff - halfwidth the lower-risk table is in full
        effect; at rho >= cutoff + halfwidth the higher-risk
        table is in full effect; in between, weights are linearly
        interpolated. Default ``RHO_TRANSITION_HALFWIDTH``.
        Setting halfwidth=0.0 recovers a hard step at the cutoff.

    Returns
    -------
    Weight in [0, 1]. Unknown actions return 0.0 in both bands.
    """
    canonical = _resolve_action(action)
    w_low = HIERARCHY_WEIGHT_LOW_RISK.get(canonical, 0.0)
    w_high = HIERARCHY_WEIGHT_HIGH_RISK.get(canonical, 0.0)
    if halfwidth <= 0.0:
        return w_low if rho <= cutoff else w_high
    lo = cutoff - halfwidth
    hi = cutoff + halfwidth
    if rho <= lo:
        return w_low
    if rho >= hi:
        return w_high
    # Linear interpolation across the transition band. At rho=lo,
    # alpha=0 (full lower-risk); at rho=hi, alpha=1 (full higher-risk);
    # at rho=cutoff, alpha=0.5 (midpoint blend).
    alpha = (rho - lo) / (hi - lo)
    return float((1.0 - alpha) * w_low + alpha * w_high)


# ---------------------------------------------------------------------------
# ARI
# ---------------------------------------------------------------------------

def compute_ari(waste: float, slca_composite: float, rho: float) -> float:
    """Compute the Adaptive Resilience Index for a single timestep.

    ARI = (1 − waste) × social_proxy × (1 − ρ)

    This is the single author-defined ARI throughout the codebase. The
    multiplicative form makes all three unit-interval components
    non-substitutable but is not an externally validated resilience index.

    Parameters
    ----------
    waste : net waste fraction after intervention, in [0, 1].
    slca_composite : attenuated sustainability/social proxy, in [0, 1]
        (legacy parameter name retained).
    rho : spoilage risk (1 − shelf_left), in [0, 1].

    Returns
    -------
    ARI value in [0, 1].
    """
    return (1.0 - waste) * slca_composite * (1.0 - rho)


# ---------------------------------------------------------------------------
# RLE (author-defined hierarchy-inspired, severity-weighted score)
# ---------------------------------------------------------------------------
# Single canonical RLE form. EU 2008/98/EC and the food-waste literature
# motivate the qualitative ordering, but the action mapping, weights, risk
# threshold, and smoothing are author-defined. The binary "routed / at_risk" form, the severity-aware
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
# The hierarchy-weighted form below contains author-set parameters. Severity
# weighting multiplies the assigned action utility by per-step rho. No method
# ranking is assumed; rankings are outputs of the benchmark.
#
# Definition:
#
#     RLE = sum_t [ rho(t) * w(a_t, rho(t)) * 1[rho(t) > theta] ]
#           ---------------------------------------------
#           sum_t [ rho(t) * w_max * 1[rho(t) > theta] ]
#
# where the author-defined mapping changes smoothly from redistribution in the
# lower-risk band toward recovery in the higher-risk band and w_max=1.00.
# A static cold-chain policy lands at 0; the preferred action depends on rho.

class RLETracker:
    """Stateful tracker for the hierarchy-inspired, severity-weighted RLE.

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
        # at-risk timestep receives the top-weighted action for its risk band.
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
            # rho-conditional weight: above RHO_ACTION_WEIGHT_CUTOFF the
            # synthetic table shifts its assigned utility toward Recovery.
            # Denominator
            # uses w_max=1.0 in both bands so the ratio stays in [0, 1].
            w = hierarchy_weight(action, rho)
            self._w_num += rho * w
            self._w_den += rho * self._w_max

    @property
    def rle(self) -> float:
        """Hierarchy-inspired, severity-weighted RLE in [0, 1].

        Returns 0.0 when no threshold-defined at-risk timesteps occurred
        (avoids division by zero and records that no rerouting demand was
        observed under this synthetic trigger).
        """
        if self._w_den <= 0.0:
            return 0.0
        return float(self._w_num / self._w_den)


def compute_rle(rho_values: List[float], actions: List[str],
                threshold: float = RLE_THRESHOLD) -> float:
    """Compute the canonical RLE over a full episode.

    Hierarchy-inspired, severity-weighted form. See the module docstring
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
# Violation disposition (outcome-side metric on the operating-envelope event set)
# ---------------------------------------------------------------------------
# constraint_violation_rate / regulatory_violation_rate / compliance_
# violation_rate are all driven by the dataset's ambient temperature and
# humidity trajectory and are computed by predicates that do not consult
# the chosen action. They are therefore *environmental signatures* of how
# stress-laden a scenario is, not measures of policy quality. Reading
# table1's ConstraintViolationRate or OperatingEnvelopeViolationRate column
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
# No cross-method ranking is assumed; the benchmark reports the observed
# disposition rates.
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
    was in a declared benchmark-envelope violation state (temperature ceiling exceeded
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
    """Stability-weighted mean proxy score (legacy names retained).

    legacy ``Equity`` key = mean(social_proxy) × (1 − std(social_proxy))

    The score is high only when the per-step proxy is both *temporally stable*
    and at a *high mean level*. A
    static cold-chain policy with mean SLCA ~0.5 cannot outscore an
    integrated policy with mean SLCA ~0.85 regardless of how flat its
    trajectory is. The formula is an author-defined simulation construct, not
    an empirical measure of distributive equity across people or groups.

    This is the single canonical temporal stability proxy throughout the
    codebase; the function and stored ``Equity`` key are retained for
    compatibility. No parallel "Sen welfare" / "Gini-based" / etc. variants are
    exposed. The Sen-welfare robustness companion that earlier
    versions also emitted (compute_equity_sen) was retired in the
    2026-04 single-version-of-the-truth pass per the user mandate
    that every metric have exactly one formulation in the
    repository.

    Parameters
    ----------
    slca_values : per-step attenuated proxy scores, in [0, 1].

    Returns
    -------
    Temporal social-performance stability proxy in [0, 1]. Higher means a
    temporally more uniform and higher mean author-declared proxy.
    """
    arr = np.asarray(slca_values, dtype=float)
    if arr.size == 0:
        return 0.0
    mean_s = float(np.mean(arr))
    std_s = float(np.std(arr))
    # The proxy is bounded in [0, 1] so std cannot exceed 0.5 in practice;
    # Clip defensively so the compatibility-key value stays in [0, 1] for downstream consumers
    # that assume a unit-interval metric.
    uniformity = max(0.0, min(1.0, 1.0 - std_s))
    return max(0.0, min(1.0, mean_s * uniformity))
