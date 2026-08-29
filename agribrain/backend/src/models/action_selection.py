"""
Regime-aware contextual softmax policy for routing decisions.

Implements the softmax action selection described in Section 4.6 of the
AGRI-BRAIN paper.  Given a 10-dimensional feature vector phi(s) extracted
from the current supply chain state, the policy computes action
probabilities via:

    pi(a | s) = softmax(Theta phi(s) + b_tau tau + bonus(mode, rho))

where the action-specific regime vector is
``b_tau = [0.25, 0.05, -0.25]`` for cold-chain, local redistribution,
and recovery.  It is deliberately non-uniform, so the binary regime flag
changes relative logits rather than adding a softmax-invariant scalar shift.

Feature vector design (10 features)
------------------------------------
Perception features 0 through 5 are the original state vector. Features
6 through 8 add the symmetric supply-demand forecast channel. Feature
9 is a demand-volatility price-pressure proxy:

    phi_0 = freshness          = 1 - rho
    phi_1 = inv_pressure       = min(inv / INV_CAPACITY, 1)
    phi_2 = demand_point       = min(y_hat_d / BASELINE_DEMAND, 1)
    phi_3 = thermal_stress     = clamp((T - T_0) / dT_max, 0, 1)
    phi_4 = spoilage_urgency   = rho
    phi_5 = interaction        = rho * inv_pressure
    phi_6 = supply_point       = clip(y_hat_s / INV_BASELINE - 1, -0.5, +0.5)
    phi_7 = supply_uncertainty = clip(sigma_s / max(|y_hat_s|, 1), 0, 1)
    phi_8 = demand_uncertainty = clip(sigma_d / max(|y_hat_d|, 1), 0, 1)
    phi_9 = price_signal       = clip(demand_bollinger_z, -1, +1)

``sigma_s`` and ``sigma_d`` are rolling one-step error-scale estimates from
the validation-selected persistence supply-proxy and non-seasonal Holt-linear
demand forecasts, respectively.  They are model-based uncertainty proxies,
not calibrated predictive standard deviations.  Features phi_7 and phi_8 are
dimensionless coefficient-of-variation scalars on a common [0, 1] scale.

``phi_6`` is *centered* on the baseline supply level: the raw ratio is
shifted by -1 and clipped to [-0.5, +0.5] so nominal supply yields zero
contribution and only deviation (surplus or shortage) drives the logit
modifier. This avoids a baseline shift when supply is at its expected
level.

``phi_9`` is a demand-volatility Bollinger z-score clipped to [-1, +1].
Positive values indicate demand above its rolling trend (shortage,
price pressure up); negative values indicate oversupply (price pressure
down). Proxy for market pressure that lets the ``adaptive_pricing``
scenario register a direct policy response rather than only an
indirect effect via temperature and inventory stress.

THETA matrix (3 actions x 10 features)
---------------------------------------
Each entry is sign-justified. The original six columns are unchanged;
the three forecast columns and the price column are documented in the
THETA block below.

                 fresh  inv_p  dem_pt  therm  spoil  inter  sup_pt  sup_unc  dem_unc  price
    ColdChain:    +      -     +       -      -      -      -       +        +        +
    LocalRedist:  0      +     -       +      +      +      +       +~0      -        -
    Recovery:     -      -     -       +      +      -      +       -        -        ~0

Mode-specific bonus terms
-------------------------
    - hybrid_rl:  Theta phi + b_tau tau                  (base learned policy)
    - no_slca:    Theta phi + b_tau tau                  (SLCA terms removed)
    - all other learned publication arms:
                   + SLCA_BONUS + SLCA_RHO_BONUS * rho
      Their MCP, retrieval, peer, and sign-projection differences are applied
      through the declared capability-controlled pathways, not hidden bonuses.
    - static:     always cold_chain                      (no optimisation)

Cyber outage handling
---------------------
Channel availability is modelled by the coordinator. Action selection always
uses the same policy equation; there is no mode-specific outage probability.

SLCA quality attenuation
-------------------------
Under physical stress (thermal or surplus), all SLCA pillars degrade:

    slca_quality = 1 / (1 + alpha_thermal * theta + alpha_surplus * surplus_ratio)

This physical attenuation equation is mode-neutral. It does not guarantee or
preordain any cross-method ordering because policies can choose different
actions and produce different endogenous trajectories.

References
----------
    - Sutton, R.S. & Barto, A.G. (2018). Reinforcement Learning: An
      Introduction (2nd ed.). MIT Press. [Softmax policy, Ch. 2.8]
    - Luce, R.D. (1959). Individual Choice Behavior. John Wiley & Sons.
      [Choice axiom / softmax derivation]
    - Hyndman, R.J. & Athanasopoulos, G. (2018). Forecasting: Principles
      and Practice, 2nd ed. OTexts, Ch. 8.7. [Residual-std prediction
      intervals for sigma_s and sigma_d.]
    - Dixit, A.K. & Pindyck, R.S. (1994). Investment Under Uncertainty.
      Princeton University Press. [Real-options logic for uncertainty
      columns phi_7, phi_8: uncertainty favours option-preserving
      actions (cold chain) over irreversible commitments (recovery).]
    - Trigeorgis, L. (1996). Real Options: Managerial Flexibility and
      Strategy in Resource Allocation. MIT Press.
    - Triantis, A. (2005). Realizing the Potential of Real Options.
      J. Applied Corporate Finance, 17(2), 8-16.
    - Fisher, M.L. (1997). What Is the Right Supply Chain for Your
      Product? Harvard Business Review 75(2), 105-116. [Supply-demand
      matching; surplus-driven redistribution for phi_6 column.]
    - Chopra, S. & Meindl, P. (2016). Supply Chain Management, 6th ed.,
      Ch. 11. [Excess-inventory dispositioning for phi_6 column.]
    - Chen, F., Drezner, Z., Ryan, J.K. & Simchi-Levi, D. (2000).
      Quantifying the Bullwhip Effect in a Simple Supply Chain.
      Management Science 46(3), 436-443. [Demand-uncertainty cold-chain
      positioning for phi_8 column.]
    - Lee, H.L., Padmanabhan, V. & Whang, S. (1997). The Bullwhip Effect
      in Supply Chains. Sloan Management Review 38(3), 93-102.
"""
from __future__ import annotations

import numpy as np

from .mode_capabilities import VALID_MODES as _DECLARED_VALID_MODES



# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
ACTIONS: list[str] = ["cold_chain", "local_redistribute", "recovery"]

ACTION_KM_KEYS: dict[str, str] = {
    "cold_chain": "km_coldchain",
    "local_redistribute": "km_local",
    "recovery": "km_recovery",
}

PRICE_FACTOR: dict[str, float] = {
    "cold_chain": 1.0,
    "local_redistribute": 0.95,
    "recovery": 0.88,
}
"""Per-action price multiplier applied to MSRP."""

VALID_MODES: list[str] = list(_DECLARED_VALID_MODES)
"""Valid operating modes for the softmax policy.

``no_context`` uses the same logits as ``agribrain`` but with
``context_modifier`` forced to None for ablation studies.
``mcp_only`` and ``pirag_only`` use agribrain logits with partial
context (MCP features only or piRAG features only).

Supply and demand forecast information (both point estimates and
uncertainties) now enters the state vector phi(s) symmetrically, so
there is no separate "supply-uncertainty ablation" mode.
"""

# ---------------------------------------------------------------------------
# Feature normalisation constants
# ---------------------------------------------------------------------------
INV_CAPACITY: float = 15_000.0
"""Inventory normalisation capacity (units). baseline_inv x 1.25 headroom."""

INV_BASELINE: float = 12_000.0
"""Baseline inventory level (units), matches ``waste.INV_BASELINE``.

Used to center the supply-point feature phi_6: at baseline supply this
feature equals zero, and deviations (surplus or shortage) produce a
signed, clipped signal in [-0.5, +0.5]."""

BASELINE_DEMAND: float = 20.0
"""Baseline demand normalisation (units / 15-min step)."""

THERMAL_T0: float = 4.0
"""Ideal cold-chain temperature (deg C)."""

THERMAL_DELTA_MAX: float = 20.0
"""Maximum temperature deviation for normalisation (deg C)."""

# ---------------------------------------------------------------------------
# THETA matrix (3 actions x 10 features)
# ---------------------------------------------------------------------------
# Columns:                 fresh   inv_p   dem_pt  therm   spoil   inter   sup_pt  sup_unc dem_unc price
# ColdChain row sign:         +       -       +       -       -       -       -       +       +       +
# LocalRedist row sign:       0       +       -       +       +       +       +       +~0     -       -
# Recovery row sign:          -       -       -       +       +       -       +       -       -       ~0
#
# Directional design rationale for the forecast columns (phi_6, phi_7, phi_8).
# The signs and magnitudes below are declared policy priors, not numerical
# estimates derived from the cited literature.
#
# phi_6 supply_point (signed surplus, clipped [-0.5, +0.5]):
#   - ColdChain -0.40: projected surplus saturates cold-chain capacity
#     and favours diversion (Chopra & Meindl 2016, Ch. 11); negative
#     coefficient also means projected shortage re-favours cold chain.
#   - LocalRedist +0.80: surplus is the canonical trigger for
#     redistribution pathways (Fisher 1997; Schoenherr & Swink 2012).
#   - Recovery +0.15: mild spillover when redistribution capacity is
#     exceeded (Kazancoglu et al. 2021).
#
# phi_7 supply_uncertainty (CV in [0, 1]):
#   - ColdChain +0.40: cold chain preserves optionality; high supply
#     uncertainty argues for deferring commitment (Dixit & Pindyck 1994;
#     Trigeorgis 1996).
#   - LocalRedist +0.05: near-zero; redistribution is moderately
#     reversible, so uncertainty's effect is second-order.
#   - Recovery -0.30: recovery is irreversible commitment; under
#     uncertainty this is strictly disfavoured (Triantis 2005).
#
# phi_8 demand_uncertainty (CV in [0, 1]):
#   - ColdChain +0.30: demand uncertainty argues for inventory
#     positioning and late-binding (Chen et al. 2000; Lee et al. 1997);
#     cold chain is the positioning option.
#   - LocalRedist -0.20: committing inventory to a specific local
#     channel when demand is uncertain risks over/under-supply.
#   - Recovery -0.30: demand uncertainty means product may still sell;
#     recovery forgoes that upside.
DECLARED_THETA: np.ndarray = np.array([
    # fresh  inv_p  dem_pt  therm  spoil  inter  sup_pt  sup_unc  dem_unc  price
    [  0.5,  -0.3,   0.4,   -0.5,  -2.0,  -1.0,  -0.40,   0.40,    0.30,    0.30],   # ColdChain
    [  0.0,   0.5,  -0.2,    0.5,   2.0,   1.5,   0.80,   0.05,   -0.20,   -0.30],   # LocalRedistribute
    [ -0.5,  -0.3,  -0.2,    0.3,   1.5,  -0.3,   0.15,  -0.30,   -0.30,   -0.05],   # Recovery
])
THETA: np.ndarray = DECLARED_THETA.copy()
"""Declared sign-constrained policy prior, shape (3, 10).

The 30 entries are author-specified benchmark hyperparameters, not fitted
coefficients and not numerical estimates taken from the cited literature.
Their intended directional effects are documented above.

Illustrative model distribution (rho=0.05, temp=5C, inv=25k, y_hat=18,
supply_hat=18, supply_std=2, demand_std=2, price_signal=0; tau=0):

  hybrid_rl : pi = [0.563, 0.330, 0.107]
  agribrain : pi = [0.347, 0.566, 0.087]   (THETA + SLCA_BONUS + 0.05*SLCA_RHO_BONUS)
  no_slca   : pi = [0.764, 0.182, 0.053]   (THETA; proxy terms removed)

At rho=0.5 (heatwave-like) agribrain shifts to pi ~ [0.007, 0.954, 0.039],
i.e. the policy commits almost entirely to local redistribution under
spoilage urgency. The earlier docstring claim of "~45 % CC / 45 % LR /
10 % Rec at baseline" is rough at best for hybrid_rl (closer to 56/33/11
at the conditions above) and does NOT describe agribrain (which is
LR-leaning by design due to SLCA_BONUS). The distribution table
above is the operative baseline.

THETA and the volatility-regime coefficients have no empirical calibration.
The separate 100-point structural-sensitivity design varies the prior scale
and independently sweeps the three action-specific regime coordinates
``b_tau = [0.25, 0.05, -0.25]`` within their declared bounds.

Column 9 (``price_signal``) is the demand-volatility Bollinger z-score
clipped to [-1, 1]. Weights +0.30 / -0.30 / -0.05 encode: cold chain
preferred under price-rise / supply shortage; local redistribution
preferred under price-drop / oversupply; recovery driven by spoilage
urgency rather than price. At ``price_signal = 0`` the column
contributes zero to every logit so the calibration of the other
columns is preserved.
"""

# ---------------------------------------------------------------------------
# Mode-specific bonus vectors
# ---------------------------------------------------------------------------
SLCA_BONUS: np.ndarray = np.array([-0.05, 0.10, 0.05])
"""Declared constant social-objective logit prior.

The vector encodes a small modelled preference for socially weighted routing.
It is an author-specified design choice, not an empirical social-effect
estimate. The locked ``no_slca`` ablation measures the benchmark's dependence
on this pathway.
"""

SLCA_RHO_BONUS: np.ndarray = np.array([-0.40, 0.35, 0.45])
"""Declared risk-dependent social-objective logit prior.

As modelled spoilage risk rises, the vector penalises cold-chain continuation
and raises redistribution and recovery. The magnitudes define the synthetic
policy and are not legal, food-safety, or empirical marketability thresholds.
"""

# ---------------------------------------------------------------------------
# Recovery knee: triage transition at high spoilage risk
# ---------------------------------------------------------------------------
RHO_RECOVERY_KNEE: float = 0.30
"""Declared policy knee for shifting modelled risk toward recovery.

This synthetic-case hyperparameter is neither a legal food-safety threshold nor
an empirically validated spinach marketability boundary. Separate constants in
the resilience and inventory modules are likewise benchmark rules and must be
reported as such.
"""

RHO_RECOVERY_KNEE_GAIN: float = 5.00
"""Additional Recovery logit gain per unit rho above the knee.

Applied as: logits[Recovery] += KNEE_GAIN * (rho - KNEE) / (1 - KNEE),
logits[LR] -= 3.00 * (rho - KNEE) / (1 - KNEE).
The value is a declared synthetic benchmark hyperparameter, not an empirical
effect estimate.
"""

RHO_RECOVERY_KNEE_LR_PENALTY: float = 3.00
"""LR logit penalty per unit rho above the knee (paired with KNEE_GAIN)."""

NO_SLCA_OFFSET: np.ndarray = np.zeros(3, dtype=float)
"""Deprecated compatibility constant.

The no-SLCA ablation now removes the SLCA terms only. A special offset would
change more than one factor and confound attribution.
"""

GOVERNANCE_CC_PROB_CEILING: float = 0.005
"""Upper bound on pi(cold_chain) that triggers the declared probability-gap override.

When the softmax probability of cold-chain falls below this ceiling AND
pi(local_redistribute) exceeds pi(cold_chain) by
``GOVERNANCE_LOCAL_ADVANTAGE_MIN``, the rule substitutes local
redistribution. It is a declared, recorded synthetic policy rule, not a regulatory
requirement or a field-calibrated probability threshold.
"""

GOVERNANCE_LOCAL_ADVANTAGE_MIN: float = 0.80
"""Minimum pi(local_redistribute) - pi(cold_chain) gap that, together
with the :data:`GOVERNANCE_CC_PROB_CEILING` condition, fires the
probability-gap override.

The value is a declared synthetic benchmark hyperparameter.
"""


def calibrate_governance_thresholds(
    prob_rollouts: np.ndarray,
    cc_quantile: float = 0.05,
    local_quantile: float = 0.50,
) -> dict[str, float]:
    """Derive governance thresholds from a rollout probability distribution.

    This optional exploratory helper derives candidate thresholds from a
    supplied rollout distribution. The publication benchmark uses the declared
    constants above; it does not present them as externally validated values.

    1. Run the simulator over benchmark scenarios with the override
       disabled (or with the previous thresholds) and collect the full
       sequence of policy probability vectors at every decision point.
    2. Pass the stacked (N, 3) probability array to this function.
    3. It returns the ceiling (``cc_prob_ceiling``) and advantage floor
       (``local_advantage_min``) at the chosen quantiles.
    4. Treat the returned values as exploratory diagnostics only; changing the
       locked constants would define a new protocol and require fresh runs.

    Note: the shipped governance constants are declared independently of this
    optional helper and must be treated as policy hyperparameters.

    Parameters
    ----------
    prob_rollouts : (N, 3) array of softmax probabilities observed at
        decision points, columns ordered (cold_chain, local_redistribute,
        recovery) to match :data:`ACTIONS`.
    cc_quantile : lower-tail quantile of pi(cold_chain) to use as the
        ceiling. Default 0.05 (5th percentile) means the override fires
        when confidence in cold-chain is in the bottom 5 percent of
        the calibration distribution.
    local_quantile : quantile of (pi(local) - pi(cold_chain)) to use as
        the advantage floor. Default 0.50 (median).

    Returns
    -------
    dict with keys ``cc_prob_ceiling`` and ``local_advantage_min``.
    """
    rollouts = np.asarray(prob_rollouts, dtype=np.float64)
    if rollouts.ndim != 2 or rollouts.shape[-1] != 3:
        raise ValueError(
            f"prob_rollouts must be shape (N, 3), got {rollouts.shape}"
        )
    if not (0.0 <= cc_quantile <= 1.0 and 0.0 <= local_quantile <= 1.0):
        raise ValueError(
            "quantile arguments must lie in [0, 1]"
        )
    cc_probs = rollouts[:, 0]
    gap = rollouts[:, 1] - rollouts[:, 0]
    return {
        "cc_prob_ceiling": float(np.quantile(cc_probs, cc_quantile)),
        "local_advantage_min": float(np.quantile(gap, local_quantile)),
    }


def governance_override_applies(probs: np.ndarray) -> bool:
    """Return whether the declared probability-space rule activates.

    The implemented guardrail has exactly two predicates: cold-chain
    probability must be strictly below
    :data:`GOVERNANCE_CC_PROB_CEILING`, and the local-redistribution minus
    cold-chain probability gap must be strictly above
    :data:`GOVERNANCE_LOCAL_ADVANTAGE_MIN`.  Compliance and spoilage
    quantities may affect these probabilities through the normal policy
    inputs, but they are not additional override predicates.
    """
    policy_probs = np.asarray(probs, dtype=float)
    if policy_probs.shape != (3,):
        raise ValueError(
            f"probs must be a length-3 probability vector, got "
            f"{policy_probs.shape}"
        )
    return bool(
        policy_probs[0] < GOVERNANCE_CC_PROB_CEILING
        and policy_probs[1] - policy_probs[0]
        > GOVERNANCE_LOCAL_ADVANTAGE_MIN
    )

# Compatibility export retained for callers from earlier releases. Cyber
# response is now generated by channel availability plus the common policy.
CYBER_REROUTE_PROB: dict[str, float] = {}

# ---------------------------------------------------------------------------
# SLCA attenuation under stress
# ---------------------------------------------------------------------------
SLCA_THERMAL_ATTEN: float = 0.25
"""Declared synthetic SLCA-proxy attenuation for thermal stress."""

SLCA_SURPLUS_ATTEN: float = 0.08
"""Declared synthetic SLCA-proxy attenuation for inventory surplus."""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax: π(a) = exp(x_a − max(x)) / Σ exp(x_i − max(x))."""
    e = np.exp(x - x.max())
    return e / e.sum()


def regime_logit_term(policy: object, tau: float) -> np.ndarray:
    """Return the exact action-specific regime term ``b_tau * tau``.

    ``tau`` is a declared binary regime flag, not a continuous severity score.
    Failing here prevents an out-of-contract value from silently becoming an
    undocumented logit multiplier. The three coefficients remain independently
    sensitivity-testable, so this helper validates finiteness without
    hard-coding their default values.
    """

    try:
        tau_value = float(tau)
        bias = np.asarray([
            policy.gamma_coldchain,
            policy.gamma_local,
            policy.gamma_recovery,
        ], dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("regime coefficients and tau must be numeric") from exc
    if not np.isfinite(tau_value) or tau_value not in (0.0, 1.0):
        raise ValueError(f"tau must be the binary flag 0 or 1, got {tau!r}")
    if bias.shape != (3,) or not np.all(np.isfinite(bias)):
        raise ValueError("regime coefficient vector must be a finite 3-vector")
    return bias * tau_value


def categorical_action_from_uniform(
    probabilities: np.ndarray,
    uniform: float,
) -> int:
    """Select an action with the locked left-closed inverse-CDF rule.

    The input variate must lie in ``[0, 1)``. Zero-probability intervals are
    skipped via ``searchsorted(..., side="right")``. The explicit algorithm is
    portable and can be replayed by the independent ledger validator.
    """
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("probabilities must be a non-empty vector")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(np.sum(probs))
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("probabilities must sum to one")
    value = float(uniform)
    if not np.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError("categorical uniform must lie in [0, 1)")
    index = int(np.searchsorted(np.cumsum(probs), value, side="right"))
    return min(index, int(probs.size - 1))


def build_feature_vector(
    rho: float,
    inv: float,
    y_hat: float,
    temp: float,
    supply_hat: float | None = None,
    supply_std: float | None = None,
    demand_std: float | None = None,
    price_signal: float | None = None,
) -> np.ndarray:
    """Construct the 10-dimensional state feature vector phi(s).

    Features 0-5 are the original physics-and-operations state (freshness,
    inventory pressure, demand point forecast, thermal stress, spoilage
    urgency, interaction). Features 6-8 add the supply-demand forecast
    channel with matching point and uncertainty quantities. Feature 9 is
    a demand-volatility-driven price signal:

        phi_6 supply_point       = clip(supply_hat / INV_BASELINE - 1, -0.5, +0.5)
        phi_7 supply_uncertainty = clip(supply_std / max(|supply_hat|, 1), 0, 1)
        phi_8 demand_uncertainty = clip(demand_std / max(|y_hat|, 1), 0, 1)
        phi_9 price_signal       = clip(price_signal, -1, +1)

    Parameters
    ----------
    rho : spoilage risk (1 - shelf_left), in [0, 1].
    inv : current inventory level (units).
    y_hat : demand point forecast (confirmatory: Holt-linear; units / step).
    temp : current temperature (deg C).
    supply_hat : optional. Supply-proxy point forecast (confirmatory:
        persistence; units).
        When omitted, phi_6 is zero (neutral).
    supply_std : optional. Rolling persistence-error scale. When omitted,
        phi_7 is zero.
    demand_std : optional. Holt-linear rolling residual scale. When omitted,
        phi_8 is zero.
    price_signal : optional. Demand-Bollinger z-score used as a
        market-pressure proxy: positive values indicate demand above
        trend (price pressure up, shortage), negative values indicate
        demand below trend (price pressure down, oversupply). Clipped
        to [-1, +1]. When omitted, phi_9 is zero.

    The optional kwargs default to None so legacy call sites still
    work. The simulator always passes the forecast and price payload;
    the REST decide endpoints compute price_signal from the demand
    history they already read for the Bollinger trigger.

    Returns
    -------
    phi : np.ndarray of shape (10,)
    """
    freshness = 1.0 - rho
    inv_pressure = min(inv / INV_CAPACITY, 1.0)
    demand_signal = min(y_hat / BASELINE_DEMAND, 1.0)
    thermal_stress = min(max((temp - THERMAL_T0) / THERMAL_DELTA_MAX, 0.0), 1.0)
    spoilage_urgency = rho
    interaction = rho * inv_pressure

    # Supply point: centered surplus/shortage signal, clipped to [-0.5, +0.5]
    # so nominal supply (ratio 1) gives zero contribution.
    if supply_hat is None or INV_BASELINE <= 0.0:
        supply_point = 0.0
    else:
        ratio = float(supply_hat) / float(INV_BASELINE) - 1.0
        supply_point = float(np.clip(ratio, -0.5, 0.5))

    # Supply uncertainty: coefficient of variation of the selected forecast's
    # rolling one-step error scale, clipped to the unit interval.
    if supply_hat is None or supply_std is None:
        supply_uncertainty = 0.0
    else:
        sh = abs(float(supply_hat))
        su = float(supply_std) / max(sh, 1.0)
        supply_uncertainty = float(np.clip(su, 0.0, 1.0))

    # Demand uncertainty: coefficient of variation of the selected forecast's
    # rolling one-step error scale, clipped to the unit interval. Uses a
    # floor of 1 unit in the denominator so near-zero-demand does not
    # produce an infinite CV.
    if y_hat is None or demand_std is None:
        demand_uncertainty = 0.0
    else:
        yh = abs(float(y_hat))
        du = float(demand_std) / max(yh, 1.0)
        demand_uncertainty = float(np.clip(du, 0.0, 1.0))

    # Price signal: demand-volatility Bollinger z-score clipped to
    # [-1, +1]. Proxy for market pressure; the adaptive_pricing scenario
    # oscillates demand which drives this channel away from zero.
    if price_signal is None:
        price_signal_phi = 0.0
    else:
        price_signal_phi = float(np.clip(float(price_signal), -1.0, 1.0))

    return np.array([
        freshness,
        inv_pressure,
        demand_signal,
        thermal_stress,
        spoilage_urgency,
        interaction,
        supply_point,
        supply_uncertainty,
        demand_uncertainty,
        price_signal_phi,
    ])


def compute_thermal_stress(
    temp: float,
    thermal_t0: float = THERMAL_T0,
    thermal_delta_max: float = THERMAL_DELTA_MAX,
) -> float:
    """Compute normalised thermal stress θ ∈ [0, 1].

    θ = clamp((T − T₀) / ΔT_max, 0, 1)

    Parameters
    ----------
    temp : ambient temperature in °C.

    Returns
    -------
    Normalised thermal stress.
    """
    return min(max((temp - thermal_t0) / thermal_delta_max, 0.0), 1.0)


def compute_slca_attenuation(
    thermal_stress: float,
    surplus_ratio: float,
    thermal_atten: float = SLCA_THERMAL_ATTEN,
    surplus_atten: float = SLCA_SURPLUS_ATTEN,
) -> float:
    """Compute stress-dependent SLCA quality attenuation factor.

    slca_quality = 1 / (1 + α_thermal × θ + α_surplus × surplus_ratio)

    Parameters
    ----------
    thermal_stress : normalised thermal stress θ ∈ [0, 1].
    surplus_ratio : inventory surplus above baseline (0 at/below baseline).
    thermal_atten : thermal attenuation coefficient.
    surplus_atten : surplus attenuation coefficient.

    Returns
    -------
    Multiplicative attenuation factor in (0, 1].
    """
    return 1.0 / (1.0 + thermal_atten * thermal_stress
                  + surplus_atten * surplus_ratio)


def select_action(
    mode: str,
    rho: float,
    inv: float,
    y_hat: float,
    temp: float,
    tau: float,
    policy,
    rng: np.random.Generator,
    scenario: str = "baseline",
    hour: float = 0.0,
    role_bias: np.ndarray | None = None,
    deterministic: bool = False,
    context_modifier: np.ndarray | None = None,
    slca_amp_coeff: float | None = None,
    supply_hat: float | None = None,
    supply_std: float | None = None,
    demand_std: float | None = None,
    price_signal: float | None = None,
    theta_delta: np.ndarray | None = None,
    slca_bonus_delta: np.ndarray | None = None,
    slca_rho_delta: np.ndarray | None = None,
    no_slca_offset_delta: np.ndarray | None = None,
    policy_temperature: float = 1.0,
    categorical_uniform: float | None = None,
    out: dict | None = None,
) -> tuple[int, np.ndarray]:
    """Select routing action based on mode-specific softmax policy.

    Parameters
    ----------
    mode : operating mode declared by ``mode_capabilities.VALID_MODES``.
    rho : spoilage risk.
    inv : current inventory.
    y_hat : demand point forecast.
    temp : current temperature (deg C).
    tau : volatility indicator (1.0 if anomaly, 0.0 otherwise).
    policy : Policy object with gamma_* and distance attributes.
    rng : numpy random generator.
    scenario : current scenario name.
    hour : hours since start (for cyber outage timing).
    role_bias : optional per-role logit bias vector (3,).
    deterministic : if True, use argmax instead of sampling.
    context_modifier : optional logit modifier vector (3,) from the
        MCP/piRAG context pipeline.  Added to logits after all other
        mode-specific and role-specific terms, before softmax.
        When ``None``, behavior is bit-identical to the original policy.
    supply_hat : supply-proxy point forecast (confirmatory: persistence;
        units). Feeds ``phi_6`` (centered supply point).
    supply_std : selected supply forecast's rolling one-step error scale
        (units). Feeds ``phi_7`` (supply uncertainty CV).
    demand_std : selected demand forecast's rolling one-step error scale
        (units). Feeds ``phi_8`` (demand uncertainty CV). The forecast kwargs default to None so
        legacy callers still work; missing values yield zero contribution
        on the corresponding phi channels.
    price_signal : optional demand-volatility Bollinger z-score used
        as a market-pressure proxy. Feeds ``phi_9`` clipped to [-1, 1].
    theta_delta : optional (3, 10) learned correction added to THETA
        at inference. Provided by PolicyDeltaLearner. The hand-calibrated
        THETA stays fixed; only this delta moves with training, and it
        is bounded at 25 percent of each entry's initial magnitude so
        the learned policy cannot drift more than a quarter away from
        the domain priors.
    out : optional mutable dict that, when provided, receives
        diagnostic side-channel data the caller may want to consume
        without changing the function's return signature. Populated
        keys (only when applicable):

          ``base_argmax`` (int): argmax of the logits *before* the
            ``context_modifier`` is added. This is an observer-only policy
            diagnostic used in channel-attribution reconstructions. The
            reported ``context_influence_rate`` instead uses a paired
            pre-selection-RNG-state context ablation in the coordinator. On
            stochastic calls both policies consume the same categorical
            variate, including when the live probability-gap override discards its
            sampled action, so a
            stochastic sample that differs from this argmax cannot by itself
            count as context influence. Set only on the regular
            logit-construction path; not set on the static path.

    Returns
    -------
    (action_index, probability_vector)
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode: {mode!r}. Must be one of {VALID_MODES}")

    # Static is ALWAYS cold chain, regardless of scenario. It is a fixed
    # decision rule and therefore consumes no categorical variate.
    if mode == "static":
        if out is not None:
            out["policy_categorical_uniform"] = None
            out["sampled_action_pre_override"] = 0
        return 0, np.array([1.0, 0.0, 0.0])

    phi = build_feature_vector(
        rho, inv, y_hat, temp,
        supply_hat=supply_hat,
        supply_std=supply_std,
        demand_std=demand_std,
        price_signal=price_signal,
    )
    regime_term = regime_logit_term(policy, tau)

    # Effective reward-shaping vectors. When RewardShapingLearner deltas
    # are provided, they are zero-init shrinkage corrections inside the
    # 25-percent per-entry cap, so at step 0 the effective vectors are
    # bit-identical to SLCA_BONUS / SLCA_RHO_BONUS / NO_SLCA_OFFSET.
    # Sign projection is enabled for the confirmatory arms and disabled only
    # for the declared sign-unconstrained secondary ablation. The deltas are
    # applied unconditionally here and the mode branches below pick the
    # relevant vectors.
    _slca_bonus = SLCA_BONUS
    _slca_rho_bonus = SLCA_RHO_BONUS
    _no_slca_offset = NO_SLCA_OFFSET
    if slca_bonus_delta is not None:
        _slca_bonus = _slca_bonus + np.asarray(slca_bonus_delta)
    if slca_rho_delta is not None:
        _slca_rho_bonus = _slca_rho_bonus + np.asarray(slca_rho_delta)
    if no_slca_offset_delta is not None:
        _no_slca_offset = _no_slca_offset + np.asarray(no_slca_offset_delta)

    if mode == "hybrid_rl":
        logits = THETA @ phi + regime_term

    elif mode == "no_slca":
        # One-factor ablation: retain the same state, context, policy weights,
        # and learning budget, but remove both SLCA shaping terms.
        logits = THETA @ phi + regime_term

    else:
        # AGRI-BRAIN, the external-channel arms, and the three secondary
        # one-factor ablations share the same base logits.  Their differences
        # are implemented structurally upstream (channel kind, retrieval kind,
        # peer delivery, or learner sign projection), never as mode-specific
        # outcome bonuses.
        logits = (
            THETA @ phi
            + regime_term
            + _slca_bonus
            + _slca_rho_bonus * rho
        )

    # Declared recovery knee: above RHO_RECOVERY_KNEE, the synthetic policy
    # increasingly favours recovery. This is not a marketability or food-safety
    # determination. It applies to rho-shaped modes but not hybrid_rl.
    if mode != "hybrid_rl" and rho > RHO_RECOVERY_KNEE:
        excess = (rho - RHO_RECOVERY_KNEE) / (1.0 - RHO_RECOVERY_KNEE)
        logits[2] += RHO_RECOVERY_KNEE_GAIN * excess
        logits[1] -= RHO_RECOVERY_KNEE_LR_PENALTY * excess

    # Learned policy correction. PolicyDeltaLearner owns a (3, 10)
    # delta trained via REINFORCE on the full phi with a 25 percent
    # per-entry magnitude cap. Sign projection is capability-controlled: on
    # for confirmatory arms and off only for the declared unconstrained
    # secondary ablation. Delta is zero-initialised and shrinks toward zero
    # under a Gaussian prior, so the default behaviour is bit-identical to the
    # hand-calibrated policy until the learner observes enough reward signal.
    if theta_delta is not None:
        logits = logits + np.asarray(theta_delta) @ phi

    if role_bias is not None:
        logits = logits + role_bias

    if out is not None:
        # Exact pre-context policy surface for every non-static arm. Recording
        # this outside the context branch lets the independent validator apply
        # one policy-equation contract to controls and context modes alike.
        out["base_logits"] = [float(v) for v in logits]
        out["regime_logit_bias"] = [float(v) for v in regime_term]
        out["slca_shaping"] = [
            float(v) for v in (_slca_bonus + _slca_rho_bonus * rho)
        ]
        out["slca_amp"] = float(
            slca_amp_coeff if slca_amp_coeff is not None else 0.0
        )
        out["policy_temperature"] = float(policy_temperature)

    if context_modifier is not None:
        # Capture the pre-modifier argmax for observer-only policy
        # diagnostics and conditional feature-group masking. Computed here -- after
        # THETA, gamma, slca_bonus, knee, theta_delta, role_bias --
        # so the comparison isolates the *modifier's* contribution
        # against the otherwise-final base logits. Note: when the
        # caller is in the cyber-outage Bernoulli branch (which
        # returns above), this code is unreachable, so out["base_argmax"]
        # is correctly left unset for those steps -- they have no
        # modifier-vs-base comparison to make.
        if out is not None:
            out["base_argmax"] = int(np.argmax(logits))
            # Observer-only side-channel for offline observed-state
            # feature-group masking. None of
            # these assignments alter the chosen action or consume the RNG,
            # so the live trajectory is bit-identical with or without ``out``.
            # ``base_logits`` is the pre-context logit vector (everything up
            # to but excluding the context_modifier and its SLCA boost).
            # Combined with ``policy_temperature``, the SLCA shaping vector
            # ``slca_shaping`` (= _slca_bonus + _slca_rho_bonus * rho) and the
            # amplification coefficient ``slca_amp``, plus the feature-group
            # masked modifiers the coordinator records, the H2 aggregator
            # reconstructs argmax(base + m + slca_boost(m)) while holding the
            # observed dispatch/retrieval results and guards fixed.
            out["base_logits"] = [float(v) for v in logits]

        # Optional interaction between context and social-proxy shaping. The
        # confirmatory learner keeps this at zero to avoid double-counting
        # context; a non-zero value is an explicitly enabled sensitivity.
        amp = slca_amp_coeff if slca_amp_coeff is not None else 0.0
        slca_amplification = 1.0 + amp * min(abs(context_modifier[1]), 1.0)
        # Use the locally computed vectors so any declared
        # RewardShapingLearner deltas are carried into the optional
        # context-by-social-proxy interaction.
        if mode == "no_slca":
            slca_boost = np.zeros_like(_slca_bonus)
        else:
            slca_boost = (_slca_bonus + _slca_rho_bonus * rho) * (slca_amplification - 1.0)
        logits = logits + context_modifier + slca_boost

    # Apply per-(mode, seed) policy temperature. T = 1 reproduces the
    # original behaviour bit-for-bit; T < 1 sharpens the softmax (more
    # confident); T > 1 smooths it (more diverse). Drawn once per (mode,
    # seed) by the caller and passed through here. The confirmatory benchmark
    # leaves T=1; alternative values are sensitivity analyses.
    if policy_temperature != 1.0 and policy_temperature > 0.0:
        logits = logits / float(policy_temperature)

    # Observer-only trace field: the exact policy logits after the external
    # context term and temperature scaling, but before the probability-gap
    # override below. Recording this side channel does not alter
    # the action, probabilities, or RNG stream.
    if out is not None:
        out["post_context_logits_pre_override"] = [float(v) for v in logits]

    probs = _softmax(logits)
    if out is not None:
        out["policy_probs_pre_override"] = [float(v) for v in probs]
        out["governance_override"] = False

    # Preserve the common-random-number schedule across experimental arms.
    # A stochastic regular-policy call consumes exactly one categorical draw
    # whether or not the probability-gap rule subsequently changes its action.
    # Before this draw was hoisted above the override, an override returned
    # early and shifted every later policy draw in context-enabled arms
    # relative to No-context and the single-channel controls.  Explicit
    # deterministic calls remain draw-free.
    sampled_action = None
    if not deterministic:
        # Record one portable uniform variate and apply the locked inverse-CDF
        # sampler. This binds the chosen action to the Merkle-covered
        # probabilities without depending on NumPy's version-specific
        # implementation of Generator.choice.
        categorical_uniform = float(
            rng.random()
            if categorical_uniform is None else categorical_uniform
        )
        sampled_action = categorical_action_from_uniform(
            probs, categorical_uniform,
        )
        if out is not None:
            out["policy_categorical_uniform"] = categorical_uniform
            out["sampled_action_pre_override"] = int(sampled_action)
    elif out is not None:
        out["policy_categorical_uniform"] = None
        out["sampled_action_pre_override"] = int(np.argmax(probs))

    # Author-declared probability-gap override: fires only for context-enabled modes (those that
    # build a context_modifier). Stated in probability space so the
    # condition is auditable without reference to the raw logit scale: it
    # fires when the policy's confidence in cold-chain is below the
    # declared ceiling and local-redistribute dominates cold-chain by the
    # declared margin.
    if context_modifier is not None:
        if governance_override_applies(probs):
            if out is not None:
                out["governance_override"] = True
            return 1, np.array([0.0, 1.0, 0.0])

    if deterministic:
        return int(np.argmax(probs)), probs
    assert sampled_action is not None
    return sampled_action, probs
