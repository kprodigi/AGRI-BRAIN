"""
Regime-aware contextual softmax policy for routing decisions.

Implements the softmax action selection described in Section 4.6 of the
AGRI-BRAIN paper.  Given a 9-dimensional feature vector phi(s) extracted
from the current supply chain state, the policy computes action
probabilities via:

    pi(a | s) = softmax(Theta phi(s) + gamma tau + bonus(mode, rho))

Feature vector design (9 features)
-----------------------------------
Perception features 0 through 5 are the original state vector. Features
6 through 8 (added for architectural symmetry between supply and demand)
extend the state vector with supply and demand forecast information:

    phi_0 = freshness          = 1 - rho
    phi_1 = inv_pressure       = min(inv / INV_CAPACITY, 1)
    phi_2 = demand_point       = min(y_hat_d / BASELINE_DEMAND, 1)
    phi_3 = thermal_stress     = clamp((T - T_0) / dT_max, 0, 1)
    phi_4 = spoilage_urgency   = rho
    phi_5 = interaction        = rho * inv_pressure
    phi_6 = supply_point       = clip(y_hat_s / INV_BASELINE - 1, -0.5, +0.5)
    phi_7 = supply_uncertainty = clip(sigma_s / max(|y_hat_s|, 1), 0, 1)
    phi_8 = demand_uncertainty = clip(sigma_d / max(|y_hat_d|, 1), 0, 1)

``sigma_s`` and ``sigma_d`` are in-sample one-step-ahead residual standard
deviations from the Holt-Winters supply forecaster and the LSTM demand
forecaster respectively (Hyndman & Athanasopoulos 2018, Ch. 8.7). They
capture forecast uncertainty in the same statistical sense for both
series, so phi_7 and phi_8 are dimensionless coefficient-of-variation
scalars on a common scale.

``phi_6`` is *centered* on the baseline supply level: the raw ratio is
shifted by -1 and clipped to [-0.5, +0.5] so nominal supply yields zero
contribution and only deviation (surplus or shortage) drives the logit
modifier. This avoids a baseline shift when supply is at its expected
level.

THETA matrix (3 actions x 9 features)
--------------------------------------
Each entry is sign-justified. The original six columns are unchanged;
the three new columns are documented in the THETA block below.

                 fresh  inv_p  dem_pt  therm  spoil  inter  sup_pt  sup_unc  dem_unc
    ColdChain:    +      -     +       -      -      -      -       +        +
    LocalRedist:  0      +     -       +      +      +      +       +~0      -
    Recovery:     -      -     -       +      +      -      +       -        -

Mode-specific bonus terms
-------------------------
    - hybrid_rl:  Theta phi + gamma tau              (base RL, no SLCA/PINN)
    - no_pinn:    + 0.5 * SLCA_bonus                 (SLCA at reduced strength)
    - no_slca:    + NO_SLCA_OFFSET                   (conservative / CC-heavy)
    - agribrain:  + SLCA_BONUS + SLCA_RHO_BONUS * rho  (full system)
    - static:     always cold_chain                  (no optimisation)

Cyber outage handling
---------------------
During a cyber outage (processor offline from hour 24), rerouting success
depends on each mode's autonomous intelligence (edge computing, cached
policies). When rerouting fails, the shipment defaults to cold chain.

SLCA quality attenuation
-------------------------
Under physical stress (thermal or surplus), all SLCA pillars degrade:

    slca_quality = 1 / (1 + alpha_thermal * theta + alpha_surplus * surplus_ratio)

This is applied equally to all modes within a scenario, preserving
cross-method orderings.

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

from .action_aliases import resolve_action


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

VALID_MODES: list[str] = [
    "static", "hybrid_rl", "no_pinn", "no_slca",
    "agribrain", "no_context", "mcp_only", "pirag_only",
]
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
# THETA matrix (3 actions x 9 features)
# ---------------------------------------------------------------------------
# Columns:                 fresh   inv_p   dem_pt  therm   spoil   inter   sup_pt  sup_unc dem_unc
# ColdChain row sign:         +       -       +       -       -       -       -       +       +
# LocalRedist row sign:       0       +       -       +       +       +       +       +~0     -
# Recovery row sign:          -       -       -       +       +       -       +       -       -
#
# Sign justifications for the three new columns (phi_6, phi_7, phi_8):
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
THETA: np.ndarray = np.array([
    # fresh  inv_p  dem_pt  therm  spoil  inter  sup_pt  sup_unc  dem_unc  price
    [  0.5,  -0.3,   0.4,   -0.5,  -2.0,  -1.0,  -0.40,   0.40,    0.30,    0.30],   # ColdChain
    [  0.0,   0.5,  -0.2,    0.5,   2.0,   1.5,   0.80,   0.05,   -0.20,   -0.30],   # LocalRedistribute
    [ -0.5,  -0.3,  -0.2,    0.3,   1.5,  -0.3,   0.15,  -0.30,   -0.30,   -0.05],   # Recovery
])
"""Policy weight matrix, shape (3, 10).

Calibrated so that the base policy (hybrid_rl) produces approximately
45 % CC / 45 % LR / 10 % Rec at baseline conditions, shifting toward
more LR/Rec under thermal stress or spoilage urgency. At nominal
supply and nominal forecast uncertainty (phi_6 ~ 0, phi_7 ~ 0.05,
phi_8 ~ 0.05), the new columns contribute less than 0.04 per logit
in absolute value, so the calibration of the original six columns is
effectively preserved. Under overproduction (phi_6 = +0.5) the
redistribution logit gains +0.40 and cold chain loses 0.20, matching
the intended behaviour of the overproduction scenario. Under combined
supply and demand uncertainty at CV = 0.8, cold chain gains +0.56,
local redistribution loses 0.12, and recovery loses 0.48, consistent
with the real-options literature cited in the module docstring.

Column 9 is the ``price_signal`` channel, a demand-volatility proxy
for market pressure (Bollinger z-score of demand, clipped to [-1, 1]).
Weights +0.30 / -0.30 / -0.05 encode the first-order economic
intuition: when prices rise (positive z, supply shortage) the cold
chain is preferred to preserve high-value inventory; when prices
drop (negative z, oversupply) local redistribution is preferred to
clear volume. The recovery row is near zero because the recovery
decision is driven by spoilage urgency rather than price. At
``price_signal = 0`` (baseline demand volatility) the column
contributes zero to every logit so the calibration of the other
columns is preserved.
"""

# ---------------------------------------------------------------------------
# Mode-specific bonus vectors
# ---------------------------------------------------------------------------
SLCA_BONUS: np.ndarray = np.array([-0.35, 0.60, -0.1])
"""Constant SLCA bonus for agribrain and no_pinn modes.

Represents the system's baseline ability to identify socially beneficial
routing through SLCA feedback.
"""

SLCA_RHO_BONUS: np.ndarray = np.array([-0.5, 1.0, 0.15])
"""Rho-dependent SLCA bonus for proactive rerouting.

The PINN spoilage prediction enables proactive rerouting of at-risk produce.
Moderate magnitude prevents overcompensation under stress.
"""

# NOPINN_SLCA_SCALE was a 0.5x SLCA-bonus attenuation applied only in
# no_pinn mode. It was removed so no_pinn measures PINN's effect in
# isolation: the simulator routes rho through plain Arrhenius
# (compute_spoilage) instead of compute_spoilage_pinn (gated by
# _PINN_MODES in mvp/simulation/generate_results.py), and the SLCA
# reward shape stays identical to agribrain. The previous combined
# ablation conflated "no PINN spoilage correction" with "halved SLCA
# reward signal" and was not a clean attribution of the PINN component.

NO_SLCA_OFFSET: np.ndarray = np.array([0.6, -0.3, -0.4])
"""Logit offset for no_slca mode (conservative, CC-heavy routing).

Without SLCA feedback, the system defaults toward cold chain the "safe"
choice, since it cannot assess social value of alternatives.
"""

GOVERNANCE_CC_PROB_CEILING: float = 0.05
"""Upper bound on pi(cold_chain) that triggers the governance override.

When the softmax probability of cold-chain falls below this ceiling AND
pi(local_redistribute) exceeds pi(cold_chain) by
``GOVERNANCE_LOCAL_ADVANTAGE_MIN``, the override mandates local
redistribution. Stated in probability space rather than raw logits so
the condition is auditable without reference to the rest of the
distribution: regulators can verify the policy "overrides when
confidence in cold-chain is below 5 percent."

Default derivation: the 5th percentile of pi(cold_chain) observed
across benchmark-scenario rollouts at decision points where the policy
eventually selected cold-chain. See
:func:`calibrate_governance_thresholds` for the helper that recomputes
this value from a rollout probability array so the calibration is
scripted rather than hand-picked.
"""

GOVERNANCE_LOCAL_ADVANTAGE_MIN: float = 0.50
"""Minimum pi(local_redistribute) - pi(cold_chain) gap that, together
with the :data:`GOVERNANCE_CC_PROB_CEILING` condition, fires the
governance override.

Default derivation: the median of (pi(local) - pi(cold_chain))
observed across rollouts where the context modifier pushed the policy
away from cold-chain. A median is used rather than a lower quantile so
the override requires unambiguous dominance of local-redistribute, not
merely any preference over cold-chain.
"""


def calibrate_governance_thresholds(
    prob_rollouts: np.ndarray,
    cc_quantile: float = 0.05,
    local_quantile: float = 0.50,
) -> dict[str, float]:
    """Derive governance thresholds from a rollout probability distribution.

    The project's paper-facing story is that governance override thresholds
    have statistical provenance rather than being hand-picked numbers. This
    helper implements the calibration step:

    1. Run the simulator over benchmark scenarios with the override
       disabled (or with the previous thresholds) and collect the full
       sequence of policy probability vectors at every decision point.
    2. Pass the stacked (N, 3) probability array to this function.
    3. It returns the ceiling (``cc_prob_ceiling``) and advantage floor
       (``local_advantage_min``) at the chosen quantiles.
    4. Write the returned values into
       :data:`GOVERNANCE_CC_PROB_CEILING` and
       :data:`GOVERNANCE_LOCAL_ADVANTAGE_MIN` before the main benchmark
       run.

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

# ---------------------------------------------------------------------------
# Cyber outage: rerouting success probabilities
# ---------------------------------------------------------------------------
CYBER_REROUTE_PROB: dict[str, float] = {
    # static mode returns ColdChain before reaching cyber logic; no entry needed.
    "hybrid_rl": 0.55,
    "no_pinn": 0.65,
    "no_slca": 0.60,
    "agribrain": 0.82,
    "no_context": 0.82,
    "mcp_only": 0.82,
    "pirag_only": 0.82,
}
"""Mode-dependent probability of successful rerouting during cyber outage.

Reflects each mode's autonomous intelligence (edge computing, cached
policies, local PINN + SLCA inference capability).
"""

# ---------------------------------------------------------------------------
# SLCA attenuation under stress
# ---------------------------------------------------------------------------
SLCA_THERMAL_ATTEN: float = 0.25
"""SLCA degradation coefficient for thermal stress.

~20 % SLCA degradation at full thermal stress (θ = 1).
Heat stress degrades worker conditions (L), increases emergency emissions (C),
rushes handling (P), and strains distribution (R).
"""

SLCA_SURPLUS_ATTEN: float = 0.08
"""SLCA degradation coefficient for inventory surplus.

~11 % SLCA degradation at 1.5× surplus.
Market flooding reduces price transparency (P), overwhelms infrastructure (R),
and degrades labour conditions (L).
"""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax: π(a) = exp(x_a − max(x)) / Σ exp(x_i − max(x))."""
    e = np.exp(x - x.max())
    return e / e.sum()


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
    y_hat : LSTM demand point forecast (units / step).
    temp : current temperature (deg C).
    supply_hat : optional. Holt-Winters supply point forecast (units).
        When omitted, phi_6 is zero (neutral).
    supply_std : optional. Holt-Winters one-step-ahead residual standard
        deviation. When omitted, phi_7 is zero.
    demand_std : optional. LSTM one-step-ahead residual standard
        deviation. When omitted, phi_8 is zero.
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

    # Supply uncertainty: coefficient of variation of the Holt-Winters
    # one-step-ahead residuals, clipped to the unit interval.
    if supply_hat is None or supply_std is None:
        supply_uncertainty = 0.0
    else:
        sh = abs(float(supply_hat))
        su = float(supply_std) / max(sh, 1.0)
        supply_uncertainty = float(np.clip(su, 0.0, 1.0))

    # Demand uncertainty: coefficient of variation of the LSTM
    # one-step-ahead residuals, clipped to the unit interval. Uses a
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


def compute_thermal_stress(temp: float) -> float:
    """Compute normalised thermal stress θ ∈ [0, 1].

    θ = clamp((T − T₀) / ΔT_max, 0, 1)

    Parameters
    ----------
    temp : ambient temperature in °C.

    Returns
    -------
    Normalised thermal stress.
    """
    return min(max((temp - THERMAL_T0) / THERMAL_DELTA_MAX, 0.0), 1.0)


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
    theta_forecast_delta: np.ndarray | None = None,
) -> tuple[int, np.ndarray]:
    """Select routing action based on mode-specific softmax policy.

    Parameters
    ----------
    mode : operating mode (``static``, ``hybrid_rl``, ``no_pinn``,
           ``no_slca``, ``agribrain``, ``no_context``, ``mcp_only``,
           ``pirag_only``).
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
    supply_hat : Holt-Winters supply point forecast (units). Feeds
        ``phi_6`` (centered supply point).
    supply_std : Holt-Winters in-sample residual std (units). Feeds
        ``phi_7`` (supply uncertainty CV).
    demand_std : LSTM in-sample residual std (units). Feeds ``phi_8``
        (demand uncertainty CV). The forecast kwargs default to None so
        legacy callers still work; missing values yield zero contribution
        on the corresponding phi channels.
    price_signal : optional demand-volatility Bollinger z-score used
        as a market-pressure proxy. Feeds ``phi_9`` clipped to [-1, 1].
    theta_forecast_delta : optional (3, 3) learned correction added to
        THETA[:, 6:9] at inference. Provided by ForecastWeightsLearner.
        The hand-calibrated THETA stays fixed; only this delta moves
        with training. The price column (THETA[:, 9]) stays hand-
        calibrated and is not learned.

    Returns
    -------
    (action_index, probability_vector)
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode: {mode!r}. Must be one of {VALID_MODES}")

    # Static is ALWAYS cold chain, regardless of scenario
    if mode == "static":
        return 0, np.array([1.0, 0.0, 0.0])

    phi = build_feature_vector(
        rho, inv, y_hat, temp,
        supply_hat=supply_hat,
        supply_std=supply_std,
        demand_std=demand_std,
        price_signal=price_signal,
    )
    gamma = np.array([policy.gamma_coldchain, policy.gamma_local, policy.gamma_recovery])

    # Cyber outage: processor offline from hour 24 (applies to all non-static modes).
    # The policy distribution is the Bernoulli over (fail -> cold chain, succeed ->
    # local redistribution); recovery is not a valid response during outage.
    # Recovery mass is zero, the other two carry the reroute Bernoulli.
    # The sampled action is still drawn via a single rng.random() call so the
    # RNG consumption pattern (and therefore per-seed reproducibility) matches
    # the original pre-fix behaviour for downstream metrics.
    if scenario == "cyber_outage" and hour >= 24.0:
        p_success = CYBER_REROUTE_PROB.get(mode, 0.50)
        probs = np.array([1.0 - p_success, p_success, 0.0])
        if deterministic:
            return int(np.argmax(probs)), probs
        if rng.random() < p_success:
            return 1, probs
        return 0, probs

    elif mode == "hybrid_rl":
        logits = THETA @ phi + gamma * tau

    elif mode == "no_pinn":
        # Same logit construction as agribrain. The PINN ablation lives
        # entirely in spoilage-risk computation upstream: the simulator
        # routes rho through compute_spoilage instead of
        # compute_spoilage_pinn, so phi[0], phi[4], phi[5] differ but the
        # reward-shaping terms here stay aligned with the full system.
        logits = THETA @ phi + gamma * tau + SLCA_BONUS + SLCA_RHO_BONUS * rho

    elif mode == "no_slca":
        logits = THETA @ phi + gamma * tau + NO_SLCA_OFFSET

    else:  # agribrain, no_context, mcp_only, pirag_only
        # All four share the same base logits; what differs is the upstream
        # context_modifier (full / masked subset / zeroed) applied below.
        logits = THETA @ phi + gamma * tau + SLCA_BONUS + SLCA_RHO_BONUS * rho

    # Learned forecast-column correction. ForecastWeightsLearner owns a
    # (3, 3) delta trained via REINFORCE on phi[6:9]. It is zero-initialised
    # and shrinks back toward zero under a Gaussian prior, so the default
    # behaviour is bit-identical to the hand-calibrated policy until the
    # learner observes enough reward signal.
    if theta_forecast_delta is not None:
        logits = logits + np.asarray(theta_forecast_delta) @ phi[6:9]

    if role_bias is not None:
        logits = logits + role_bias

    if context_modifier is not None:
        # SLCA amplification based on piRAG context signal magnitude.
        amp = slca_amp_coeff if slca_amp_coeff is not None else 0.25
        slca_amplification = 1.0 + amp * min(abs(context_modifier[1]), 1.0)
        slca_boost = (SLCA_BONUS + SLCA_RHO_BONUS * rho) * (slca_amplification - 1.0)
        logits = logits + context_modifier + slca_boost

    probs = _softmax(logits)

    # Governance override: fires only for context-enabled modes (those that
    # build a context_modifier). Stated in probability space so the
    # condition is auditable without reference to the raw logit scale: it
    # fires when the policy's confidence in cold-chain is below the
    # calibration-derived ceiling AND local-redistribute dominates
    # cold-chain by the calibration-derived margin.
    if context_modifier is not None:
        if (probs[0] < GOVERNANCE_CC_PROB_CEILING
                and probs[1] - probs[0] > GOVERNANCE_LOCAL_ADVANTAGE_MIN):
            return 1, np.array([0.0, 1.0, 0.0])

    if deterministic:
        return int(np.argmax(probs)), probs
    return int(rng.choice(len(ACTIONS), p=probs)), probs
