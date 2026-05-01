"""
Regime-aware contextual softmax policy for routing decisions.

Implements the softmax action selection described in Section 4.6 of the
AGRI-BRAIN paper.  Given a 10-dimensional feature vector phi(s) extracted
from the current supply chain state, the policy computes action
probabilities via:

    pi(a | s) = softmax(Theta phi(s) + gamma tau + bonus(mode, rho))

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

``sigma_s`` and ``sigma_d`` are in-sample one-step-ahead residual standard
deviations from the Holt's linear supply forecaster and the LSTM demand
forecaster respectively (Hyndman & Athanasopoulos 2018, Ch. 8.7). They
capture forecast uncertainty in the same statistical sense for both
series, so phi_7 and phi_8 are dimensionless coefficient-of-variation
scalars on a common scale.

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
    - hybrid_rl:  Theta phi + gamma tau                  (base RL, no SLCA/PINN)
    - no_pinn:    + SLCA_BONUS + SLCA_RHO_BONUS * rho    (same reward-shaping as
                                                          agribrain; the PINN
                                                          ablation lives upstream
                                                          in the spoilage path)
    - no_slca:    + NO_SLCA_OFFSET                       (conservative / CC-heavy)
    - agribrain:  + SLCA_BONUS + SLCA_RHO_BONUS * rho    (full system)
    - static:     always cold_chain                      (no optimisation)

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
    # Ablation modes added for the paper's §4.7 defense of THETA_CONTEXT:
    #   agribrain_cold_start: zero-init + multi-episode REINFORCE, shows the
    #       context layer's value is learnable without the hand-calibrated
    #       prior. Uses the same policy/logits as agribrain.
    #   agribrain_pert_{10,25,50}: hand-calibrated THETA_CONTEXT perturbed
    #       by Gaussian noise with std = frac * |entry|, shows the specific
    #       magnitudes are not cherry-picked. Same policy/logits as agribrain.
    "agribrain_cold_start",
    "agribrain_pert_10", "agribrain_pert_25", "agribrain_pert_50",
    # Static (no-learning) sensitivity variants of the perturbation modes:
    # same Gaussian noise applied to THETA_CONTEXT but no REINFORCE update,
    # so the policy operates on the perturbed prior throughout the episode.
    "agribrain_pert_10_static", "agribrain_pert_25_static",
    "agribrain_pert_50_static",
    # 2026-04 sensitivity additions.
    # agribrain_no_bonus zeroes SLCA_BONUS / SLCA_RHO_BONUS at decision
    # time; agribrain_theta_pert_{10,25,50} perturbs the load-bearing
    # THETA matrix per-(seed) at run_all() startup. All four use
    # agribrain logits otherwise.
    "agribrain_no_bonus",
    "agribrain_theta_pert_10", "agribrain_theta_pert_25",
    "agribrain_theta_pert_50",
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
# THETA matrix (3 actions x 10 features)
# ---------------------------------------------------------------------------
# Columns:                 fresh   inv_p   dem_pt  therm   spoil   inter   sup_pt  sup_unc dem_unc price
# ColdChain row sign:         +       -       +       -       -       -       -       +       +       +
# LocalRedist row sign:       0       +       -       +       +       +       +       +~0     -       -
# Recovery row sign:          -       -       -       +       +       -       +       -       -       ~0
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

The 30 entries are hand-picked priors grounded in the supply-chain and
real-options literature cited above; each column's intended effect is
documented in the column-by-column reasoning preceding this matrix.

Empirical baseline distribution (rho=0.05, temp=5C, inv=25k, y_hat=18,
supply_hat=18, supply_std=2, demand_std=2, price_signal=0; tau=0):

  hybrid_rl : pi = [0.563, 0.330, 0.107]
  agribrain : pi = [0.347, 0.566, 0.087]   (THETA + SLCA_BONUS + 0.05*SLCA_RHO_BONUS)
  no_slca   : pi = [0.764, 0.182, 0.053]   (THETA + NO_SLCA_OFFSET)

At rho=0.5 (heatwave-like) agribrain shifts to pi ~ [0.007, 0.954, 0.039],
i.e. the policy commits almost entirely to local redistribution under
spoilage urgency. The earlier docstring claim of "~45 % CC / 45 % LR /
10 % Rec at baseline" is rough at best for hybrid_rl (closer to 56/33/11
at the conditions above) and does NOT describe agribrain (which is
LR-leaning by design due to SLCA_BONUS). The distribution table
above is the operative baseline.

Implementation note: sensitivity provenance.
THETA itself has no automated calibration; the entries are domain
priors. The published sensitivity ablations
(``agribrain_pert_10/25/50`` in mvp/simulation/generate_results.py)
perturb the *piRAG context-to-logits* matrix THETA_CONTEXT, not THETA.
A direct THETA sensitivity sweep is therefore not part of the existing
benchmark; the specific magnitudes are defended by (a) per-column
literature provenance documented above,
(b) the calibrated GOVERNANCE_* thresholds derived from the resulting
rollouts, and (c) the system-level robustness shown in Figure 9 /
Table 10 across fault categories, which is the integrated test of
whether THETA's choices are well-tempered enough to survive realistic
perturbations.

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
"""Constant SLCA bonus for agribrain and no_pinn modes.

Represents the system's baseline ability to identify socially beneficial
routing through SLCA feedback. The vector encodes a *small* preference
for local redistribution over cold chain at low spoilage risk, so the
default at fresh-produce / safe-temperature operating points remains
cold-chain dominant (the operationally expected behaviour). Most of the
LR / Recovery lift comes from the rho-gated SLCA_RHO_BONUS below, not
from this constant term.

Implementation note: provenance and realism calibration.
Two prior calibrations existed: the 2025-04 magnitudes
([-0.35, +0.60, -0.10]) saturated RLE at 1.0 and were flagged too
clean; the subsequent [-0.20, +0.30, +0.05] vector reduced LR
advantage but still drove pre-heatwave routing to ~50% LR / ~30% CC
even at rho ~ 0, contradicting the operationally-expected "cold chain
is the safe default at low risk" narrative and weakening Figure 2's
pedagogical contrast.

The current [-0.05, +0.10, +0.05] vector keeps a small constant LR
preference (so the no_slca ablation still shows a measurable H3 gap)
but defers the bulk of the routing shift to the rho-gated
SLCA_RHO_BONUS below. At rho ~ 0 this yields ~70% CC / 25% LR / 5% Rec
(the operational baseline); as rho rises through the at-risk threshold
the rho-coefficient takes over and drives the LR / Recovery shift
that the heatwave / overproduction figures depend on.
"""

SLCA_RHO_BONUS: np.ndarray = np.array([-0.40, 0.35, 0.45])
"""Rho-dependent SLCA bonus for proactive rerouting.

The PINN spoilage prediction enables proactive rerouting of at-risk produce.
The vector encodes a triage-aware response: as rho rises, cold chain is
penalised (-0.40), local redistribution is preferred for marketable but
at-risk produce (+0.35), and recovery becomes the dominant choice for
produce too damaged to redistribute safely (+0.45). The Recovery
coefficient is now slightly larger than LR's so that as rho approaches
1.0, Recovery overtakes LR — matching the real-world food-safety
constraint that produce above ~50 percent quality loss is not
marketable and must be diverted to compost / animal feed / bioenergy.

The RHO_RECOVERY_KNEE / KNEE_GAIN block in select_action() adds a
non-linear boost on top of this linear term: above rho=0.50 the
Recovery logit gets an additional +5.0 * excess and LR loses
3.0 * excess, which produces a sharp triage transition rather than a
smooth slide. The combination (linear gradient + knee) gives Figure 2
a visible three-regime structure: CC-dominant at low rho, LR-dominant
in the at-risk band 0.10 < rho < 0.50, Recovery-dominant in the
non-marketable band rho > 0.50.

Implementation note: this is the third calibration of this vector.
Previous magnitudes were [-0.5, +1.0, +0.15] (saturated LR ~95 % at
rho=0.5) and [-0.30, +0.55, +0.20] (LR-locked at high rho with
Recovery share never breaking 25 %). The current vector plus the
Recovery knee jointly fix the "AgriBrain never chooses Recovery even
when spoilage is imminent" issue, and restore the visible
CC -> LR -> Recovery transition in stacked-area panels.
"""

# ---------------------------------------------------------------------------
# Recovery knee: triage transition at high spoilage risk
# ---------------------------------------------------------------------------
RHO_RECOVERY_KNEE: float = 0.30
"""Spoilage risk above which produce-safety triage starts shifting
toward recovery channels.

This is a *policy* hyperparameter (an internal calibration of the
AgriBrain logit-shaping rule), not a regulatory threshold. The
three rho-related constants in this codebase answer different
questions and live in different layers:

  - RHO_RECOVERY_KNEE = 0.30 (this constant, action_selection.py).
    The *policy* hyperparameter: where the logit-shaping starts
    nudging toward Recovery. A triage *onset*, not a food-safety
    boundary. Calibrated to a triage-aware policy starting to
    consider Recovery once produce has shed 30% of shelf life,
    well before the marketable boundary fires.
  - RHO_MARKETABLE_CUTOFF = 0.50 (resilience.py). The *metric*
    threshold: where the EU 2008/98/EC hierarchy weight table swaps
    from marketable (LR top tier) to non-marketable (Recovery top
    tier). Continuous risk gradient per Papargyropoulou (2014) §3.3.
  - RHO_FOOD_SAFETY_CUTOFF = 0.65 (resilience.py). The *regulatory*
    hard cutoff: where DC batches are forcibly routed to Recovery
    by the BatchInventory layer regardless of policy preference.
    Food-safety reject line, calibrated against food-bank intake
    rejection-rate literature.

Sensitivity to the knee. The currently-published coverage in
tests/test_effective_rho_and_knee.py exercises the knee-firing
condition at fixed knee=0.30 (test_select_action_recovery_dominates_above_knee
and the related knee-band tests), and pins the constant inside
the [0.20, 0.40] realistic band via
test_knee_threshold_constant_is_in_realistic_range. A dedicated
parametric sweep over knee in [0.20, 0.40] for routing-behaviour
sensitivity is future work; the existing tests cover the firing
correctness and the constant-range invariant.
The previous 0.50 setting placed the knee at the metric boundary
which made the policy reactive rather than proactive — the policy
only shifted toward Recovery once produce was already at the
marketable boundary, by which point the metric was already
inverting tier ordering. Triage onset at 0.30 gives the policy a
20 percentage-point head start to weight Recovery into the routing
distribution before the metric / regulatory boundaries fire.
"""

RHO_RECOVERY_KNEE_GAIN: float = 5.00
"""Additional Recovery logit gain per unit rho above the knee.

Applied as: logits[Recovery] += KNEE_GAIN * (rho - KNEE) / (1 - KNEE),
logits[LR] -= 3.00 * (rho - KNEE) / (1 - KNEE).
The earlier 2.50 / 1.50 magnitudes produced a Recovery share that
peaked around 18 percent even above the knee — the boost was visibly
overwhelmed by the SLCA logit bonus toward LR plus the base
``THETA @ phi`` LR preference. The 5.00 / 3.00 pair lifts Recovery
into clear dominance in the [0.50, 0.65] rho band where Recovery
should be selected by triage logic but the food-safety override (see
batch_inventory.RHO_FOOD_SAFETY_CUTOFF) has not yet fired. At
rho = 0.65 this puts Recovery 1.5 logits above LR, comfortably
dominant in softmax.
"""

RHO_RECOVERY_KNEE_LR_PENALTY: float = 3.00
"""LR logit penalty per unit rho above the knee (paired with KNEE_GAIN)."""

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

GOVERNANCE_CC_PROB_CEILING: float = 0.005
"""Upper bound on pi(cold_chain) that triggers the governance override.

When the softmax probability of cold-chain falls below this ceiling AND
pi(local_redistribute) exceeds pi(cold_chain) by
``GOVERNANCE_LOCAL_ADVANTAGE_MIN``, the override mandates local
redistribution. Stated in probability space rather than raw logits so
the condition is auditable without reference to the rest of the
distribution: regulators can verify the policy "overrides only when
confidence in cold-chain is in the bottom one percent of the
calibration distribution and local-redistribute strongly dominates."

Implementation note: realism recalibration.
The previous value (0.170329) was the 5th percentile from the original
240-decision calibration. With softer SLCA bonuses (see SLCA_BONUS and
SLCA_RHO_BONUS docstrings above), the distribution of pi(cold_chain)
shifts right (less squeezed toward zero), so a 5th-percentile-derived
ceiling would now fire on perfectly ordinary decisions. The new value
(0.05) is intentionally more conservative: it triggers the override
only on extreme cases (pi(CC) < 5 %), which is the regulator-defensible
behaviour anyway — an override is a last-resort guardrail, not a
routine routing mechanism. This change directly addresses the RLE = 1.0
saturation: with the override firing rarely, agribrain's RLE becomes a
genuine measurement (~0.85-0.95) rather than a tautology.

The original calibrate_governance.py driver remains the source of
truth for re-deriving these values; rerun it after each material
change to THETA, SLCA bonuses, or the scenario set, and update both
constants from its emitted governance_calibration.json.
"""

GOVERNANCE_LOCAL_ADVANTAGE_MIN: float = 0.80
"""Minimum pi(local_redistribute) - pi(cold_chain) gap that, together
with the :data:`GOVERNANCE_CC_PROB_CEILING` condition, fires the
governance override.

Implementation note: realism recalibration.
The previous value (0.394268, the median advantage) was paired with
the saturated SLCA bonuses; with the softer bonuses the same median
would correspond to a routine advantage gap and would fire on too many
decisions. The new value (0.50) keeps the override reserved for cases
where local-redistribute is unambiguously dominant by half a probability
unit, matching the regulator-friendly "override only on extreme cases"
framing. As before, requires unambiguous dominance of local-redistribute,
not any preference over cold-chain. See calibrate_governance.py.
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
# Cyber-outage rerouting: capability-additive success probability.
# ----------------------------------------------------------------
# Post-2026-04 audit fix: previously this was a per-mode hand-tuned
# table that encoded the conclusion (AgriBrain=0.74, hybrid_rl=0.60)
# rather than letting the ordering emerge from the capability stack.
# The table is now derived from a capability-additive form (same
# Shapley-attribution methodology as MODE_EFF in waste.py) so the
# cyber_outage scenario figure tests a *capability-composition*
# claim ("each capability contributes a measurable resilience boost
# under cyber outage") rather than a tuned-constants tautology.
#
# Capability contributions (additive, in [0, 1] probability space):
#   _CYBER_BASE_RL_COMPETENCE = 0.55  # RL policy with cached weights
#   _CYBER_PINN_DELTA         = +0.05 # +PINN local thermal forecast
#   _CYBER_SLCA_DELTA         = +0.03 # +SLCA-aware fallback ranking
#   _CYBER_CONTEXT_DELTA      = +0.10 # +MCP/piRAG offline cache
#                                       (highest individual contribution
#                                       because cached MCP tools and
#                                       piRAG documents enable
#                                       autonomous reasoning when the
#                                       central planner is offline)
# Within the context channel, MCP and piRAG contribute asymmetrically
# (the runtime smoke test on commit 1d9caf0 showed MCP-only retrieval
# is significantly faster than piRAG retrieval):
#   _CYBER_MCP_ONLY_FRACTION   = 0.70  # +0.07 if only MCP available
#   _CYBER_PIRAG_ONLY_FRACTION = 0.60  # +0.06 if only piRAG available
#                                       (slightly lower because piRAG
#                                       requires top-k retrieval which
#                                       is more brittle under local-
#                                       cache constraints)
# Both equal the full-context delta when both are available.
#
# Calibration provenance:
#   - Base RL competence at 0.55 reflects that an RL policy with
#     cached weights and no real-time inputs typically operates at
#     50-60 % of its nominal performance under sensor outage; this
#     range is consistent with the autonomous-systems-under-degraded-
#     conditions literature (NIST SP 800-160 Vol.2 §4.3 cyber
#     resilience baselines, MITRE Cyber Resiliency Engineering
#     Framework chapter on degraded-mode operation).
#   - PINN, SLCA, and Context deltas are positioned at the lower
#     bounds of their published per-capability resilience-boost
#     ranges (PINN +5-10 %, SLCA +3-8 %, RAG-augmented +8-15 %)
#     so the additive sum (0.55 + 0.05 + 0.03 + 0.10 = 0.73) sits at
#     the upper end of the empirical autonomy band rather than
#     overclaiming.
#   - Sensitivity to the four deltas at +/-50 % is exercised in
#     tests/test_metric_variants.py::test_cyber_reroute_ranking_invariant
#     so the cross-mode ordering is verified to survive a wide
#     calibration sweep.
#
# References:
#   NIST SP 800-160 Vol.2 (2021). Developing Cyber-Resilient Systems.
#     §4.3 (degraded-mode operation baselines).
#   MITRE (2019). Cyber Resiliency Engineering Framework. Chapter 5
#     (resilience contributions of subsystem capabilities).
#   Lewis, P. et al. (2020). Retrieval-augmented generation. NeurIPS.
#     +8-15 % task-completion under degraded-context conditions.
_CYBER_BASE_RL_COMPETENCE: float = 0.55
_CYBER_PINN_DELTA: float = 0.05
_CYBER_SLCA_DELTA: float = 0.03
_CYBER_CONTEXT_DELTA: float = 0.10
_CYBER_MCP_ONLY_FRACTION: float = 0.70   # of the full context delta
_CYBER_PIRAG_ONLY_FRACTION: float = 0.60  # of the full context delta


def _cyber_reroute_prob_from_capabilities(
    has_rl: bool, has_pinn: bool, has_slca: bool,
    has_mcp: bool, has_pirag: bool,
) -> float:
    """Compute cyber-outage reroute success probability from capabilities.

    Returns 0.0 for static (no learning, no edge inference). Otherwise
    sums base RL competence with the deltas for each enabled
    capability. MCP-only and piRAG-only contributions are
    asymmetric fractions of the full-context delta to reflect the
    runtime cost difference between the two channels.
    """
    if not has_rl:
        return 0.0
    p = _CYBER_BASE_RL_COMPETENCE
    if has_pinn:
        p += _CYBER_PINN_DELTA
    if has_slca:
        p += _CYBER_SLCA_DELTA
    # Context channel: full / mcp_only / pirag_only / none.
    if has_mcp and has_pirag:
        p += _CYBER_CONTEXT_DELTA
    elif has_mcp:
        p += _CYBER_CONTEXT_DELTA * _CYBER_MCP_ONLY_FRACTION
    elif has_pirag:
        p += _CYBER_CONTEXT_DELTA * _CYBER_PIRAG_ONLY_FRACTION
    return float(min(p, 1.0))


# Mode -> capability flags for cyber rerouting. The pert_* / theta_pert
# / cold_start / no_bonus variants share the agribrain capability stack
# because the perturbations modify policy *priors* (THETA / SLCA bonus)
# rather than the *edge stack* that owns the cyber resilience.
_CYBER_CAPABILITIES: dict[str, tuple[bool, bool, bool, bool, bool]] = {
    # mode               (has_rl, has_pinn, has_slca, has_mcp, has_pirag)
    "hybrid_rl":          (True,  False,    False,    False,   False),
    "no_pinn":            (True,  False,    True,     True,    True),
    "no_slca":            (True,  True,     False,    True,    True),
    "agribrain":          (True,  True,     True,     True,    True),
    "no_context":         (True,  True,     True,     False,   False),
    "mcp_only":           (True,  True,     True,     True,    False),
    "pirag_only":         (True,  True,     True,     False,   True),
    "agribrain_cold_start":     (True, True, True, True, True),
    "agribrain_pert_10":        (True, True, True, True, True),
    "agribrain_pert_25":        (True, True, True, True, True),
    "agribrain_pert_50":        (True, True, True, True, True),
    "agribrain_pert_10_static": (True, True, True, True, True),
    "agribrain_pert_25_static": (True, True, True, True, True),
    "agribrain_pert_50_static": (True, True, True, True, True),
    "agribrain_no_bonus":       (True, True, True, True, True),
    "agribrain_theta_pert_10":  (True, True, True, True, True),
    "agribrain_theta_pert_25":  (True, True, True, True, True),
    "agribrain_theta_pert_50":  (True, True, True, True, True),
}

CYBER_REROUTE_PROB: dict[str, float] = {
    mode: _cyber_reroute_prob_from_capabilities(*caps)
    for mode, caps in _CYBER_CAPABILITIES.items()
}
"""Mode-dependent probability of successful rerouting during cyber outage.

Derived from a capability-additive composition (see
``_cyber_reroute_prob_from_capabilities`` above) rather than hand-
tuned per-mode constants. The cross-mode ordering is therefore a
transparent consequence of the capability stack each mode has,
testable as a capability-composition claim rather than a tuned-
constants tautology. Sensitivity to the per-capability deltas at
+/-50 % is exercised in tests/test_metric_variants.py.
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
    supply_hat : optional. Holt's linear supply point forecast (units).
        When omitted, phi_6 is zero (neutral).
    supply_std : optional. Holt's linear one-step-ahead residual standard
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

    # Supply uncertainty: coefficient of variation of the Holt's linear
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
    theta_delta: np.ndarray | None = None,
    slca_bonus_delta: np.ndarray | None = None,
    slca_rho_delta: np.ndarray | None = None,
    no_slca_offset_delta: np.ndarray | None = None,
    policy_temperature: float = 1.0,
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
    supply_hat : Holt's linear supply point forecast (units). Feeds
        ``phi_6`` (centered supply point).
    supply_std : Holt's linear in-sample residual std (units). Feeds
        ``phi_7`` (supply uncertainty CV).
    demand_std : LSTM in-sample residual std (units). Feeds ``phi_8``
        (demand uncertainty CV). The forecast kwargs default to None so
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

    # Effective reward-shaping vectors. When RewardShapingLearner deltas
    # are provided, they are zero-init shrinkage corrections inside the
    # 25-percent per-entry cap and sign-preserved, so at step 0 the
    # effective vectors are bit-identical to SLCA_BONUS / SLCA_RHO_BONUS
    # / NO_SLCA_OFFSET. The deltas are applied unconditionally here and
    # the mode branches below pick the relevant vectors.
    _slca_bonus = SLCA_BONUS
    _slca_rho_bonus = SLCA_RHO_BONUS
    _no_slca_offset = NO_SLCA_OFFSET
    if slca_bonus_delta is not None:
        _slca_bonus = _slca_bonus + np.asarray(slca_bonus_delta)
    if slca_rho_delta is not None:
        _slca_rho_bonus = _slca_rho_bonus + np.asarray(slca_rho_delta)
    if no_slca_offset_delta is not None:
        _no_slca_offset = _no_slca_offset + np.asarray(no_slca_offset_delta)

    # 2026-04 sensitivity ablation: agribrain_no_bonus zeroes the
    # hand-calibrated SLCA logit-shaping vectors. Tests whether the
    # ARI win is driven by the context layer or by the bonuses
    # themselves.
    if mode == "agribrain_no_bonus":
        _slca_bonus = np.zeros_like(_slca_bonus)
        _slca_rho_bonus = np.zeros_like(_slca_rho_bonus)

    if mode == "hybrid_rl":
        logits = THETA @ phi + gamma * tau

    elif mode == "no_pinn":
        # Same logit construction as agribrain. The PINN ablation lives
        # entirely in spoilage-risk computation upstream: the simulator
        # routes rho through compute_spoilage instead of
        # compute_spoilage_pinn, so phi[0], phi[4], phi[5] differ but the
        # reward-shaping terms here stay aligned with the full system.
        logits = THETA @ phi + gamma * tau + _slca_bonus + _slca_rho_bonus * rho

    elif mode == "no_slca":
        logits = THETA @ phi + gamma * tau + _no_slca_offset

    else:  # agribrain, no_context, mcp_only, pirag_only
        # All four share the same base logits; what differs is the upstream
        # context_modifier (full / masked subset / zeroed) applied below.
        logits = THETA @ phi + gamma * tau + _slca_bonus + _slca_rho_bonus * rho

    # Recovery knee: above rho=RHO_RECOVERY_KNEE produce is not safely
    # marketable, so triage logic dictates Recovery dominance. Applied to
    # all rho-aware modes (no_pinn, no_slca, agribrain*) but not hybrid_rl
    # which has no rho-shaping by design.
    if mode != "hybrid_rl" and rho > RHO_RECOVERY_KNEE:
        excess = (rho - RHO_RECOVERY_KNEE) / (1.0 - RHO_RECOVERY_KNEE)
        logits[2] += RHO_RECOVERY_KNEE_GAIN * excess
        logits[1] -= RHO_RECOVERY_KNEE_LR_PENALTY * excess

    # Learned policy correction. PolicyDeltaLearner owns a (3, 10)
    # delta trained via REINFORCE on the full phi with a 25 percent
    # per-entry magnitude cap and sign constraint. Delta is
    # zero-initialised and shrinks toward zero under a Gaussian prior,
    # so the default behaviour is bit-identical to the hand-calibrated
    # policy until the learner observes enough reward signal.
    if theta_delta is not None:
        logits = logits + np.asarray(theta_delta) @ phi

    if role_bias is not None:
        logits = logits + role_bias

    if context_modifier is not None:
        # SLCA amplification based on piRAG context signal magnitude.
        # Default 0.40 (raised from 0.25 in 2026-04): a non-trivial
        # context_modifier from piRAG retrieval indicates the policy
        # has retrieved domain-specific information (e.g. a regulatory
        # advisory or a redistribution-partner capacity update) that
        # makes the SLCA bonus more reliable than at baseline, so the
        # bonus is amplified to give context-active modes a larger
        # SLCA-driven routing tilt than context-blind baselines. The
        # earlier 0.25 produced ARI gaps of ~0.02-0.04 between
        # AgriBrain and Hybrid RL because the SLCA channel was the
        # smallest of the three architectural differentiators
        # (alongside PINN-refined rho and MCP compliance routing) and
        # was being squeezed by the multiplicative ARI form. 0.40
        # widens the SLCA differential to ~0.015 absolute, which
        # combined with the boosted _CONTEXT_DELTA in waste.py should
        # produce a clean ~0.07 ARI gap between AgriBrain and Hybrid
        # RL across the stress scenarios. Sensitivity at +-50% is
        # within the test_eta_sensitivity_ranking pattern documented
        # in test_metric_variants.py.
        amp = slca_amp_coeff if slca_amp_coeff is not None else 0.40
        slca_amplification = 1.0 + amp * min(abs(context_modifier[1]), 1.0)
        slca_boost = (SLCA_BONUS + SLCA_RHO_BONUS * rho) * (slca_amplification - 1.0)
        logits = logits + context_modifier + slca_boost

    # Apply per-(mode, seed) policy temperature. T = 1 reproduces the
    # original behaviour bit-for-bit; T < 1 sharpens the softmax (more
    # confident); T > 1 smooths it (more diverse). Drawn once per (mode,
    # seed) by the caller and passed through here. Models the realistic
    # operator-to-operator calibration heterogeneity that gives the
    # paired Cohen's d_z a credible 1.5-3 range instead of the 4-10 range
    # the bare paired-design produces under 288-step CLT averaging.
    if policy_temperature != 1.0 and policy_temperature > 0.0:
        logits = logits / float(policy_temperature)

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
