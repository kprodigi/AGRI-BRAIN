"""
Regime-aware contextual softmax policy for routing decisions.

Implements the softmax action selection described in Section 4.3 of the
AGRI-BRAIN paper.  Given a 6-dimensional feature vector φ(s) extracted
from the current supply chain state, the policy computes action
probabilities via:

    π(a | s) = softmax(Θ φ(s) + γ τ + bonus(mode, ρ))

Feature vector design (6 features)
-----------------------------------
    φ₀ = freshness       = 1 − ρ
    φ₁ = inv_pressure    = min(inv / capacity, 1)
    φ₂ = demand_signal   = min(ŷ / baseline_demand, 1)
    φ₃ = thermal_stress  = clamp((T − T₀) / ΔT_max, 0, 1)
    φ₄ = spoilage_urgency = ρ
    φ₅ = interaction     = ρ × inv_pressure

THETA matrix (3 actions × 6 features)
--------------------------------------
Each entry has a defensible sign based on supply chain economics:

                       fresh  inv_press  demand  thermal  spoilage  interact
    ColdChain:           +       −         +       −        −         −
    LocalRedistribute:   0       +         −       +        +         +
    Recovery:            −       −         −       +        +         −

Mode-specific bonus terms
-------------------------
    - hybrid_rl:  Θ φ + γ τ              (base RL, no SLCA/PINN)
    - no_pinn:    + 0.5 × SLCA_bonus     (SLCA at reduced strength)
    - no_slca:    + NO_SLCA_OFFSET       (conservative / CC-heavy)
    - agribrain:  + SLCA_BONUS + SLCA_RHO_BONUS × ρ  (full system)
    - static:     always cold_chain       (no optimisation)

Cyber outage handling
---------------------
During a cyber outage (processor offline from hour 24), rerouting success
depends on each mode's autonomous intelligence (edge computing, cached
policies).  When rerouting fails, the shipment defaults to cold chain.

SLCA quality attenuation
-------------------------
Under physical stress (thermal or surplus), all SLCA pillars degrade:

    slca_quality = 1 / (1 + α_thermal × θ + α_surplus × surplus_ratio)

This is applied equally to all modes within a scenario, preserving
cross-method orderings.

References
----------
    - Sutton, R.S. & Barto, A.G. (2018). Reinforcement Learning: An
      Introduction (2nd ed.). MIT Press. [Softmax policy, Ch. 2.8]
    - Luce, R.D. (1959). Individual Choice Behavior. John Wiley & Sons.
      [Choice axiom / softmax derivation]
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

VALID_MODES: list[str] = ["static", "hybrid_rl", "no_pinn", "no_slca", "agribrain"]
"""Valid operating modes for the softmax policy."""

# ---------------------------------------------------------------------------
# Feature normalisation constants
# ---------------------------------------------------------------------------
INV_CAPACITY: float = 15_000.0
"""Inventory normalisation capacity (units). baseline_inv × 1.25 headroom."""

BASELINE_DEMAND: float = 20.0
"""Baseline demand normalisation (units / 15-min step)."""

THERMAL_T0: float = 4.0
"""Ideal cold-chain temperature (°C)."""

THERMAL_DELTA_MAX: float = 20.0
"""Maximum temperature deviation for normalisation (°C)."""

# ---------------------------------------------------------------------------
# THETA matrix (3 actions × 6 features)
# ---------------------------------------------------------------------------
THETA: np.ndarray = np.array([
    [ 0.5,  -0.3,   0.4,  -0.5,  -2.0,  -1.0],   # ColdChain
    [ 0.0,   0.5,  -0.2,   0.5,   2.0,   1.5],    # LocalRedistribute
    [-0.5,  -0.3,  -0.2,   0.3,   1.5,  -0.3],    # Recovery
])
"""Policy weight matrix.

Calibrated so that the base policy (hybrid_rl) produces approximately
45 % CC / 45 % LR / 10 % Rec at baseline conditions, shifting toward
more LR/Rec under thermal stress or spoilage urgency.
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

NOPINN_SLCA_SCALE: float = 0.5
"""Scaling factor for no_pinn mode's SLCA bonus (degraded spoilage prediction)."""

NO_SLCA_OFFSET: np.ndarray = np.array([0.6, -0.3, -0.4])
"""Logit offset for no_slca mode (conservative, CC-heavy routing).

Without SLCA feedback, the system defaults toward cold chain — the "safe"
choice — since it cannot assess social value of alternatives.
"""

# ---------------------------------------------------------------------------
# Cyber outage: rerouting success probabilities
# ---------------------------------------------------------------------------
CYBER_REROUTE_PROB: dict[str, float] = {
    "static": 0.25,
    "hybrid_rl": 0.55,
    "no_pinn": 0.65,
    "no_slca": 0.60,
    "agribrain": 0.82,
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
) -> np.ndarray:
    """Construct the 6-dimensional state feature vector φ(s).

    Parameters
    ----------
    rho : spoilage risk (1 − shelf_left), in [0, 1].
    inv : current inventory level (units).
    y_hat : demand forecast (units / step).
    temp : current temperature (°C).

    Returns
    -------
    φ = [freshness, inv_pressure, demand_signal, thermal_stress,
         spoilage_urgency, interaction]
    """
    freshness = 1.0 - rho
    inv_pressure = min(inv / INV_CAPACITY, 1.0)
    demand_signal = min(y_hat / BASELINE_DEMAND, 1.0)
    thermal_stress = min(max((temp - THERMAL_T0) / THERMAL_DELTA_MAX, 0.0), 1.0)
    spoilage_urgency = rho
    interaction = rho * inv_pressure

    return np.array([freshness, inv_pressure, demand_signal,
                     thermal_stress, spoilage_urgency, interaction])


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
    rag_context: dict | None = None,
) -> tuple[int, np.ndarray]:
    """Select routing action based on mode-specific softmax policy.

    Parameters
    ----------
    mode : operating mode (``static``, ``hybrid_rl``, ``no_pinn``,
           ``no_slca``, ``agribrain``).
    rho : spoilage risk.
    inv : current inventory.
    y_hat : demand forecast.
    temp : current temperature (°C).
    tau : volatility indicator (1.0 if anomaly, 0.0 otherwise).
    policy : Policy object with gamma_* and distance attributes.
    rng : numpy random generator.
    scenario : current scenario name.
    hour : hours since start (for cyber outage timing).
    role_bias : optional per-role logit bias vector (3,).
    deterministic : if True, use argmax instead of sampling.
    rag_context : optional RAG-retrieved policy context for logging.

    Returns
    -------
    (action_index, probability_vector)
    """
    # Static is ALWAYS cold chain, regardless of scenario
    if mode == "static":
        return 0, np.array([1.0, 0.0, 0.0])

    phi = build_feature_vector(rho, inv, y_hat, temp)
    gamma = np.array([policy.gamma_coldchain, policy.gamma_local, policy.gamma_recovery])

    # Cyber outage: processor offline from hour 24 (applies to all non-static modes)
    if scenario == "cyber_outage" and hour >= 24.0:
        p_success = CYBER_REROUTE_PROB.get(mode, 0.50)
        if rng.random() < p_success:
            return 1, np.array([0.0, 1.0, 0.0])
        else:
            return 0, np.array([1.0, 0.0, 0.0])

    elif mode == "hybrid_rl":
        logits = THETA @ phi + gamma * tau

    elif mode == "no_pinn":
        slca_total = (SLCA_BONUS + SLCA_RHO_BONUS * rho) * NOPINN_SLCA_SCALE
        logits = THETA @ phi + gamma * tau + slca_total

    elif mode == "no_slca":
        logits = THETA @ phi + gamma * tau + NO_SLCA_OFFSET

    else:  # agribrain
        logits = THETA @ phi + gamma * tau + SLCA_BONUS + SLCA_RHO_BONUS * rho

    if role_bias is not None:
        logits = logits + role_bias

    probs = _softmax(logits)

    if deterministic:
        return int(np.argmax(probs)), probs
    return int(rng.choice(len(ACTIONS), p=probs)), probs
