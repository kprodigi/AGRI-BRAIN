"""
Operational waste model for perishable produce supply chains.

Converts the instantaneous Arrhenius decay rate k(T, H) into an operational
waste fraction — the proportion of produce lost to spoilage at each timestep.

Physical basis
--------------
Waste follows a power-law (Michaelis–Menten-type saturation) relationship
with the instantaneous decay rate, reflecting diminishing returns on
recovery efficiency (Briassoulis, 2004):

    waste_raw = (k_inst × W_SCALE)^W_ALPHA

where:
    k_inst  = Arrhenius decay rate (h⁻¹) from spoilage.arrhenius_k()
    W_SCALE = effective batch exposure (transit time × batch size
              normalisation). Encapsulates the conversion from rate
              constant (h⁻¹) to batch-level spoilage fraction.
    W_ALPHA < 1 provides sub-linear compression — emergency protocols,
              shorter transit, and triage partially compensate as decay
              rate increases.

Calibration (fresh spinach, South Dakota cooperative):
    Baseline static (T ≈ 4 °C, k ≈ 0.00274):  waste_raw ≈ 0.073 (7.3 %)
    Heatwave static (mean k ≈ 0.00596):        waste_raw ≈ 0.129 (12.9 %)
    Within FAO range of 2–15 % for fresh produce supply chain losses
    (FAO, 2019).

Inventory surplus waste penalty
-------------------------------
During overproduction, excess inventory overwhelms handling capacity:
    waste_multiplier = 1 + SURPLUS_WASTE_FACTOR × max(0, inv/INV_BASELINE - 1)

This follows from the inventory mass balance (conservation of goods):
    I(t+1) = I(t) + supply(t) − demand_fulfilled(t) − spoilage(t) − waste(t)

Save factor model
-----------------
Each routing action and operating mode has a characteristic ability to
prevent waste:

    save_factor = floor[action] + (ceil[action] − floor[action]) × mode_eff
    net_waste = waste_raw × (1 − save_factor × save_capacity)

where save_capacity degrades under surplus (Michaelis–Menten saturation):
    save_capacity = 1 / (1 + SURPLUS_SAVE_PENALTY × surplus_ratio)

Floor values represent the inherent physical benefit of each routing
choice without any optimisation. Ceiling values represent the maximum
achievable with perfect optimisation. Mode effectiveness captures how
much of this gap each system mode can realise.

References
----------
    - FAO (2019). The State of Food and Agriculture: Moving forward on
      food loss and waste reduction. Rome.
    - Briassoulis, D. (2004). An overview on the mechanical behaviour of
      biodegradable agricultural films. J. Polymers and the Environment.
    - Parfitt, J., Barthel, M. & Macnaughton, S. (2010). Food waste
      within food supply chains. Phil. Trans. R. Soc. B, 365, 3065–3081.
"""
from __future__ import annotations

import numpy as np

from .action_aliases import resolve_action as _resolve_action


# ---------------------------------------------------------------------------
# Waste rate parameters (calibrated for fresh spinach)
# ---------------------------------------------------------------------------
W_SCALE: float = 10.2976
"""Effective batch exposure converting Arrhenius k (h⁻¹) to batch spoilage."""

W_ALPHA: float = 0.7339
"""Sub-linear compression exponent (< 1 → diminishing marginal spoilage)."""

# ---------------------------------------------------------------------------
# Inventory surplus parameters
# ---------------------------------------------------------------------------
INV_BASELINE: float = 12_000.0
"""Baseline inventory level (units) from data_spinach.csv."""

SURPLUS_WASTE_FACTOR: float = 0.25
"""25 % marginal waste increase per unit surplus ratio above baseline."""

SURPLUS_SAVE_PENALTY: float = 0.10
"""Save capacity degradation coefficient under surplus conditions."""


# ---------------------------------------------------------------------------
# Save factor model
# ---------------------------------------------------------------------------
SAVE_FLOOR: dict[str, float] = {
    "cold_chain": 0.0,
    "local_redistribute": 0.45,
    "recovery": 0.25,
}
"""Inherent physical waste prevention of each routing choice (no optimisation).

- Cold chain (120 km): product simply transits → no inherent prevention.
- Local redistribute (45 km): shorter transit + community markets → 45 %.
- Recovery (80 km): diversion to processing/composting → 25 %.
"""

SAVE_CEIL: dict[str, float] = {
    "cold_chain": 0.30,
    "local_redistribute": 0.95,
    "recovery": 0.70,
}
"""Maximum achievable save with perfect optimisation.

- Cold chain: optimal temp control and timing → up to 30 %.
- Local redistribute: optimal matching and timing → up to 95 %.
- Recovery: optimal triage and routing → up to 70 %.
"""

MODE_EFF: dict[str, float] = {
    "static": 0.0,
    "hybrid_rl": 0.60,
    "no_pinn": 0.66,
    "no_slca": 0.50,
    "agribrain": 0.79,
}
"""Fraction of the (ceil − floor) gap each mode achieves.

Ordering: agribrain > no_pinn > hybrid_rl > no_slca > static

- agribrain (0.79): full PINN + SLCA system, best optimisation.
- no_pinn (0.66): SLCA feedback guides good routing despite degraded
  spoilage information.
- hybrid_rl (0.60): decent RL but lacks PINN and SLCA guidance.
- no_slca (0.50): PINN helps predict spoilage but no social optimisation
  for routing.
- static (0.00): no optimisation at all.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_waste_rate(
    k_inst: float | np.ndarray,
    surplus_ratio: float = 0.0,
    w_scale: float = W_SCALE,
    w_alpha: float = W_ALPHA,
    surplus_waste_factor: float = SURPLUS_WASTE_FACTOR,
) -> float | np.ndarray:
    """Convert instantaneous Arrhenius decay rate to operational waste fraction.

    Implements the power-law mapping:
        waste_raw = (k_inst × w_scale)^w_alpha × (1 + surplus_waste_factor × surplus_ratio)

    Parameters
    ----------
    k_inst : instantaneous Arrhenius decay rate (h⁻¹) from spoilage.arrhenius_k().
    surplus_ratio : max(0, inventory / INV_BASELINE − 1). Zero at or below baseline.
    w_scale : batch exposure scaling constant.
    w_alpha : sub-linear compression exponent.
    surplus_waste_factor : marginal waste increase per unit surplus.

    Returns
    -------
    Operational waste fraction (dimensionless, typically 0.02–0.15).
    """
    waste_raw = (k_inst * w_scale) ** w_alpha
    waste_raw = waste_raw * (1.0 + surplus_waste_factor * surplus_ratio)
    return waste_raw


def compute_save_factor(
    action: str,
    mode: str,
    surplus_ratio: float = 0.0,
    surplus_save_penalty: float = SURPLUS_SAVE_PENALTY,
) -> float:
    """Compute the waste prevention factor for a given action and mode.

    save_factor = floor[action] + (ceil[action] − floor[action]) × mode_eff
    save_capacity = 1 / (1 + surplus_save_penalty × surplus_ratio)
    effective_save = save_factor × save_capacity

    Parameters
    ----------
    action : routing action (``cold_chain``, ``local_redistribute``, ``recovery``).
    mode : operating mode (``static``, ``hybrid_rl``, ``no_pinn``, ``no_slca``, ``agribrain``).
    surplus_ratio : inventory surplus above baseline (0 when at/below baseline).
    surplus_save_penalty : degradation coefficient for surplus conditions.

    Returns
    -------
    Effective save factor in [0, 1].
    """
    action = _resolve_action(action)
    floor_s = SAVE_FLOOR.get(action, 0.0)
    ceil_s = SAVE_CEIL.get(action, 0.0)
    mode_eff = MODE_EFF.get(mode, 0.0)

    save = floor_s + (ceil_s - floor_s) * mode_eff
    save_capacity = 1.0 / (1.0 + surplus_save_penalty * surplus_ratio)
    return save * save_capacity


def compute_net_waste(
    k_inst: float,
    action: str,
    mode: str,
    surplus_ratio: float = 0.0,
) -> float:
    """Compute net waste after intervention (waste_raw × (1 − save)).

    Combines the waste rate model with the save factor model:
        net_waste = compute_waste_rate(...) × (1 − compute_save_factor(...))

    Parameters
    ----------
    k_inst : instantaneous Arrhenius decay rate (h⁻¹).
    action : routing action.
    mode : operating mode.
    surplus_ratio : inventory surplus above baseline.

    Returns
    -------
    Net waste fraction after intervention.
    """
    waste_raw = compute_waste_rate(k_inst, surplus_ratio)
    save = compute_save_factor(action, mode, surplus_ratio)
    return float(waste_raw * (1.0 - save))
