"""
Operational waste model for perishable produce supply chains.

Converts the instantaneous Arrhenius decay rate k(T, H) into an operational
waste fraction — the proportion of produce lost to spoilage at each timestep.

Physical basis
--------------
Waste follows a bounded, sub-linear power-law mapping from the instantaneous
decay rate. This is a stylised simulation outcome model, not a fitted spinach
loss model:

    waste_raw = (k_inst × W_SCALE)^W_ALPHA

where:
    k_inst  = Arrhenius decay rate (h⁻¹) from spoilage.arrhenius_k()
    W_SCALE = effective batch exposure (transit time × batch size
              normalisation). Encapsulates the conversion from rate
              constant (h⁻¹) to batch-level spoilage fraction.
    W_ALPHA < 1 provides sub-linear compression — emergency protocols,
              shorter transit, and triage partially compensate as decay
              rate increases.

Synthetic parameterization
--------------------------
W_SCALE and W_ALPHA were selected to place the two declared benchmark anchors
at approximately:

    Baseline static (T ≈ 4 °C, k ≈ 0.00274):  waste_raw ≈ 0.07  (7 %)
    Heatwave  static (mean k ≈ 0.00596):       waste_raw ≈ 0.13 (13 %)

These anchors are modelling assumptions used to give the synthetic case study
an interpretable range. FAO and related aggregate loss reports motivate the
order of magnitude only; they do not provide a two-point spinach dataset from
which these parameters could be estimated. Accordingly, absolute waste values
are simulation outputs rather than externally validated estimates.

Inventory surplus waste penalty
-------------------------------
During overproduction, excess inventory overwhelms handling capacity:
    waste_multiplier = 1 + SURPLUS_WASTE_FACTOR × max(0, inv/INV_BASELINE - 1)

This follows from the inventory mass balance (conservation of goods):
    I(t+1) = I(t) + supply(t) − demand_fulfilled(t) − spoilage(t) − waste(t)

Save factor model
-----------------
Each routing action has a fixed, mode-independent ability to prevent waste:

    action_save = {cold_chain: 0.00, local_redistribute: 0.45,
                   recovery: 0.25}[action]
    net_waste = waste_raw × (1 − action_save × save_capacity)

where save_capacity degrades under a declared reciprocal saturation rule:
    save_capacity = 1 / (1 + SURPLUS_SAVE_PENALTY × surplus_ratio)

The benchmark uses these exact action coefficients; there is no interpolation
to an action ceiling or mode-specific efficacy. Identical actions under
identical physical conditions therefore have identical outcomes; architectural
comparisons can differ only because their policies select different actions.

References
----------
    - FAO (2019). The State of Food and Agriculture: Moving forward on
      food loss and waste reduction. FAO, Rome. ISBN 978-92-5-131789-1.
    - Gustavsson, J., Cederberg, C., Sonesson, U., van Otterdijk, R.
      & Meybeck, A. (2011). Global Food Losses and Food Waste:
      Extent, Causes and Prevention. FAO, Rome.
    - Tijskens, L.M.M. & Polderdijk, J.J. (1996). A generic model for
      keeping quality of vegetable produce during storage and
      distribution. Journal of Food Engineering, 30(1), 105–123.
    - van Boekel, M.A.J.S. (2008). Kinetic modeling of food quality:
      a critical review. Comprehensive Reviews in Food Science and
      Food Safety, 7(1), 144–158.
    - Parfitt, J., Barthel, M. & Macnaughton, S. (2010). Food waste
      within food supply chains: quantification and potential for
      change to 2050. Philosophical Transactions of the Royal
      Society B, 365(1554), 3065–3081.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .action_aliases import resolve_action as _resolve_action
from .mode_capabilities import PUBLICATION_BENCHMARK_MODES

# ---------------------------------------------------------------------------
# Waste-rate parameters for the synthetic spinach case study
# ---------------------------------------------------------------------------
W_SCALE: float = 10.2976
"""Effective batch exposure converting Arrhenius k (h⁻¹) to batch spoilage.

Algebraically fitted to two declared simulation anchors:

    (k=0.00274 h⁻¹, waste≈0.07)  → developed-country refrigerated baseline
    (k=0.00596 h⁻¹, waste≈0.13)  → temperate heatwave stress

The fitted precision is retained for run-to-run reproducibility. The anchors
are not field measurements and do not constitute external validation.
"""

W_ALPHA: float = 0.7339
"""Sub-linear compression exponent (< 1 → diminishing marginal spoilage).

Co-fitted with ``W_SCALE`` to the two declared simulation anchors above.
"""

WASTE_CAP: float = 0.15
"""Declared upper bound applied after the inventory-surplus multiplier."""

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
"""Exact action-specific waste-saving coefficients for the synthetic case.

``SAVE_FLOOR`` is a legacy public name retained for compatibility; there is no
paired ceiling in the live equation. The values are modelling assumptions, not
measured intervention effects.
"""

# Backward-compatible exports. They are deliberately mode-neutral and cover
# exactly the locked public modes; callers must not use a model name as an
# input to a physical outcome equation.
_KNOWN_MODES = PUBLICATION_BENCHMARK_MODES
MODE_EFF: dict[str, float] = {mode: 0.0 for mode in _KNOWN_MODES}
MODE_CARBON_EFF: dict[str, float] = {mode: 1.0 for mode in _KNOWN_MODES}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_waste_rate(
    k_inst: float | np.ndarray,
    surplus_ratio: float = 0.0,
    w_scale: float = W_SCALE,
    w_alpha: float = W_ALPHA,
    surplus_waste_factor: float = SURPLUS_WASTE_FACTOR,
    waste_cap: float = WASTE_CAP,
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
    k_array = np.asarray(k_inst, dtype=float)
    if not np.all(np.isfinite(k_array)) or np.any(k_array < 0.0):
        raise ValueError("k_inst must be finite and non-negative")
    for label, value, positive in (
        ("surplus_ratio", surplus_ratio, False),
        ("w_scale", w_scale, True),
        ("w_alpha", w_alpha, True),
        ("surplus_waste_factor", surplus_waste_factor, False),
        ("waste_cap", waste_cap, True),
    ):
        value = float(value)
        if not np.isfinite(value) or (value <= 0.0 if positive else value < 0.0):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"{label} must be finite and {qualifier}")
    if float(waste_cap) > 1.0:
        raise ValueError("waste_cap must not exceed one")

    waste_raw = (k_array * float(w_scale)) ** float(w_alpha)
    waste_raw = waste_raw * (
        1.0 + float(surplus_waste_factor) * float(surplus_ratio)
    )
    # Apply cap after surplus amplification to enforce a true physical upper bound.
    waste_raw = np.minimum(waste_raw, float(waste_cap))
    return float(waste_raw) if k_array.ndim == 0 else waste_raw


def context_waste_penalty(mcp_compliance: dict | None = None, action: str = "cold_chain") -> float:
    """Compatibility hook retained as a neutral multiplier.

    Context may change the selected action, but merely receiving compliance
    information cannot change the physical outcome of a fixed action. The
    arguments are therefore intentionally ignored.
    """
    return 1.0


def compute_save_factor(
    action: str,
    mode: str,
    surplus_ratio: float = 0.0,
    surplus_save_penalty: float = SURPLUS_SAVE_PENALTY,
    compliance_data: dict | None = None,
    save_floor: Mapping[str, float] | None = None,
) -> float:
    """Compute the mode-neutral waste prevention factor for an action.

    action_save = SAVE_FLOOR[action]  # legacy name; exact live coefficient
    save_capacity = 1 / (1 + surplus_save_penalty × surplus_ratio)
    effective_save = action_save × save_capacity

    Parameters
    ----------
    action : routing action (``cold_chain``, ``local_redistribute``, ``recovery``).
    mode : retained for API compatibility; ignored by the physical model.
    surplus_ratio : inventory surplus above baseline (0 when at/below baseline).
    surplus_save_penalty : degradation coefficient for surplus conditions.
    compliance_data : retained for API compatibility and ignored. Context may
        alter action selection but not the outcome of an identical action.

    Returns
    -------
    Effective save factor in [0, 1].
    """
    surplus_ratio = float(surplus_ratio)
    surplus_save_penalty = float(surplus_save_penalty)
    if not np.isfinite(surplus_ratio) or surplus_ratio < 0.0:
        raise ValueError("surplus_ratio must be finite and non-negative")
    if not np.isfinite(surplus_save_penalty) or surplus_save_penalty < 0.0:
        raise ValueError(
            "surplus_save_penalty must be finite and non-negative"
        )

    action = _resolve_action(action)
    action_save = (
        SAVE_FLOOR.get(action, 0.0)
        if save_floor is None
        else save_floor.get(action, 0.0)
    )

    # ``compliance_data`` is accepted for API compatibility but does not alter
    # physical outcomes for a fixed action.

    action_save = float(action_save)
    if not np.isfinite(action_save) or not 0.0 <= action_save <= 1.0:
        raise ValueError("action save fraction must be finite and within [0, 1]")
    save_capacity = 1.0 / (1.0 + surplus_save_penalty * surplus_ratio)
    return action_save * save_capacity


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
