"""
Carbon emissions model for cold chain transport.

Implements a stylized activity-based transport-emissions proxy. The GHG
Protocol motivates the activity-times-emission-factor structure, and transport
refrigeration literature motivates a thermal penalty. The route distances,
emission factor, and penalty magnitude are declared simulation assumptions, not
fleet measurements or factors extracted from those sources.

Transport emissions
-------------------
Base transport emissions follow the GHG Protocol activity-based method:

    E_transport = distance × EF_vehicle        [kg CO₂-eq]

where:
    distance    = route distance in km (policy-defined per action)
    EF_vehicle  = carbon_per_km (kg CO₂-eq/km) for refrigerated truck
                  transport, including both propulsion and baseline
                  refrigeration energy

The benchmark default EF_vehicle = 0.12 kg CO₂-eq/km is author-specified and
must be replaced with region-, vehicle-, load-, fuel-, and refrigeration-specific
inventory data before deployment.

COP degradation under thermal stress
-------------------------------------
Higher ambient temperatures reduce the coefficient of performance (COP)
of transport refrigeration units (TRUs), increasing energy consumption:

    COP(T) = COP_design / (1 + β_COP × θ)

where:
    θ = thermal_stress = clamp((T − T₀) / ΔT_max, 0, 1)
    T₀ = 4 °C (design cold-chain temperature)
    ΔT_max = 20 °C (extreme heatwave deviation)
    β_COP = REFRIG_COP_PENALTY = 0.40

The actual carbon emission is then:

    E_actual = E_transport × (1 + β_COP × θ)

This makes the synthetic emissions proxy increase under modelled thermal stress.

Cold chain energy model (Tassou et al., 2009):
    P_refrigeration = (UA × ΔT + Q_product) / COP
    E_cold = P_refrigeration × time × EF_electricity

References
----------
    - WRI/WBCSD (2004). The Greenhouse Gas Protocol: A Corporate
      Accounting and Reporting Standard (Revised Edition).
    - Tassou, S.A., De-Lille, G. & Ge, Y.T. (2009). Food transport
      refrigeration — Approaches to reduce energy consumption and
      environmental impacts of road transport. Applied Thermal
      Engineering, 29(8-9), 1467–1477.
"""
from __future__ import annotations

from math import isfinite

# ---------------------------------------------------------------------------
# COP degradation constant
# ---------------------------------------------------------------------------
REFRIG_COP_PENALTY: float = 0.40
"""Declared fractional emissions increase at full synthetic thermal stress."""


def compute_transport_carbon(
    km: float,
    carbon_per_km: float,
    thermal_stress: float = 0.0,
    cop_penalty: float = REFRIG_COP_PENALTY,
    eff_factor: float = 1.0,
) -> float:
    """Compute carbon emissions for a routing action.

    Combines an activity-based emissions proxy with a declared linear thermal
    penalty and an optional physical
    efficiency multiplier:

        E = km × carbon_per_km × eff_factor × (1 + cop_penalty × thermal_stress)

    Parameters
    ----------
    km : route distance in kilometres.
    carbon_per_km : emission factor (kg CO₂-eq / km) for refrigerated transport.
    thermal_stress : normalised thermal stress θ ∈ [0, 1].
        θ = clamp((T_ambient − T₀) / ΔT_max, 0, 1)
    cop_penalty : COP degradation coefficient (default 0.40).
    eff_factor : physical vehicle/fuel efficiency multiplier in (0, 1].
        The confirmatory comparison holds this at 1.0 for every mode.

    Returns
    -------
    Total carbon emissions in kg CO₂-eq.
    """
    values = {
        "km": float(km),
        "carbon_per_km": float(carbon_per_km),
        "thermal_stress": float(thermal_stress),
        "cop_penalty": float(cop_penalty),
        "eff_factor": float(eff_factor),
    }
    if any(not isfinite(value) for value in values.values()):
        raise ValueError("transport-carbon inputs must all be finite")
    if values["km"] < 0.0:
        raise ValueError("km must be non-negative")
    if values["carbon_per_km"] < 0.0:
        raise ValueError("carbon_per_km must be non-negative")
    if not 0.0 <= values["thermal_stress"] <= 1.0:
        raise ValueError("thermal_stress must be within [0, 1]")
    if values["cop_penalty"] < 0.0:
        raise ValueError("cop_penalty must be non-negative")
    if values["eff_factor"] <= 0.0:
        raise ValueError("eff_factor must be positive")
    base_carbon = (
        values["km"] * values["carbon_per_km"] * values["eff_factor"]
    )
    return base_carbon * (
        1.0 + values["cop_penalty"] * values["thermal_stress"]
    )


def compute_carbon_efficiency(mean_ari: float, episode_carbon_kg: float) -> float:
    """Return resilience per episode emissions proxy in ARI·kg⁻¹ CO2e.

    The numerator is the episode mean Adaptive Resilience Index and the
    denominator is the episode sum of the standardized-routing-opportunity
    emissions proxy. No factor of 1,000 is applied. The result is undefined
    for a non-positive denominator, which is treated as invalid input rather
    than silently reported as zero.
    """
    mean_ari = float(mean_ari)
    episode_carbon_kg = float(episode_carbon_kg)
    if not isfinite(mean_ari) or mean_ari < 0.0:
        raise ValueError("mean_ari must be finite and non-negative")
    if not isfinite(episode_carbon_kg) or episode_carbon_kg <= 0.0:
        raise ValueError("episode_carbon_kg must be finite and positive")
    return mean_ari / episode_carbon_kg
