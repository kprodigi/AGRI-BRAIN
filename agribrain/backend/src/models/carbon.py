"""
Carbon emissions model for cold chain transport.

Implements activity-based carbon accounting following the GHG Protocol
Corporate Standard (WRI/WBCSD, 2004) with refrigeration COP degradation
under thermal stress (Tassou et al., 2009).

Transport emissions
-------------------
Base transport emissions follow the GHG Protocol activity-based method:

    E_transport = distance × EF_vehicle        [kg CO₂-eq]

where:
    distance    = route distance in km (policy-defined per action)
    EF_vehicle  = carbon_per_km (kg CO₂-eq/km) for refrigerated truck
                  transport, including both propulsion and baseline
                  refrigeration energy

Default EF_vehicle = 0.12 kg CO₂-eq/km based on EPA emission factors
for medium-duty refrigerated vehicles (range: 0.08–0.18).

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

This ensures carbon footprint increases during heatwave conditions,
creating a physically realistic cascading effect through the SLCA
carbon component and into the reward function.

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
    - EPA (2021). Emission Factors for Greenhouse Gas Inventories.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# COP degradation constant
# ---------------------------------------------------------------------------
REFRIG_COP_PENALTY: float = 0.40
"""Fractional increase in carbon emissions at full thermal stress (θ = 1).

Based on COP sensitivity analysis for transport refrigeration units
(Tassou et al., 2009): COP degrades roughly linearly with ambient
temperature above the design point, with up to ~40 % efficiency loss
at extreme conditions (+20 °C above set point).
"""


def compute_transport_carbon(
    km: float,
    carbon_per_km: float,
    thermal_stress: float = 0.0,
    cop_penalty: float = REFRIG_COP_PENALTY,
    eff_factor: float = 1.0,
) -> float:
    """Compute carbon emissions for a routing action.

    Combines GHG Protocol activity-based transport emissions with
    COP degradation under thermal stress and an optional mode-
    conditional efficiency multiplier:

        E = km × carbon_per_km × eff_factor × (1 + cop_penalty × thermal_stress)

    Parameters
    ----------
    km : route distance in kilometres.
    carbon_per_km : emission factor (kg CO₂-eq / km) for refrigerated transport.
    thermal_stress : normalised thermal stress θ ∈ [0, 1].
        θ = clamp((T_ambient − T₀) / ΔT_max, 0, 1)
    cop_penalty : COP degradation coefficient (default 0.40).
    eff_factor : mode-conditional efficiency multiplier in (0, 1].
        1.0 means baseline (no optimisation); values < 1 represent the
        per-dispatch carbon reduction from PINN-timed dispatching,
        SLCA-shaped partner selection, and context-aware route
        optimisation. See ``waste.MODE_CARBON_EFF`` for the per-mode
        values and provenance. Default 1.0 preserves the previous
        behaviour for any caller that has not migrated to the
        mode-conditional API.

    Returns
    -------
    Total carbon emissions in kg CO₂-eq.
    """
    base_carbon = km * carbon_per_km * float(eff_factor)
    return base_carbon * (1.0 + cop_penalty * thermal_stress)
