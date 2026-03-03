"""
Extended Policy configuration model.

Backward-compatible with the original fields (max_temp_c, min_shelf_reroute,
min_shelf_expedite, carbon_per_km, km_farm_to_dc, km_dc_to_retail,
km_expedited, msrp) while adding all paper-derived parameters.

Every parameter has a brief comment explaining:
  - Physical / economic meaning
  - Realistic range
  - Why the default was chosen
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Policy(BaseModel):
    # ---- original fields (backward compatible) ----
    max_temp_c: float = Field(
        8.0,
        description="Max acceptable cold-chain temperature (°C). "
        "FDA recommends ≤5°C for leafy greens; 8°C is the upper bound "
        "before accelerated spoilage. Range: 5-10°C.",
    )
    min_shelf_reroute: float = Field(
        0.70,
        description="Remaining shelf fraction triggering reroute consideration. "
        "At 70% quality, produce should be diverted to local markets. Range: 0.50-0.80.",
    )
    min_shelf_expedite: float = Field(
        0.50,
        description="Remaining shelf fraction triggering expedited delivery. "
        "At 50% quality, produce needs immediate processing or recovery. Range: 0.30-0.60.",
    )
    carbon_per_km: float = Field(
        0.12,
        description="kg CO2-eq per km for refrigerated truck transport. "
        "Based on EPA emission factors for medium-duty refrigerated vehicles "
        "(0.10-0.15 kg/km including refrigeration). Range: 0.08-0.18.",
    )
    km_farm_to_dc: float = Field(
        280.0, description="Farm-to-distribution-center distance (km). Legacy field."
    )
    km_dc_to_retail: float = Field(
        50.0, description="Distribution-center-to-retail distance (km). Legacy field."
    )
    km_expedited: float = Field(
        160.0, description="Expedited route distance (km). Legacy field."
    )
    msrp: float = Field(1.50, description="Manufacturer suggested retail price (USD/unit).")

    # ---- SLCA weights ----
    # Based on stakeholder analysis for perishable produce supply chains.
    # Carbon gets the highest weight (0.30) reflecting climate priorities.
    # Social components split the remaining 0.70 roughly equally.
    w_c: float = Field(0.30, description="SLCA weight: carbon reduction. Range: 0.20-0.40.")
    w_l: float = Field(0.20, description="SLCA weight: labour fairness. Range: 0.10-0.30.")
    w_r: float = Field(0.25, description="SLCA weight: community resilience. Range: 0.15-0.35.")
    w_p: float = Field(0.25, description="SLCA weight: price transparency. Range: 0.15-0.35.")

    # ---- waste penalty ----
    eta: float = Field(
        0.50,
        description="Waste penalty coefficient in the reward function. "
        "Controls the trade-off between SLCA improvement and waste reduction. "
        "Range: 0.3-1.0. Higher values make the policy more waste-averse.",
    )

    # ---- energy / water penalty coefficients ----
    alpha_E: float = Field(0.05, description="Energy penalty coefficient for Green AI tracking.")
    beta_W: float = Field(0.03, description="Water penalty coefficient for Green AI tracking.")

    # ---- Bollinger volatility parameters ----
    boll_window: int = Field(
        20,
        description="Bollinger rolling window size (number of 15-min steps = 5 hours). "
        "Chosen to capture intra-day demand patterns. Range: 10-30.",
    )
    boll_k: float = Field(
        2.0,
        description="Bollinger z-score threshold for anomaly detection. "
        "2.0 corresponds to ~95% confidence interval. Range: 1.5-3.0.",
    )

    # ---- volatility tilt parameters ----
    # When volatility is detected (tau=1), these shift the softmax logits.
    # Positive gamma_coldchain encourages safe routing during uncertainty.
    gamma_coldchain: float = Field(
        0.3,
        description="Volatility tilt toward cold-chain (positive = prefer CC under volatility). "
        "Kept small to avoid over-conservative routing during demand noise, which "
        "would degrade SLCA scores and ARI. Range: 0.1-0.8.",
    )
    gamma_local: float = Field(
        0.05,
        description="Volatility tilt for local redistribution (near-neutral: local markets "
        "are relatively adaptive to demand changes). Range: -0.5 to 0.5.",
    )
    gamma_recovery: float = Field(
        -0.3,
        description="Volatility tilt for recovery (discouraged under volatility "
        "since recovery capacity may be strained). Range: -1.5 to 0.0.",
    )

    # ---- PINN spoilage parameters (Arrhenius form) ----
    # The Arrhenius model k(T) = k_ref * exp[Ea_R * (1/T_ref - 1/T_K)] is
    # the standard in food science for temperature-dependent quality loss
    # (Labuza & Riboh, 1982; Giannakourou & Taoukis, 2003).
    k_ref: float = Field(
        0.0021,
        description="Reference decay rate at T_ref_K (h^-1). Calibrated for "
        "fresh spinach so that quality loss reaches ~12% over 72h at 4°C with "
        "lag phase. Range: 0.001-0.01 for leafy greens.",
    )
    Ea_R: float = Field(
        8000.0,
        description="Arrhenius activation energy / gas constant (K). "
        "Ea_R = Ea/R where Ea is in J/mol and R = 8.314 J/(mol·K). "
        "8000 K corresponds to Ea ≈ 66.5 kJ/mol, consistent with enzymatic "
        "browning and microbial growth in leafy greens (typical range: "
        "5000-12000 K for produce spoilage reactions).",
    )
    T_ref_K: float = Field(
        277.15,
        description="Reference temperature in Kelvin (= 4.0°C). "
        "Standard cold storage temperature for leafy greens per FDA guidelines.",
    )
    beta_humidity: float = Field(
        0.25,
        description="Humidity coupling coefficient. Higher water activity (a_w ≈ RH/100) "
        "accelerates microbial growth and enzymatic degradation. At RH=89% "
        "(typical cold storage), this increases the effective rate by ~22%. "
        "Range: 0.10-0.50.",
    )
    lag_lambda: float = Field(
        12.0,
        description="Baranyi lag phase parameter (hours). Fresh produce has an "
        "initial lag before exponential quality loss begins, due to microbial "
        "adaptation time. 12h is typical for spinach at 4°C with standard "
        "post-harvest handling. Range: 6-24h depending on initial microbial "
        "load and temperature.",
    )

    # ---- Legacy PINN parameters (kept for backward compatibility) ----
    k0: float = Field(0.04, description="Legacy: PINN base decay rate (h^-1). Use k_ref instead.")
    alpha_decay: float = Field(
        0.12, description="Legacy: PINN thermal sensitivity (°C^-1). Use Ea_R instead."
    )
    T0: float = Field(
        4.0, description="Legacy: PINN reference temperature (°C). Use T_ref_K instead."
    )

    # ---- route distances ----
    # Based on typical South Dakota cooperative cold chain logistics.
    # Farm-to-DC-to-retail via standard cold chain: ~120 km total.
    # Local redistribution to food banks / community markets: ~45 km.
    # Recovery (composting, bioenergy, animal feed facilities): ~80 km.
    km_coldchain: float = Field(
        120.0,
        description="Cold-chain route distance (km). Typical farm-to-DC-to-retail "
        "distance for a South Dakota cooperative. Range: 80-200 km.",
    )
    km_local: float = Field(
        45.0,
        description="Local redistribution distance (km). Distance to nearby food "
        "banks, community markets, or secondary outlets. Range: 20-80 km.",
    )
    km_recovery: float = Field(
        80.0,
        description="Recovery route distance (km). Distance to composting, "
        "bioenergy, or animal feed facilities. Range: 40-120 km.",
    )

    # ---- SLCA carbon normalization ----
    carbon_cap: float = Field(
        50.0,
        description="Carbon normalization cap (kg CO2-eq per step) for SLCA "
        "carbon component. C = max(0, 1 - carbon_kg/carbon_cap). "
        "Default 50 provides good dynamic range across actions. "
        "Range: 20-100.",
    )
