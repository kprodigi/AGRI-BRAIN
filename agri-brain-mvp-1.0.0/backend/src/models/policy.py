"""
Extended Policy configuration model.

Backward-compatible with the original fields (max_temp_c, min_shelf_reroute,
min_shelf_expedite, carbon_per_km, km_farm_to_dc, km_dc_to_retail,
km_expedited, msrp) while adding all paper-derived parameters.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Policy(BaseModel):
    # ---- original fields (backward compatible) ----
    max_temp_c: float = Field(8.0, description="Max acceptable cold-chain temp (C)")
    min_shelf_reroute: float = Field(0.70, description="Shelf fraction triggering reroute")
    min_shelf_expedite: float = Field(0.50, description="Shelf fraction triggering expedite")
    carbon_per_km: float = Field(0.12, description="kg CO2-eq per km")
    km_farm_to_dc: float = Field(280.0, description="Farm-to-DC distance (km)")
    km_dc_to_retail: float = Field(50.0, description="DC-to-retail distance (km)")
    km_expedited: float = Field(160.0, description="Expedited route distance (km)")
    msrp: float = Field(1.50, description="Manufacturer suggested retail price")

    # ---- SLCA weights ----
    w_c: float = Field(0.30, description="SLCA weight: carbon reduction")
    w_l: float = Field(0.20, description="SLCA weight: labour fairness")
    w_r: float = Field(0.25, description="SLCA weight: community resilience")
    w_p: float = Field(0.25, description="SLCA weight: price transparency")

    # ---- waste penalty ----
    eta: float = Field(0.50, description="Waste penalty coefficient")

    # ---- energy / water penalty coefficients ----
    alpha_E: float = Field(0.05, description="Energy penalty coefficient")
    beta_W: float = Field(0.03, description="Water penalty coefficient")

    # ---- Bollinger volatility parameters ----
    boll_window: int = Field(20, description="Bollinger rolling window size")
    boll_k: float = Field(2.0, description="Bollinger z-score threshold")

    # ---- volatility tilt parameters ----
    gamma_coldchain: float = Field(1.5, description="Volatility tilt: cold-chain")
    gamma_local: float = Field(-0.3, description="Volatility tilt: local redistribute")
    gamma_recovery: float = Field(-0.5, description="Volatility tilt: recovery")

    # ---- PINN decay parameters ----
    k0: float = Field(0.04, description="PINN base decay rate (h^-1)")
    alpha_decay: float = Field(0.12, description="PINN thermal sensitivity (C^-1)")
    T0: float = Field(4.0, description="PINN reference temperature (C)")
    beta_humidity: float = Field(0.25, description="PINN humidity coupling coefficient")

    # ---- route distances ----
    km_coldchain: float = Field(120.0, description="Cold-chain route distance (km)")
    km_local: float = Field(45.0, description="Local redistribution distance (km)")
    km_recovery: float = Field(80.0, description="Recovery route distance (km)")
