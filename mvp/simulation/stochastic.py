#!/usr/bin/env python3
"""Dual-mode stochastic perturbation layer for simulation.

Seven realistic uncertainty sources:
  1. Sensor noise — temperature ±1.5°C, humidity ±5.0%
  2. Demand variability — multiplicative CV 18%
  3. Inventory/yield uncertainty — multiplicative CV 15%
  4. Transport distance jitter — route CV 15% (detours, traffic, loading)
  5. Spoilage model error — per-episode k_ref CV 15%, Ea_R CV 10%
  6. Scenario onset jitter — ±4 hour uniform shift
  7. Policy weight perturbation — THETA noise sigma 0.03

DETERMINISTIC_MODE=false (default) enables seeded, bounded perturbations.
DETERMINISTIC_MODE=true disables all perturbations for strict reproducibility.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


def _is_deterministic() -> bool:
    """Read DETERMINISTIC_MODE at call time, not import time."""
    return os.environ.get("DETERMINISTIC_MODE", "false").lower() == "true"


# Property-like module attribute for backward-compatible imports.
class _DetFlag:
    def __bool__(self): return _is_deterministic()
    def __repr__(self): return str(_is_deterministic())
    def __eq__(self, other): return _is_deterministic() == other

DETERMINISTIC_MODE = _DetFlag()


@dataclass(frozen=True)
class StochasticLayer:
    rng: np.random.Generator
    enabled: bool
    # --- Source 1: Sensor noise ---
    temp_std_c: float
    rh_std: float
    # --- Source 2: Demand variability ---
    demand_frac_std: float
    # --- Source 3: Inventory/yield uncertainty ---
    inventory_frac_std: float
    # --- Source 4: Transport distance jitter ---
    transport_km_frac_std: float
    # --- Source 5: Spoilage model error (per-episode) ---
    k_ref_frac_std: float
    ea_r_frac_std: float
    # --- Source 6: Scenario onset jitter ---
    onset_jitter_hours: float
    # --- Source 7: Policy weight perturbation ---
    theta_noise_std: float
    # --- Telemetry lag (kept from original) ---
    delay_prob: float

    # ---- Source 1: Sensor noise ----

    def perturb_temperature(self, temp_c: float) -> float:
        if not self.enabled or self.temp_std_c <= 0.0:
            return float(temp_c)
        return float(np.clip(temp_c + self.rng.normal(0.0, self.temp_std_c), -5.0, 55.0))

    def perturb_humidity(self, rh: float) -> float:
        if not self.enabled or self.rh_std <= 0.0:
            return float(rh)
        return float(np.clip(rh + self.rng.normal(0.0, self.rh_std), 0.0, 100.0))

    # ---- Source 2: Demand variability ----

    def perturb_demand(self, demand: float) -> float:
        if not self.enabled or self.demand_frac_std <= 0.0:
            return float(demand)
        mult = 1.0 + float(self.rng.normal(0.0, self.demand_frac_std))
        return float(max(0.0, demand * mult))

    # ---- Source 3: Inventory/yield uncertainty ----

    def perturb_inventory(self, inv: float) -> float:
        if not self.enabled or self.inventory_frac_std <= 0.0:
            return float(inv)
        mult = 1.0 + float(self.rng.normal(0.0, self.inventory_frac_std))
        return float(max(0.0, inv * mult))

    # ---- Source 4: Transport distance jitter ----

    def perturb_transport_km(self, km: float) -> float:
        """Jitter transport distance (detours, traffic, loading delays)."""
        if not self.enabled or self.transport_km_frac_std <= 0.0:
            return float(km)
        mult = 1.0 + float(self.rng.normal(0.0, self.transport_km_frac_std))
        return float(max(0.0, km * mult))

    # ---- Source 5: Spoilage model error (call ONCE per episode) ----

    def perturb_k_ref(self, k_ref: float) -> float:
        """Batch-to-batch variation in produce decay rate."""
        if not self.enabled or self.k_ref_frac_std <= 0.0:
            return float(k_ref)
        mult = 1.0 + float(self.rng.normal(0.0, self.k_ref_frac_std))
        return float(max(1e-6, k_ref * mult))

    def perturb_ea_r(self, ea_r: float) -> float:
        """Batch-to-batch variation in activation energy."""
        if not self.enabled or self.ea_r_frac_std <= 0.0:
            return float(ea_r)
        mult = 1.0 + float(self.rng.normal(0.0, self.ea_r_frac_std))
        return float(max(100.0, ea_r * mult))

    # ---- Source 6: Scenario onset jitter ----

    def jitter_onset_hour(self, base_hour: float) -> float:
        """Shift scenario onset by ±onset_jitter_hours (uniform)."""
        if not self.enabled or self.onset_jitter_hours <= 0.0:
            return float(base_hour)
        shift = float(self.rng.uniform(-self.onset_jitter_hours, self.onset_jitter_hours))
        return float(max(0.0, base_hour + shift))

    # ---- Source 7: Policy weight perturbation (call ONCE per seed) ----

    def perturb_theta(self, theta: np.ndarray) -> np.ndarray:
        """Add small Gaussian noise to policy weight matrix."""
        if not self.enabled or self.theta_noise_std <= 0.0:
            return theta.copy()
        noise = self.rng.normal(0.0, self.theta_noise_std, size=theta.shape)
        return theta + noise

    # ---- Telemetry lag ----

    def should_delay(self) -> bool:
        """Return True with probability delay_prob (telemetry lag event)."""
        if not self.enabled or self.delay_prob <= 0.0:
            return False
        return float(self.rng.random()) < self.delay_prob


_DISABLED = StochasticLayer(
    rng=np.random.default_rng(0),
    enabled=False,
    temp_std_c=0.0,
    rh_std=0.0,
    demand_frac_std=0.0,
    inventory_frac_std=0.0,
    transport_km_frac_std=0.0,
    k_ref_frac_std=0.0,
    ea_r_frac_std=0.0,
    onset_jitter_hours=0.0,
    theta_noise_std=0.0,
    delay_prob=0.0,
)


def make_stochastic_layer(rng: np.random.Generator) -> StochasticLayer:
    if _is_deterministic():
        return _DISABLED
    return StochasticLayer(
        rng=rng,
        enabled=True,
        temp_std_c=float(os.environ.get("STOCH_TEMP_STD_C", "1.5")),
        rh_std=float(os.environ.get("STOCH_RH_STD", "5.0")),
        demand_frac_std=float(os.environ.get("STOCH_DEMAND_FRAC_STD", "0.18")),
        inventory_frac_std=float(os.environ.get("STOCH_INVENTORY_FRAC_STD", "0.15")),
        transport_km_frac_std=float(os.environ.get("STOCH_TRANSPORT_KM_STD", "0.15")),
        k_ref_frac_std=float(os.environ.get("STOCH_K_REF_STD", "0.15")),
        ea_r_frac_std=float(os.environ.get("STOCH_EA_R_STD", "0.10")),
        onset_jitter_hours=float(os.environ.get("STOCH_ONSET_JITTER_H", "4.0")),
        theta_noise_std=float(os.environ.get("STOCH_THETA_NOISE_STD", "0.03")),
        delay_prob=float(os.environ.get("STOCH_DELAY_PROB", "0.05")),
    )
