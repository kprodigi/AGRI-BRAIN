#!/usr/bin/env python3
"""Dual-mode stochastic perturbation layer for simulation.

DETERMINISTIC_MODE=false (default) enables seeded, bounded perturbations.
DETERMINISTIC_MODE=true disables perturbations for strict reproducibility.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


def _is_deterministic() -> bool:
    """Read DETERMINISTIC_MODE at call time, not import time."""
    return os.environ.get("DETERMINISTIC_MODE", "false").lower() == "true"

# Deprecated: prefer _is_deterministic() for call-time evaluation.
# Kept for backward compatibility with existing imports.
DETERMINISTIC_MODE: bool = _is_deterministic()


@dataclass(frozen=True)
class StochasticLayer:
    rng: np.random.Generator
    enabled: bool
    temp_std_c: float
    rh_std: float
    demand_frac_std: float
    inventory_frac_std: float
    delay_prob: float

    def perturb_temperature(self, temp_c: float) -> float:
        if not self.enabled or self.temp_std_c <= 0.0:
            return float(temp_c)
        return float(np.clip(temp_c + self.rng.normal(0.0, self.temp_std_c), -5.0, 55.0))

    def perturb_humidity(self, rh: float) -> float:
        if not self.enabled or self.rh_std <= 0.0:
            return float(rh)
        return float(np.clip(rh + self.rng.normal(0.0, self.rh_std), 0.0, 100.0))

    def perturb_demand(self, demand: float) -> float:
        if not self.enabled or self.demand_frac_std <= 0.0:
            return float(demand)
        mult = 1.0 + float(self.rng.normal(0.0, self.demand_frac_std))
        return float(max(0.0, demand * mult))

    def perturb_inventory(self, inv: float) -> float:
        if not self.enabled or self.inventory_frac_std <= 0.0:
            return float(inv)
        mult = 1.0 + float(self.rng.normal(0.0, self.inventory_frac_std))
        return float(max(0.0, inv * mult))

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
    delay_prob=0.0,
)


def make_stochastic_layer(rng: np.random.Generator) -> StochasticLayer:
    if _is_deterministic():
        return _DISABLED
    return StochasticLayer(
        rng=rng,
        enabled=True,
        temp_std_c=float(os.environ.get("STOCH_TEMP_STD_C", "0.35")),
        rh_std=float(os.environ.get("STOCH_RH_STD", "1.5")),
        demand_frac_std=float(os.environ.get("STOCH_DEMAND_FRAC_STD", "0.04")),
        inventory_frac_std=float(os.environ.get("STOCH_INVENTORY_FRAC_STD", "0.03")),
        delay_prob=float(os.environ.get("STOCH_DELAY_PROB", "0.02")),
    )
