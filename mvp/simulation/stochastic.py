"""
Dual-mode stochastic perturbation layer for AgriBrain simulation.

DETERMINISTIC_MODE=true  (default) -> all perturbations are no-ops, exact reproducibility.
DETERMINISTIC_MODE=false            -> seeded Gaussian/uniform noise on sensor, demand, inventory.

Perturbation amplitudes are physically plausible and configurable via env vars.
The StochasticLayer is stateless except for the RNG, so given the same seed it
always produces the same perturbation sequence (stochastic-but-reproducible).
"""
from __future__ import annotations

import os
import numpy as np

# ---------------------------------------------------------------------------
# Configuration (read once at import time)
# ---------------------------------------------------------------------------
DETERMINISTIC_MODE: bool = os.environ.get("DETERMINISTIC_MODE", "false").lower() == "true"

# Perturbation amplitudes (sensible defaults, all overridable)
STOCH_TEMP_SIGMA: float = float(os.environ.get("STOCH_TEMP_SIGMA", "0.3"))       # degrees C
STOCH_RH_SIGMA: float = float(os.environ.get("STOCH_RH_SIGMA", "1.5"))           # percent
STOCH_DEMAND_CV: float = float(os.environ.get("STOCH_DEMAND_CV", "0.05"))         # coefficient of variation
STOCH_INVENTORY_CV: float = float(os.environ.get("STOCH_INVENTORY_CV", "0.03"))   # coefficient of variation
STOCH_LATENCY_MAX: float = float(os.environ.get("STOCH_LATENCY_MAX", "15.0"))     # ms (future use)


class StochasticLayer:
    """Per-episode perturbation engine.

    When ``enabled=False`` every method is an identity function.
    When ``enabled=True`` each call draws from *rng*, advancing its state
    deterministically so repeated runs with the same seed are reproducible.
    """

    __slots__ = ("_rng", "_enabled")

    def __init__(self, rng: np.random.Generator, enabled: bool) -> None:
        self._rng = rng
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def perturb_temperature(self, temp: float) -> float:
        if not self._enabled:
            return temp
        return temp + self._rng.normal(0.0, STOCH_TEMP_SIGMA)

    def perturb_humidity(self, rh: float) -> float:
        if not self._enabled:
            return rh
        return float(np.clip(rh + self._rng.normal(0.0, STOCH_RH_SIGMA), 0.0, 100.0))

    def perturb_demand(self, demand: float) -> float:
        if not self._enabled:
            return demand
        return max(0.0, demand * (1.0 + self._rng.normal(0.0, STOCH_DEMAND_CV)))

    def perturb_inventory(self, inv: float) -> float:
        if not self._enabled:
            return inv
        return max(0.0, inv * (1.0 + self._rng.normal(0.0, STOCH_INVENTORY_CV)))

    def perturb_latency(self, latency_ms: float) -> float:
        """Additive latency jitter (for future latency-quality frontier work)."""
        if not self._enabled:
            return latency_ms
        return latency_ms + self._rng.uniform(0.0, STOCH_LATENCY_MAX)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_stochastic_layer(rng: np.random.Generator) -> StochasticLayer:
    """Create a StochasticLayer that respects the global DETERMINISTIC_MODE flag."""
    return StochasticLayer(rng=rng, enabled=not DETERMINISTIC_MODE)


# Disabled singleton for backward-compatible callers
_DISABLED = StochasticLayer(rng=np.random.default_rng(0), enabled=False)
