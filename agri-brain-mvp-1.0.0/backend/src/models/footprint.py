"""
Green AI footprint meter.

Tracks cumulative energy (Joules) and water (Litres) consumed per inference
step so the cost of running the decision engine is transparent.

Default per-step estimates are based on the Strubell et al. (2019) and
Patterson et al. (2021) benchmarks for lightweight ML inference on CPU:
    energy_per_step_J  = 0.05 J   (~50 mJ per forward pass)
    water_per_step_L   = 1.8e-6 L (cooling water per server-second)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class FootprintMeter:
    """Session-scoped cumulative Green-AI meter."""

    energy_per_step_J: float = 0.05
    water_per_step_L: float = 1.8e-6

    _total_energy_J: float = field(default=0.0, init=False, repr=False)
    _total_water_L: float = field(default=0.0, init=False, repr=False)
    _step_count: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # -- public API --------------------------------------------------------

    def compute_footprint(
        self,
        steps: int = 1,
        energy_override_J: float | None = None,
        water_override_L: float | None = None,
    ) -> Dict[str, float]:
        """Record *steps* inference steps and return per-call + cumulative totals.

        Parameters
        ----------
        steps : number of inference forward passes in this call.
        energy_override_J : override the default energy-per-step (J).
        water_override_L  : override the default water-per-step (L).

        Returns
        -------
        dict with per-call and cumulative metrics.
        """
        e_step = energy_override_J if energy_override_J is not None else self.energy_per_step_J
        w_step = water_override_L if water_override_L is not None else self.water_per_step_L

        call_energy = e_step * steps
        call_water = w_step * steps

        with self._lock:
            self._total_energy_J += call_energy
            self._total_water_L += call_water
            self._step_count += steps
            cum_e = self._total_energy_J
            cum_w = self._total_water_L
            cnt = self._step_count

        return {
            "steps": steps,
            "energy_J": round(call_energy, 8),
            "water_L": round(call_water, 10),
            "cumulative_energy_J": round(cum_e, 8),
            "cumulative_water_L": round(cum_w, 10),
            "total_steps": cnt,
        }

    def summary(self) -> Dict[str, float]:
        """Return cumulative footprint without recording new steps."""
        with self._lock:
            return {
                "cumulative_energy_J": round(self._total_energy_J, 8),
                "cumulative_water_L": round(self._total_water_L, 10),
                "total_steps": self._step_count,
            }

    def reset(self) -> None:
        """Zero the counters (useful between test runs)."""
        with self._lock:
            self._total_energy_J = 0.0
            self._total_water_L = 0.0
            self._step_count = 0


# Module-level singleton so the whole backend shares one meter
footprint_meter = FootprintMeter()


def compute_footprint(steps: int = 1, **kwargs) -> Dict[str, float]:
    """Convenience wrapper around the module-level meter."""
    return footprint_meter.compute_footprint(steps=steps, **kwargs)
