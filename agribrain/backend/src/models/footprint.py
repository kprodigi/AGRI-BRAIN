"""Activity-based Green-AI estimator with an explicit system boundary.

The estimator is not hardware telemetry. Time-based quantities multiply the
measured decision-path wall time supplied by the caller by declared nominal
rates. Historical per-step constants are retained only as separately named
proxies; they are never reported as elapsed-time estimates.

Default declared assumptions:

``assumed_active_power_W = 10``
    Nominal active CPU/edge power used to estimate energy from elapsed seconds.

``water_per_server_second_L = 1.8e-6``
    Nominal cooling-water rate used to estimate water from elapsed seconds.

``energy_per_step_proxy_J = 0.05`` and
``water_per_step_proxy_L = 1.8e-6``
    Legacy fixed-step proxies retained for sensitivity/backward compatibility.
    The caller must declare what one proxy step represents. Output labels
    always include ``per_step_proxy`` so these cannot be confused with
    measured-time estimates or with counted hardware forward passes.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Dict


DEFAULT_ASSUMED_ACTIVE_POWER_W: float = 10.0
DEFAULT_WATER_RATE_L_PER_SERVER_SECOND: float = 1.8e-6
DEFAULT_ENERGY_PER_PROXY_STEP_J: float = 0.05
DEFAULT_WATER_PER_PROXY_STEP_L: float = 1.8e-6


@dataclass
class FootprintMeter:
    """Session-scoped activity estimator; not direct resource metering."""

    assumed_active_power_W: float = DEFAULT_ASSUMED_ACTIVE_POWER_W
    water_per_server_second_L: float = (
        DEFAULT_WATER_RATE_L_PER_SERVER_SECOND
    )
    measurement_scope: str = "caller-supplied timed operation wall time"
    proxy_step_unit: str = "caller-declared proxy activity unit"
    # Explicitly named proxy coefficients. They do not feed the elapsed-time
    # estimates below.
    energy_per_step_proxy_J: float = DEFAULT_ENERGY_PER_PROXY_STEP_J
    water_per_step_proxy_L: float = DEFAULT_WATER_PER_PROXY_STEP_L

    _total_energy_J: float = field(default=0.0, init=False, repr=False)
    _total_water_L: float = field(default=0.0, init=False, repr=False)
    _total_elapsed_seconds: float = field(default=0.0, init=False, repr=False)
    _total_energy_per_step_proxy_J: float = field(
        default=0.0, init=False, repr=False,
    )
    _total_water_per_step_proxy_L: float = field(
        default=0.0, init=False, repr=False,
    )
    _step_count: int = field(default=0, init=False, repr=False)
    _timed_call_count: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # -- public API --------------------------------------------------------

    def compute_footprint(
        self,
        steps: int = 1,
        elapsed_seconds: float | None = None,
        active_power_override_W: float | None = None,
        water_rate_override_L_per_second: float | None = None,
        energy_override_J: float | None = None,
        water_override_L: float | None = None,
    ) -> Dict[str, Any]:
        """Record decision activity and return estimates plus named proxies.

        Parameters
        ----------
        steps : number of caller-declared proxy activity units in this call.
            The benchmark declares one unit per standardized routing decision;
            the meter does not infer or measure neural forward-pass counts.
        elapsed_seconds : measured wall time for the declared decision scope.
            When omitted, time-based energy and water are unavailable; only
            explicitly labelled per-step proxies are recorded.
        active_power_override_W : nominal power rate for this timed call.
        water_rate_override_L_per_second : nominal water rate for this timed
            call.
        energy_override_J : legacy override for the energy-per-step proxy.
        water_override_L : legacy override for the water-per-step proxy.

        Returns
        -------
        dict with per-call and cumulative estimates, measurement scope, and
        explicitly labelled per-step proxies.
        """
        if isinstance(steps, bool) or int(steps) != steps or int(steps) < 0:
            raise ValueError("steps must be a non-negative integer")
        steps = int(steps)
        if elapsed_seconds is not None:
            elapsed_seconds = float(elapsed_seconds)
            if not isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
                raise ValueError("elapsed_seconds must be finite and non-negative")

        active_power = (
            self.assumed_active_power_W
            if active_power_override_W is None
            else float(active_power_override_W)
        )
        water_rate = (
            self.water_per_server_second_L
            if water_rate_override_L_per_second is None
            else float(water_rate_override_L_per_second)
        )
        if not isfinite(active_power) or active_power < 0.0:
            raise ValueError("active power must be finite and non-negative")
        if not isfinite(water_rate) or water_rate < 0.0:
            raise ValueError("water rate must be finite and non-negative")

        energy_proxy_per_step = (
            self.energy_per_step_proxy_J
            if energy_override_J is None else float(energy_override_J)
        )
        water_proxy_per_step = (
            self.water_per_step_proxy_L
            if water_override_L is None else float(water_override_L)
        )
        for label, value in (
            ("energy per-step proxy", energy_proxy_per_step),
            ("water per-step proxy", water_proxy_per_step),
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be finite and non-negative")

        has_timing = elapsed_seconds is not None
        call_energy = active_power * elapsed_seconds if has_timing else None
        call_water = water_rate * elapsed_seconds if has_timing else None
        call_energy_proxy = energy_proxy_per_step * steps
        call_water_proxy = water_proxy_per_step * steps

        with self._lock:
            if has_timing:
                self._total_energy_J += float(call_energy)
                self._total_water_L += float(call_water)
                self._total_elapsed_seconds += float(elapsed_seconds)
                self._timed_call_count += 1
            self._total_energy_per_step_proxy_J += call_energy_proxy
            self._total_water_per_step_proxy_L += call_water_proxy
            self._step_count += steps
            cum_e = self._total_energy_J
            cum_w = self._total_water_L
            cum_seconds = self._total_elapsed_seconds
            cum_e_proxy = self._total_energy_per_step_proxy_J
            cum_w_proxy = self._total_water_per_step_proxy_L
            cnt = self._step_count
            timed_calls = self._timed_call_count

        return {
            "steps": steps,
            "elapsed_seconds": (
                round(float(elapsed_seconds), 12) if has_timing else None
            ),
            "time_based_estimate_available": has_timing,
            "estimate_basis": (
                "measured_elapsed_seconds_x_declared_rates"
                if has_timing else "unavailable_without_elapsed_seconds"
            ),
            "measurement_scope": self.measurement_scope,
            "proxy_step_unit": self.proxy_step_unit,
            "estimation_status": "activity-based estimate; not hardware telemetry",
            "assumed_active_power_W": active_power,
            "water_rate_L_per_server_second": water_rate,
            "energy_J": round(float(call_energy), 8) if has_timing else None,
            "water_L": round(float(call_water), 12) if has_timing else None,
            "cumulative_energy_J": round(cum_e, 8),
            "cumulative_water_L": round(cum_w, 12),
            "cumulative_elapsed_seconds": round(cum_seconds, 12),
            "energy_per_step_proxy_J": round(energy_proxy_per_step, 8),
            "water_per_step_proxy_L": round(water_proxy_per_step, 12),
            "step_count_energy_proxy_J": round(call_energy_proxy, 8),
            "step_count_water_proxy_L": round(call_water_proxy, 12),
            "cumulative_energy_per_step_proxy_J": round(cum_e_proxy, 8),
            "cumulative_water_per_step_proxy_L": round(cum_w_proxy, 12),
            "total_steps": cnt,
            "timed_call_count": timed_calls,
        }

    def summary(self) -> Dict[str, Any]:
        """Return cumulative footprint without recording new steps."""
        with self._lock:
            return {
                "cumulative_energy_J": round(self._total_energy_J, 8),
                "cumulative_water_L": round(self._total_water_L, 12),
                "cumulative_elapsed_seconds": round(
                    self._total_elapsed_seconds, 12,
                ),
                "cumulative_energy_per_step_proxy_J": round(
                    self._total_energy_per_step_proxy_J, 8,
                ),
                "cumulative_water_per_step_proxy_L": round(
                    self._total_water_per_step_proxy_L, 12,
                ),
                "total_steps": self._step_count,
                "timed_call_count": self._timed_call_count,
                "time_based_estimate_available": self._timed_call_count > 0,
                "estimate_basis": (
                    "measured_elapsed_seconds_x_declared_rates"
                    if self._timed_call_count > 0
                    else "unavailable_without_elapsed_seconds"
                ),
                "measurement_scope": self.measurement_scope,
                "proxy_step_unit": self.proxy_step_unit,
                "estimation_status": (
                    "activity-based estimate; not hardware telemetry"
                ),
                "assumed_active_power_W": self.assumed_active_power_W,
                "water_rate_L_per_server_second": (
                    self.water_per_server_second_L
                ),
            }

    def reset(self) -> None:
        """Zero the counters (useful between test runs)."""
        with self._lock:
            self._total_energy_J = 0.0
            self._total_water_L = 0.0
            self._total_elapsed_seconds = 0.0
            self._total_energy_per_step_proxy_J = 0.0
            self._total_water_per_step_proxy_L = 0.0
            self._step_count = 0
            self._timed_call_count = 0


# Module-level singleton so the whole backend shares one meter
footprint_meter = FootprintMeter()


def compute_footprint(steps: int = 1, **kwargs) -> Dict[str, Any]:
    """Convenience wrapper around the module-level meter."""
    return footprint_meter.compute_footprint(steps=steps, **kwargs)
