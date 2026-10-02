"""Legacy fixed-rate Green-AI proxy query for the MCP server.

This tool has no wall-time or hardware-telemetry input. It therefore reports
only explicitly labelled step-count proxies. Confirmatory episode results use
``src.models.footprint.FootprintMeter`` with measured action-selection elapsed
time and declared rates; these proxy counters must not be substituted for that
estimate.
"""
from __future__ import annotations

from math import isfinite
from typing import Any, Dict

from src.models.footprint import (
    DEFAULT_ENERGY_PER_PROXY_STEP_J,
    DEFAULT_WATER_PER_PROXY_STEP_L,
)

# Author-declared grid-intensity proxy for sensitivity/monitoring examples.
# It is not a measured regional factor for the benchmark execution site.
CO2_PER_KWH_PROXY: float = 0.42  # kg CO2-eq per kWh


def query_footprint(
    steps_completed: int,
    energy_per_step_j: float = DEFAULT_ENERGY_PER_PROXY_STEP_J,
    water_per_step_l: float = DEFAULT_WATER_PER_PROXY_STEP_L,
) -> Dict[str, Any]:
    """Return fixed per-step proxy counters, not measured resource use.

    Parameters
    ----------
    steps_completed : number of inference steps completed.
    energy_per_step_j : declared energy proxy per step (default 50 mJ).
    water_per_step_l : declared water proxy per step (default 1.8 uL).

    Returns
    -------
    Dict with explicitly labelled per-step and cumulative proxy quantities.
    """
    if (
        isinstance(steps_completed, bool)
        or int(steps_completed) != steps_completed
        or int(steps_completed) < 0
    ):
        raise ValueError("steps_completed must be a non-negative integer")
    steps_completed = int(steps_completed)
    energy_per_step_j = float(energy_per_step_j)
    water_per_step_l = float(water_per_step_l)
    if not isfinite(energy_per_step_j) or energy_per_step_j < 0.0:
        raise ValueError("energy_per_step_j proxy must be finite and non-negative")
    if not isfinite(water_per_step_l) or water_per_step_l < 0.0:
        raise ValueError("water_per_step_l proxy must be finite and non-negative")

    cumulative_energy_j = steps_completed * energy_per_step_j
    cumulative_water_l = steps_completed * water_per_step_l
    cumulative_energy_kwh = cumulative_energy_j / 3_600_000.0
    cumulative_co2_kg = cumulative_energy_kwh * CO2_PER_KWH_PROXY

    efficiency_flag = "normal"
    if energy_per_step_j > 0.100:
        efficiency_flag = "review_required"
    elif energy_per_step_j > 0.050:
        efficiency_flag = "above_baseline"

    return {
        "steps_completed": steps_completed,
        "estimate_basis": "fixed_per_step_proxy_not_elapsed_time",
        "estimation_status": "proxy only; not hardware telemetry",
        "time_based_estimate_available": False,
        "proxy_step_unit": "simulation/inference step count supplied by caller",
        "per_step_proxy": {
            "energy_j": energy_per_step_j,
            "water_l": water_per_step_l,
        },
        "cumulative_step_count_proxy": {
            "energy_j": round(cumulative_energy_j, 6),
            "energy_kwh": round(cumulative_energy_kwh, 10),
            "water_l": round(cumulative_water_l, 10),
            "co2_kg": round(cumulative_co2_kg, 10),
        },
        "grid_carbon_intensity_proxy_kg_per_kwh": CO2_PER_KWH_PROXY,
        "efficiency_flag_basis": "declared_energy_per_step_proxy_threshold",
        "efficiency_flag": efficiency_flag,
    }
