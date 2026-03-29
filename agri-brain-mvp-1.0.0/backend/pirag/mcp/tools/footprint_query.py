"""Green AI footprint query tool for the MCP server.

Returns cumulative and per-step energy and water usage estimates,
following the reporting requirements from the green_ai_reporting
knowledge base document.
"""
from __future__ import annotations

from typing import Any, Dict

# Baseline constants (from green_ai_reporting.txt)
CO2_PER_KWH: float = 0.42  # kg CO2-eq per kWh


def query_footprint(
    steps_completed: int,
    energy_per_step_j: float = 0.050,
    water_per_step_l: float = 1.8e-6,
) -> Dict[str, Any]:
    """Return cumulative and per-step energy and water footprint.

    Parameters
    ----------
    steps_completed : number of inference steps completed.
    energy_per_step_j : energy per step in joules (default 50 mJ).
    water_per_step_l : water per step in litres (default 1.8 uL).

    Returns
    -------
    Dict with per-step and cumulative energy, water, and carbon metrics.
    """
    cumulative_energy_j = steps_completed * energy_per_step_j
    cumulative_water_l = steps_completed * water_per_step_l
    cumulative_energy_kwh = cumulative_energy_j / 3_600_000.0
    cumulative_co2_kg = cumulative_energy_kwh * CO2_PER_KWH

    efficiency_flag = "normal"
    if energy_per_step_j > 0.050:
        efficiency_flag = "above_baseline"
    elif energy_per_step_j > 0.100:
        efficiency_flag = "review_required"

    return {
        "steps_completed": steps_completed,
        "per_step": {
            "energy_j": energy_per_step_j,
            "water_l": water_per_step_l,
        },
        "cumulative": {
            "energy_j": round(cumulative_energy_j, 6),
            "energy_kwh": round(cumulative_energy_kwh, 10),
            "water_l": round(cumulative_water_l, 10),
            "co2_kg": round(cumulative_co2_kg, 10),
        },
        "efficiency_flag": efficiency_flag,
    }
