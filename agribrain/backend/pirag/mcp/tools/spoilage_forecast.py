"""Arrhenius-Baranyi ODE forward integration for spoilage prediction.

Wraps the physics model as an MCP tool so agents can request forward-looking
spoilage forecasts without circular imports to ``src.models.spoilage``.
Uses the same calibrated parameters as the backend model.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

# Calibrated parameters (spinach — identical to src.models.spoilage)
K_REF: float = 0.0021      # h^-1 at T_ref
EA_R: float = 8000.0        # E_a / R  (K)
T_REF: float = 277.15       # 4 °C in Kelvin
BETA: float = 0.25           # humidity coupling
LAMBDA_LAG: float = 12.0     # Baranyi lag phase (h)
DT: float = 0.25             # integration step (15 min)


def _arrhenius_k(temp_c: float, humidity_frac: float) -> float:
    """Compute effective decay rate constant k(T, H)."""
    t_k = temp_c + 273.15
    k = K_REF * math.exp(EA_R * (1.0 / T_REF - 1.0 / t_k))
    a_w = max(0.0, min(humidity_frac, 1.0))
    return k * (1.0 + BETA * a_w)


def _baranyi_alpha(t_hours: float) -> float:
    """Baranyi lag-phase adjustment alpha(t) = t / (t + lambda)."""
    return t_hours / (t_hours + LAMBDA_LAG) if (t_hours + LAMBDA_LAG) > 0 else 0.0


def forecast_spoilage(
    current_rho: float,
    temperature: float,
    humidity: float,
    hours_ahead: int = 6,
) -> Dict[str, Any]:
    """Integrate Arrhenius-Baranyi ODE forward from current state.

    Parameters
    ----------
    current_rho : current spoilage risk in [0, 1].
    temperature : ambient temperature in Celsius.
    humidity : relative humidity in percent (0–100).
    hours_ahead : forecast horizon in hours.

    Returns
    -------
    Dict with forecast_rho, quality_trajectory, urgency, and parameters.
    """
    humidity_frac = max(0.0, min(humidity / 100.0, 1.0))
    k_base = _arrhenius_k(temperature, humidity_frac)

    rho = max(0.0, min(current_rho, 1.0))
    quality = 1.0 - rho
    steps = max(1, int(hours_ahead / DT))

    trajectory: List[float] = [round(rho, 4)]
    elapsed = 0.0

    for _ in range(steps):
        elapsed += DT
        alpha = _baranyi_alpha(elapsed)
        k_eff = k_base * alpha
        dq = -k_eff * quality * DT
        quality = max(0.0, quality + dq)
        rho = 1.0 - quality
        trajectory.append(round(rho, 4))

    forecast_rho = round(rho, 4)

    if forecast_rho > 0.60:
        urgency = "critical"
    elif forecast_rho > 0.40:
        urgency = "high"
    elif forecast_rho > 0.20:
        urgency = "medium"
    else:
        urgency = "low"

    return {
        "forecast_rho": forecast_rho,
        "quality_trajectory": trajectory,
        "urgency": urgency,
        "hours_ahead": hours_ahead,
        "k_effective": round(k_base, 6),
        "parameters": {
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "current_rho": current_rho,
        },
    }
