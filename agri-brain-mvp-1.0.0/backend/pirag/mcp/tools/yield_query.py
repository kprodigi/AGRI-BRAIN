"""MCP tool: yield_query.

Wraps :func:`backend.src.models.yield_forecast.yield_supply_forecast` and
exposes a normalised supply-uncertainty signal (`uncertainty`) used as the
sixth context feature psi_5 in :mod:`backend.pirag.context_to_logits`.

The uncertainty signal is the coefficient of variation of the Holt's linear
forecast, clamped to the unit interval:

    uncertainty = clip( std / max(|forecast[0]|, 1.0), 0.0, 1.0 )

Scale-invariant, intuitive, matches the [0, 1] domain of the other psi
features.

Path B note: when the simulator pre-computes the uncertainty (in
``mvp/simulation/generate_results.py``) and exposes it via
``obs.raw["supply_uncertainty"]``, this tool short-circuits and returns
the cached value rather than re-running Holt's linear. This avoids
duplicate computation per step per agent without changing the MCP-facing
contract.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.models.yield_forecast import yield_supply_forecast


def query_yield(
    inventory_history: Optional[List[float]] = None,
    horizon: int = 1,
    cached_uncertainty: Optional[float] = None,
    cached_forecast: Optional[List[float]] = None,
    cached_std: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a Holt's linear yield/supply forecast plus a normalised
    supply-uncertainty signal in [0, 1].

    When ``cached_uncertainty`` is provided (typically by the simulator
    that already ran Holt's linear this step), the call short-circuits
    and returns the cached values without re-running the forecast.
    """
    if cached_uncertainty is not None:
        u = float(cached_uncertainty)
        u = min(max(u, 0.0), 1.0)
        return {
            "forecast": list(cached_forecast) if cached_forecast else [],
            "ci_lower": [],
            "ci_upper": [],
            "std": float(cached_std) if cached_std is not None else 0.0,
            "uncertainty": round(u, 4),
            "source": "cached",
        }

    if not inventory_history:
        return {
            "forecast": [],
            "ci_lower": [],
            "ci_upper": [],
            "std": 0.0,
            "uncertainty": 0.0,
            "source": "computed",
        }

    df = pd.DataFrame({"inventory_units": [float(v) for v in inventory_history]})
    fc = yield_supply_forecast(df, horizon=horizon)

    point = fc["forecast"][0] if fc["forecast"] else 1.0
    std = float(fc["std"])
    cv = std / max(abs(point), 1.0)
    uncertainty = min(max(cv, 0.0), 1.0)

    return {
        "forecast": fc["forecast"],
        "ci_lower": fc["ci_lower"],
        "ci_upper": fc["ci_upper"],
        "std": fc["std"],
        "uncertainty": round(uncertainty, 4),
        "source": "computed",
    }
