"""MCP tool: yield_query.

Wraps the locked persistence supply-proxy forecast.  Holt's linear method
remains an explicit diagnostic alternative.  The tool exposes a normalised
supply-uncertainty signal (``uncertainty``) used in the policy state.

The uncertainty signal is the coefficient of variation of the selected
forecast, clamped to the unit interval:

    uncertainty = clip( std / max(|forecast[0]|, 1.0), 0.0, 1.0 )

Scale-invariant, intuitive, matches the [0, 1] domain of the other
psi features.

**Cached vs computed semantics (honest framing).** The simulator's
hot path (``mvp/simulation/generate_results.py``) runs the selected forecaster
once per step and threads the result into ``obs.raw["supply_uncertainty"]``
to avoid running persistence twice (once outside MCP for the state
vector, once inside MCP for the tool contract). When that cache is
present this tool returns the cached value verbatim with
``"source": "cached"`` — the MCP layer is then a thin wrapper, not
the place where the selected forecast ran. When the cache is absent (e.g.,
the FastAPI ``/decide`` path or a direct MCP client invocation),
the tool runs ``yield_supply_forecast`` itself and returns
``"source": "computed"``. The previous prose in this file
(and the paper) gave MCP credit for the forecast in *every* call;
the simulator-cached calls credit MCP only with the contract layer,
not the numerics. The ``source`` field distinguishes the two.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.models.yield_forecast import yield_supply_forecast
from src.models.persistence_forecast import persistence_forecast


_METHOD_ALIASES = {
    "persistence": "persistence",
    "holt_linear": "holt_linear",
    "holt_winters": "holt_linear",
}


def _normalise_method(method: str) -> str:
    key = str(method).strip().lower()
    try:
        return _METHOD_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "supply forecast method must be persistence or holt_linear "
            "(holt_winters is a legacy alias)"
        ) from exc


def query_yield(
    inventory_history: Optional[List[float]] = None,
    horizon: int = 1,
    method: str = "persistence",
    cached_uncertainty: Optional[float] = None,
    cached_forecast: Optional[List[float]] = None,
    cached_std: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a yield/supply-proxy forecast plus a normalised
    supply-uncertainty signal in [0, 1].

    When ``cached_uncertainty`` is provided (typically by the simulator
    that already ran the selected method this step), the call short-circuits
    and returns the cached values without re-running the forecast.
    """
    selected_method = _normalise_method(method)

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
            "method": selected_method,
        }

    if not inventory_history:
        return {
            "forecast": [],
            "ci_lower": [],
            "ci_upper": [],
            "std": 0.0,
            "uncertainty": 0.0,
            "source": "computed",
            "method": selected_method,
        }

    df = pd.DataFrame({"inventory_units": [float(v) for v in inventory_history]})
    if selected_method == "persistence":
        fc = persistence_forecast(
            df, horizon=horizon, series_col="inventory_units",
        )
    else:
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
        "method": selected_method,
    }
