"""MCP tool: demand_query.

Symmetric counterpart to ``yield_query``. Wraps the demand forecaster
(Holt's linear for the locked confirmatory protocol; LSTM and persistence
remain explicit diagnostic alternatives) and exposes a normalised demand-uncertainty
signal plus the point forecast and residual standard deviation. Feeds
the same slot in ``obs.raw`` that the simulator uses to populate
``phi_8`` (demand uncertainty CV).

**Cached vs computed semantics (honest framing).** Mirrors the
contract used by ``yield_query``. When the simulator's hot path
provides the cached forecast and uncertainty via ``obs.raw``, this
tool short-circuits and returns ``"source": "cached"`` — MCP is then
a contract layer, not the place where the LSTM/Holt's-linear
computation ran. When the cache is absent, the tool runs the
forecaster itself and returns ``"source": "computed"``. The
simulator's published runs are overwhelmingly cached; the ``source``
field in the recorded protocol traces distinguishes the two paths.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.models.forecast import yield_demand_forecast
from src.models.lstm_demand import lstm_demand_forecast
from src.models.persistence_forecast import persistence_forecast


_METHOD_ALIASES = {
    "holt_linear": "holt_linear",
    # Backward-compatible name only.  The implementation has no seasonal
    # component and therefore is not Holt-Winters.
    "holt_winters": "holt_linear",
    "lstm": "lstm",
    "persistence": "persistence",
}


def _normalise_method(method: str) -> str:
    key = str(method).strip().lower()
    try:
        return _METHOD_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "demand forecast method must be one of: holt_linear, lstm, "
            "persistence (holt_winters is a legacy alias)"
        ) from exc


def query_demand(
    demand_history: Optional[List[float]] = None,
    horizon: int = 1,
    method: str = "holt_linear",
    cached_uncertainty: Optional[float] = None,
    cached_forecast: Optional[List[float]] = None,
    cached_std: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a demand forecast plus a normalised demand-uncertainty signal in [0, 1].

    Parameters
    ----------
    demand_history : recent demand observations used for the forecast.
    horizon : number of future steps to forecast.
    method : ``"holt_linear"`` (confirmatory default), ``"lstm"``, or
        ``"persistence"`` for the underlying forecaster.  The legacy
        ``"holt_winters"`` spelling maps to non-seasonal Holt-linear.  The
        families are kept pluggable so
        the simulator's ``FORECAST_METHOD`` knob can route through this
        tool without changing numerics.
    cached_uncertainty, cached_forecast, cached_std : when provided by
        a caller that already ran the forecaster this step, the call
        short-circuits and returns the cached values directly.
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

    if not demand_history:
        return {
            "forecast": [],
            "ci_lower": [],
            "ci_upper": [],
            "std": 0.0,
            "uncertainty": 0.0,
            "source": "computed",
            "method": selected_method,
        }

    df = pd.DataFrame({"demand_units": [float(v) for v in demand_history]})
    if selected_method == "holt_linear":
        fc = yield_demand_forecast(df, horizon=horizon)
    elif selected_method == "lstm":
        fc = lstm_demand_forecast(df, horizon=horizon)
    else:
        fc = persistence_forecast(
            df, horizon=horizon, series_col="demand_units",
        )

    point = fc["forecast"][0] if fc["forecast"] else 1.0
    std = float(fc.get("std", 0.0) or 0.0)
    cv = std / max(abs(point), 1.0)
    uncertainty = min(max(cv, 0.0), 1.0)

    return {
        "forecast": fc["forecast"],
        "ci_lower": fc.get("ci_lower", []),
        "ci_upper": fc.get("ci_upper", []),
        "std": std,
        "uncertainty": round(uncertainty, 4),
        "source": "computed",
        "method": selected_method,
    }
