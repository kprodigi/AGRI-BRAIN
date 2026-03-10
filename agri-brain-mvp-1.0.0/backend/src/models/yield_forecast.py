"""Short-term yield/supply forecaster using Holt-Winters double exponential smoothing.

Operates on the inventory_units series as a supply/yield proxy, providing a
distinct supply-side forecast separate from the LSTM demand forecaster.

This module implements the "Short-term yield forecaster" component shown in the
AGRI-BRAIN architectural figure, using the freed Holt-Winters method (previously
used for demand forecasting, now replaced by the LSTM demand model).

References
----------
    - Holt, C.C. (1957). Forecasting seasonals and trends by exponentially
      weighted averages. ONR Memorandum No. 52.
    - Winters, P.R. (1960). Forecasting sales by exponentially weighted
      moving averages. Management Science, 6(3), 324-342.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def yield_supply_forecast(
    df: pd.DataFrame,
    horizon: int = 24,
    ema_alpha: float = 0.5,
    lookback: int = 48,
    ci_z: float = 1.96,
    series_col: str = "inventory_units",
    trend_beta: float = 0.2,
) -> Dict[str, object]:
    """Produce a horizon-tiled yield/supply forecast with confidence interval.

    Uses Holt's double exponential smoothing (level + trend) on the
    inventory_units series as a supply proxy.

    Parameters
    ----------
    df : DataFrame with at least a *series_col* column.
    horizon : number of future steps to forecast.
    ema_alpha : level smoothing factor (0 < alpha <= 1).
    lookback : number of most-recent observations used to seed the smoother.
    ci_z : z-score multiplier for the confidence interval (default 1.96 = 95%).
    series_col : column name to forecast (default: inventory_units).
    trend_beta : trend smoothing factor (0 < beta <= 1).

    Returns
    -------
    dict with keys:
        ``forecast``   - list[float] of length *horizon* (point forecast)
        ``ci_lower``   - list[float] lower bound of CI
        ``ci_upper``   - list[float] upper bound of CI
        ``std``        - float, historical rolling std used for CI
    """
    d = df[series_col].astype(float).to_numpy()

    if len(d) == 0:
        zeros = [0.0] * horizon
        return {"forecast": zeros, "ci_lower": zeros, "ci_upper": zeros, "std": 0.0}

    # Use the most recent observations
    tail = d[-min(lookback, len(d)):]

    # Holt-Winters double exponential smoothing
    level = tail[0]
    if len(tail) > 1:
        trend = float(tail[1] - tail[0])
    else:
        trend = 0.0

    for x in tail[1:]:
        prev_level = level
        level = ema_alpha * x + (1.0 - ema_alpha) * (level + trend)
        trend = trend_beta * (level - prev_level) + (1.0 - trend_beta) * trend

    # Multi-step forecast
    forecast = []
    for h in range(1, horizon + 1):
        y_hat = max(0.0, level + h * trend)
        forecast.append(round(y_hat, 4))

    # Confidence interval from rolling std
    if len(d) >= 2:
        std = float(np.std(tail))
    else:
        std = 0.0

    ci_lower = [round(max(0.0, f - ci_z * std), 4) for f in forecast]
    ci_upper = [round(f + ci_z * std, 4) for f in forecast]

    return {
        "forecast": forecast,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": round(std, 6),
    }
