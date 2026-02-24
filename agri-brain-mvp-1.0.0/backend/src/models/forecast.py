"""
Lightweight single-step forecaster with horizon tiling.

Implements the approach from Section 4.2.2 of the AGRI-BRAIN paper:
  1. Compute a single-step forecast y_hat_{t+1} via exponential smoothing
     (stand-in for the LSTM single-step head when training data is absent).
  2. Tile the one-step estimate across the full horizon:
         Y_hat[t+1 : t+h] = [y_hat_{t+1}, ..., y_hat_{t+1}]
  3. Return both the forecast array and a simple confidence band.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def yield_demand_forecast(
    df: pd.DataFrame,
    horizon: int = 24,
    ema_alpha: float = 0.6,
    lookback: int = 48,
    ci_z: float = 1.96,
    series_col: str = "demand_units",
) -> Dict[str, object]:
    """Produce a horizon-tiled forecast with confidence interval.

    Parameters
    ----------
    df : DataFrame with at least a *series_col* column.
    horizon : number of future steps to forecast.
    ema_alpha : smoothing factor for exponential moving average (0 < a <= 1).
    lookback : number of most-recent observations used to seed the EMA.
    ci_z : z-score multiplier for the confidence interval (default 1.96 = 95 %).
    series_col : column name to forecast.

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

    # --- single-step EMA forecast ---
    tail = d[-min(lookback, len(d)):]
    s = tail[0]
    for x in tail[1:]:
        s = ema_alpha * x + (1.0 - ema_alpha) * s

    # Incorporate a mild trend correction (dampened)
    if len(tail) > 1:
        trend = float(np.gradient(tail).mean())
        y_hat = max(0.0, s + 0.2 * trend)
    else:
        y_hat = max(0.0, s)

    # --- horizon tiling (Section 4.2.2) ---
    forecast = [round(y_hat, 4)] * horizon

    # --- confidence interval from rolling std ---
    if len(d) >= 2:
        std = float(np.std(tail))
    else:
        std = 0.0

    ci_lower = [round(max(0.0, y_hat - ci_z * std), 4)] * horizon
    ci_upper = [round(y_hat + ci_z * std, 4)] * horizon

    return {
        "forecast": forecast,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": round(std, 6),
    }
