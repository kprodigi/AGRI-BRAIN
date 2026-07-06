"""Short-term yield/supply forecaster using Holt's linear (double exponential smoothing) method.

Operates on the inventory_units series as a supply/yield proxy, providing a
distinct supply-side forecast separate from the LSTM demand forecaster.

This module implements the "Short-term yield forecaster" component shown in the
AGRI-BRAIN architectural figure, using the freed Holt's-linear (level + trend)
method previously used for demand forecasting (now replaced by the LSTM demand
model). No seasonal indices are computed, so this is Holt's method (Holt,
1957) rather than the seasonal Holt-Winters extension.

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
        ``forecast``     - list[float] of length *horizon* (point forecast)
        ``ci_lower``     - list[float] lower bound of CI
        ``ci_upper``     - list[float] upper bound of CI
        ``std``          - float, in-sample one-step-ahead residual
                           standard deviation (prediction-uncertainty
                           estimate; Hyndman & Athanasopoulos 2018,
                           *Forecasting: Principles and Practice* 2nd
                           ed., Ch. 8.7, eq. 8.16).
        ``series_std``   - float, raw dispersion of the training tail,
                           retained for backward compatibility.
    """
    d = df[series_col].astype(float).to_numpy()

    if len(d) == 0:
        zeros = [0.0] * horizon
        return {
            "forecast": zeros, "ci_lower": zeros, "ci_upper": zeros,
            "std": 0.0, "series_std": 0.0,
        }

    # Use the most recent observations
    tail = d[-min(lookback, len(d)):]

    # Holt's linear (level + trend) double exponential smoothing, recording
    # the one-step-ahead forecast at each timestep to reconstruct residuals
    # afterwards. No seasonal component is fit, so this is Holt 1957 rather
    # than seasonal Holt-Winters.
    level = tail[0]
    if len(tail) > 1:
        trend = float(tail[1] - tail[0])
    else:
        trend = 0.0

    in_sample_preds = []
    for x in tail[1:]:
        # Forecast for x using state *before* absorbing it.
        in_sample_preds.append(level + trend)
        prev_level = level
        level = ema_alpha * x + (1.0 - ema_alpha) * (level + trend)
        trend = trend_beta * (level - prev_level) + (1.0 - trend_beta) * trend

    # Multi-step forecast
    forecast = []
    for h in range(1, horizon + 1):
        y_hat = max(0.0, level + h * trend)
        forecast.append(round(y_hat, 4))

    # Prediction uncertainty: standard deviation of in-sample one-step-
    # ahead residuals on the recent tail. Proper residual-std estimate of
    # prediction-error sigma, not raw series dispersion.
    if len(in_sample_preds) >= 2:
        preds_arr = np.asarray(in_sample_preds, dtype=float)
        targets_arr = tail[1:]
        residuals = targets_arr - preds_arr
        # Use only the most recent 8 residuals so uncertainty tracks the
        # current regime, symmetric with lstm_demand.LSTMDemandModel.
        if len(residuals) > 8:
            residuals = residuals[-8:]
        residual_std = float(np.std(residuals, ddof=0))
    else:
        residual_std = 0.0

    series_std = float(np.std(tail)) if len(d) >= 2 else 0.0

    ci_lower = [round(max(0.0, f - ci_z * residual_std), 4) for f in forecast]
    ci_upper = [round(f + ci_z * residual_std, 4) for f in forecast]

    return {
        "forecast": forecast,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": round(residual_std, 6),
        "series_std": round(series_std, 6),
    }
