"""Transparent persistence forecast with rolling empirical uncertainty.

This is the confirmatory fallback when a more complex forecaster does not beat
one-step persistence on the locked validation segment.  It predicts the most
recent observed value and estimates one-step error scale from recent first
differences, which are exactly the historical errors of the persistence rule.
No target at or after the forecast origin is used.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def persistence_forecast(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    lookback: int = 48,
    residual_tail: int = 8,
    ci_z: float = 1.96,
    series_col: str,
) -> Dict[str, object]:
    """Forecast the last value and return Gaussian rolling-difference bands."""
    values = df[series_col].astype(float).to_numpy()
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if len(values) == 0:
        zeros = [0.0] * horizon
        return {
            "forecast": zeros,
            "ci_lower": zeros,
            "ci_upper": zeros,
            "std": 0.0,
            "series_std": 0.0,
        }
    tail = values[-min(lookback, len(values)):]
    point = max(0.0, float(tail[-1]))
    errors = np.diff(tail)
    if len(errors) > residual_tail:
        errors = errors[-residual_tail:]
    residual_std = float(np.std(errors, ddof=0)) if len(errors) >= 2 else 0.0
    forecast = [round(point, 4)] * horizon
    lower = [
        round(max(0.0, point - ci_z * residual_std * np.sqrt(step)), 4)
        for step in range(1, horizon + 1)
    ]
    upper = [
        round(point + ci_z * residual_std * np.sqrt(step), 4)
        for step in range(1, horizon + 1)
    ]
    return {
        "forecast": forecast,
        "ci_lower": lower,
        "ci_upper": upper,
        "std": round(residual_std, 6),
        "series_std": round(float(np.std(tail, ddof=0)), 6),
    }
