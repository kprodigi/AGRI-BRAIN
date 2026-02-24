"""
PINN-based first-order decay spoilage model.

ODE:  dC/dt = -k(T,H) * C
where k(T,H) = k0 * exp(alpha * (T - T0)) * (1 + beta * H)

Default parameters calibrated per Section 4.1 of the AGRI-BRAIN paper:
    k0    = 0.04   (base decay rate, h^-1)
    alpha = 0.12   (thermal sensitivity, C^-1)
    T0    = 4.0    (reference cold-storage temperature, deg C)
    beta  = 0.25   (humidity coupling coefficient)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PINN spoilage: numerical ODE integration along (T, H) trajectory
# ---------------------------------------------------------------------------

def compute_spoilage(
    df: pd.DataFrame,
    k0: float = 0.04,
    alpha: float = 0.12,
    T0: float = 4.0,
    beta: float = 0.25,
) -> pd.DataFrame:
    """Integrate dC/dt = -k(T,H)*C along the sensor trajectory.

    Parameters
    ----------
    df : DataFrame with columns ``tempC``, ``RH``, ``timestamp``.
    k0, alpha, T0, beta : PINN decay parameters (see module docstring).

    Returns
    -------
    df with two new columns:
        ``shelf_left``   - remaining quality fraction C(t) in [0, 1]
        ``spoilage_risk`` - rho(t) = 1 - C(t), monotonically non-decreasing
    """
    df = df.copy()

    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Time deltas in hours from first reading
    dt_sec = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    dt_h = dt_sec.to_numpy(dtype=np.float64) / 3600.0

    temp = df["tempC"].to_numpy(dtype=np.float64)
    rh = df["RH"].to_numpy(dtype=np.float64) / 100.0  # normalise to [0,1]

    n = len(df)
    C = np.ones(n, dtype=np.float64)  # quality fraction starts at 1.0

    for i in range(1, n):
        delta_t = dt_h[i] - dt_h[i - 1]
        if delta_t <= 0.0:
            C[i] = C[i - 1]
            continue

        # Reaction rate at midpoint temperature / humidity (trapezoidal approx)
        T_mid = 0.5 * (temp[i - 1] + temp[i])
        H_mid = 0.5 * (rh[i - 1] + rh[i])

        k = k0 * np.exp(alpha * (T_mid - T0)) * (1.0 + beta * H_mid)

        # Exact integration of dC/dt = -k*C  =>  C(t+dt) = C(t)*exp(-k*dt)
        C[i] = C[i - 1] * np.exp(-k * delta_t)

    C = np.clip(C, 0.0, 1.0)

    df["shelf_left"] = C
    df["spoilage_risk"] = 1.0 - C

    return df


# ---------------------------------------------------------------------------
# Bollinger z-score volatility flags
# ---------------------------------------------------------------------------

def volatility_flags(
    df: pd.DataFrame,
    window: int = 20,
    k: float = 2.0,
    series_col: str | None = None,
) -> np.ndarray:
    """Flag anomalous readings using a Bollinger-band z-score trigger.

    For each point the z-score is computed as:
        z_i = (x_i - mu_w) / sigma_w
    where mu_w, sigma_w are the rolling mean / std over the last *window*
    observations.  A point is flagged ``'anomaly'`` when |z| > k.

    Parameters
    ----------
    df : DataFrame containing at least one numeric series.
    window : rolling-window size (default 20).
    k : Bollinger threshold in standard deviations (default 2.0).
    series_col : column to analyse.  When *None* the function checks for
        ``demand_units`` then ``yield`` then falls back to ``tempC``.

    Returns
    -------
    numpy array of strings ``'anomaly'`` / ``'normal'``.
    """
    if series_col is None:
        for col in ("demand_units", "yield", "tempC"):
            if col in df.columns:
                series_col = col
                break
        else:
            raise KeyError("No suitable series column found for volatility_flags")

    series = df[series_col].astype(float)

    rolling_mean = series.rolling(window, min_periods=1).mean()
    rolling_std = series.rolling(window, min_periods=1).std().fillna(0.0)

    # Avoid division by zero: when std is zero, z-score is 0
    z_score = np.where(
        rolling_std > 1e-12,
        (series - rolling_mean) / rolling_std,
        0.0,
    )

    return np.where(np.abs(z_score) > k, "anomaly", "normal")
