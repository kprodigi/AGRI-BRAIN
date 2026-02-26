"""
PINN-based spoilage model with Arrhenius temperature dependence and Baranyi lag phase.

ODE:  dC/dt = -k_eff(t, T, H) * C

where:
    k_eff(t, T, H) = k(T, H) * alpha(t)

    k(T, H) = k_ref * exp[Ea_R * (1/T_ref - 1/T_K)] * (1 + beta * a_w)
        Arrhenius-form temperature dependence with humidity coupling.
        - k_ref: reference decay rate at T_ref (h^-1)
        - Ea_R = Ea/R: activation energy divided by gas constant (K)
        - T_K = T_C + 273.15: temperature in Kelvin
        - a_w ≈ RH/100: water activity (approximation)
        - beta: humidity coupling coefficient

    alpha(t) = t / (t + lambda)
        Baranyi lag adjustment. Fresh produce has an initial lag before
        exponential quality loss begins, due to microbial adaptation time
        (Baranyi & Roberts, 1994). alpha(0)=0, alpha(lambda)=0.5, alpha→1.

Parameters calibrated for fresh spinach (Spinacia oleracea):
    k_ref     = 0.0021 h^-1  (at 4°C, gives ~12% quality loss over 72h)
    Ea_R      = 8000 K        (Ea ≈ 66.5 kJ/mol, typical for leafy greens;
                               Giannakourou & Taoukis, 2003)
    T_ref     = 277.15 K      (4°C, standard cold storage per FDA)
    beta      = 0.25          (humidity coupling; Labuza, 1982)
    lambda    = 12.0 h        (lag phase for spinach at 4°C)

References:
    - Labuza, T.P. & Riboh, D. (1982). Theory and application of Arrhenius
      kinetics to the prediction of nutrient losses in foods.
    - Giannakourou, M.C. & Taoukis, P.S. (2003). Kinetic modelling of
      vitamin C loss in frozen green vegetables under variable storage
      conditions. Food Chemistry.
    - Baranyi, J. & Roberts, T.A. (1994). A dynamic approach to predicting
      bacterial growth in food. Int. J. Food Microbiology.
    - FDA (2017). Guidelines for the safe handling and storage of fresh-cut
      leafy greens.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Arrhenius decay rate helper (also used by generate_results.py for waste)
# ---------------------------------------------------------------------------

def arrhenius_k(
    temp_C: float | np.ndarray,
    k_ref: float = 0.0021,
    Ea_R: float = 8000.0,
    T_ref_K: float = 277.15,
    rh_frac: float | np.ndarray = 0.89,
    beta: float = 0.25,
) -> float | np.ndarray:
    """Compute Arrhenius decay rate k(T, H) without lag adjustment.

    This gives the instantaneous decay rate based purely on environmental
    conditions (temperature, humidity). The lag adjustment alpha(t) is
    applied separately in the ODE integration.

    Parameters
    ----------
    temp_C : temperature in degrees Celsius.
    k_ref : reference decay rate at T_ref_K (h^-1).
    Ea_R : activation energy / gas constant (K).
    T_ref_K : reference temperature (K).
    rh_frac : relative humidity as fraction [0, 1].
    beta : humidity coupling coefficient.

    Returns
    -------
    Decay rate k (h^-1), same shape as temp_C.
    """
    T_K = np.asarray(temp_C, dtype=np.float64) + 273.15
    # Arrhenius: k = k_ref * exp[Ea_R * (1/T_ref - 1/T)]
    k = k_ref * np.exp(Ea_R * (1.0 / T_ref_K - 1.0 / T_K))
    # Humidity coupling: higher water activity accelerates microbial growth
    k = k * (1.0 + beta * np.asarray(rh_frac, dtype=np.float64))
    return k


# ---------------------------------------------------------------------------
# PINN spoilage: numerical ODE integration along (T, H) trajectory
# ---------------------------------------------------------------------------

def compute_spoilage(
    df: pd.DataFrame,
    k_ref: float = 0.0021,
    Ea_R: float = 8000.0,
    T_ref_K: float = 277.15,
    beta: float = 0.25,
    lag_lambda: float = 12.0,
) -> pd.DataFrame:
    """Integrate dC/dt = -k_eff(t,T,H)*C along the sensor trajectory.

    Uses the Arrhenius temperature model with Baranyi lag phase for
    physically realistic quality degradation of fresh spinach.

    The effective rate k_eff(t) = k(T,H) * alpha(t) where:
    - k(T,H) is the Arrhenius decay rate (see arrhenius_k)
    - alpha(t) = t/(t + lag_lambda) is the Baranyi lag adjustment

    Integration uses the midpoint rule (trapezoidal approximation) for
    temperature, humidity, and time.

    Parameters
    ----------
    df : DataFrame with columns ``tempC``, ``RH``, ``timestamp``.
    k_ref : reference decay rate at T_ref_K (h^-1).
    Ea_R : Arrhenius activation energy / gas constant (K).
    T_ref_K : reference temperature in Kelvin.
    beta : humidity coupling coefficient.
    lag_lambda : Baranyi lag phase parameter (hours). Set to 0 to disable.

    Returns
    -------
    df with two new columns:
        ``shelf_left``    - remaining quality fraction C(t) in [0, 1]
        ``spoilage_risk`` - rho(t) = 1 - C(t), monotonically non-decreasing
    """
    df = df.copy()

    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Time in hours from first reading
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

        # Midpoint temperature, humidity, and time for trapezoidal integration
        T_mid = 0.5 * (temp[i - 1] + temp[i])
        H_mid = 0.5 * (rh[i - 1] + rh[i])
        t_mid = 0.5 * (dt_h[i - 1] + dt_h[i])

        # Arrhenius decay rate at current conditions
        k = arrhenius_k(T_mid, k_ref, Ea_R, T_ref_K, H_mid, beta)

        # Baranyi lag adjustment: alpha(t) = t / (t + lambda)
        # At t=0: alpha=0 (no decay), at t=lambda: alpha=0.5, at t>>lambda: alpha→1
        if lag_lambda > 0.0 and (t_mid + lag_lambda) > 0.0:
            alpha = t_mid / (t_mid + lag_lambda)
        else:
            alpha = 1.0

        k_eff = k * alpha

        # Exact integration of dC/dt = -k_eff*C => C(t+dt) = C(t)*exp(-k_eff*dt)
        C[i] = C[i - 1] * np.exp(-k_eff * delta_t)

    # Enforce monotone decay: C should never increase
    for i in range(1, n):
        if C[i] > C[i - 1]:
            C[i] = C[i - 1]

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
