"""
Mechanistic spoilage-risk model for the synthetic benchmark.

The public entry point is ``compute_spoilage``: a deterministic first-order
Arrhenius ODE integrator with an author-declared rational lag factor. It is the
common baseline used by every simulation mode. The separately versioned
``pinn_residual`` module applies one frozen residual trained against an
independent synthetic DGP; ``no_pinn`` intentionally retains this mechanistic
baseline alone. Neither path is an empirically validated spinach predictor.

The equations use Arrhenius temperature dependence and the rational lag factor
``alpha(t)=t/(t+lambda)``. Microbial lag literature motivates including a lag
phase, but this approximation is not the full Baranyi–Roberts model. The
parameters below are declared
simulation assumptions rather than estimates validated against observed spinach
quality labels. Accordingly, outputs are modelled risk trajectories, not
empirical shelf-life predictions.

ODE:  dC/dt = -k_eff(t, T, H) * C

where:
    k_eff(t, T, H) = k(T, H) * alpha(t)

    k(T, H) = k_ref * exp[Ea_R * (1/T_ref - 1/T_K)] * (1 + beta * a_w)
        Arrhenius-form temperature dependence with humidity coupling.
        - k_ref: dry-condition base rate coefficient at T_ref, before the
          declared humidity multiplier (h^-1)
        - Ea_R = Ea/R: activation energy divided by gas constant (K)
        - T_K = T_C + 273.15: temperature in Kelvin
        - a_w ≈ RH/100: water activity (approximation)
        - beta: humidity coupling coefficient

    alpha(t) = t / (t + lambda)
        Author-declared rational lag adjustment. It represents a gradual onset
        of the decay rate: alpha(0)=0, alpha(lambda)=0.5, alpha→1.

Declared benchmark parameterization:
    k_ref     = 0.0021 h^-1   dry-condition base coefficient at 4°C
    Ea_R      = 8000 K        activation parameter expressed as Ea/R
    T_ref     = 277.15 K      reference temperature (4°C)
    beta      = 0.25          humidity-coupling coefficient
    lambda    = 12.0 h        lag-shape parameter

These numerical values define the synthetic case study. They have not been
fitted to the sensor trace or validated as spinach-specific kinetic constants.

References:
    - Arrhenius, S. (1889). Über die Reaktionsgeschwindigkeit bei der
      Inversion von Rohrzucker durch Säuren. Z. Physikalische Chemie,
      4, 226–248.
    - Baranyi, J. & Roberts, T.A. (1994). A dynamic approach to
      predicting bacterial growth in food. International Journal of
      Food Microbiology, 23(3-4), 277–294.
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
    rh_frac: float | np.ndarray = 0.915,
    beta: float = 0.25,
) -> float | np.ndarray:
    """Compute Arrhenius decay rate k(T, H) without lag adjustment.

    This gives the instantaneous decay rate based purely on environmental
    conditions (temperature, humidity). The lag adjustment alpha(t) is
    applied separately in the ODE integration.

    Parameters
    ----------
    temp_C : temperature in degrees Celsius.
    k_ref : dry-condition base rate coefficient at T_ref_K, before the
        humidity multiplier (h^-1).
    Ea_R : activation energy / gas constant (K).
    T_ref_K : reference temperature (K).
    rh_frac : relative humidity as fraction [0, 1].
    beta : humidity coupling coefficient.

    Returns
    -------
    Decay rate k (h^-1), same shape as temp_C.
    """
    T_K = np.asarray(temp_C, dtype=np.float64) + 273.15
    # Arrhenius equation (Arrhenius, 1889): temperature-dependent rate constant
    #   k(T) = A * exp(-Ea / (R * T))
    # Rearranged with reference conditions:
    #   k(T) = k_ref * exp[Ea_R * (1/T_ref - 1/T)]
    # where Ea_R = Ea/R (K), R = 8.314 J/(mol*K)
    k = k_ref * np.exp(Ea_R * (1.0 / T_ref_K - 1.0 / T_K))
    # Declared benchmark humidity coupling: water activity accelerates decay
    #   k_eff = k(T) * (1 + beta * a_w), where a_w ≈ RH/100
    k = k * (1.0 + beta * np.asarray(rh_frac, dtype=np.float64))
    return k


def advance_spoilage_risk_midpoint(
    previous_rho: float,
    *,
    previous_temp_C: float,
    current_temp_C: float,
    previous_rh_pct: float,
    current_rh_pct: float,
    previous_hour: float,
    current_hour: float,
    k_ref: float = 0.0021,
    Ea_R: float = 8000.0,
    T_ref_K: float = 277.15,
    beta: float = 0.25,
    lag_lambda: float = 12.0,
) -> float:
    """Advance cumulative spoilage risk by one midpoint-rule interval.

    This is the single state-transition used by both the vector trajectory
    generator and the online policy-observation path.  Updating the surviving
    quality fraction, rather than rescaling an already cumulative risk by an
    instantaneous-rate ratio, guarantees an irreversible ``rho`` trajectory.
    """
    rho0 = float(np.clip(previous_rho, 0.0, 1.0))
    delta_t = float(current_hour) - float(previous_hour)
    if delta_t <= 0.0 or rho0 >= 1.0:
        return rho0

    t_mid = 0.5 * (float(previous_hour) + float(current_hour))
    temp_mid = 0.5 * (float(previous_temp_C) + float(current_temp_C))
    rh_mid = np.clip(
        0.5 * (float(previous_rh_pct) + float(current_rh_pct)) / 100.0,
        0.0,
        1.0,
    )
    rate = float(arrhenius_k(
        temp_mid,
        k_ref=k_ref,
        Ea_R=Ea_R,
        T_ref_K=T_ref_K,
        rh_frac=rh_mid,
        beta=beta,
    ))
    if lag_lambda > 0.0 and (t_mid + lag_lambda) > 0.0:
        alpha = max(0.0, t_mid / (t_mid + lag_lambda))
    else:
        alpha = 1.0

    quality0 = 1.0 - rho0
    quality1 = quality0 * np.exp(-rate * alpha * delta_t)
    return float(np.clip(max(rho0, 1.0 - quality1), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Mechanistic spoilage: numerical ODE integration along (T, H) trajectory
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

    Uses the Arrhenius temperature model with a declared rational lag factor
    for a synthetic quality-risk trajectory.

    The effective rate k_eff(t) = k(T,H) * alpha(t) where:
    - k(T,H) is the Arrhenius decay rate (see arrhenius_k)
    - alpha(t) = t/(t + lag_lambda) is the rational lag adjustment

    Integration uses the midpoint rule (trapezoidal approximation) for
    temperature, humidity, and time.

    Parameters
    ----------
    df : DataFrame with columns ``tempC``, ``RH``, ``timestamp``.
    k_ref : dry-condition base rate coefficient at T_ref_K, before the
        humidity multiplier (h^-1).
    Ea_R : Arrhenius activation energy / gas constant (K).
    T_ref_K : reference temperature in Kelvin.
    beta : humidity coupling coefficient.
    lag_lambda : rational lag parameter (hours). Set to 0 to disable.

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
    rh_pct = df["RH"].to_numpy(dtype=np.float64)

    n = len(df)
    rho = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        delta_t = dt_h[i] - dt_h[i - 1]
        if delta_t <= 0.0:
            rho[i] = rho[i - 1]
            continue
        rho[i] = advance_spoilage_risk_midpoint(
            rho[i - 1],
            previous_temp_C=temp[i - 1],
            current_temp_C=temp[i],
            previous_rh_pct=rh_pct[i - 1],
            current_rh_pct=rh_pct[i],
            previous_hour=dt_h[i - 1],
            current_hour=dt_h[i],
            k_ref=k_ref,
            Ea_R=Ea_R,
            T_ref_K=T_ref_K,
            beta=beta,
            lag_lambda=lag_lambda,
        )

    rho = np.maximum.accumulate(np.clip(rho, 0.0, 1.0))
    df["shelf_left"] = 1.0 - rho
    df["spoilage_risk"] = rho

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
