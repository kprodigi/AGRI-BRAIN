"""Pure scenario perturbation engine.

Holds the canonical implementation of the four perturbation scenarios
(``heatwave``, ``overproduction``, ``cyber_outage``, ``adaptive_pricing``)
in a router-free module so that the simulator (``mvp/simulation/generate_results.py``)
and the FastAPI ``/scenarios`` router can both consume it without the
domain layer importing router internals or HTTP-coupled state.

Each function is a pure transformation ``(df, policy, intensity) -> df``.
The ``Policy`` instance supplies the spoilage kinetics used by
``_recompute_derived``; passing it in explicitly removes the previous
hidden dependency on the FastAPI router's module-level ``_APP_STATE``.

The router (``src.routers.scenarios``) re-exports the underscored
functions for backward compatibility with downstream callers.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from src.models.policy import Policy
from src.models.spoilage import compute_spoilage, volatility_flags
from src.models.waste import INV_BASELINE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hours_from_start(df: pd.DataFrame) -> np.ndarray:
    """Return the elapsed-hours array for a telemetry dataframe."""
    ts = pd.to_datetime(df["timestamp"])
    return ((ts - ts.iloc[0]).dt.total_seconds() / 3600.0).to_numpy(dtype=np.float64)


def recompute_derived(df: pd.DataFrame, policy: Optional[Policy]) -> pd.DataFrame:
    """Re-run PINN spoilage + Bollinger volatility against ``policy``.

    Used after every scenario perturbation so the spoilage and
    volatility columns reflect the modified telemetry. ``policy`` may be
    ``None`` (callers without a Policy in scope), in which case Policy's
    field defaults are used.
    """
    p = policy or Policy()
    df = compute_spoilage(
        df,
        k_ref=p.k_ref,
        Ea_R=p.Ea_R,
        T_ref_K=p.T_ref_K,
        beta=p.beta_humidity,
        lag_lambda=p.lag_lambda,
    )
    df["volatility"] = volatility_flags(df, window=p.boll_window, k=p.boll_k)
    return df


# ---------------------------------------------------------------------------
# Scenario perturbations
# ---------------------------------------------------------------------------

def apply_heatwave(df: pd.DataFrame, policy: Optional[Policy] = None,
                   intensity: float = 1.0) -> pd.DataFrame:
    """Inject +20 C sigmoid onset hours 24-48, exponential tail after, +10 % RH.

    Uses sigmoid onset (reaches ~95% of peak by h ~= 30) rather than a
    linear ramp -- heatwaves build rapidly once they onset (WMO, 2018).
    """
    df = df.copy()
    hours = hours_from_start(df)
    n = len(df)
    temp_add = np.zeros(n)
    rh_add = np.zeros(n)

    for i in range(n):
        h = hours[i]
        if 24.0 <= h <= 48.0:
            onset = 1.0 - np.exp(-0.5 * (h - 24.0))
            temp_add[i] = 20.0 * onset * intensity
            rh_add[i] = 10.0 * onset * intensity
        elif h > 48.0:
            temp_add[i] = 20.0 * intensity * np.exp(-0.1 * (h - 48.0))
            rh_add[i] = 10.0 * intensity * np.exp(-0.1 * (h - 48.0))

    df["tempC"] = df["tempC"].astype(float) + temp_add
    df["RH"] = (df["RH"].astype(float) + rh_add).clip(0, 100)
    return recompute_derived(df, policy)


def apply_overproduction(df: pd.DataFrame, policy: Optional[Policy] = None,
                         intensity: float = 1.0) -> pd.DataFrame:
    """Multiply inventory by 2.5x during hours 12-60 with progressive cold storage excursion.

    Overloaded cold storage: at 2.5x capacity, reduced airflow and compressor
    strain raise cold-room temperature by up to +8 C (James & James, 2010).
    Sigmoid onset (~95% by h ~= 22), exponential recovery after hour 60.
    """
    df = df.copy()
    df["inventory_units"] = df["inventory_units"].astype(float)
    df["tempC"] = df["tempC"].astype(float)
    hours = hours_from_start(df)
    n = len(df)
    mask = (hours >= 12.0) & (hours <= 60.0)
    df.loc[mask, "inventory_units"] = df.loc[mask, "inventory_units"] * 2.5 * intensity

    temp_add = np.zeros(n)
    for i in range(n):
        h = hours[i]
        if 12.0 <= h <= 60.0:
            onset = 1.0 - np.exp(-0.3 * (h - 12.0))
            temp_add[i] = 8.0 * onset * intensity
        elif h > 60.0:
            temp_add[i] = 8.0 * intensity * np.exp(-0.15 * (h - 60.0))
    df["tempC"] = df["tempC"] + temp_add
    return recompute_derived(df, policy)


def apply_cyber_outage(df: pd.DataFrame, policy: Optional[Policy] = None,
                       intensity: float = 1.0) -> pd.DataFrame:
    """Processor offline from hour 24: demand drops to 15 %, inventory accumulates.

    IT-controlled cooling fails, causing a +10 C sigmoid temperature
    excursion (reaches ~95 % of peak within ~5 h of onset). Inventory
    stays at 100 % -- produce keeps arriving from farms, creating an
    accumulation crisis while the processor is offline.
    """
    df = df.copy()
    df["demand_units"] = df["demand_units"].astype(float)
    df["tempC"] = df["tempC"].astype(float)
    hours = hours_from_start(df)
    n = len(df)
    mask = hours >= 24.0
    df.loc[mask, "demand_units"] = df.loc[mask, "demand_units"] * 0.15 * intensity
    temp_add = np.zeros(n)
    for i in range(n):
        h = hours[i]
        if h >= 24.0:
            onset = 1.0 - np.exp(-0.2 * (h - 24.0))
            temp_add[i] = 10.0 * onset * intensity
    df["tempC"] = df["tempC"] + temp_add
    return recompute_derived(df, policy)


def apply_adaptive_pricing(df: pd.DataFrame, policy: Optional[Policy] = None,
                           intensity: float = 1.0) -> pd.DataFrame:
    """Add demand oscillation (amplitude 45, period 60) + noise (std 14).

    Cold-storage stress from demand volatility: frequent dock openings,
    variable loading patterns, and supply-demand mismatch degrade
    temperature management (Mercier et al., 2017).
    """
    df = df.copy()
    df["demand_units"] = df["demand_units"].astype(float)
    df["tempC"] = df["tempC"].astype(float)
    df["inventory_units"] = df["inventory_units"].astype(float)
    n = len(df)
    rng = np.random.default_rng(42)
    oscillation = 45.0 * intensity * np.sin(2.0 * np.pi * np.arange(n) / 60.0)
    noise = rng.normal(0.0, 14.0 * intensity, size=n)
    df["demand_units"] = (df["demand_units"] + oscillation + noise).clip(0)
    demand = df["demand_units"].to_numpy()
    inv = df["inventory_units"].to_numpy()
    demand_dev = np.abs(demand - np.median(demand)) / (np.median(demand) + 1.0)
    surplus_signal = np.clip((inv / INV_BASELINE - 1.0), 0, 2.0)
    temp_add = 1.5 * intensity * np.clip(demand_dev, 0, 1) + 2.0 * intensity * surplus_signal
    df["tempC"] = df["tempC"] + temp_add
    return recompute_derived(df, policy)


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

#: Canonical scenario id -> perturbation function map. Use ``apply()`` for
#: a more permissive lookup that returns a recomputed-baseline copy when
#: the id is not recognised.
SCENARIO_FUNCTIONS: Dict[str, Callable[..., pd.DataFrame]] = {
    "heatwave": apply_heatwave,
    "overproduction": apply_overproduction,
    "cyber_outage": apply_cyber_outage,
    "adaptive_pricing": apply_adaptive_pricing,
}


def apply(name: str, df: pd.DataFrame, policy: Optional[Policy] = None,
          intensity: float = 1.0) -> pd.DataFrame:
    """Apply a named scenario or return a recomputed baseline copy.

    A name that is not in :data:`SCENARIO_FUNCTIONS` (including the
    sentinel ``"baseline"``) returns ``df`` with derived columns
    refreshed -- this is the same semantics as the previous router-side
    ``_apply_to_state`` baseline branch, lifted out so the simulator and
    router share it.
    """
    fn = SCENARIO_FUNCTIONS.get(name)
    if fn is None:
        return recompute_derived(df.copy(), policy)
    return fn(df.copy(), policy=policy, intensity=intensity)


# Backward-compatible aliases so older imports keep working.
_apply_heatwave = apply_heatwave
_apply_overproduction = apply_overproduction
_apply_cyber_outage = apply_cyber_outage
_apply_adaptive_pricing = apply_adaptive_pricing
_hours_from_start = hours_from_start
_recompute_derived = recompute_derived


__all__ = [
    "apply",
    "SCENARIO_FUNCTIONS",
    "hours_from_start",
    "recompute_derived",
    "apply_heatwave",
    "apply_overproduction",
    "apply_cyber_outage",
    "apply_adaptive_pricing",
    # legacy underscore aliases
    "_apply_heatwave",
    "_apply_overproduction",
    "_apply_cyber_outage",
    "_apply_adaptive_pricing",
    "_hours_from_start",
    "_recompute_derived",
]
