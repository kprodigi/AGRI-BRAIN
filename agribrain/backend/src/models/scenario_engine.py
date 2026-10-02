"""Pure scenario perturbation engine.

Holds the canonical implementation of the four perturbation scenarios
(``heatwave``, ``overproduction``, ``cyber_outage``, ``adaptive_pricing``)
in a router-free module so that the simulator (``mvp/simulation/generate_results.py``)
and the FastAPI ``/scenarios`` router can both consume it without the
domain layer importing router internals or HTTP-coupled state.

Each function is a transformation of the supplied inputs. The stochastic
adaptive-pricing treatment additionally accepts an explicit random generator;
publication callers must pass their per-seed scenario generator.
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


def validate_scenario_controls(
    intensity: float, onset_offset_hours: float = 0.0,
) -> tuple[float, float]:
    """Validate public scenario controls and return canonical floats.

    ``intensity=0`` is the identity treatment and values above one are
    permitted for the interactive sensitivity slider. Negative or non-finite
    values are rejected instead of silently inverting a disruption or
    contaminating the telemetry trace with NaNs.
    """
    intensity_value = float(intensity)
    onset_value = float(onset_offset_hours)
    if not np.isfinite(intensity_value) or intensity_value < 0.0:
        raise ValueError("scenario intensity must be finite and non-negative")
    if not np.isfinite(onset_value):
        raise ValueError("scenario onset offset must be finite")
    return intensity_value, onset_value


def recompute_derived(df: pd.DataFrame, policy: Optional[Policy]) -> pd.DataFrame:
    """Re-run mechanistic spoilage risk and Bollinger volatility.

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
                   intensity: float = 1.0,
                   onset_offset_hours: float = 0.0) -> pd.DataFrame:
    """Apply a declared +20 C exponential approach over hours 24-48 and tail.

    The approach ``1-exp(-0.5*(h-24))`` reaches about 95% at hour 30.
    Relative humidity receives a matching +10 percentage-point approach and
    is clipped to [0,100]. These are synthetic benchmark perturbations, not
    field-calibrated heatwave magnitudes.
    """
    intensity, onset_offset_hours = validate_scenario_controls(
        intensity, onset_offset_hours,
    )
    df = df.copy()
    hours = hours_from_start(df) - onset_offset_hours
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
                         intensity: float = 1.0,
                         onset_offset_hours: float = 0.0) -> pd.DataFrame:
    """Apply declared overproduction and temperature perturbations.

    At nominal intensity, inventory is multiplied by 2.5 during hours 12-60.
    Temperature follows the synthetic +8 C exponential approach
    ``1-exp(-0.3*(h-12))`` (about 95% by hour 22) and an exponential tail
    after hour 60. These values are declared benchmark assumptions.
    """
    intensity, onset_offset_hours = validate_scenario_controls(
        intensity, onset_offset_hours,
    )
    df = df.copy()
    df["inventory_units"] = df["inventory_units"].astype(float)
    df["tempC"] = df["tempC"].astype(float)
    hours = hours_from_start(df) - onset_offset_hours
    n = len(df)
    mask = (hours >= 12.0) & (hours <= 60.0)
    inventory_multiplier = 1.0 + 1.5 * intensity
    df.loc[mask, "inventory_units"] = (
        df.loc[mask, "inventory_units"] * inventory_multiplier
    )

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
                       intensity: float = 1.0,
                       onset_offset_hours: float = 0.0) -> pd.DataFrame:
    """Apply the declared synthetic cyber-outage perturbation from hour 24.

    The synthetic cooling disturbance causes a +10 C exponential temperature
    excursion: 63.2 % of the asymptotic rise is reached after 5 h and about
    95 % after 15 h. The transform does not directly overwrite inventory, and
    ordinary processor-stage decision ownership remains active.
    """
    intensity, onset_offset_hours = validate_scenario_controls(
        intensity, onset_offset_hours,
    )
    df = df.copy()
    df["demand_units"] = df["demand_units"].astype(float)
    df["tempC"] = df["tempC"].astype(float)
    hours = hours_from_start(df) - onset_offset_hours
    n = len(df)
    mask = hours >= 24.0
    demand_multiplier = 1.0 - 0.85 * intensity
    df.loc[mask, "demand_units"] = (
        df.loc[mask, "demand_units"] * max(0.0, demand_multiplier)
    )
    temp_add = np.zeros(n)
    for i in range(n):
        h = hours[i]
        if h >= 24.0:
            onset = 1.0 - np.exp(-0.2 * (h - 24.0))
            temp_add[i] = 10.0 * onset * intensity
    df["tempC"] = df["tempC"] + temp_add
    return recompute_derived(df, policy)


def apply_adaptive_pricing(df: pd.DataFrame, policy: Optional[Policy] = None,
                           intensity: float = 1.0,
                           onset_offset_hours: float = 0.0,
                           rng: Optional[np.random.Generator] = None,
                           ) -> pd.DataFrame:
    """Add the declared demand oscillation and synthetic temperature coupling.

    At nominal intensity, demand receives a sinusoid of amplitude 45 and period
    60 plus Gaussian noise with standard deviation 14. Temperature receives
    the author-declared demand-deviation and surplus adjustments below.
    """
    intensity, onset_offset_hours = validate_scenario_controls(
        intensity, onset_offset_hours,
    )
    df = df.copy()
    df["demand_units"] = df["demand_units"].astype(float)
    df["tempC"] = df["tempC"].astype(float)
    df["inventory_units"] = df["inventory_units"].astype(float)
    n = len(df)
    # The publication driver always supplies its scenario/seed-specific RNG.
    # A fresh generator is retained only for interactive callers that do not
    # request reproducibility; never silently collapse every benchmark seed to
    # one hard-coded adaptive-pricing realization.
    rng = rng if rng is not None else np.random.default_rng()
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

#: Canonical non-baseline scenario id -> perturbation function map.
SCENARIO_FUNCTIONS: Dict[str, Callable[..., pd.DataFrame]] = {
    "heatwave": apply_heatwave,
    "overproduction": apply_overproduction,
    "cyber_outage": apply_cyber_outage,
    "adaptive_pricing": apply_adaptive_pricing,
}


def apply(name: str, df: pd.DataFrame, policy: Optional[Policy] = None,
          intensity: float = 1.0,
          onset_offset_hours: float = 0.0,
          rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """Apply a named scenario, with an explicit unperturbed baseline path.

    Only the exact sentinel ``"baseline"`` returns a recomputed baseline copy.
    Every other unknown identifier raises before a scenario result is produced.
    """
    intensity, onset_offset_hours = validate_scenario_controls(
        intensity, onset_offset_hours,
    )
    if name == "baseline":
        return recompute_derived(df.copy(), policy)
    fn = SCENARIO_FUNCTIONS.get(name)
    if fn is None:
        raise ValueError(
            f"unknown scenario {name!r}; expected baseline or one of "
            f"{sorted(SCENARIO_FUNCTIONS)}"
        )
    kwargs = {
        "policy": policy,
        "intensity": intensity,
        "onset_offset_hours": onset_offset_hours,
    }
    if fn is apply_adaptive_pricing:
        kwargs["rng"] = rng
    return fn(df.copy(), **kwargs)


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
    "validate_scenario_controls",
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
