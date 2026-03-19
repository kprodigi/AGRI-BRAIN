# backend/src/routers/scenarios.py
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, Optional

import numpy as np

router = APIRouter()

# ---- in-memory active scenario ----
ACTIVE: Dict[str, Any] = {"name": None, "intensity": 1.0}

# ---- reference to the app-level state dict (set by app.py at startup) ----
_APP_STATE: Optional[Dict[str, Any]] = None


def register_app_state(st: Dict[str, Any]) -> None:
    """Called once by app.py at startup so scenarios can modify the DataFrame."""
    global _APP_STATE
    _APP_STATE = st


# ---- catalog shown in Admin -> Scenarios ----
SCENARIOS = [
    {"id": "baseline",         "label": "Baseline (no perturbation)",
     "desc": "Original sensor data with no modifications."},
    {"id": "heatwave",         "label": "Climate-Induced Heatwave",
     "desc": "72 h heatwave: +20 C sigmoid onset (hours 24-48) with exponential tail; +10 % RH."},
    {"id": "overproduction",   "label": "Overproduction / Glut",
     "desc": "Inventory multiplied 2.5x during hours 12-60 with progressive +8°C cold storage excursion."},
    {"id": "cyber_outage",     "label": "Cyber Threat & Node Outage",
     "desc": "Yield drops to 15 % and inventory to 25 % from hour 24 onward."},
    {"id": "adaptive_pricing", "label": "Adaptive Pricing & Demand Oscillation",
     "desc": "Demand oscillation (amp 45, period 60) plus Gaussian noise (std 14)."},
]


class RunRequest(BaseModel):
    name: str
    intensity: float | int | None = 1.0


# ---------------------------------------------------------------------------
# Scenario application helpers
# ---------------------------------------------------------------------------

def _hours_from_start(df) -> np.ndarray:
    """Return array of hours elapsed since the first timestamp."""
    import pandas as pd
    ts = pd.to_datetime(df["timestamp"])
    return ((ts - ts.iloc[0]).dt.total_seconds() / 3600.0).to_numpy(dtype=np.float64)


def _recompute_derived(df):
    """Re-run PINN spoilage (Arrhenius model) + Bollinger volatility."""
    from src.models.spoilage import compute_spoilage, volatility_flags

    p = None
    if _APP_STATE:
        p = _APP_STATE.get("policy")

    k_ref = getattr(p, "k_ref", 0.0021) if p else 0.0021
    Ea_R = getattr(p, "Ea_R", 8000.0) if p else 8000.0
    T_ref_K = getattr(p, "T_ref_K", 277.15) if p else 277.15
    beta = getattr(p, "beta_humidity", 0.25) if p else 0.25
    lag_lambda = getattr(p, "lag_lambda", 12.0) if p else 12.0
    window = getattr(p, "boll_window", 20) if p else 20
    k_boll = getattr(p, "boll_k", 2.0) if p else 2.0

    df = compute_spoilage(df, k_ref=k_ref, Ea_R=Ea_R, T_ref_K=T_ref_K,
                          beta=beta, lag_lambda=lag_lambda)
    df["volatility"] = volatility_flags(df, window=window, k=k_boll)
    return df


def _apply_heatwave(df, intensity: float = 1.0):
    """Inject +20 C sigmoid onset hours 24-48, exponential tail after, +10 % RH.

    Uses sigmoid onset (reaches ~95% of peak by h ~= 30) rather than a
    linear ramp — heatwaves build rapidly once they onset (WMO, 2018).
    Matches the implementation in generate_results.py.
    """
    df = df.copy()
    hours = _hours_from_start(df)
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
    return _recompute_derived(df)


def _apply_overproduction(df, intensity: float = 1.0):
    """Multiply inventory by 2.5x during hours 12-60 with progressive cold storage excursion.

    Overloaded cold storage: at 2.5x capacity, reduced airflow and compressor
    strain raise cold-room temperature by up to +8°C (James & James, 2010).
    Sigmoid onset (~95% by h ~= 22), exponential recovery after hour 60.
    Matches the implementation in generate_results.py.
    """
    df = df.copy()
    df["inventory_units"] = df["inventory_units"].astype(float)
    df["tempC"] = df["tempC"].astype(float)
    hours = _hours_from_start(df)
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
    return _recompute_derived(df)


def _apply_cyber_outage(df, intensity: float = 1.0):
    """Processor offline from hour 24: demand drops to 15 %, inventory accumulates.

    IT-controlled cooling fails, causing a +10 °C sigmoid temperature
    excursion (reaches ~95 % of peak within ~5 h of onset).  Inventory
    stays at 100 % — produce keeps arriving from farms, creating an
    accumulation crisis while the processor is offline.
    """
    df = df.copy()
    df["demand_units"] = df["demand_units"].astype(float)
    df["tempC"] = df["tempC"].astype(float)
    hours = _hours_from_start(df)
    n = len(df)
    mask = hours >= 24.0
    df.loc[mask, "demand_units"] = df.loc[mask, "demand_units"] * 0.15 * intensity
    # Refrigeration degradation: IT-controlled cooling fails (+10 °C)
    temp_add = np.zeros(n)
    for i in range(n):
        h = hours[i]
        if h >= 24.0:
            onset = 1.0 - np.exp(-0.2 * (h - 24.0))
            temp_add[i] = 10.0 * onset * intensity
    df["tempC"] = df["tempC"] + temp_add
    return _recompute_derived(df)


def _apply_adaptive_pricing(df, intensity: float = 1.0):
    """Add demand oscillation (amplitude 45, period 60) + noise (std 14).

    Cold-storage stress from demand volatility: frequent dock openings,
    variable loading patterns, and supply-demand mismatch degrade
    temperature management (Mercier et al., 2017).
    """
    from src.models.waste import INV_BASELINE
    df = df.copy()
    df["demand_units"] = df["demand_units"].astype(float)
    df["tempC"] = df["tempC"].astype(float)
    df["inventory_units"] = df["inventory_units"].astype(float)
    n = len(df)
    rng = np.random.default_rng(42)
    oscillation = 45.0 * intensity * np.sin(2.0 * np.pi * np.arange(n) / 60.0)
    noise = rng.normal(0.0, 14.0 * intensity, size=n)
    df["demand_units"] = (df["demand_units"] + oscillation + noise).clip(0)
    # Temperature stress from demand volatility and inventory surplus
    demand = df["demand_units"].to_numpy()
    inv = df["inventory_units"].to_numpy()
    demand_dev = np.abs(demand - np.median(demand)) / (np.median(demand) + 1.0)
    surplus_signal = np.clip((inv / INV_BASELINE - 1.0), 0, 2.0)
    temp_add = 1.5 * intensity * np.clip(demand_dev, 0, 1) + 2.0 * intensity * surplus_signal
    df["tempC"] = df["tempC"] + temp_add
    return _recompute_derived(df)


_SCENARIO_FN = {
    "heatwave": _apply_heatwave,
    "overproduction": _apply_overproduction,
    "cyber_outage": _apply_cyber_outage,
    "adaptive_pricing": _apply_adaptive_pricing,
}


def _apply_to_state(name: str, intensity: float) -> bool:
    """Modify the app DataFrame in-place according to the named scenario."""
    if _APP_STATE is None:
        return False

    fn = _SCENARIO_FN.get(name)
    if fn is None:
        # baseline or unknown -> restore original
        orig = _APP_STATE.get("df_original")
        if orig is not None:
            _APP_STATE["df"] = orig.copy()
        return True

    # Always start from the pristine original
    orig = _APP_STATE.get("df_original")
    if orig is None:
        return False

    _APP_STATE["df"] = fn(orig.copy(), intensity)
    return True


# ---------- API used by the Admin panel ----------
@router.get("/list")
def list_scenarios():
    return {"scenarios": SCENARIOS, "active": ACTIVE if ACTIVE["name"] else None}


@router.post("/run")
def run_scenario(req: RunRequest):
    ACTIVE["name"] = req.name
    try:
        ACTIVE["intensity"] = float(req.intensity or 1.0)
    except (TypeError, ValueError):
        ACTIVE["intensity"] = 1.0

    ok = _apply_to_state(req.name, ACTIVE["intensity"])
    return {"ok": ok, "active": ACTIVE}


@router.post("/reset")
def reset_scenario():
    ACTIVE["name"] = None
    ACTIVE["intensity"] = 1.0
    _apply_to_state("baseline", 1.0)
    return {"ok": True, "active": None}


# ---------- LEGACY FALLBACK (old UI calling POST /scenarios) ----------
@router.post("", include_in_schema=False)
def legacy_apply(id: str | None = None, name: str | None = None):
    chosen = (name or id or "").strip()
    if not chosen:
        return {"ok": False, "error": "missing scenario id"}
    ACTIVE["name"] = chosen
    ACTIVE["intensity"] = 1.0
    _apply_to_state(chosen, 1.0)
    return {"ok": True, "active": ACTIVE}
