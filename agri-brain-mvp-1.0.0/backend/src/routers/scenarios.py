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
     "desc": "72 h heatwave: +20 C ramp (hours 24-48) with exponential tail; +10 % RH."},
    {"id": "overproduction",   "label": "Overproduction / Glut",
     "desc": "Inventory multiplied 2.5x during hours 12-60; triggers redistribution."},
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
    """Re-run PINN spoilage + Bollinger volatility on the (modified) DataFrame."""
    from src.models.spoilage import compute_spoilage, volatility_flags

    p = None
    if _APP_STATE:
        p = _APP_STATE.get("policy")

    k0 = getattr(p, "k0", 0.04) if p else 0.04
    alpha = getattr(p, "alpha_decay", 0.12) if p else 0.12
    T0 = getattr(p, "T0", 4.0) if p else 4.0
    beta = getattr(p, "beta_humidity", 0.25) if p else 0.25
    window = getattr(p, "boll_window", 20) if p else 20
    k_boll = getattr(p, "boll_k", 2.0) if p else 2.0

    df = compute_spoilage(df, k0=k0, alpha=alpha, T0=T0, beta=beta)
    df["volatility"] = volatility_flags(df, window=window, k=k_boll)
    return df


def _apply_heatwave(df, intensity: float = 1.0):
    """Inject +20 C ramp hours 24-48, exponential tail after, +10 % RH."""
    df = df.copy()
    hours = _hours_from_start(df)
    n = len(df)
    temp_add = np.zeros(n)
    rh_add = np.zeros(n)

    for i in range(n):
        h = hours[i]
        if 24.0 <= h <= 48.0:
            # linear ramp from 0 to +20 over 24 hours
            frac = (h - 24.0) / 24.0
            temp_add[i] = 20.0 * frac * intensity
            rh_add[i] = 10.0 * intensity
        elif h > 48.0:
            # exponential tail  decay constant ~0.1 h^-1
            temp_add[i] = 20.0 * intensity * np.exp(-0.1 * (h - 48.0))
            rh_add[i] = 10.0 * intensity * np.exp(-0.1 * (h - 48.0))

    df["tempC"] = df["tempC"].astype(float) + temp_add
    df["RH"] = (df["RH"].astype(float) + rh_add).clip(0, 100)
    return _recompute_derived(df)


def _apply_overproduction(df, intensity: float = 1.0):
    """Multiply inventory by 2.5x during hours 12-60."""
    df = df.copy()
    hours = _hours_from_start(df)
    factor = 1.0 + 1.5 * intensity    # intensity=1 -> 2.5x
    mask = (hours >= 12.0) & (hours <= 60.0)
    df.loc[mask, "inventory_units"] = (
        df.loc[mask, "inventory_units"].astype(float) * factor
    )
    return _recompute_derived(df)


def _apply_cyber_outage(df, intensity: float = 1.0):
    """Set demand to 15 % and inventory to 25 % from hour 24 onward."""
    df = df.copy()
    hours = _hours_from_start(df)
    mask = hours >= 24.0
    demand_frac = 0.15 * intensity
    inv_frac = 0.25 * intensity
    df.loc[mask, "demand_units"] = (
        df.loc[mask, "demand_units"].astype(float) * demand_frac
    )
    df.loc[mask, "inventory_units"] = (
        df.loc[mask, "inventory_units"].astype(float) * inv_frac
    )
    return _recompute_derived(df)


def _apply_adaptive_pricing(df, intensity: float = 1.0):
    """Add demand oscillation (amplitude 45, period 60) + noise (std 14)."""
    df = df.copy()
    n = len(df)
    rng = np.random.default_rng(42)
    oscillation = 45.0 * intensity * np.sin(2.0 * np.pi * np.arange(n) / 60.0)
    noise = rng.normal(0.0, 14.0 * intensity, size=n)
    df["demand_units"] = (df["demand_units"].astype(float) + oscillation + noise).clip(0)
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
    except Exception:
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
