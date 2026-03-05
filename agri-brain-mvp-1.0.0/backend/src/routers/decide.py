# backend/src/routers/decide.py
"""
Decision router -- regime-aware contextual softmax policy.

Scientific logic (constants, feature-vector construction, softmax, carbon
accounting, reward computation) lives in the Layer 1 backend model files:

    src.models.action_selection  -- THETA, ACTIONS, feature vector, softmax
    src.models.carbon            -- transport carbon emissions
    src.models.reward            -- multi-objective reward function
    src.models.resilience        -- adaptive resilience index

This router is imported by compat.py for legacy endpoints.
The primary /decide endpoint lives in app.py; this module provides
a standalone fallback that also implements the softmax policy.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from time import time

import numpy as np

# --- Layer 1 model imports (scientific logic) ---
from src.models.action_selection import (
    ACTIONS,
    THETA,
    SLCA_BONUS,
    SLCA_RHO_BONUS,
    _softmax,
    build_feature_vector,
    INV_CAPACITY,
    BASELINE_DEMAND,
    THERMAL_T0,
    THERMAL_DELTA_MAX,
)
from src.models.carbon import compute_transport_carbon, REFRIG_COP_PENALTY
from src.models.reward import compute_reward, compute_reward_extended
from src.models.resilience import compute_ari
from src.models.spoilage import arrhenius_k
from src.models.waste import INV_BASELINE, compute_waste_rate, compute_save_factor

# --- Optional shared state for PDF and others
try:
    from src.routers.case import STATE as CASE_STATE
except Exception:
    CASE_STATE = {}

# --- Optional scenario state
try:
    from src.routers.scenarios import ACTIVE as SCENARIO_ACTIVE
except Exception:
    SCENARIO_ACTIVE = {"name": None, "intensity": 1.0}

# --- Optional chain config
try:
    from src.routers.governance import CHAIN as CHAIN_CFG
except Exception:
    CHAIN_CFG = {}

router = APIRouter()

# ---------------------------------------------------------------------------
# Router-local fallback constants (not duplicated from model modules)
# ---------------------------------------------------------------------------
GAMMA_DEFAULT = np.array([0.3, 0.05, -0.3])


# ---------- Request / Response models ----------
class DecideRequest(BaseModel):
    agent: str = "farm"
    role: str = ""
    step: int | None = None
    deterministic: bool = True
    mode: str = "agribrain"          # operating mode for waste save factor

    # Optional knobs used by the QuickDecision panel
    inventory_units: float | None = None
    demand_units: float | None = None
    temp_c: float | None = None
    volatility: float | None = None


# ---------- In-memory log ----------
DECISIONS: list[dict] = []
LAST: dict | None = None


def _persist_last(memo: dict) -> None:
    DECISIONS.append(memo)
    globals()["LAST"] = memo
    try:
        CASE_STATE["last_decision"] = memo
    except Exception:
        pass


def _compat_from_memo(m: dict) -> dict:
    """Map to the exact keys the PDF expects."""
    return {
        "time":       m.get("ts"),
        "agent":      m.get("agent") or "",
        "role":       m.get("role") or "",
        "decision":   m.get("action") or m.get("decision") or "",
        "shelf_left": m.get("shelf_left") or 0.0,
        "volatility": m.get("volatility") or 0.0,
        "km":         m.get("km") or 0.0,
        "carbon_kg":  m.get("carbon_kg") or m.get("carbon") or 0.0,
        "unit_price": m.get("unit_price") or 0.0,
        "slca":       m.get("slca_score") or m.get("slca") or 0.0,
        "tx":         m.get("tx_hash") or m.get("tx") or "",
        "note":       m.get("note") or "",
    }


# ---------- Core decision logic ----------
def _get_app_state():
    """Lazy import to access the main app state (avoids circular imports)."""
    try:
        from src.app import state
        return state
    except Exception:
        return None


def _decide_with_app(req: DecideRequest) -> dict | None:
    """Delegate to the main app.py /decide endpoint if available."""
    try:
        from src.app import decide as _app_decide, DecideIn
        result = _app_decide(DecideIn(
            agent_id=req.agent,
            role=req.role,
            step=req.step,
            deterministic=req.deterministic,
            mode=req.mode,
        ))
        return result
    except Exception:
        return None


def _decide_standalone(req: DecideRequest) -> dict:
    """Standalone softmax decision when app state is not available."""
    app_st = _get_app_state()
    df = app_st.get("df") if app_st else None
    policy = app_st.get("policy") if app_st else None

    if df is not None and len(df) > 0:
        idx = req.step if req.step is not None else len(df) - 1
        idx = max(0, min(idx, len(df) - 1))
        row = df.iloc[idx]
        rho = float(row.get("spoilage_risk", 1.0 - row.get("shelf_left", 0.5)))
        inv = float(row.get("inventory_units", 100.0))
        temp = float(row.get("tempC", 4.0))
        rh_val = float(row.get("RH", 50.0))
        tau = 1.0 if str(row.get("volatility", "normal")) == "anomaly" else 0.0
        shelf = float(row.get("shelf_left", 0.5))
        vol = str(row.get("volatility", "normal"))
    else:
        rho = 0.2
        inv = req.inventory_units or 100.0
        temp = req.temp_c or 4.0
        rh_val = 50.0
        tau = 1.0 if (req.volatility or 0) > 0.5 else 0.0
        shelf = 1.0 - rho
        vol = "anomaly" if tau else "normal"

    y_hat = 100.0
    phi = build_feature_vector(rho, inv, y_hat, temp)

    gamma = GAMMA_DEFAULT
    if policy:
        gamma = np.array([
            getattr(policy, "gamma_coldchain", 0.3),
            getattr(policy, "gamma_local", 0.05),
            getattr(policy, "gamma_recovery", -0.3),
        ])

    logits = THETA @ phi + gamma * tau + SLCA_BONUS + SLCA_RHO_BONUS * rho
    probs = _softmax(logits)

    if req.deterministic:
        action_idx = int(np.argmax(probs))
    else:
        action_idx = int(np.random.choice(len(ACTIONS), p=probs))
    action = ACTIONS[action_idx]

    # Transport emission model (GHG Protocol, WRI/WBCSD, 2004)
    km_map = {"cold_chain": 120.0, "local_redistribute": 45.0, "recovery": 80.0}
    carbon_per_km = 0.12
    if policy:
        km_map = {
            "cold_chain": getattr(policy, "km_coldchain", 120.0),
            "local_redistribute": getattr(policy, "km_local", 45.0),
            "recovery": getattr(policy, "km_recovery", 80.0),
        }
        carbon_per_km = getattr(policy, "carbon_per_km", 0.12)

    km = km_map[action]
    carbon = compute_transport_carbon(km, carbon_per_km)

    try:
        from src.models.slca import slca_score
        slca_result = slca_score(carbon, action)
    except Exception:
        slca_result = {"C": 0.7, "L": 0.5, "R": 0.4, "P": 0.45, "composite": 0.55,
                       "action_family": "coldchain"}

    slca_composite = slca_result["composite"]

    try:
        from src.models.footprint import compute_footprint
        fp = compute_footprint(steps=1)
    except Exception:
        fp = {"energy_J": 0.05, "water_L": 1.8e-6}

    # Full waste model (matching generate_results.py)
    p_k_ref = getattr(policy, "k_ref", 0.0021) if policy else 0.0021
    p_Ea_R = getattr(policy, "Ea_R", 8000.0) if policy else 8000.0
    p_T_ref_K = getattr(policy, "T_ref_K", 277.15) if policy else 277.15
    p_beta_h = getattr(policy, "beta_humidity", 0.25) if policy else 0.25
    k_inst = arrhenius_k(temp, p_k_ref, p_Ea_R, p_T_ref_K,
                         rh_val / 100.0, p_beta_h)
    surplus_ratio = max(0.0, inv / INV_BASELINE - 1.0)
    waste_raw = compute_waste_rate(k_inst, surplus_ratio)
    save = compute_save_factor(action, req.mode, surplus_ratio)
    waste = float(waste_raw * (1.0 - save))
    eta = getattr(policy, "eta", 0.5) if policy else 0.5
    alpha_E = getattr(policy, "alpha_E", 0.05) if policy else 0.05
    beta_W = getattr(policy, "beta_W", 0.03) if policy else 0.03
    msrp = getattr(policy, "msrp", 1.50) if policy else 1.50

    price_factor = {"cold_chain": 1.0, "local_redistribute": 0.95, "recovery": 0.88}
    price = msrp * price_factor.get(action, 1.0)

    # Multi-objective reward via imported reward model
    reward_decomp = compute_reward_extended(
        slca_composite=slca_composite,
        waste=waste,
        energy_J=fp["energy_J"],
        water_L=fp["water_L"],
        eta=eta,
        alpha_E=alpha_E,
        beta_W=beta_W,
    )

    memo = {
        "time": None,
        "ts": int(time()),
        "agent": req.agent,
        "role": req.role,
        "mode": req.mode,
        "decision": action,
        "action": action,
        "shelf_left": round(shelf, 4),
        "spoilage_risk": round(rho, 4),
        "volatility": vol,
        "km": round(km, 2),
        "carbon_kg": round(carbon, 4),
        "waste": round(waste, 4),
        "unit_price": round(price, 2),
        "slca": round(slca_composite, 4),
        "slca_score": round(slca_composite, 4),
        "action_probabilities": {
            "cold_chain": round(float(probs[0]), 4),
            "local_redistribute": round(float(probs[1]), 4),
            "recovery": round(float(probs[2]), 4),
        },
        "slca_components": {
            "carbon": slca_result["C"],
            "labor": slca_result["L"],
            "resilience": slca_result["R"],
            "transparency": slca_result["P"],
            "composite": slca_result["composite"],
        },
        "footprint": {
            "energy_J": fp["energy_J"],
            "water_L": fp["water_L"],
        },
        "regime": {
            "tau": tau,
            "bollinger_z": 0.0,
        },
        "reward_decomposition": reward_decomp,
        "tx_hash": "0x0",
        "note": (
            f"Softmax policy: action={action} "
            f"P={probs[action_idx]:.3f} rho={rho:.3f} tau={tau}"
        ),
    }

    _persist_last(memo)

    # Optional: log to blockchain
    try:
        from src.chain.eth import log_decision_onchain
        txh = log_decision_onchain(memo, CHAIN_CFG or {})
        if txh:
            memo["tx_hash"] = txh
    except Exception:
        pass

    return {"ok": True, "memo": memo}


# ---------- Routes ----------
@router.post("/decide")
def decide(req: DecideRequest):
    # Try delegating to app.py first (canonical implementation)
    result = _decide_with_app(req)
    if result is not None:
        return result
    # Fallback to standalone
    return _decide_standalone(req)


# GET alias for quick testing in a browser
@router.get("/decide")
def decide_get(agent: str = "farm", role: str = ""):
    return decide(DecideRequest(agent=agent, role=role))


# Legacy/compat endpoint some frontends call
@router.api_route("/decision/take", methods=["GET", "POST"])
def decision_take(agent: str = "farm", role: str = ""):
    return decide(DecideRequest(agent=agent, role=role))


# Public read used by the PDF (compatibility dict)
@router.get("/last-decision", response_model=dict)
def last_decision():
    try:
        data = CASE_STATE.get("last_decision")
        if data:
            return _compat_from_memo(data) if not isinstance(data, dict) else data
    except Exception:
        pass
    if LAST:
        return _compat_from_memo(LAST)
    return {}


# Optional: recent memos feed
@router.get("/decisions", response_model=list[dict])
def decisions_feed():
    return list(reversed(DECISIONS[-50:]))
