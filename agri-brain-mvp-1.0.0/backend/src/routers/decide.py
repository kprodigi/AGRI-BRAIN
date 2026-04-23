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

import logging
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel
from time import time

import numpy as np

# --- Layer 1 model imports (scientific logic) ---
from src.models.action_selection import (
    ACTIONS,
    PRICE_FACTOR,
    select_action,
    compute_thermal_stress,
    compute_slca_attenuation,
)
from src.models.carbon import compute_transport_carbon
from src.models.reward import compute_reward_extended
from src.models.spoilage import arrhenius_k
from src.models.waste import INV_BASELINE, compute_waste_rate, compute_save_factor
from src.models.policy import Policy

logger = logging.getLogger(__name__)

# --- Optional shared state for PDF and others
try:
    from src.routers.case import STATE as CASE_STATE
except ImportError:
    CASE_STATE = {}

# --- Optional scenario state
try:
    from src.routers.scenarios import ACTIVE as SCENARIO_ACTIVE
except ImportError:
    SCENARIO_ACTIVE = {"name": None, "intensity": 1.0}

# --- Optional chain config
try:
    from src.routers.governance import CHAIN as CHAIN_CFG
except ImportError:
    CHAIN_CFG = {}

router = APIRouter()

# ---------- Request / Response models ----------
class DecideRequest(BaseModel):
    agent: str = "farm"
    role: str = ""
    step: int | None = None
    deterministic: bool = True
    mode: Literal["static", "hybrid_rl", "no_pinn", "no_slca", "agribrain", "no_context", "mcp_only", "pirag_only"] = "agribrain"

    # Optional knobs used by the QuickDecision panel
    inventory_units: float | None = None
    demand_units: float | None = None
    temp_c: float | None = None
    volatility: float | None = None

    # Optional forecast payload. When provided, these flow into phi_6..phi_8
    # so this fallback endpoint produces the same state vector the simulator
    # and primary /decide handler use. When omitted, phi_6..phi_8 default to
    # zero (legacy behavior).
    y_hat: float | None = None
    supply_hat: float | None = None
    supply_std: float | None = None
    demand_std: float | None = None


# ---------- In-memory log ----------
DECISIONS: list[dict] = []
LAST: dict | None = None


def _persist_last(memo: dict) -> None:
    DECISIONS.append(memo)
    globals()["LAST"] = memo
    try:
        CASE_STATE["last_decision"] = memo
    except (TypeError, KeyError):
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
    except ImportError:
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
    except (ImportError, ValueError, KeyError) as e:
        logger.debug("app delegation failed: %s", e)
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

    y_hat = float(req.y_hat) if req.y_hat is not None else 100.0
    _defaults = Policy()
    _policy = policy if policy else _defaults

    # Use canonical select_action() for mode-aware softmax policy
    rng = np.random.default_rng()
    action_idx, probs = select_action(
        mode=req.mode, rho=rho, inv=inv, y_hat=y_hat, temp=temp,
        tau=tau, policy=_policy, rng=rng,
        deterministic=req.deterministic,
        supply_hat=req.supply_hat,
        supply_std=req.supply_std,
        demand_std=req.demand_std,
    )
    action = ACTIONS[action_idx]

    # Transport emission model (GHG Protocol, WRI/WBCSD, 2004)
    km_map = {
        "cold_chain": getattr(policy, "km_coldchain", _defaults.km_coldchain) if policy else _defaults.km_coldchain,
        "local_redistribute": getattr(policy, "km_local", _defaults.km_local) if policy else _defaults.km_local,
        "recovery": getattr(policy, "km_recovery", _defaults.km_recovery) if policy else _defaults.km_recovery,
    }
    carbon_per_km = getattr(policy, "carbon_per_km", _defaults.carbon_per_km) if policy else _defaults.carbon_per_km

    km = km_map[action]
    thermal_stress = compute_thermal_stress(temp)
    carbon = compute_transport_carbon(km, carbon_per_km, thermal_stress)

    try:
        from src.models.slca import slca_score
        slca_result = slca_score(carbon, action)
    except ImportError:
        slca_result = {"C": 0.7, "L": 0.5, "R": 0.4, "P": 0.45, "composite": 0.55,
                       "action_family": "cold_chain"}

    # Apply SLCA stress attenuation (Eq. 12) matching generate_results.py
    surplus_ratio = max(0.0, inv / INV_BASELINE - 1.0)
    slca_quality = compute_slca_attenuation(thermal_stress, surplus_ratio)
    slca_composite = slca_result["composite"] * slca_quality

    try:
        from src.models.footprint import compute_footprint
        fp = compute_footprint(steps=1)
    except ImportError:
        fp = {"energy_J": 0.05, "water_L": 1.8e-6}

    # Full waste model (matching generate_results.py)
    p_k_ref = getattr(policy, "k_ref", _defaults.k_ref) if policy else _defaults.k_ref
    p_Ea_R = getattr(policy, "Ea_R", _defaults.Ea_R) if policy else _defaults.Ea_R
    p_T_ref_K = getattr(policy, "T_ref_K", _defaults.T_ref_K) if policy else _defaults.T_ref_K
    p_beta_h = getattr(policy, "beta_humidity", _defaults.beta_humidity) if policy else _defaults.beta_humidity
    k_inst = arrhenius_k(temp, p_k_ref, p_Ea_R, p_T_ref_K,
                         rh_val / 100.0, p_beta_h)
    waste_raw = compute_waste_rate(k_inst, surplus_ratio)
    save = compute_save_factor(action, req.mode, surplus_ratio)
    waste = float(waste_raw * (1.0 - save))
    eta = getattr(policy, "eta", _defaults.eta) if policy else _defaults.eta
    alpha_E = getattr(policy, "alpha_E", _defaults.alpha_E) if policy else _defaults.alpha_E
    beta_W = getattr(policy, "beta_W", _defaults.beta_W) if policy else _defaults.beta_W
    msrp = getattr(policy, "msrp", _defaults.msrp) if policy else _defaults.msrp

    price = msrp * PRICE_FACTOR.get(action, 1.0)

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
            "composite": round(slca_composite, 4),
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

    # --- Explainability enrichment (best-effort) ---
    try:
        from pirag.context_provider import get_policy_context
        from pirag.context_to_logits import extract_context_features, THETA_CONTEXT
        from pirag.keyword_extractor import extract_keywords_by_type
        from pirag.explain_decision import explain_decision

        _rc = get_policy_context(
            scenario="baseline", spoilage_risk=rho, temperature=temp,
            role=req.role or "farm", humidity=rh_val,
            inventory=inv, surplus_ratio=surplus_ratio, tau=tau,
        )
        _mcp_res = _rc.get("mcp_results", {})

        class _Obs:
            pass
        _obs = _Obs()
        _obs.rho = rho; _obs.temp = temp; _obs.rh = rh_val; _obs.inv = inv
        _obs.tau = tau; _obs.hour = 0.0; _obs.surplus_ratio = surplus_ratio
        _obs.y_hat = y_hat

        _psi = extract_context_features(_mcp_res, _rc, _obs)
        _modifier = THETA_CONTEXT @ _psi

        _, _cf_probs = select_action(
            mode=req.mode, rho=rho, inv=inv, y_hat=y_hat, temp=temp,
            tau=tau, policy=_policy, rng=np.random.default_rng(),
            deterministic=req.deterministic,
            supply_hat=req.supply_hat,
            supply_std=req.supply_std,
            demand_std=req.demand_std,
        )
        _cf_action = ACTIONS[int(np.argmax(_cf_probs))]

        _kw = {}
        for _gf, _kt in [("regulatory_guidance", "regulatory"), ("relevant_sops", "sop"),
                          ("waste_hierarchy_guidance", "waste_hierarchy")]:
            _txt = _rc.get(_gf, "")
            if _txt:
                _kw[_kt] = extract_keywords_by_type(_txt)

        _expl = explain_decision(
            action=action, role=req.role or "farm", hour=0.0, obs=_obs,
            mcp_results=_mcp_res, rag_context=_rc,
            slca_score=slca_composite, carbon_kg=carbon, waste=waste,
            context_features=_psi, logit_adjustment=_modifier,
            action_probs=probs, counterfactual_action=_cf_action,
            counterfactual_probs=_cf_probs, keywords=_kw,
        )

        memo["explainability"] = {
            "context_features": {
                "compliance_severity": round(float(_psi[0]), 3),
                "forecast_urgency": round(float(_psi[1]), 3),
                "retrieval_confidence": round(float(_psi[2]), 3),
                "regulatory_pressure": round(float(_psi[3]), 3),
                "recovery_saturation": round(float(_psi[4]), 3),
            },
            "logit_adjustment": {
                "cold_chain": round(float(_modifier[0]), 3),
                "local_redistribute": round(float(_modifier[1]), 3),
                "recovery": round(float(_modifier[2]), 3),
            },
            "mcp_tools_invoked": _mcp_res.get("_tools_invoked", []),
            "compliance": _mcp_res.get("check_compliance", {}),
            "forecast": _mcp_res.get("spoilage_forecast", {}),
            "pirag_top_doc": _rc.get("top_doc_id", ""),
            "pirag_top_score": round(_rc.get("top_citation_score", 0), 3),
            "keywords": _kw,
            "provenance": {
                "evidence_hashes": _rc.get("evidence_hashes", [])[:5],
                "guards_passed": _rc.get("guards_passed", True),
                "merkle_root": _expl.get("merkle_root", ""),
            },
            "causal_text": _expl.get("full_explanation", ""),
            "causal_chain": _expl.get("causal_chain", {}),
            "counterfactual": _expl.get("counterfactual", {}),
            "summary": _expl.get("summary", ""),
        }
    except (ImportError, KeyError, AttributeError, TypeError, ValueError) as _e:
        logger.warning("explainability enrichment skipped: %s", _e)
        memo["explainability"] = None

    _persist_last(memo)

    # Optional: log to blockchain
    try:
        from src.chain.eth import log_decision_onchain
        txh = log_decision_onchain(memo, CHAIN_CFG or {})
        if txh:
            memo["tx_hash"] = txh
    except (ImportError, ConnectionError, TimeoutError, ValueError) as e:
        logger.debug("on-chain log skipped: %s", e)

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


# Note: GET /decide and /decision/take aliases live in compat.py.
# This router is not mounted directly; only decide() is imported.
# Data endpoints (/last-decision, /decisions) live in app.py.
