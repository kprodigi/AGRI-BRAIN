# backend/src/app.py
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import uuid

import inspect
import logging
import os
import time as _time
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.responses import RedirectResponse, FileResponse, HTMLResponse
from pydantic import BaseModel, field_validator

# PiRAG / MCP routers
from pirag.api.routes.rag import router as rag_router
from pirag.mcp.server import router as mcp_router

# Your routers
from src.routers import case as _case
from src.routers import audit as _audit
from src.routers import governance as _gov
from src.routers import scenarios as _scn
from src.routers import compat as _compat
from src.routers import debug as _debug
from src.routers import stream as _stream
from src.routers import results as _results
from src.routers import phase as _phase
from src.agents.runtime import start_agent_runtime
from src.settings import SETTINGS

# Your models/utilities
from .models.spoilage import compute_spoilage, volatility_flags, arrhenius_k
from .models.forecast import yield_demand_forecast
from .models.lstm_demand import lstm_demand_forecast
from .models.yield_forecast import yield_supply_forecast
from .models.slca import slca_score
from .models.policy import Policy
from .models.governance_models import ChainConfig
from .models.footprint import compute_footprint, footprint_meter
from .models.resilience import RLE_THRESHOLD
from .models.action_selection import (
    ACTIONS, ACTION_KM_KEYS, PRICE_FACTOR,
    select_action, compute_thermal_stress, compute_slca_attenuation,
)
from .models.waste import (
    INV_BASELINE, MODE_CARBON_EFF,
    compute_waste_rate, compute_save_factor,
)
from .models.carbon import compute_transport_carbon
from .models.reverse_logistics import evaluate_recovery_options, compute_circular_economy_score
from .models.policy_learner import PolicyLearner
from .chain.eth import log_decision_onchain

logger = logging.getLogger(__name__)

# Forecast method selection (default: LSTM, fallback: Holt's linear level+trend)
FORECAST_METHOD = SETTINGS.forecast_method


def _demand_forecast(df, horizon=1, **kwargs):
    """Demand forecast routed through the MCP demand_query tool.

    Both simulator and REST share this single forecasting code path so
    the paper benchmark and live inference stay aligned. Falls back to
    the inline forecaster on import error so the endpoint keeps working
    if the MCP layer is unavailable.
    """
    try:
        from pirag.mcp.tools.demand_query import query_demand
        series = df["demand_units"].astype(float).tolist()
        return query_demand(demand_history=series, horizon=horizon, method=FORECAST_METHOD)
    except Exception:
        if FORECAST_METHOD == "holt_winters":
            return yield_demand_forecast(df, horizon=horizon, **kwargs)
        return lstm_demand_forecast(df, horizon=horizon, **kwargs)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Lifespan context manager replacing deprecated @on_event('startup')."""
    # --- startup ---
    try:
        _scn.register_app_state(state)
    except (ImportError, AttributeError, TypeError):
        pass
    try:
        _gov.register_app_state(state)
    except (ImportError, AttributeError, TypeError):
        pass
    try:
        case_load()
        logger.info("startup spinach_csv_loaded=true")
    except (FileNotFoundError, ValueError, KeyError) as e:
        logger.warning("startup case_load_skipped error=%s", e)
    try:
        sig = inspect.signature(start_agent_runtime)
        if len(sig.parameters) == 0:
            await start_agent_runtime()
        else:
            await start_agent_runtime(app)
    except (ImportError, RuntimeError, OSError) as e:
        logger.warning("startup agent_runtime_skipped error=%s", e)

    yield
    # --- shutdown (none needed) ---


API = FastAPI(title="AGRI BRAIN MVP API", lifespan=_lifespan)


# CORS (register once)
API.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.cors_origins,
    allow_credentials=False if SETTINGS.cors_origins == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Always-exempt paths (health probe, root redirect, favicon, static assets).
# The Swagger / ReDoc / OpenAPI surface is exempted only when
# SETTINGS.protect_docs is False -- production deployments keep the docs
# behind the API key so an unauthenticated GET /openapi.json cannot
# enumerate the full route schema for reconnaissance.
_ALWAYS_EXEMPT_PATHS = {"/health", "/favicon.ico", "/"}
_DOCS_PATHS = {"/docs", "/redoc", "/openapi.json"}


@API.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Enforce API key on all routes when REQUIRE_API_KEY=true."""
    path = request.url.path.rstrip("/") or "/"
    is_static = path.startswith("/static")
    is_docs = path in _DOCS_PATHS
    is_exempt = path in _ALWAYS_EXEMPT_PATHS or is_static
    # Docs are exempt only when explicitly opted out via PROTECT_DOCS=false.
    if is_docs and not SETTINGS.protect_docs:
        is_exempt = True

    if not is_exempt:
        from src.security import enforce_api_key
        try:
            enforce_api_key(request, request.headers.get("x-api-key"))
        except HTTPException as exc:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=exc.status_code,
                                content={"detail": exc.detail})
    return await call_next(request)

@API.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = _time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (_time.perf_counter() - start) * 1000.0
    response.headers["x-request-id"] = req_id
    logger.info(
        "http request_id=%s method=%s path=%s status=%s latency_ms=%.2f",
        req_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response

# ---------------------------------------------------------------------------
# Mount routers (each once)
#
# Scoped API-key dependencies (governance, chain, phase, mcp) live as
# router-level Depends so a single leaked credential cannot mutate
# every privileged surface. The global REQUIRE_API_KEY middleware
# authenticates *any* valid key; the scoped check then narrows that to
# the keys configured for the specific scope. Both layers are no-ops
# when REQUIRE_API_KEY=false (the dev default).
# ---------------------------------------------------------------------------
from fastapi import Depends as _Depends  # noqa: E402  (after other top-level imports)
from src.security import require_scope_api_key  # noqa: E402

API.include_router(_case.router,        prefix="/case",       tags=["case"])
API.include_router(_audit.router,       prefix="/audit",      tags=["audit"])
API.include_router(
    _gov.router, prefix="/governance", tags=["governance"],
    dependencies=[_Depends(require_scope_api_key("governance"))],
)
API.include_router(_scn.router,         prefix="/scenarios",  tags=["scenarios"])
API.include_router(_compat.router,                          tags=["compat"])
API.include_router(_debug.router,                           tags=["debug"])
API.include_router(_stream.router)  # no prefix => /stream (websocket)

API.include_router(_results.router,    prefix="/results",    tags=["results"])
API.include_router(
    _phase.router, prefix="/phase", tags=["phase"],
    dependencies=[_Depends(require_scope_api_key("phase"))],
)

API.include_router(rag_router, prefix="/rag", tags=["pirag"])
API.include_router(
    mcp_router, prefix="/mcp", tags=["mcp"],
    dependencies=[_Depends(require_scope_api_key("mcp"))],
)

# ---------------------------------------------------------------------------
# Static + Swagger branding (logo, favicon, CSS)
# Resolves: .../backend/src/app.py  -> static at .../backend/static
# ---------------------------------------------------------------------------
_STATIC_DIR = (Path(__file__).resolve().parent.parent / "static").resolve()
API.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

@API.get("/docs", include_in_schema=False)
def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=API.openapi_url,
        title="AGRI-BRAIN Admin",
        swagger_favicon_url="/static/branding/favicon.png",     # PNG favicon
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui-bundle.js",
        swagger_css_url="/static/branding/custom.v2.css",       # cache-busting v2 file
    )

@API.get("/redoc", include_in_schema=False)
def redoc_html():
    return get_redoc_html(
        openapi_url=API.openapi_url,
        title="AGRI-BRAIN Admin — ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc/bundles/redoc.standalone.js",
        with_google_fonts=False,
    )

# Serve favicon so browsers stop 404-ing /favicon.ico
@API.get("/favicon.ico", include_in_schema=False)
def favicon():
    png = _STATIC_DIR / "branding" / "favicon.png"
    if png.exists():
        return FileResponse(str(png), media_type="image/png")
    return HTMLResponse(status_code=404, content="")

# Make / redirect to /docs (so navigating to the root shows your branded docs)
@API.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# ---------------------------------------------------------------------------
# In-memory state + config
# ---------------------------------------------------------------------------
DATA = Path(SETTINGS.data_csv) if SETTINGS.data_csv else Path(__file__).parent / "data_spinach.csv"
# Bootstrap the chain signing key from env so on-chain calls work even
# when /chain/config has never been POSTed. The 2026-04 hardening
# removed the body-based ingestion path (private_key is no longer a
# field on ChainConfig); this env-only path stays as the sole
# configuration channel for the signing key.
_BOOTSTRAP_CHAIN_PRIVKEY = os.environ.get("CHAIN_PRIVKEY", "") or None
_BOOTSTRAP_CHAIN_RPC = os.environ.get("CHAIN_RPC", "") or None
state: Dict[str, Any] = {
    "df": None,
    "df_original": None,        # pristine copy for scenario reset
    "policy": Policy(),
    "chain": {
        "rpc": _BOOTSTRAP_CHAIN_RPC,
        "addresses": {},
        "chain_id": 31337,
        "private_key": _BOOTSTRAP_CHAIN_PRIVKEY,
    },
}


# ---------------------------------------------------------------------------
# PINN overlay cache for the /predictions endpoint.
# Keyed on (df_signature, policy_signature) so a stable telemetry +
# policy combination only trains the PINN once per backend lifetime.
# Cleared explicitly on case_load() because the dataframe shape and
# content fully define df_signature anyway.
# ---------------------------------------------------------------------------
_PINN_OVERLAY_CACHE: Dict[Any, Any] = {}


def _df_signature(df) -> str:
    """Stable, cheap signature of the loaded telemetry dataframe.

    Uses pandas's internal index hash plus the (rows, cols) shape and
    the first/last timestamp + temperature so two distinct CSV loads
    are guaranteed to produce different signatures even when row count
    matches. Avoids hashing the full dataframe (slow on every poll).

    Best-effort: a malformed dataframe (missing columns, empty index,
    non-numeric tempC) degrades to a no-cache signature, which is the
    correct fallback. Narrow the catch to the realistic shapes we
    encounter; an unexpected exception type now propagates.
    """
    try:
        first_ts = str(df["timestamp"].iloc[0])
        last_ts = str(df["timestamp"].iloc[-1])
        first_t = float(df["tempC"].iloc[0])
        last_t = float(df["tempC"].iloc[-1])
        return f"{df.shape[0]}x{df.shape[1]}|{first_ts}|{last_ts}|{first_t:.4f}|{last_t:.4f}"
    except (KeyError, AttributeError, IndexError, ValueError, TypeError) as exc:
        logger.debug("PINN cache signature fell back to nosig: %s", exc)
        return f"{getattr(df, 'shape', '?')}|nosig"


def _policy_signature(p) -> str:
    """Signature of the policy parameters that feed the PINN/ODE model.

    Only the kinetic parameters matter (k_ref, Ea_R, T_ref_K,
    beta_humidity, lag_lambda) — changing carbon factors or SLCA
    weights does not change the spoilage trajectory, so they are
    excluded from the cache key.

    Best-effort: a missing attribute on the policy object falls back
    to a no-cache signature (correct fallback). Narrow the catch.
    """
    try:
        return (
            f"k={p.k_ref:.6g}|Ea_R={p.Ea_R:.6g}|Tref={p.T_ref_K:.6g}|"
            f"beta={p.beta_humidity:.6g}|lam={p.lag_lambda:.6g}"
        )
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("policy signature fell back to nosig: %s", exc)
        return "policy:nosig"

# ---------------------------------------------------------------------------
# Role-specific profiles for the live REST decision endpoint.
#
# logit_bias is sourced from agents.roles.ROLE_BIASES so the live
# endpoint and the simulator share a single source of truth — the
# 2026-04 cleanup caught a stale parallel set of biases here that no
# longer matched the simulator's. The remaining keys (km_overrides,
# slca_weights) are REST-side configuration: km_overrides drive the
# carbon and SLCA computations against the cooperative's own route
# distances (the simulator uses Policy.km_* defaults instead) and
# slca_weights tilt the SLCA composite for stakeholders who opted out
# of equal weighting in the dashboard. Both are intentionally
# different from simulator defaults and kept here, not in roles.py,
# because they are a property of the live operations cooperative,
# not of the modelled agent personality.
# ---------------------------------------------------------------------------
from .agents.roles import ROLE_BIASES as _ROLE_BIASES

ROLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "farm": {
        "logit_bias": _ROLE_BIASES["farm"].copy(),
        "km_overrides": {"km_coldchain": 80.0, "km_local": 25.0, "km_recovery": 40.0},
        "slca_weights": {"w_c": 0.25, "w_l": 0.30, "w_r": 0.25, "w_p": 0.20},
    },
    "processor": {
        "logit_bias": _ROLE_BIASES["processor"].copy(),
        "km_overrides": {"km_coldchain": 110.0, "km_local": 50.0, "km_recovery": 60.0},
        "slca_weights": {"w_c": 0.30, "w_l": 0.25, "w_r": 0.20, "w_p": 0.25},
    },
    "distributor": {
        "logit_bias": _ROLE_BIASES["distributor"].copy(),
        "km_overrides": {"km_coldchain": 180.0, "km_local": 65.0, "km_recovery": 100.0},
        "slca_weights": {"w_c": 0.35, "w_l": 0.15, "w_r": 0.30, "w_p": 0.20},
    },
    "recovery": {
        "logit_bias": _ROLE_BIASES["recovery"].copy(),
        "km_overrides": {"km_coldchain": 130.0, "km_local": 40.0, "km_recovery": 50.0},
        "slca_weights": {"w_c": 0.25, "w_l": 0.15, "w_r": 0.25, "w_p": 0.35},
    },
    "cooperative": {
        "logit_bias": _ROLE_BIASES["cooperative"].copy(),
        "km_overrides": {"km_coldchain": 100.0, "km_local": 35.0, "km_recovery": 55.0},
        "slca_weights": {"w_c": 0.25, "w_l": 0.25, "w_r": 0.25, "w_p": 0.25},
    },
}

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@API.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}

# (startup logic moved to _lifespan context manager above)

# ---------------------------------------------------------------------------
# Data endpoints
# ---------------------------------------------------------------------------
def case_load():
    """Load data_spinach.csv, compute PINN spoilage and volatility flags."""
    p = state["policy"]
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = compute_spoilage(
        df,
        k_ref=p.k_ref,
        Ea_R=p.Ea_R,
        T_ref_K=p.T_ref_K,
        beta=p.beta_humidity,
        lag_lambda=p.lag_lambda,
    )
    df["volatility"] = volatility_flags(df, window=p.boll_window, k=p.boll_k)
    state["df"] = df
    state["df_original"] = df.copy()
    # Drop the PINN overlay cache so the next /predictions call recomputes
    # against the freshly loaded telemetry.
    _PINN_OVERLAY_CACHE.clear()
    return {"ok": True, "records": len(df)}

@API.get("/kpis")
def kpis():
    if state["df"] is None:
        case_load()
    df = state["df"]
    p = state["policy"]
    # Compute realistic waste rates using the Arrhenius-based waste model
    # across all timesteps for static (baseline) and agribrain modes.
    _waste_baseline_vals = []
    _waste_agri_vals = []
    for _, row in df.iterrows():
        _t = float(row["tempC"])
        _rh = float(row.get("RH", 50.0)) / 100.0
        _inv = float(row.get("inventory_units", INV_BASELINE))
        _k = arrhenius_k(_t, p.k_ref, p.Ea_R, p.T_ref_K, _rh, p.beta_humidity)
        _surplus = max(0.0, _inv / INV_BASELINE - 1.0)
        _waste_baseline_vals.append(float(compute_waste_rate(_k, _surplus)))
        _save = compute_save_factor("local_redistribute", "agribrain", _surplus)
        _waste_agri_vals.append(float(compute_waste_rate(_k, _surplus) * (1.0 - _save)))
    waste_baseline = float(np.mean(_waste_baseline_vals))
    waste_agri = float(np.mean(_waste_agri_vals))

    # --- Extended KPIs from decision logs ---------------------------------
    logs = state.get("log", [])
    ari_vals: list[float] = []
    rle_at_risk = 0
    rle_routed = 0
    slca_vals: list[float] = []
    total_carbon = 0.0

    for m in logs:
        sc = m.get("slca_components") or {}
        slca_c = m.get("slca", sc.get("composite", 0.0))
        slca_vals.append(slca_c)

        rho = m.get("spoilage_risk", 1.0 - m.get("shelf_left", 1.0))
        waste = m.get("waste", rho)
        ari_vals.append((1.0 - waste) * slca_c * (1.0 - rho))

        total_carbon += m.get("carbon_kg", 0.0)

        # RLE: at-risk = rho > RLE_THRESHOLD (from resilience.py)
        if rho > RLE_THRESHOLD:
            rle_at_risk += 1
            act = m.get("action", "")
            if act in ("local_redistribute", "recovery"):
                rle_routed += 1

    fp_summary = footprint_meter.summary()

    return {
        "records": len(df),
        "avg_tempC": float(df["tempC"].mean()),
        "anomaly_points": int((df["volatility"] == "anomaly").sum()),
        "waste_rate_baseline": waste_baseline,
        "waste_rate_agri": waste_agri,
        # --- new KPIs ---
        "mean_ari": round(float(np.mean(ari_vals)), 4) if ari_vals else 0.0,
        "mean_rle": round(rle_routed / max(rle_at_risk, 1), 4),
        "mean_slca_composite": round(float(np.mean(slca_vals)), 4) if slca_vals else 0.0,
        "total_carbon_kg": round(total_carbon, 4),
        "total_energy_J": fp_summary["cumulative_energy_J"],
        "total_water_L": fp_summary["cumulative_water_L"],
    }

@API.get("/telemetry")
def telemetry():
    if state["df"] is None:
        case_load()
    df = state["df"]
    return {
        "timestamp": df["timestamp"].astype(str).tolist(),
        "tempC": df["tempC"].tolist(),
        "RH": df["RH"].tolist(),
        "ambientC": df["ambientC"].tolist(),
        "shockG": df["shockG"].tolist(),
        "inventory_units": df["inventory_units"].tolist(),
        "demand_units": df["demand_units"].tolist(),
    }

@API.get("/predictions")
def predictions():
    if state["df"] is None:
        case_load()
    df = state["df"]
    demand_fc = _demand_forecast(df, horizon=24)
    supply_fc = yield_supply_forecast(df, horizon=24)

    # PINN vs ODE comparison (Section 4.13, Figure 13b).
    # `df["shelf_left"]` is the ODE-only Arrhenius-Baranyi integral.
    # The PINN-corrected trajectory is the same ODE plus a bounded
    # neural residual (compute_spoilage_pinn). We surface both so the
    # Quality panel can render the real overlay instead of synthesising
    # an ODE baseline client-side.
    p = state["policy"]
    shelf_ode = df["shelf_left"].round(4).tolist()
    risk_ode = df["spoilage_risk"].round(4).tolist() if "spoilage_risk" in df.columns else []

    # PINN overlay cache. compute_spoilage_pinn trains a small NN from
    # scratch on every call (~1-3 s for 200 epochs over 288 timesteps),
    # so the QualityPage's 5-second poll would race the training and
    # waste compute. The cache is keyed on (df_hash, policy_hash) so any
    # change to the loaded telemetry (case_load) or the policy
    # parameters that feed the kinetic model (k_ref, Ea_R, T_ref_K,
    # beta_humidity, lag_lambda) busts the entry; pure UI re-polls hit
    # the cache and return in microseconds.
    df_key = _df_signature(df)
    pol_key = _policy_signature(p)
    cache_key = (df_key, pol_key)
    cached = _PINN_OVERLAY_CACHE.get(cache_key)
    if cached is not None:
        shelf_pinn, risk_pinn, pinn_available = cached
    else:
        shelf_pinn = list(shelf_ode)
        risk_pinn = list(risk_ode)
        pinn_available = False
        try:
            from .models.spoilage import compute_spoilage_pinn
            df_pinn = compute_spoilage_pinn(
                df,
                k_ref=p.k_ref,
                Ea_R=p.Ea_R,
                T_ref_K=p.T_ref_K,
                beta=p.beta_humidity,
                lag_lambda=p.lag_lambda,
                pinn_epochs=200,  # cheaper than the simulator's 1000
            )
            shelf_pinn = df_pinn["shelf_left"].round(4).tolist()
            risk_pinn = df_pinn["spoilage_risk"].round(4).tolist()
            pinn_available = True
        except (ImportError, RuntimeError, ValueError, KeyError) as exc:
            # ImportError -> torch not installed; RuntimeError -> training
            # divergence or shape mismatch; ValueError/KeyError -> bad
            # dataframe shape. The /predictions response sets
            # pinn_available=false so the frontend already conveys this
            # to the user; we log at INFO with structured cause for
            # operators reading server logs.
            logger.info(
                "predictions.pinn_overlay_skipped exception_type=%s exception=%s",
                type(exc).__name__, exc,
            )
        _PINN_OVERLAY_CACHE[cache_key] = (shelf_pinn, risk_pinn, pinn_available)
        # Keep the cache bounded — at most a handful of (df, policy)
        # combinations per session in practice (case_load happens once,
        # policy edits in the Admin panel are rare).
        if len(_PINN_OVERLAY_CACHE) > 16:
            # Evict the oldest entry to keep behaviour deterministic.
            _PINN_OVERLAY_CACHE.pop(next(iter(_PINN_OVERLAY_CACHE)))

    return {
        "timestamp": df["timestamp"].astype(str).tolist(),
        # Headline series (PINN-corrected when available, ODE-only otherwise).
        # `shelf_left` and `spoilage_risk` keep their pre-existing semantics
        # for backward compatibility with the Decisions / Ops dashboards.
        "shelf_left": shelf_pinn,
        "spoilage_risk": risk_pinn,
        # Explicit PINN vs ODE channels for the Quality panel overlay.
        "shelf_left_pinn": shelf_pinn,
        "spoilage_risk_pinn": risk_pinn,
        "shelf_left_ode": shelf_ode,
        "spoilage_risk_ode": risk_ode,
        "pinn_available": pinn_available,
        "volatility": df["volatility"].tolist(),
        "demand_forecast": demand_fc,
        "yield_forecast": supply_fc,
        # Legacy fields — mapped to supply (yield) forecast, not demand
        "yield_forecast_24h": supply_fc["forecast"],
        "yield_forecast_ci_lower": supply_fc["ci_lower"],
        "yield_forecast_ci_upper": supply_fc["ci_upper"],
        "yield_forecast_std": supply_fc["std"],
    }

# ---------------------------------------------------------------------------
# Policy (local) — canonical Policy object lives in state["policy"]
# ---------------------------------------------------------------------------
@API.get("/policy")
def get_policy():
    """Return the full Policy object (used internally and by /governance/policy)."""
    return state["policy"].model_dump()

@API.post("/policy")
def set_policy(p: Policy):
    """Update the canonical Policy object."""
    state["policy"] = p
    return {"ok": True, "policy": p.model_dump()}

# ---------------------------------------------------------------------------
# Memo text generation — detailed human-readable decision narrative
# ---------------------------------------------------------------------------
_ACTION_LABELS = {
    "cold_chain": "cold chain transport",
    "local_redistribute": "local redistribution",
    "recovery": "recovery diversion",
}

_ROLE_CONTEXT = {
    "farm": (
        "As a farm-level agent, this decision prioritizes minimizing post-harvest "
        "losses through proximity-based redistribution and fair labor practices. "
        "Local redistribution keeps produce within regional markets, supporting "
        "smallholder income and reducing food miles."
    ),
    "processor": (
        "As a processing facility agent, this decision prioritizes product quality "
        "preservation and cold chain integrity. Maintaining unbroken refrigeration "
        "ensures compliance with FDA leafy greens safety guidelines and maximizes "
        "shelf life for downstream retailers."
    ),
    "cooperative": (
        "As a cooperative agent, this decision balances equity across all supply "
        "chain stakeholders. The cooperative model weighs carbon, labor, resilience, "
        "and transparency equally, ensuring no single SLCA pillar is sacrificed for "
        "short-term efficiency gains."
    ),
    "distributor": (
        "As a distribution agent, this decision optimizes logistics efficiency and "
        "transport carbon cost. The distributor manages the longest transport legs "
        "in the supply chain and must balance delivery speed against emissions and "
        "thermal degradation risk during transit."
    ),
    "recovery": (
        "As a recovery agent, this decision prioritizes diverting at-risk produce "
        "from landfill into composting, animal feed, or food bank channels. Recovery "
        "pathways capture residual value from spoiled or surplus inventory while "
        "reducing the environmental burden of organic waste."
    ),
}

def _build_memo_text(
    action: str, role_key: str, mode: str,
    rho: float, inv: float, temp: float, y_hat: float,
    shelf: float, vol: str, tau: float, boll_z: float,
    probs: np.ndarray, action_idx: int,
    carbon: float, waste: float, km: float, price: float,
    slca_composite: float, slca_result: dict,
    circular_score: float, reward_total: float,
    energy_penalty: float, water_penalty: float, waste_penalty: float,
    recovery_opts: dict, rag_context: dict,
) -> str:
    """Build a detailed, multi-paragraph decision memo."""

    # --- risk level ---
    if rho < 0.3:
        risk_label, risk_desc = "low", "well within acceptable limits"
    elif rho < 0.6:
        risk_label, risk_desc = "moderate", "approaching the rerouting threshold"
    else:
        risk_label, risk_desc = "high", "exceeding safe thresholds and requiring immediate intervention"

    shelf_hours = max(shelf * 72, 0)  # shelf_left is fraction of 72h window
    regime = "anomalous (Bollinger band breach)" if tau > 0.5 else "normal"

    # --- paragraph 1: observation ---
    obs = (
        f"Current conditions indicate {risk_label} spoilage risk "
        f"(\u03c1 = {rho:.3f}), {risk_desc}. "
        f"Cold chain temperature is {temp:.1f} \u00b0C with "
        f"{shelf_hours:.1f} hours of estimated shelf life remaining. "
        f"Inventory stands at {inv:.0f} units against a forecasted demand "
        f"of {y_hat:.1f} units (LSTM). "
        f"The demand regime is {regime} (Bollinger z = {boll_z:+.2f})."
    )

    # --- paragraph 2: decision rationale ---
    chosen = _ACTION_LABELS.get(action, action)
    alt_actions = [(a, float(probs[i])) for i, a in enumerate(["cold_chain", "local_redistribute", "recovery"])]
    alt_actions.sort(key=lambda x: x[1], reverse=True)
    prob_str = ", ".join(f"{_ACTION_LABELS.get(a, a)} (P = {p:.3f})" for a, p in alt_actions)

    rationale = (
        f"The {mode} policy selected {chosen} with probability "
        f"{float(probs[action_idx]):.3f}. "
        f"Full action distribution: {prob_str}. "
    )
    if tau > 0.5:
        rationale += (
            "The anomaly trigger (\u03c4 = 1) activated regime-aware rerouting, "
            "shifting probability mass toward redistribution and recovery channels. "
        )
    if vol == "anomaly":
        rationale += "Volatility flags indicate abnormal sensor readings in this window. "

    # --- paragraph 3: impact assessment ---
    sc = slca_result
    impact = (
        f"This routing decision covers {km:.0f} km of transport, producing "
        f"{carbon:.2f} kg CO\u2082-eq in emissions. "
        f"The projected waste rate is {waste:.4f} ({waste*100:.2f}%), "
        f"with a unit price of ${price:.2f}. "
        f"SLCA composite score: {slca_composite:.3f} "
        f"(Carbon {sc['C']:.2f}, Labor {sc['L']:.2f}, "
        f"Resilience {sc['R']:.2f}, Transparency {sc['P']:.2f}). "
        f"Circular economy score: {circular_score:.3f}. "
        f"Net reward after penalties (energy {energy_penalty:.4f}, "
        f"water {water_penalty:.6f}, waste {waste_penalty:.4f}): "
        f"{reward_total:.4f}."
    )

    # --- paragraph 4: role context ---
    role_text = _ROLE_CONTEXT.get(role_key, _ROLE_CONTEXT["cooperative"])

    # --- optional RAG guidance ---
    rag_text = ""
    guidance = (rag_context or {}).get("regulatory_guidance", "")
    if guidance:
        rag_text = f"\n\nRegulatory guidance: {guidance}"

    return f"{obs}\n\n{rationale}\n\n{impact}\n\n{role_text}{rag_text}"

# ---------------------------------------------------------------------------
# Decisions — regime-aware contextual softmax policy  (Section 5)
# ---------------------------------------------------------------------------
class DecideIn(BaseModel):
    agent_id: str
    role: str
    step: int | None = None          # optional row index (None → last row)
    deterministic: bool = True       # argmax when True, sample when False
    # Mode is validated at request time against ``action_selection.VALID_MODES``
    # (the single source of truth used by the simulator). The previous
    # ``Literal[...]`` only listed the 8 canonical modes -- the §4.7
    # ablation modes (cold_start, pert_*, theta_pert_*, no_bonus) returned
    # 422 from REST even though the simulator runs them happily, which
    # broke the "REST mirrors simulator" claim a paper reviewer would
    # expect. Validating against VALID_MODES instead keeps the two paths
    # in lock-step automatically.
    mode: str = "agribrain"

    @field_validator("mode")
    @classmethod
    def _mode_in_valid_modes(cls, v: str) -> str:
        from src.models.action_selection import VALID_MODES
        if v not in VALID_MODES:
            raise ValueError(
                f"mode={v!r} is not one of {sorted(VALID_MODES)!r}"
            )
        return v


@API.post("/decide")
def decide(d: DecideIn):
    if state["df"] is None:
        case_load()

    p = state["policy"]
    df = state["df"]

    # Pick the observation row
    idx = d.step if d.step is not None else len(df) - 1
    idx = max(0, min(idx, len(df) - 1))
    row = df.iloc[idx]

    # ---- state vector s_t = [rho, I, Y_hat, T, tau] ----------------------
    # The fallbacks for ``inv`` and ``y_hat`` use the simulator's
    # canonical baseline constants (``INV_BASELINE``=12000 units,
    # ``BASELINE_DEMAND``=20 units/step) rather than 100.0. The pre-
    # 2026-05 hardcoded 100.0 made phi_1 = 100/15000 = 0.007 vs the
    # simulator's ~0.83 for the same row, which produced a different
    # action distribution from REST vs the published benchmark.
    from src.models.action_selection import (
        BASELINE_DEMAND as _BASELINE_DEMAND,
        INV_BASELINE as _INV_BASELINE,
    )

    rho = float(row.get("spoilage_risk", 1.0 - row["shelf_left"]))
    inv = float(row.get("inventory_units", _INV_BASELINE))
    temp = float(row["tempC"])

    # Demand and supply forecasts both routed through the MCP tools so
    # this endpoint shares the simulator's forecasting code path.
    demand_fc = _demand_forecast(df.iloc[: idx + 1], horizon=1)
    y_hat = float(demand_fc["forecast"][0]) if demand_fc["forecast"] else _BASELINE_DEMAND
    demand_std = float(demand_fc.get("std", 0.0) or 0.0)
    try:
        from pirag.mcp.tools.yield_query import query_yield
        _inv_hist = df["inventory_units"].iloc[: idx + 1].astype(float).tolist()
        supply_fc = query_yield(inventory_history=_inv_hist, horizon=1)
    except Exception:
        supply_fc = yield_supply_forecast(df.iloc[: idx + 1], horizon=1)
    _supply_forecast_list = supply_fc.get("forecast") if isinstance(supply_fc, dict) else None
    supply_hat = (
        float(_supply_forecast_list[0])
        if _supply_forecast_list
        else None
    )
    supply_std = float(supply_fc.get("std", 0.0) or 0.0) if isinstance(supply_fc, dict) else 0.0

    # Bollinger trigger  (tau = 1 if anomaly)
    tau = 1.0 if str(row.get("volatility", "normal")) == "anomaly" else 0.0

    # Bollinger z-score at this point
    demand_series = df["demand_units"].astype(float).iloc[: idx + 1]
    roll_mean = demand_series.rolling(p.boll_window, min_periods=1).mean()
    roll_std = demand_series.rolling(p.boll_window, min_periods=1).std().fillna(0.0)
    boll_z = float(
        (demand_series.iloc[-1] - roll_mean.iloc[-1])
        / max(float(roll_std.iloc[-1]), 1e-12)
    )
    # price_signal feeds phi_9: clip the Bollinger z-score to [-1, 1]
    # so it lands in the same range build_feature_vector expects.
    price_signal = float(np.clip(boll_z, -1.0, 1.0))

    # ---- role-specific profile ---------------------------------------------
    role_key = (d.role or "").strip().lower()
    profile = ROLE_PROFILES.get(role_key, {})
    role_bias = profile.get("logit_bias", np.zeros(3))
    km_ov = profile.get("km_overrides", {})
    slca_w = profile.get("slca_weights", {})

    # ---- RAG context (best effort) — computed before action selection -----
    # The active scenario flows in from /scenarios/run via the scenarios
    # router's ACTIVE container. Earlier this was hardcoded to "baseline",
    # which kept compliance/regulatory retrieval pointed at the
    # baseline-scenario corpus even after operators selected heatwave /
    # cyber_outage / etc. via the Admin panel -- explainability and
    # decision logits could then disagree about the operating context.
    rag_context = {}
    try:
        from pirag.context_provider import get_policy_context
        try:
            from src.routers.scenarios import ACTIVE as _SCN_ACTIVE
            _scenario_name = (_SCN_ACTIVE.get("name") or "baseline")
        except (ImportError, AttributeError):
            _scenario_name = "baseline"
        rag_context = get_policy_context(
            scenario=_scenario_name, spoilage_risk=rho, temperature=temp,
        )
    except Exception as _exc:
        logger.debug("RAG policy context skipped: %s", _exc)

    # ---- action selection via canonical select_action() -------------------
    rng = np.random.default_rng()
    action_idx, probs = select_action(
        mode=d.mode, rho=rho, inv=inv, y_hat=y_hat, temp=temp,
        tau=tau, policy=p, rng=rng,
        role_bias=role_bias, deterministic=d.deterministic,
        supply_hat=supply_hat, supply_std=supply_std,
        demand_std=demand_std, price_signal=price_signal,
    )

    action = ACTIONS[action_idx]
    km_key = ACTION_KM_KEYS[action]
    km = km_ov.get(km_key, getattr(p, km_key))

    # ---- carbon with COP degradation (Eq. 18) ----------------------------
    # Mode-conditional eff_factor (waste.MODE_CARBON_EFF) scales emissions
    # down for context-aware modes that route through lower-carbon
    # partners and PINN-time dispatches into cooler windows.
    thermal_stress = compute_thermal_stress(temp)
    carbon = compute_transport_carbon(
        km, p.carbon_per_km, thermal_stress,
        eff_factor=MODE_CARBON_EFF.get(d.mode, 1.0),
    )

    # ---- SLCA with stress attenuation (Eq. 12) ---------------------------
    surplus_ratio = max(0.0, inv / INV_BASELINE - 1.0)
    slca_result = slca_score(
        carbon, action,
        w_c=slca_w.get("w_c", p.w_c),
        w_l=slca_w.get("w_l", p.w_l),
        w_r=slca_w.get("w_r", p.w_r),
        w_p=slca_w.get("w_p", p.w_p),
    )
    slca_quality = compute_slca_attenuation(thermal_stress, surplus_ratio)
    slca_composite = slca_result["composite"] * slca_quality

    # ---- footprint -------------------------------------------------------
    fp = compute_footprint(steps=1)

    # ---- waste (full model matching generate_results.py) -----------------
    rh_val = float(row.get("RH", 50.0))
    k_inst = arrhenius_k(temp, p.k_ref, p.Ea_R, p.T_ref_K,
                         rh_val / 100.0, p.beta_humidity)
    waste_raw = compute_waste_rate(k_inst, surplus_ratio)
    save = compute_save_factor(action, d.mode, surplus_ratio)
    waste = float(waste_raw * (1.0 - save))
    price = p.msrp * PRICE_FACTOR.get(action, 1.0)

    # ---- circular economy score ------------------------------------------
    recovery_opts = evaluate_recovery_options(rho, inv, temp)
    circular_score = compute_circular_economy_score(action, recovery_opts)

    # ---- composite reward ------------------------------------------------
    # R = w_c*C + w_l*L + w_r*R + w_p*P  - alpha_E*E - beta_W*W - eta*waste
    energy_penalty = p.alpha_E * fp["energy_J"]
    water_penalty = p.beta_W * fp["water_L"]
    waste_penalty = p.eta * waste
    reward_total = slca_composite - energy_penalty - water_penalty - waste_penalty

    shelf = float(row["shelf_left"])
    vol = str(row.get("volatility", "normal"))

    memo = {
        "time": datetime.now(timezone.utc).isoformat(),
        "ts": int(_time.time()),
        "step": idx,
        "agent": d.agent_id,
        "role": d.role,
        "mode": d.mode,
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
            "bollinger_z": round(boll_z, 4),
        },
        "reward_decomposition": {
            "slca": round(slca_composite, 4),
            "energy_penalty": round(energy_penalty, 6),
            "water_penalty": round(water_penalty, 8),
            "waste_penalty": round(waste_penalty, 4),
            "total": round(reward_total, 4),
        },
        "circular_economy_score": circular_score,
        "recovery_options": recovery_opts,
        "rag_context": {
            "regulatory_guidance": rag_context.get("regulatory_guidance", ""),
            "relevant_sops": rag_context.get("relevant_sops", ""),
            "source_documents": rag_context.get("source_documents", []),
        },
        "demand_forecast": {"method": FORECAST_METHOD, "y_hat": round(y_hat, 4)},
        "yield_forecast": {"y_hat": round(float(supply_fc["forecast"][0]) if supply_fc["forecast"] else 0.0, 4)},
        "note": (
            f"Softmax policy: action={action} "
            f"P={probs[action_idx]:.3f} rho={rho:.3f} tau={tau}"
        ),
        "memo_text": _build_memo_text(
            action=action, role_key=role_key, mode=d.mode,
            rho=rho, inv=inv, temp=temp, y_hat=y_hat,
            shelf=shelf, vol=vol, tau=tau, boll_z=boll_z,
            probs=probs, action_idx=action_idx,
            carbon=carbon, waste=waste, km=km, price=price,
            slca_composite=slca_composite, slca_result=slca_result,
            circular_score=circular_score, reward_total=reward_total,
            energy_penalty=energy_penalty, water_penalty=water_penalty,
            waste_penalty=waste_penalty,
            recovery_opts=recovery_opts, rag_context=rag_context,
        ),
    }

    # tx_hash is set by the phase finaliser (see _finalize_decision_side_effects
    # at module bottom). Initialised here so the explainability enrichment
    # block below can read a consistent value even when the decision is
    # held in the advisory queue or running in monitoring-only mode.
    memo["tx"] = None
    memo["tx_hash"] = None

    # --- Explainability enrichment (best-effort) ---
    try:
        from pirag.context_to_logits import extract_context_features, THETA_CONTEXT
        from pirag.keyword_extractor import extract_keywords_by_type
        from pirag.explain_decision import explain_decision

        class _Obs:
            pass
        _obs = _Obs()
        _obs.rho = rho
        _obs.temp = temp
        _obs.rh = rh_val
        _obs.inv = inv
        _obs.tau = tau
        _obs.hour = float(idx)
        _obs.surplus_ratio = surplus_ratio
        _obs.y_hat = y_hat

        _mcp_res = rag_context.get("mcp_results", {})
        _ctx_mod = rag_context.get("context_modifier")

        _psi = extract_context_features(_mcp_res, rag_context, _obs)
        _modifier = THETA_CONTEXT @ _psi if _ctx_mod is None else np.array(_ctx_mod)

        # Counterfactual (probs without context)
        _, _cf_probs = select_action(
            mode=d.mode, rho=rho, inv=inv, y_hat=y_hat, temp=temp,
            tau=tau, policy=p, rng=np.random.default_rng(),
            role_bias=role_bias, deterministic=d.deterministic,
            supply_hat=supply_hat, supply_std=supply_std,
            demand_std=demand_std, price_signal=price_signal,
        )
        _cf_action = ACTIONS[int(np.argmax(_cf_probs))]

        # Keywords from guidance text
        _kw = {}
        for _gf, _kt in [("regulatory_guidance", "regulatory"), ("relevant_sops", "sop"),
                          ("waste_hierarchy_guidance", "waste_hierarchy")]:
            _txt = rag_context.get(_gf, "")
            if _txt:
                _kw[_kt] = extract_keywords_by_type(_txt)

        _expl = explain_decision(
            action=action, role=role_key, hour=float(idx), obs=_obs,
            mcp_results=_mcp_res, rag_context=rag_context,
            slca_score=slca_composite, carbon_kg=carbon, waste=waste,
            context_features=_psi, logit_adjustment=_modifier,
            action_probs=probs, ablation_action=_cf_action,
            ablation_probs=_cf_probs, keywords=_kw,
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
            "pirag_top_doc": rag_context.get("top_doc_id", ""),
            "pirag_top_score": round(rag_context.get("top_citation_score", 0), 3),
            "keywords": _kw,
            "provenance": {
                "evidence_hashes": rag_context.get("evidence_hashes", [])[:5],
                "guards_passed": rag_context.get("guards_passed", True),
                "guard_breakdown": rag_context.get("guard_breakdown", {}),
                "merkle_root": _expl.get("merkle_root", ""),
            },
            "causal_text": _expl.get("full_explanation", ""),
            # New honest field names + legacy aliases for backward compat.
            "attribution_chain": _expl.get("attribution_chain", {}),
            "ablation_delta": _expl.get("ablation_delta", {}),
            "causal_chain": _expl.get("causal_chain", {}),
            "counterfactual": _expl.get("counterfactual", {}),
            "summary": _expl.get("summary", ""),
        }
    except Exception as _e:
        logger.debug("explainability enrichment skipped: %s", _e)
        memo["explainability"] = None

    # PolicyLearner: record experience for optional online learning.
    # Earlier this stub fed (supply_hat=None, supply_std=None,
    # demand_std=None) into build_feature_vector even though the policy
    # selection upstream had used the real query_yield / query_demand
    # outputs -- so the learner's gradient was being computed against a
    # *different* state vector from the one the action was sampled
    # under. The 2026-05 fix mirrors the same supply_hat / supply_std /
    # demand_std / price_signal that select_action() consumed, so the
    # phi(s) recorded in the buffer matches the phi(s) the policy
    # actually saw.
    if PolicyLearner.is_enabled():
        _learner = state.setdefault("_policy_learner", PolicyLearner())
        from .models.action_selection import build_feature_vector
        phi = build_feature_vector(
            rho, inv, y_hat, temp,
            supply_hat=supply_hat,
            supply_std=supply_std,
            demand_std=demand_std,
            price_signal=price_signal,
        )
        _learner.record(phi, action_idx, reward_total)

    # Apply the deployment-phase semantics:
    #   * autonomous  -> finalise immediately (chain log, broadcast, mirror)
    #   * monitoring  -> compute and return; no side effects
    #   * advisory    -> queue for operator approve / reject; on approve we
    #                    re-run the same finaliser
    memo = _phase.finalize_or_queue_decision(memo, finalize=_finalize_decision_side_effects)

    return {"ok": True, "memo": memo}


def _finalize_decision_side_effects(memo: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the side-effects of an *approved* decision.

    Called immediately for ``autonomous`` mode and after operator
    approval for ``advisory`` mode. Skipped entirely in ``monitoring``
    mode (the recommendation is surfaced but the supply chain is not
    mutated).

    Side-effects:
      * write the memo to the in-memory decision log
      * mirror to ``case.STATE['last_decision']`` for /case/last_decision
      * best-effort on-chain log via DecisionLogger (chain.eth)
      * best-effort websocket broadcast on the ``decision`` channel
    """
    # tx_hash semantics:
    #   None  : chain not configured / submission failed (default)
    #   "0x0" : legacy sentinel from older deployments (treated as
    #           "not anchored" by the frontend's verification logic)
    #   "0x..": real on-chain transaction hash
    #
    # Best-effort posture is governed by CHAIN_BEST_EFFORT (default
    # false outside dev). When false, an on-chain submission failure
    # propagates so operators do not silently believe an anchor
    # happened when it did not. When true, the failure is logged at
    # WARNING with structured fields for monitoring counters and
    # tx remains None.
    tx = None
    chain_best_effort = os.environ.get(
        "CHAIN_BEST_EFFORT",
        "true" if SETTINGS.env == "dev" else "false",
    ).strip().lower() in {"1", "true", "yes", "on"}
    try:
        txh = log_decision_onchain(memo, state.get("chain", {}))
        if txh:
            tx = txh
    except (ConnectionError, TimeoutError, ValueError) as exc:
        # Network / RPC failures and explicit revert ValueErrors. Reraise
        # in strict mode so /decide returns 500 rather than silently
        # dropping the on-chain log.
        logger.warning(
            "chain.log_decision_onchain rpc_failure exception_type=%s exception=%s "
            "best_effort=%s mode=%s",
            type(exc).__name__, exc, chain_best_effort, memo.get("mode"),
        )
        if not chain_best_effort:
            raise
    except RuntimeError as exc:
        # On-chain transaction reverted (raised by log_decision_onchain
        # itself). Same posture: surface in strict mode so the operator
        # cannot mistake "0x0 / null tx_hash" for "the anchor went
        # through and I just missed seeing it".
        logger.warning(
            "chain.log_decision_onchain tx_reverted exception=%s best_effort=%s",
            exc, chain_best_effort,
        )
        if not chain_best_effort:
            raise
    memo["tx"] = tx
    memo["tx_hash"] = tx

    # append to in-memory log
    state.setdefault("log", []).append(memo)

    # mirror last decision into case.STATE so /case/last_decision stays current
    try:
        from src.routers.case import STATE as _case_state
        _case_state["last_decision"] = memo
    except (ImportError, KeyError):
        pass

    # Broadcast to websockets (best effort by intent: a missing event
    # bus must not block the supply-chain decision pipeline). Narrow
    # the catch and log at WARN so operators can see broadcast loss.
    # We catch RuntimeError/OSError (transport-level), ImportError
    # (anyio not installed in some test contexts), and Exception only
    # as a last-resort safety net with type-info preserved in the log.
    try:
        from src.agents.bus import BUS
        import anyio
        anyio.from_thread.run(BUS.emit, "decision", memo)
    except (ImportError, RuntimeError, OSError) as exc:
        logger.warning(
            "decision.broadcast_failed exception_type=%s exception=%s",
            type(exc).__name__, exc,
        )

    return memo


@API.get("/decision/take")
def decision_take(agent: str = "farm", role: str = "farm"):
    return decide(DecideIn(agent_id=agent, role=role))


@API.get("/last-decision")
def last_decision():
    """Return the most recent decision memo."""
    log = state.get("log", [])
    return log[-1] if log else {}


@API.get("/decisions")
def list_decisions():
    """Return recent decision memos (newest first)."""
    return {"decisions": list(reversed(state.get("log", [])[-500:]))}

# ---------------------------------------------------------------------------
# Chain config (local)
# ---------------------------------------------------------------------------
def _redacted_chain_config(chain_cfg: dict) -> dict:
    """Return a copy of the chain config with the signing key redacted.

    ``private_key`` must never cross process or proxy boundaries through
    a response body: even on localhost, logs, reverse proxies, or
    developer screenshots can leak the key. The redacted copy reports
    only whether a key is configured (``"set"`` / ``"unset"``) so
    operators can still verify configuration without exposing the
    secret itself.
    """
    if not isinstance(chain_cfg, dict):
        return {}
    redacted = {k: v for k, v in chain_cfg.items() if k != "private_key"}
    redacted["private_key"] = (
        "set" if chain_cfg.get("private_key") else "unset"
    )
    return redacted


@API.post(
    "/chain/config",
    dependencies=[_Depends(require_scope_api_key("chain"))],
)
def chain_config(cfg: ChainConfig):
    """Update RPC / chain id / addresses; the signing key is env-only.

    The signing key is never accepted in the HTTP request body. We pull
    it from ``CHAIN_PRIVKEY`` once here (not on every decision) so that
    operators see whether it is configured in the response, but the
    flag is redacted to ``set`` / ``unset`` in the response. Submitting
    a body field named ``private_key`` will fail Pydantic validation
    with a 422 because :class:`ChainConfig` declares ``extra="forbid"``.
    """
    chain_cfg = cfg.model_dump()
    # Preserve any previously-loaded private key (from env or a prior
    # process-internal write); the body never mutates the key.
    existing_key = (state.get("chain") or {}).get("private_key")
    env_key = os.environ.get("CHAIN_PRIVKEY", "")
    chain_cfg["private_key"] = existing_key or (env_key or None)
    state["chain"] = chain_cfg
    # Keep governance module config in sync with local chain config endpoint.
    try:
        _gov.CHAIN.update({
            "rpc": chain_cfg.get("rpc"),
            "chain_id": chain_cfg.get("chain_id"),
            "private_key": chain_cfg.get("private_key"),
            "addresses": chain_cfg.get("addresses") or {},
        })
    except Exception as exc:
        logger.debug("governance chain sync skipped: %s", exc)
    # Never echo the signing key back in the HTTP response: redact it to
    # a presence flag instead. The key stays in the in-memory state dict
    # so subsequent on-chain calls work, but no HTTP response body or
    # proxy log ever carries it.
    return {"ok": True, "chain": _redacted_chain_config(state["chain"])}

# ---------------------------------------------------------------------------
# Simple PDF
# ---------------------------------------------------------------------------
@API.get("/report/pdf")
def report_pdf(role: str = ""):
    if state["df"] is None:
        case_load()
    kp = kpis()
    logs = state.get("log") or []
    if role:
        role_logs = [m for m in logs if m.get("role", "") == role]
        last = role_logs[-1] if role_logs else {}
    else:
        last = logs[-1] if logs else {}

    from io import BytesIO
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    )
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("MemoTitle", parent=styles["Title"],
                                  fontSize=18, spaceAfter=4*mm)
    heading_style = ParagraphStyle("SectionHead", parent=styles["Heading2"],
                                    fontSize=13, spaceBefore=6*mm, spaceAfter=3*mm,
                                    textColor=colors.HexColor("#1a5632"))
    body_style = styles["BodyText"]
    small_style = ParagraphStyle("Small", parent=body_style, fontSize=9,
                                  textColor=colors.grey)

    story = []

    # --- Title ---
    story.append(Paragraph("AGRI-BRAIN Decision Memo", title_style))
    ts_str = last.get("time", "N/A")
    agent_str = f"{last.get('agent', 'N/A')} ({last.get('role', 'N/A')})"
    story.append(Paragraph(f"<b>Timestamp:</b> {ts_str} &nbsp;&nbsp; "
                            f"<b>Agent:</b> {agent_str} &nbsp;&nbsp; "
                            f"<b>Mode:</b> {last.get('mode', 'N/A')}", small_style))
    story.append(Spacer(1, 4*mm))

    def _table(headers, rows, col_widths=None):
        data = [headers] + rows
        t = Table(data, colWidths=col_widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5632")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f5f5f5"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return t

    if not last:
        story.append(Paragraph("No decisions recorded yet. Run a decision first.", body_style))
    else:
        # --- Executive Summary (memo_text) ---
        memo_text = last.get("memo_text", "")
        if memo_text:
            # Replace Unicode subscript/superscript chars that ReportLab can't render
            memo_text = memo_text.replace("CO\u2082", "CO<sub>2</sub>")
            memo_text = memo_text.replace("\u2082", "<sub>2</sub>")
            memo_text = memo_text.replace("\u00b2", "<sup>2</sup>")
            story.append(Paragraph("Executive Summary", heading_style))
            for para in memo_text.split("\n\n"):
                story.append(Paragraph(para.strip(), body_style))
                story.append(Spacer(1, 2*mm))

        # --- Current Conditions ---
        story.append(Paragraph("Current Conditions", heading_style))
        regime = last.get("regime", {})
        story.append(_table(
            ["Parameter", "Value"],
            [
                ["Temperature", f"{kp.get('avg_tempC', 'N/A')} \u00b0C"],
                ["Shelf Life Remaining", f"{float(last.get('shelf_left', 0)) * 72:.1f} hours ({float(last.get('shelf_left', 0)):.3f})"],
                ["Spoilage Risk (\u03c1)", f"{last.get('spoilage_risk', 'N/A')}"],
                ["Volatility", f"{last.get('volatility', 'N/A')}"],
                ["Regime Trigger (\u03c4)", f"{regime.get('tau', 'N/A')}"],
                ["Bollinger z-score", f"{regime.get('bollinger_z', 'N/A')}"],
            ],
            col_widths=[55*mm, 80*mm],
        ))

        # --- Decision ---
        story.append(Paragraph("Decision", heading_style))
        ap = last.get("action_probabilities", {})
        story.append(_table(
            ["Field", "Value"],
            [
                ["Selected Action", last.get("action", "N/A")],
                ["P(cold_chain)", f"{ap.get('cold_chain', 'N/A')}"],
                ["P(local_redistribute)", f"{ap.get('local_redistribute', 'N/A')}"],
                ["P(recovery)", f"{ap.get('recovery', 'N/A')}"],
                ["Transport Distance", f"{last.get('km', 'N/A')} km"],
            ],
            col_widths=[55*mm, 80*mm],
        ))

        # --- Impact Assessment ---
        story.append(Paragraph("Impact Assessment", heading_style))
        story.append(_table(
            ["Metric", "Value"],
            [
                ["Carbon Emissions", f"{last.get('carbon_kg', 'N/A')} kg CO2-eq"],
                ["Waste Rate", f"{last.get('waste', 'N/A')}"],
                ["Unit Price", f"${last.get('unit_price', 'N/A')}"],
                ["SLCA Composite", f"{last.get('slca', 'N/A')}"],
                ["Circular Economy Score", f"{last.get('circular_economy_score', 'N/A')}"],
            ],
            col_widths=[55*mm, 80*mm],
        ))

        # --- SLCA Breakdown ---
        sc = last.get("slca_components", {})
        if sc:
            story.append(Paragraph("SLCA Breakdown", heading_style))
            story.append(_table(
                ["Pillar", "Score"],
                [
                    ["Carbon (C)", f"{sc.get('carbon', 'N/A')}"],
                    ["Labor (L)", f"{sc.get('labor', 'N/A')}"],
                    ["Resilience (R)", f"{sc.get('resilience', 'N/A')}"],
                    ["Transparency (P)", f"{sc.get('transparency', 'N/A')}"],
                    ["Composite", f"{sc.get('composite', 'N/A')}"],
                ],
                col_widths=[55*mm, 80*mm],
            ))

        # --- Reward Decomposition ---
        rd = last.get("reward_decomposition", {})
        if rd:
            story.append(Paragraph("Reward Decomposition", heading_style))
            story.append(_table(
                ["Component", "Value"],
                [
                    ["SLCA Reward", f"{rd.get('slca', 'N/A')}"],
                    ["Energy Penalty", f"{rd.get('energy_penalty', 'N/A')}"],
                    ["Water Penalty", f"{rd.get('water_penalty', 'N/A')}"],
                    ["Waste Penalty", f"{rd.get('waste_penalty', 'N/A')}"],
                    ["Net Total", f"{rd.get('total', 'N/A')}"],
                ],
                col_widths=[55*mm, 80*mm],
            ))

        # --- Blockchain ---
        tx = last.get("tx_hash") or last.get("tx", "")
        if tx and tx != "0x0":
            story.append(Paragraph("Blockchain Verification", heading_style))
            story.append(Paragraph(f"Transaction hash: <font face='Courier'>{tx}</font>", body_style))

    # --- KPI Summary ---
    story.append(Paragraph("System KPI Summary", heading_style))
    story.append(_table(
        ["KPI", "Value"],
        [
            ["Records Loaded", str(kp.get("records", 0))],
            ["Avg Temperature", f"{kp.get('avg_tempC', 0):.2f} \u00b0C"],
            ["Anomaly Points", str(kp.get("anomaly_points", 0))],
            ["Waste Rate (Baseline)", f"{kp.get('waste_rate_baseline', 0):.4f}"],
            ["Waste Rate (AGRI-BRAIN)", f"{kp.get('waste_rate_agri', 0):.4f}"],
            ["Mean ARI", f"{kp.get('mean_ari', 0)}"],
            ["Total Carbon", f"{kp.get('total_carbon_kg', 0)} kg"],
        ],
        col_widths=[55*mm, 80*mm],
    ))

    doc.build(story)
    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="application/pdf")

# (agent runtime startup moved to _lifespan context manager above)
