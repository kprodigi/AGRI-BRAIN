# backend/src/app.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import inspect
import time as _time
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
from src.agents.runtime import start_agent_runtime

# Your models/utilities
from .models.spoilage import compute_spoilage, volatility_flags
from .models.forecast import yield_demand_forecast
from .models.slca import slca_score
from .models.policy import Policy
from .models.footprint import compute_footprint, footprint_meter
from .chain.client import ChainClient

# Static/docs branding
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.responses import RedirectResponse, FileResponse, HTMLResponse

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
API = FastAPI(title="AGRI BRAIN MVP API")
#API = FastAPI(
#    title="AGRI BRAIN MVP API",
#    docs_url=None,      # <- disable default /docs
#    redoc_url=None      # <- disable default /redoc
#)


# CORS (register once)
API.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Mount routers (each once)
# ---------------------------------------------------------------------------
API.include_router(_case.router,        prefix="/case",       tags=["case"])
API.include_router(_audit.router,       prefix="/audit",      tags=["audit"])
API.include_router(_gov.router,         prefix="/governance", tags=["governance"])
API.include_router(_scn.router,         prefix="/scenarios",  tags=["scenarios"])
API.include_router(_compat.router,                          tags=["compat"])
API.include_router(_debug.router,                           tags=["debug"])
API.include_router(_stream.router)  # no prefix => /stream (websocket)

API.include_router(_results.router,    prefix="/results",    tags=["results"])

API.include_router(rag_router, prefix="/rag", tags=["pirag"])
API.include_router(mcp_router, prefix="/mcp", tags=["mcp"])

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
DATA = Path(__file__).parent / "data_spinach.csv"
state: Dict[str, Any] = {
    "df": None,
    "df_original": None,        # pristine copy for scenario reset
    "policy": Policy(),
    "chain": {"rpc": None, "addresses": {}, "chain_id": 31337, "private_key": None},
}

# ---------------------------------------------------------------------------
# Softmax policy constants (Section 5 — contextual regime-aware policy)
# ---------------------------------------------------------------------------
ACTIONS = ["cold_chain", "local_redistribute", "recovery"]
ACTION_KM_KEYS = {"cold_chain": "km_coldchain",
                  "local_redistribute": "km_local",
                  "recovery": "km_recovery"}
PRICE_FACTOR = {"cold_chain": 1.0, "local_redistribute": 0.95, "recovery": 0.88}

# theta matrix  (3 actions x 6 features)
THETA = np.array([
    [ 1.0, -0.5,  0.3, -0.8, -2.0, -1.0],   # ColdChain
    [-0.3,  1.2, -0.2,  0.3,  1.5,  2.0],   # LocalRedist
    [-0.8, -0.3, -0.3,  0.8,  1.8,  0.5],   # Recovery
])

class ChainConfig(BaseModel):
    rpc: Optional[str] = None
    chain_id: int = 31337
    private_key: Optional[str] = None
    addresses: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@API.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}

# ---------------------------------------------------------------------------
# Startup: load CSV (sync) — you can have multiple startup handlers
# ---------------------------------------------------------------------------
@API.on_event("startup")
def _warm_case() -> None:
    try:
        case_load()
        print("[startup] spinach CSV loaded")
    except Exception as e:
        print("[startup] case_load skipped:", e)
    # Let the scenarios router access our state for data modifications
    try:
        _scn.register_app_state(state)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Data endpoints
# ---------------------------------------------------------------------------
@API.post("/case/load")
def case_load():
    p = state["policy"]
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = compute_spoilage(df, k0=p.k0, alpha=p.alpha_decay, T0=p.T0, beta=p.beta_humidity)
    df["volatility"] = volatility_flags(df, window=p.boll_window, k=p.boll_k)
    state["df"] = df
    state["df_original"] = df.copy()
    return {"ok": True, "records": len(df)}

@API.get("/kpis")
def kpis():
    if state["df"] is None:
        case_load()
    df = state["df"]
    waste_baseline = float((df["shelf_left"] < 0.0).sum() / len(df))
    waste_agri = float((df["shelf_left"] < 0.0).rolling(4).max().fillna(0).mean() * 0.6)

    # --- Extended KPIs from decision logs ---------------------------------
    logs = state.get("log", [])
    ari_vals: list[float] = []
    rle_at_risk = 0
    rle_routed = 0
    slca_vals: list[float] = []
    total_carbon = 0.0

    for m in logs:
        sc = m.get("slca_components") or {}
        slca_c = sc.get("composite", m.get("slca", 0.0))
        slca_vals.append(slca_c)

        rho = m.get("spoilage_risk", 1.0 - m.get("shelf_left", 1.0))
        waste = rho
        ari_vals.append((1.0 - waste) * slca_c * (1.0 - rho))

        total_carbon += m.get("carbon_kg", 0.0)

        # RLE: at-risk = rho > 0.3  (shelf_left < 0.7)
        if rho > 0.3:
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
    yf = yield_demand_forecast(df, horizon=24)
    return {
        "timestamp": df["timestamp"].astype(str).tolist(),
        "shelf_left": df["shelf_left"].round(4).tolist(),
        "spoilage_risk": df["spoilage_risk"].round(4).tolist() if "spoilage_risk" in df.columns else [],
        "volatility": df["volatility"].tolist(),
        "yield_forecast_24h": yf["forecast"],
        "yield_forecast_ci_lower": yf["ci_lower"],
        "yield_forecast_ci_upper": yf["ci_upper"],
        "yield_forecast_std": yf["std"],
    }

# ---------------------------------------------------------------------------
# Policy (local)
# ---------------------------------------------------------------------------
@API.get("/policy")
def get_policy():
    return state["policy"].model_dump()

@API.post("/policy")
def set_policy(p: Policy):
    state["policy"] = p
    return {"ok": True, "policy": p.model_dump()}

# ---------------------------------------------------------------------------
# Decisions — regime-aware contextual softmax policy  (Section 5)
# ---------------------------------------------------------------------------
class DecideIn(BaseModel):
    agent_id: str
    role: str
    step: int | None = None          # optional row index (None → last row)
    deterministic: bool = True       # argmax when True, sample when False


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


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
    rho = float(row.get("spoilage_risk", 1.0 - row["shelf_left"]))
    inv = float(row.get("inventory_units", 100.0))
    temp = float(row["tempC"])

    # Yield forecast (single-step)
    yf = yield_demand_forecast(df.iloc[: idx + 1], horizon=1)
    y_hat = float(yf["forecast"][0]) if yf["forecast"] else 100.0

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

    # ---- feature vector  phi(s_t) ----------------------------------------
    inv_norm = min(inv / 1000.0, 1.0)
    yhat_norm = min(y_hat / 1000.0, 1.0)
    temp_norm = min(max(temp / 40.0, 0.0), 1.0)

    phi = np.array([1.0 - rho, inv_norm, yhat_norm, temp_norm, rho, rho * inv_norm])

    # ---- logits  l_a = theta_a @ phi + gamma_a * tau ---------------------
    gamma = np.array([p.gamma_coldchain, p.gamma_local, p.gamma_recovery])
    logits = THETA @ phi + gamma * tau

    probs = _softmax(logits)

    # ---- action selection ------------------------------------------------
    if d.deterministic:
        action_idx = int(np.argmax(probs))
    else:
        action_idx = int(np.random.choice(len(ACTIONS), p=probs))

    action = ACTIONS[action_idx]
    km = getattr(p, ACTION_KM_KEYS[action])

    # ---- carbon & SLCA ---------------------------------------------------
    carbon = km * p.carbon_per_km
    slca_result = slca_score(
        carbon, action,
        w_c=p.w_c, w_l=p.w_l, w_r=p.w_r, w_p=p.w_p,
    )
    slca_composite = slca_result["composite"]

    # ---- footprint -------------------------------------------------------
    fp = compute_footprint(steps=1)

    # ---- waste & price ---------------------------------------------------
    waste = rho
    price = p.msrp * PRICE_FACTOR.get(action, 1.0)

    # ---- composite reward ------------------------------------------------
    # R = w_c*C + w_l*L + w_r*R + w_p*P  - alpha_E*E - beta_W*W - eta*waste
    energy_penalty = p.alpha_E * fp["energy_J"]
    water_penalty = p.beta_W * fp["water_L"]
    waste_penalty = p.eta * waste
    reward_total = slca_composite - energy_penalty - water_penalty - waste_penalty

    shelf = float(row["shelf_left"])
    vol = str(row.get("volatility", "normal"))

    memo = {
        "time": datetime.utcnow().isoformat(),
        "ts": int(_time.time()),
        "step": idx,
        "agent": d.agent_id,
        "role": d.role,
        "decision": action,
        "action": action,
        "shelf_left": round(shelf, 4),
        "spoilage_risk": round(rho, 4),
        "volatility": vol,
        "km": round(km, 2),
        "carbon_kg": round(carbon, 4),
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
            "bollinger_z": round(boll_z, 4),
        },
        "reward_decomposition": {
            "slca": round(slca_composite, 4),
            "energy_penalty": round(energy_penalty, 6),
            "water_penalty": round(water_penalty, 8),
            "waste_penalty": round(waste_penalty, 4),
            "total": round(reward_total, 4),
        },
        "note": (
            f"Softmax policy: action={action} "
            f"P={probs[action_idx]:.3f} rho={rho:.3f} tau={tau}"
        ),
    }

    # best-effort on-chain log (never break)
    tx = "0x0"
    try:
        chain = ChainClient(**state["chain"])
        tx = chain.log_decision(d.agent_id, action, int(slca_composite * 1e6), "")
    except Exception:
        pass
    memo["tx"] = tx
    memo["tx_hash"] = tx

    # append to in-memory log
    state.setdefault("log", []).append(memo)

    # broadcast to websockets (best effort)
    try:
        from src.agents.bus import BUS
        import anyio
        anyio.from_thread.run(BUS.emit, "decision", memo)
    except Exception:
        pass

    return {"ok": True, "memo": memo}


@API.get("/decision/take")
def decision_take(agent: str = "farm", role: str = ""):
    return decide(DecideIn(agent_id=agent, role=role))


@API.get("/decisions")
def list_decisions():
    return {"decisions": state.get("log", [])[-500:]}

# ---------------------------------------------------------------------------
# Chain config (local)
# ---------------------------------------------------------------------------
@API.post("/chain/config")
def chain_config(cfg: ChainConfig):
    state["chain"] = cfg.model_dump()
    return {"ok": True, "chain": state["chain"]}

# ---------------------------------------------------------------------------
# Simple PDF
# ---------------------------------------------------------------------------
@API.get("/report/pdf")
def report_pdf():
    if state["df"] is None:
        case_load()
    kp = kpis()
    last = (state.get("log") or [{}])[-1] if state.get("log") else {}

    from io import BytesIO
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 20 * mm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, y, "AGRI BRAIN Spinach — Decision Memo")
    y -= 10 * mm

    c.setFont("Helvetica", 10)
    for k in ("records", "avg_tempC", "anomaly_points", "waste_rate_baseline", "waste_rate_agri"):
        c.drawString(20 * mm, y, f"{k}: {kp.get(k)}"); y -= 6 * mm

    y -= 4 * mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "Last Decision")
    y -= 7 * mm
    c.setFont("Helvetica", 10)
    for k in ("time","agent","role","decision","shelf_left","volatility","km","carbon_kg","unit_price","slca","tx","note"):
        c.drawString(20 * mm, y, f"{k}: {last.get(k, '')}"); y -= 6 * mm
        if y < 20 * mm:
            c.showPage(); y = h - 20 * mm

    c.showPage(); c.save()

    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="application/pdf")

# ---------------------------------------------------------------------------
# Startup: agent runtime (async)
# ---------------------------------------------------------------------------
@API.on_event("startup")
async def _start_agentic_runtime():
    try:
        sig = inspect.signature(start_agent_runtime)
        if len(sig.parameters) == 0:
            await start_agent_runtime()
        else:
            await start_agent_runtime(API)
    except Exception as e:
        print(f"[startup] agent runtime skipped: {e}")
