# backend/src/app.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import inspect
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
from src.agents.runtime import start_agent_runtime

# Your models/utilities
from .models.spoilage import compute_spoilage, volatility_flags
from .models.forecast import yield_demand_forecast
from .models.slca import slca_score
from .models.policy import Policy
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
    "policy": Policy(),
    "chain": {"rpc": None, "addresses": {}, "chain_id": 31337, "private_key": None},
}

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

# ---------------------------------------------------------------------------
# Data endpoints
# ---------------------------------------------------------------------------
@API.post("/case/load")
def case_load():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = compute_spoilage(df)
    df["volatility"] = volatility_flags(df)
    state["df"] = df
    return {"ok": True, "records": len(df)}

@API.get("/kpis")
def kpis():
    if state["df"] is None:
        case_load()
    df = state["df"]
    waste_baseline = float((df["shelf_left"] < 0.0).sum() / len(df))
    waste_agri = float((df["shelf_left"] < 0.0).rolling(4).max().fillna(0).mean() * 0.6)
    return {
        "records": len(df),
        "avg_tempC": float(df["tempC"].mean()),
        "anomaly_points": int((df["volatility"] == "anomaly").sum()),
        "waste_rate_baseline": waste_baseline,
        "waste_rate_agri": waste_agri,
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
        "volatility": df["volatility"].tolist(),
        "yield_forecast_24h": yf
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
# Decisions (local)
# ---------------------------------------------------------------------------
class DecideIn(BaseModel):
    agent_id: str
    role: str

@API.post("/decide")
def decide(d: DecideIn):
    if state["df"] is None:
        case_load()

    p = state["policy"]
    df = state["df"]
    row = df.iloc[-1]
    shelf = float(row["shelf_left"])
    vol = str(row["volatility"])

    if shelf < p.min_shelf_expedite:
        action = "expedite_to_retail"; km = p.km_expedited; price = p.msrp * 0.92
    elif shelf < p.min_shelf_reroute or vol == "anomaly":
        action = "reroute_to_near_dc"; km = p.km_farm_to_dc * 0.6; price = p.msrp * 0.95
    else:
        action = "standard_cold_chain"; km = p.km_farm_to_dc + p.km_dc_to_retail; price = p.msrp

    carbon = km * p.carbon_per_km
    slca = slca_score(carbon)

    import time
    memo = {
        "time": datetime.utcnow().isoformat(),
        "ts": int(time.time()),
        "agent": d.agent_id,
        "role": d.role,
        "decision": action,
        "action": action,
        "shelf_left": round(shelf, 3),
        "volatility": vol,
        "km": km,
        "carbon_kg": round(carbon, 2),
        "unit_price": round(price, 2),
        "slca": round(slca, 3),
        "slca_score": round(slca, 3),
        "note": f"Decision={action} because shelf_left={shelf:.2f} and volatility={vol}.",
    }

    # best-effort on-chain log (never break)
    tx = "0x0"
    try:
        chain = ChainClient(**state["chain"])
        tx = chain.log_decision(d.agent_id, action, int(slca * 1e6), "")
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
