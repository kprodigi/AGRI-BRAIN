# backend/src/routers/scenarios.py
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# ---- in-memory active scenario ----
ACTIVE = {"name": None, "intensity": 1.0}

# ---- catalog shown in Admin -> Scenarios ----
SCENARIOS = [
    {"id": "climate_shock",     "label": "Climate-Induced Supply Shock",
     "desc": "72h heatwave; accelerated spoilage; reconfigure routes."},
    {"id": "reverse_logistics", "label": "Reverse Logistics of Spoiled Food",
     "desc": "Glut / overproduction; trigger redistribution and recovery."},
    {"id": "cyber_outage",      "label": "Cyber Threat & Node Outage",
     "desc": "Processor offline; unauthorized tx blocked; reroute flows."},
    {"id": "adaptive_pricing",  "label": "Adaptive Pricing & Cooperative Auctions",
     "desc": "Learned pricing; equity-aware redistribution when saturated."},
]

class RunRequest(BaseModel):
    name: str
    intensity: float | int | None = 1.0

# ---------- NEW API used by the Admin panel ----------
@router.get("/list")
def list_scenarios():
    # shape AdminPanel expects
    return {"scenarios": SCENARIOS, "active": ACTIVE if ACTIVE["name"] else None}

@router.post("/run")
def run_scenario(req: RunRequest):
    ACTIVE["name"] = req.name
    try:
        ACTIVE["intensity"] = float(req.intensity or 1.0)
    except Exception:
        ACTIVE["intensity"] = 1.0
    return {"ok": True, "active": ACTIVE}

@router.post("/reset")
def reset_scenario():
    ACTIVE["name"] = None
    ACTIVE["intensity"] = 1.0
    return {"ok": True, "active": None}

# ---------- LEGACY FALLBACK (old UI calling POST /scenarios) ----------
@router.post("", include_in_schema=False)  # path resolves to /scenarios
def legacy_apply(id: str | None = None, name: str | None = None):
    # keep this extremely simple and avoid NameError:
    chosen = (name or id or "").strip()
    if not chosen:
        return {"ok": False, "error": "missing scenario id"}
    ACTIVE["name"] = chosen
    ACTIVE["intensity"] = 1.0
    return {"ok": True, "active": ACTIVE}
