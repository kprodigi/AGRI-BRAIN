# backend/src/routers/scenarios.py
"""Scenario HTTP layer.

Pure perturbation functions live in :mod:`src.models.scenario_engine`.
This router holds the *active scenario* container and the small bit of
state-mutation glue that makes the live FastAPI app respond to scenario
selections from the Admin panel. The simulator imports the engine
directly, not this router, so the simulator does not depend on
HTTP-coupled state.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, Optional

from src.models import scenario_engine as _engine

# Re-export the canonical perturbation functions and helpers under their
# legacy underscore names so that earlier callers
# (``from src.routers.scenarios import _apply_heatwave``) keep working.
# New code should import ``src.models.scenario_engine`` directly.
from src.models.scenario_engine import (
    _apply_heatwave as _apply_heatwave,
    _apply_overproduction as _apply_overproduction,
    _apply_cyber_outage as _apply_cyber_outage,
    _apply_adaptive_pricing as _apply_adaptive_pricing,
    _hours_from_start as _hours_from_start,
    _recompute_derived as _recompute_derived,
    SCENARIO_FUNCTIONS as _SCENARIO_FN,
)

router = APIRouter()

# ---- in-memory active scenario ----
ACTIVE: Dict[str, Any] = {"name": None, "intensity": 1.0}

# ---- reference to the app-level state dict (set by app.py at startup) ----
_APP_STATE: Optional[Dict[str, Any]] = None


def register_app_state(st: Dict[str, Any]) -> None:
    """Called once by app.py at startup so scenarios can modify the DataFrame."""
    global _APP_STATE
    _APP_STATE = st


def get_active_scenario() -> Dict[str, Any]:
    """Return a snapshot of the active scenario for downstream consumers.

    Decision-time callers (``/decide``, the standalone fallback in
    :mod:`src.routers.decide`, the policy-context retriever) read this
    via :data:`ACTIVE` directly; this helper exists so test code and
    routers outside this module do not have to reach into the global
    container's keys to format a {"name", "intensity"} pair.
    """
    return {"name": ACTIVE.get("name"), "intensity": float(ACTIVE.get("intensity") or 1.0)}


# ---- catalog shown in Admin -> Scenarios ----
SCENARIOS = [
    {"id": "baseline",         "label": "Baseline (no perturbation)",
     "desc": "Original sensor data with no modifications."},
    {"id": "heatwave",         "label": "Climate-Induced Heatwave",
     "desc": "72 h heatwave: +20 C sigmoid onset (hours 24-48) with exponential tail; +10 % RH."},
    {"id": "overproduction",   "label": "Overproduction / Glut",
     "desc": "Inventory multiplied 2.5x during hours 12-60 with progressive +8°C cold storage excursion."},
    {"id": "cyber_outage",     "label": "Cyber Threat & Node Outage",
     "desc": "Processor offline from hour 24: demand drops to 15 %, inventory accumulates, +10 C refrigeration degradation."},
    {"id": "adaptive_pricing", "label": "Adaptive Pricing & Demand Oscillation",
     "desc": "Demand oscillation (amp 45, period 60) plus Gaussian noise (std 14)."},
]


class RunRequest(BaseModel):
    name: str
    intensity: float | int | None = 1.0


# ---------------------------------------------------------------------------
# State application (router-only glue around the pure engine)
# ---------------------------------------------------------------------------

def _apply_to_state(name: str, intensity: float) -> bool:
    """Modify the app DataFrame in-place according to the named scenario."""
    if _APP_STATE is None:
        return False

    orig = _APP_STATE.get("df_original")
    if orig is None:
        # Nothing to perturb against and no baseline to restore to.
        return False

    policy = _APP_STATE.get("policy")

    if name not in _SCENARIO_FN:
        # baseline or unknown -> restore original (with derived columns
        # refreshed against the active policy).
        _APP_STATE["df"] = _engine.recompute_derived(orig.copy(), policy)
        return True

    _APP_STATE["df"] = _engine.apply(name, orig, policy=policy, intensity=intensity)
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
class LegacyApplyBody(BaseModel):
    id: str | None = None
    name: str | None = None

@router.post("", include_in_schema=False)
def legacy_apply(body: LegacyApplyBody | None = None,
                 id: str | None = None, name: str | None = None):
    # Accept both JSON body and query params
    bid = getattr(body, "id", None) or getattr(body, "name", None) if body else None
    chosen = (name or id or bid or "").strip()
    if not chosen:
        return {"ok": False, "error": "missing scenario id"}
    ACTIVE["name"] = chosen
    ACTIVE["intensity"] = 1.0
    _apply_to_state(chosen, 1.0)
    return {"ok": True, "active": ACTIVE}
