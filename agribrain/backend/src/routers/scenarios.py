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

from fastapi import APIRouter, HTTPException
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
    raw_intensity = ACTIVE.get("intensity")
    return {
        "name": ACTIVE.get("name"),
        "intensity": float(1.0 if raw_intensity is None else raw_intensity),
    }


# ---- catalog shown in Admin -> Scenarios ----
SCENARIOS = [
    {"id": "baseline",         "label": "Baseline (no perturbation)",
     "desc": "Unperturbed synthetic benchmark series."},
    {"id": "heatwave",         "label": "Synthetic Heatwave",
     "desc": "+20 C exponential approach over hours 24-48 with an exponential tail; +10 percentage-point RH adjustment."},
    {"id": "overproduction",   "label": "Synthetic Overproduction",
     "desc": "Inventory multiplied 2.5x during hours 12-60 with progressive +8°C cold storage excursion."},
    {"id": "cyber_outage",     "label": "Synthetic Cyber-Outage",
     "desc": "From hour 24, demand is multiplied by 0.15 and temperature follows a +10 C exponential excursion; MCP becomes unavailable while processor-stage decisions remain active."},
    {"id": "adaptive_pricing", "label": "Synthetic Adaptive-Pricing Oscillation",
     "desc": "Demand oscillation (amplitude 45, period 60) plus Gaussian noise (standard deviation 14)."},
]


class RunRequest(BaseModel):
    name: str
    intensity: float | int | None = 1.0


# ---------------------------------------------------------------------------
# State application (router-only glue around the pure engine)
# ---------------------------------------------------------------------------

def _apply_to_state(name: str, intensity: float) -> bool:
    """Modify the app DataFrame in-place according to the named scenario."""
    # Fail closed before touching the active DataFrame.  Historically every
    # unknown name fell through to the baseline restoration branch, so a typo
    # could both erase the current perturbation and later be advertised as the
    # active scenario.
    if name != "baseline" and name not in _SCENARIO_FN:
        return False
    if _APP_STATE is None:
        return False

    orig = _APP_STATE.get("df_original")
    if orig is None:
        # Nothing to perturb against and no baseline to restore to.
        return False

    policy = _APP_STATE.get("policy")

    if name == "baseline":
        # Restore the original with derived columns refreshed against the
        # active policy.
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
    name = req.name.strip().lower()
    if name != "baseline" and name not in _SCENARIO_FN:
        raise HTTPException(
            status_code=422,
            detail=f"unknown scenario: {req.name}",
        )
    try:
        requested = 1.0 if req.intensity is None else float(req.intensity)
        intensity, _ = _engine.validate_scenario_controls(requested)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ok = _apply_to_state(name, intensity)
    # Publish the new active state only after name/intensity validation and the
    # attempted application. This preserves the legacy ``ok`` response when no
    # DataFrame has been registered while ensuring invalid names never mutate
    # either ACTIVE or the data.
    ACTIVE["name"] = name
    ACTIVE["intensity"] = intensity
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
    chosen = chosen.lower()
    if chosen != "baseline" and chosen not in _SCENARIO_FN:
        raise HTTPException(status_code=422, detail=f"unknown scenario: {chosen}")
    ok = _apply_to_state(chosen, 1.0)
    ACTIVE["name"] = chosen
    ACTIVE["intensity"] = 1.0
    return {"ok": ok, "active": ACTIVE}
