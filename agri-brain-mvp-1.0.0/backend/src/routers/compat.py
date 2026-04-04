import json
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Any, Dict, Optional
from src.routers.decide import decide, DecideRequest

router = APIRouter()

def _coerce(payload: Dict[str, Any] | None) -> DecideRequest:
    d = dict(payload or {})
    return DecideRequest(agent=str(d.get("agent") or "farm"))

async def _payload(req: Request) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if req.method in ("POST","PUT","PATCH"):
        try: data = await req.json()
        except (json.JSONDecodeError, ValueError): pass
    for k,v in req.query_params.items(): data.setdefault(k, v)
    return data

@router.api_route("/decision/take",      methods=["GET","POST"])
@router.api_route("/decisions/take",     methods=["GET","POST"])
@router.api_route("/decision",           methods=["GET","POST"])
@router.api_route("/case/decide",        methods=["GET","POST"])
@router.api_route("/api/decision/take",  methods=["GET","POST"])
async def legacy_any(req: Request):
    return decide(_coerce(await _payload(req)))


# ---------------------------------------------------------------------------
# /sim/validate — feasibility guard endpoint (piRAG feasibility_guard.py)
# ---------------------------------------------------------------------------
class SimValidateRequest(BaseModel):
    answer: str = ""
    context: Optional[Dict[str, Any]] = None

@router.post("/sim/validate")
def sim_validate(req: SimValidateRequest):
    """Basic feasibility check used by the piRAG feasibility guard.

    Returns feasible=true unless the answer contains obviously out-of-range
    numeric values relative to the context constraints.
    """
    from pirag.guards.feasibility_guard import within_ranges
    constraints = (req.context or {}).get("constraints", {})
    feasible = within_ranges(req.answer, constraints)
    return {"feasible": feasible}
