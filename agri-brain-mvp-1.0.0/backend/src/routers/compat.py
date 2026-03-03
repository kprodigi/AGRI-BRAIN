from fastapi import APIRouter, Request
from typing import Any, Dict
from src.routers.decide import decide, DecideRequest

router = APIRouter()

def _coerce(payload: Dict[str, Any] | None) -> DecideRequest:
    d = dict(payload or {})
    return DecideRequest(agent=str(d.get("agent") or "farm"))

async def _payload(req: Request) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if req.method in ("POST","PUT","PATCH"):
        try: data = await req.json()
        except Exception: pass
    for k,v in req.query_params.items(): data.setdefault(k, v)
    return data

@router.api_route("/decision/take",      methods=["GET","POST"])
@router.api_route("/decisions/take",     methods=["GET","POST"])
@router.api_route("/decision",           methods=["GET","POST"])
@router.api_route("/case/decide",        methods=["GET","POST"])
@router.api_route("/api/decision/take",  methods=["GET","POST"])
async def legacy_any(req: Request):
    return decide(_coerce(await _payload(req)))
