
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from .tools import calculator, units, simulator, policy_oracle

router = APIRouter()

TOOLS = {
    "calculator": {"fn": calculator.calculate, "schema": {"expr": "str"}},
    "convert_units": {"fn": units.convert, "schema": {"value": "float", "from_unit": "str", "to_unit": "str"}},
    "simulate": {"fn": simulator.simulate, "schema": {"endpoint": "str", "payload": "dict"}},
    "policy_check": {"fn": policy_oracle.check_access, "schema": {"user_id": "str", "tool_name": "str"}},
}

class CallReq(BaseModel):
    name: str
    args: Dict[str, Any]

@router.get("/tools")
def list_tools():
    return {"tools": [{"name": k, "schema": v["schema"]} for k,v in TOOLS.items()]}

@router.post("/call")
def call_tool(req: CallReq):
    if req.name not in TOOLS:
        raise HTTPException(404, f"Unknown tool: {req.name}")
    try:
        fn = TOOLS[req.name]["fn"]
        out = fn(**req.args)
        return {"ok": True, "result": out}
    except Exception as e:
        raise HTTPException(400, f"Tool error: {e}")
