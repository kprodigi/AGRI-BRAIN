# backend/src/agents/runtime.py
import asyncio
from typing import Optional
from fastapi import FastAPI

from src.agents.bus import BUS

# ---- Helpers for chain polling (JSON-RPC, no extra deps) ----
import json, urllib.request

def _rpc_numeric_hex(rpc_url: str, method: str, params=None) -> Optional[int]:
    if not rpc_url:
        return None
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []}).encode()
    req = urllib.request.Request(rpc_url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as resp:
        data = json.loads(resp.read().decode())
    if "error" in data:
        return None
    val = data.get("result")
    if isinstance(val, str) and val.startswith("0x"):
        return int(val, 16)
    return None

async def _chain_head_loop():
    """
    Poll the RPC head and broadcast block number changes.
    """
    from src.routers.governance import CHAIN as CHAIN_CFG  # runtime read
    last = None
    while True:
        try:
            rpc_url = (CHAIN_CFG or {}).get("rpc")
            head = _rpc_numeric_hex(rpc_url, "eth_blockNumber") if rpc_url else None
            if head is not None and head != last:
                last = head
                await BUS.emit("chain/head", {"block": head})
        except Exception as e:
            await BUS.emit("chain/error", {"error": str(e)})
        await asyncio.sleep(2.0)

async def _policy_watch_loop():
    """
    Simple agentic loop:
    - Watch case KPIs
    - If waste_agri exceeds baseline by >2%, auto-take a decision
    Broadcast the memo so the UI updates immediately.
    """
    while True:
        try:
            # pull KPIs from the case state (or compute via route)
            try:
                from src.routers.case import STATE as _STATE
                metrics = (_STATE.get("metrics") or {})
            except Exception:
                metrics = {}
            wa = float(metrics.get("waste_rate_agri") or 0.0)
            wb = float(metrics.get("waste_rate_baseline") or 0.0)

            if (wa - wb) > 0.02:
                # Use your existing decision logic; call the same FastAPI function
                from src.app import decide, DecideIn  # your working app.py
                rsp = decide(DecideIn(agent_id="agentic:watcher", role="farm"))
                memo = (rsp or {}).get("memo", rsp) or {}

                # push to all clients
                await BUS.emit("decision", memo)

            # back off a bit (keeps CPU low but still feels reactive)
            await asyncio.sleep(15)
        except Exception as e:
            await BUS.emit("agent/error", {"error": str(e)})
            await asyncio.sleep(15)

async def start_agent_runtime():
    """
    Start background tasks on FastAPI startup.
    """
    asyncio.create_task(_chain_head_loop())
    asyncio.create_task(_policy_watch_loop())


async def start_agent_runtime(app: FastAPI) -> None:
    """
    Start any background agentic tasks.
    Currently a no-op so the app can start without extra deps.
    """
    return