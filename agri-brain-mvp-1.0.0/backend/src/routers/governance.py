# backend/src/routers/governance.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import os, json

router = APIRouter()

# ---------------- In-memory stores (safe defaults) ----------------
POLICY: Dict[str, Any] = {
    "min_shelf_reroute": 0.70,
    "min_shelf_expedite": 0.50,
    "carbon_transport": 0.12,
    "carbon_cold_chain": 0.08,
}

# CHAIN keeps on-chain connection + deployed addresses
CHAIN: Dict[str, Any] = {
    "rpc": "http://127.0.0.1:8545",
    "chain_id": 31337,
    "private_key": "",   # optional in MVP
    "addresses": {},     # auto-filled if 'auto' below is True
    "auto": True,
}

# Track last autoload mtime so we only parse when the file changes
_AUTO_STATE = {"last_mtime": 0.0}

# ---------------- Auto-sync helper ----------------
def _auto_base_dir() -> Path:
    """
    Where deploy.js writes addresses by default:
      backend/runtime/chain/
    Allow override with env CHAIN_DIR.
    """
    env_dir = os.environ.get("CHAIN_DIR")
    if env_dir:
        return Path(env_dir)
    # governance.py is backend/src/routers/governance.py
    # parents[2] -> backend/
    return Path(__file__).resolve().parents[2] / "runtime" / "chain"

def _auto_paths():
    base = _auto_base_dir()
    return [
        base / "deployed-addresses.latest.json",
        base / "deployed-addresses.localhost.json",
    ]

def _try_autoload():
    """If auto is enabled, read addresses JSON when it changes."""
    if not CHAIN.get("auto", True):
        return
    for p in _auto_paths():
        if p.exists():
            mt = p.stat().st_mtime
            if mt > _AUTO_STATE["last_mtime"]:
                try:
                    raw = json.loads(p.read_text())
                    addrs = raw.get("addresses") if isinstance(raw, dict) else None
                    CHAIN["addresses"] = addrs if addrs else raw
                    _AUTO_STATE["last_mtime"] = mt
                    CHAIN.setdefault("rpc", "http://127.0.0.1:8545")
                    CHAIN.setdefault("chain_id", 31337)
                    print(f"[governance] auto-synced addresses from {p}")
                    break
                except Exception as e:
                    print("[governance] auto-load failed:", e)

# ---------------- Models ----------------
class PolicyModel(BaseModel):
    min_shelf_reroute: float
    min_shelf_expedite: float
    carbon_transport: float
    carbon_cold_chain: float

class ChainModel(BaseModel):
    rpc: Optional[str] = None
    chain_id: Optional[int] = None
    private_key: Optional[str] = None
    addresses: Optional[dict] = None          # dict variant
    addresses_json: Optional[str] = None       # string variant used by UI
    auto: Optional[bool] = None

# ---------------- Policy endpoints ----------------
@router.get("/policy")
def get_policy():
    return POLICY

@router.post("/policy")
def set_policy(payload: Dict[str, Any]):
    """
    Accept both the flat shape and legacy:
      { carbon_factors: { transport, cold_chain }, ... }
    """
    data = dict(POLICY)
    data.update(payload or {})
    cf = (payload or {}).get("carbon_factors")
    if isinstance(cf, dict):
        if "transport" in cf:
            data["carbon_transport"] = float(cf["transport"])
        if "cold_chain" in cf:
            data["carbon_cold_chain"] = float(cf["cold_chain"])
    POLICY.update(
        PolicyModel(**{
            "min_shelf_reroute": float(data.get("min_shelf_reroute", 0.70)),
            "min_shelf_expedite": float(data.get("min_shelf_expedite", 0.50)),
            "carbon_transport": float(data.get("carbon_transport", 0.12)),
            "carbon_cold_chain": float(data.get("carbon_cold_chain", 0.08)),
        }).model_dump()
    )
    return POLICY

# ---------------- Chain endpoints ----------------
@router.get("/chain")
def get_chain():
    _try_autoload()
    addrs = CHAIN.get("addresses") or {}
    addresses_json = json.dumps(addrs, indent=2, ensure_ascii=False)
    return {
        "rpc": CHAIN.get("rpc"),
        "chain_id": CHAIN.get("chain_id"),
        "private_key": CHAIN.get("private_key") or "",
        "auto": CHAIN.get("auto", True),
        "addresses": addrs,
        "addresses_json": addresses_json,
    }

@router.post("/chain")
def set_chain(c: ChainModel):
    data = {k: v for k, v in (c.model_dump() if hasattr(c, "model_dump") else c.dict()).items() if v is not None}
    # Allow posting either a dict or a string JSON
    if "addresses_json" in data and isinstance(data["addresses_json"], str):
        try:
            data["addresses"] = json.loads(data["addresses_json"])
        except Exception:
            pass
    for k in ("rpc", "chain_id", "private_key", "auto"):
        if k in data:
            CHAIN[k] = data[k]
    if "addresses" in data and isinstance(data["addresses"], dict):
        CHAIN["addresses"] = data["addresses"]
    return get_chain()
